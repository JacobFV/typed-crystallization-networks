import { randomUUID } from './determinism.js';
import { spawn } from 'node:child_process';
import path from 'node:path';
import type {
  AppExecutionRecord,
  AppLaunchRequest,
  ComputerSpec,
  HostExecutionResult,
  HostExecutionRule,
  InstalledApp,
} from '@tcn-computer/protocol';
import type { InternetFabric } from './network.js';
import type { ProcessManager } from './processes.js';
import type { SoftwareEnvironment } from './software.js';
import type { VirtualFileSystem } from './vfs.js';
import { PermissionManager, type SeedPermission } from './permissions.js';
import { SyscallRouter, type SyscallAuditEntry, type SyscallHandler } from './syscalls.js';
import { runSandboxedBundle, SANDBOX_DEFAULT_TIMEOUT_MS } from './sandbox.js';

export interface ApplicationRuntimeDependencies {
  spec: ComputerSpec;
  vfs: VirtualFileSystem;
  processes: ProcessManager;
  network: InternetFabric;
  software: SoftwareEnvironment;
  executeShell(command: string): Promise<unknown>;
  installedApp(appId: string): InstalledApp | undefined;
  serviceOperation(app: InstalledApp, request: AppLaunchRequest): Promise<unknown>;
  /** Home directory of the computer's user, from the OS profile. Scopes application filesystem grants. */
  homePath?: string;
  /** Budget for a bundle's own execution; the sandbox clock pauses during kernel syscalls. */
  sandboxTimeoutMs?: number;
  /** Absolute wall-clock ceiling for one application execution. */
  sandboxHardTimeoutMs?: number;
}

export function seedJavaScriptBundle(): string {
  return `'use strict';
module.exports = async function run(seed, request) {
  if (!request || typeof request.operation !== 'string') throw new Error('operation is required');
  return seed.dispatch(request.operation, request.payload || {});
};
`;
}

function calculate(expression: string): number {
  const tokens = expression.match(/\d+(?:\.\d+)?|[()+\-*/]/g) ?? [];
  if (tokens.join('') !== expression.replace(/\s+/g, '')) throw new Error('unsupported calculator expression');
  const output: Array<number | string> = [];
  const operators: string[] = [];
  const precedence: Record<string, number> = { '+': 1, '-': 1, '*': 2, '/': 2 };
  for (const token of tokens) {
    if (/^\d/.test(token)) output.push(Number(token));
    else if (token === '(') operators.push(token);
    else if (token === ')') {
      while (operators.length && operators.at(-1) !== '(') output.push(operators.pop()!);
      if (operators.pop() !== '(') throw new Error('unbalanced expression');
    } else {
      while (operators.length && operators.at(-1) !== '(' && precedence[operators.at(-1)!]! >= precedence[token]!) output.push(operators.pop()!);
      operators.push(token);
    }
  }
  while (operators.length) {
    const operator = operators.pop()!;
    if (operator === '(') throw new Error('unbalanced expression');
    output.push(operator);
  }
  const values: number[] = [];
  for (const token of output) {
    if (typeof token === 'number') values.push(token);
    else {
      const right = values.pop(); const left = values.pop();
      if (left === undefined || right === undefined) throw new Error('invalid expression');
      values.push(token === '+' ? left + right : token === '-' ? left - right : token === '*' ? left * right : left / right);
    }
  }
  if (values.length !== 1 || !Number.isFinite(values[0])) throw new Error('invalid calculation');
  return values[0]!;
}

const GIT_OPERATIONS = ['status', 'stage', 'commit', 'branch', 'checkout', 'switch', 'fetch', 'pull', 'push'];
const PACKAGE_OPERATIONS = ['list', 'search', 'inspect', 'install', 'upgrade', 'remove'];

/**
 * Derives a default-deny security context from the installed manifest. Capabilities decide which
 * syscalls exist for the process at all; the filesystem grants scope *where* those syscalls may act.
 */
export function applicationPermissions(app: InstalledApp, spec: ComputerSpec, homePath?: string): SeedPermission {
  // The OS profile owns the real home path; without it, grant the conventional roots for the family.
  const homes = homePath ? [homePath] : spec.os === 'windows' ? ['/C/Users/agent'] : ['/Users/agent', '/home/agent'];
  const allowedSyscalls = ['proc.self', 'app.event', 'compute.calculate'];
  const fsRead = [`${app.installPath}/**`, `${app.dataPath}/**`];
  const fsWrite = [`${app.dataPath}/**`];
  const network: string[] = [];
  if (app.capabilities.includes('filesystem')) {
    allowedSyscalls.push('vfs.read', 'vfs.write', 'vfs.list');
    fsRead.push(...homes.map((home) => `${home}/**`), '/tmp/**', '/usr/share/**', '/Applications/**', '/opt/**', '/C/Program Files/**');
    fsWrite.push(...homes.map((home) => `${home}/**`), '/tmp/**');
  }
  if (app.capabilities.includes('network')) { allowedSyscalls.push('net.request'); network.push('*'); }
  if (app.serviceContracts.length) allowedSyscalls.push('service.invoke');
  if (app.entrypoint === 'app://git') allowedSyscalls.push('git.command');
  if (app.entrypoint === 'app://packages') allowedSyscalls.push('pkg.command');
  if (app.operations.includes('execute')) allowedSyscalls.push('shell.exec');
  return { label: app.id, allowedSyscalls, deniedSyscalls: [], fsRead, fsWrite, network };
}

export class SeedApplicationRuntime {
  private readonly executions: AppExecutionRecord[] = [];
  private readonly permissions = new PermissionManager();
  private readonly router: SyscallRouter;

  constructor(private readonly deps: ApplicationRuntimeDependencies) {
    this.router = new SyscallRouter(this.permissions).registerAll(this.syscalls());
  }

  listExecutions(): AppExecutionRecord[] { return this.executions.map((record) => structuredClone(record)); }

  /** Syscall names this kernel implements, for introspection and tests. */
  syscallNames(): string[] { return this.router.names(); }

  /** Recent syscall outcomes, including every permission denial. */
  syscallAudit(): SyscallAuditEntry[] { return this.router.audit(); }

  /** Escape hatch for callers that already hold a PID and want to issue a syscall directly. */
  async syscall(pid: number, syscall: string, args: Record<string, unknown> = {}) {
    return this.router.handle({ id: randomUUID(), pid, syscall, args });
  }

  async execute(appId: string, request: AppLaunchRequest): Promise<AppExecutionRecord> {
    const app = this.deps.installedApp(appId);
    if (!app) throw new Error(`application is not installed: ${appId}`);
    if (!app.operations.includes(request.operation)) throw new Error(`${app.name} does not expose operation ${request.operation}`);
    const startedAt = new Date().toISOString();
    const record: AppExecutionRecord = {
      id: randomUUID(), computerId: this.deps.spec.id, appId, runtime: app.runtime.kind,
      operation: request.operation, startedAt, completedAt: startedAt, status: 'completed',
    };
    const process = this.deps.processes.spawn({
      executable: app.entrypoint, argv: [request.operation], cwd: app.dataPath, ppid: 1,
      memoryBytes: 12 * 1024 * 1024,
    });
    this.permissions.setPermission(process.pid, applicationPermissions(app, this.deps.spec, this.deps.homePath));
    try {
      if (app.runtime.kind === 'seed-wasm') throw new Error('seed-wasm package requires an exported run function; no WASM bundle is installed');
      const result = app.runtime.kind === 'seed-js'
        ? await this.executeJavaScript(app, request, process.pid)
        : await this.dispatch(process.pid, app, request.operation, request.payload ?? {});
      record.result = structuredClone(result);
    } catch (error) {
      record.status = 'failed';
      record.error = error instanceof Error ? error.message : String(error);
    } finally {
      record.completedAt = new Date().toISOString();
      this.permissions.removePermission(process.pid);
      this.deps.processes.kill(process.pid);
      this.executions.push(record);
      if (this.executions.length > 500) this.executions.shift();
    }
    return structuredClone(record);
  }

  private async executeJavaScript(app: InstalledApp, request: AppLaunchRequest, pid: number): Promise<unknown> {
    const entryPath = `${app.installPath}/${app.runtime.entryFile}`;
    // Reading the bundle is a kernel action performed before the process has any code of its own,
    // so it deliberately bypasses the app's own filesystem grants.
    const source = await this.deps.vfs.readFile(entryPath);
    const outcome = await runSandboxedBundle({
      source,
      filename: entryPath,
      request: structuredClone(request),
      app: { id: app.id, version: app.version, capabilities: [...app.capabilities] },
      contextName: `${this.deps.spec.id}:${app.id}`,
      timeoutMs: this.deps.sandboxTimeoutMs ?? SANDBOX_DEFAULT_TIMEOUT_MS,
      hardTimeoutMs: this.deps.sandboxHardTimeoutMs,
      dispatch: (operation, payload) => this.dispatch(pid, app, operation, payload),
    });
    return outcome.value;
  }

  /**
   * Translates an application operation into a syscall and routes it through the permission check.
   * The operation vocabulary and the resulting behaviour are unchanged; only the path is new.
   */
  private async dispatch(pid: number, app: InstalledApp, operation: string, payload: Record<string, unknown>): Promise<unknown> {
    if (!app.operations.includes(operation)) throw new Error(`operation denied by manifest: ${operation}`);
    const call = this.routeOperation(app, operation, payload);
    return this.router.invoke({ id: randomUUID(), pid, syscall: call.syscall, args: call.args });
  }

  private routeOperation(app: InstalledApp, operation: string, payload: Record<string, unknown>): { syscall: string; args: Record<string, unknown> } {
    if (['navigate', 'send-request'].includes(operation)) {
      return { syscall: 'net.request', args: { url: String(payload.url ?? ''), method: String(payload.method ?? 'GET'), body: payload.body === undefined ? undefined : String(payload.body) } };
    }
    if (operation === 'calculate') return { syscall: 'compute.calculate', args: { expression: String(payload.expression ?? '') } };
    if (operation === 'execute') return { syscall: 'shell.exec', args: { command: String(payload.command ?? '') } };
    if (['open', 'open-file'].includes(operation) && payload.path) return { syscall: 'vfs.read', args: { path: String(payload.path) } };
    if (['save', 'save-as', 'edit'].includes(operation) && payload.path) {
      return { syscall: 'vfs.write', args: { path: String(payload.path), content: String(payload.content ?? '') } };
    }
    if (operation === 'list' && payload.path) return { syscall: 'vfs.list', args: { path: String(payload.path) } };
    if (app.id === 'slack' || app.id === 'teams') return { syscall: 'service.invoke', args: { operation, payload } };
    if (GIT_OPERATIONS.includes(operation) && app.entrypoint === 'app://git') {
      const args = Array.isArray(payload.args) ? payload.args.map(String) : [];
      return { syscall: 'git.command', args: { argv: [operation, ...args], cwd: String(payload.cwd ?? app.dataPath) } };
    }
    if (operation === 'source-control' && app.entrypoint === 'app://git') {
      const cwd = String(payload.cwd ?? app.dataPath);
      const argv = payload.action === 'checkout' && payload.branch ? ['checkout', String(payload.branch)] : ['status'];
      return { syscall: 'git.command', args: { argv, cwd } };
    }
    if (PACKAGE_OPERATIONS.includes(operation) && app.entrypoint === 'app://packages') {
      const manager = String(payload.manager ?? this.deps.software.supportedManagers()[0]);
      const verb = operation === 'inspect' ? 'info' : operation;
      const argv = Array.isArray(payload.args) ? payload.args.map(String) : [verb, ...(payload.name ? [String(payload.name)] : [])];
      return { syscall: 'pkg.command', args: { manager, argv, cwd: String(payload.cwd ?? app.dataPath) } };
    }
    if (app.serviceContracts.some((contract) => contract.protocol !== 'virtual')) {
      return { syscall: 'service.invoke', args: { operation, payload } };
    }
    return { syscall: 'app.event', args: { operation, payload } };
  }

  /** Handler table. Every handler receives the caller's security context and nothing else. */
  private syscalls(): Record<string, SyscallHandler> {
    const requireApp = (subject: string): InstalledApp => {
      const app = this.deps.installedApp(subject);
      if (!app) throw new Error(`application is not installed: ${subject}`);
      return app;
    };
    const requireString = (args: Record<string, unknown>, key: string, message: string): string => {
      const value = args[key];
      if (typeof value !== 'string' || !value) throw new Error(message);
      return value;
    };
    const argv = (args: Record<string, unknown>): string[] => (Array.isArray(args.argv) ? args.argv.map(String) : []);

    return {
      'proc.self': async (_args, context) => {
        const record = this.deps.processes.get(context.pid);
        if (!record) throw new Error(`process ${context.pid} is gone`);
        return { pid: record.pid, ppid: record.ppid, executable: record.executable, cwd: record.cwd, state: record.state, appId: context.subject };
      },
      'vfs.read': async (args, context) => {
        const filePath = requireString(args, 'path', 'path is required');
        context.requirePath(filePath, 'read');
        return { path: filePath, content: await this.deps.vfs.readFile(filePath) };
      },
      'vfs.write': async (args, context) => {
        const filePath = requireString(args, 'path', 'path is required');
        context.requirePath(filePath, 'write');
        const content = String(args.content ?? '');
        await this.deps.vfs.writeFile(filePath, content);
        return { path: filePath, bytes: Buffer.byteLength(content) };
      },
      'vfs.list': async (args, context) => {
        const filePath = requireString(args, 'path', 'path is required');
        context.requirePath(filePath, 'read');
        return this.deps.vfs.list(filePath);
      },
      'net.request': async (args, context) => {
        const url = requireString(args, 'url', 'url is required');
        let host: string;
        try { host = new URL(url).hostname; } catch { throw new Error(`invalid url: ${url}`); }
        context.requireHost(host);
        const app = requireApp(context.subject);
        const response = await this.deps.network.request(
          this.deps.spec.id, url, String(args.method ?? 'GET'),
          args.body === undefined ? undefined : String(args.body),
        );
        await this.updateState(app, { lastUrl: url, lastStatus: response.status, updatedAt: new Date().toISOString() });
        return response;
      },
      'shell.exec': async (args) => this.deps.executeShell(requireString(args, 'command', 'command is required')),
      'compute.calculate': async (args) => {
        const expression = String(args.expression ?? '');
        return { expression, value: calculate(expression) };
      },
      'service.invoke': async (args, context) => {
        const app = requireApp(context.subject);
        const operation = requireString(args, 'operation', 'operation is required');
        const payload = (args.payload ?? {}) as Record<string, unknown>;
        const result = await this.deps.serviceOperation(app, { operation, payload });
        await this.updateState(app, { lastServiceOperation: operation, lastServiceResult: result, updatedAt: new Date().toISOString() });
        return result;
      },
      'git.command': async (args) => this.deps.software.gitCommand(argv(args), String(args.cwd ?? '/')),
      'pkg.command': async (args) => this.deps.software.packageCommand(String(args.manager ?? ''), argv(args), String(args.cwd ?? '/')),
      'app.event': async (args, context) => {
        const app = requireApp(context.subject);
        context.requirePath(`${app.dataPath}/state.json`, 'write');
        const operation = requireString(args, 'operation', 'operation is required');
        const payload = (args.payload ?? {}) as Record<string, unknown>;
        const state = await this.readState(app);
        const event = { id: randomUUID(), operation, payload: structuredClone(payload), at: new Date().toISOString() };
        const events = Array.isArray(state.events) ? state.events : [];
        events.push(event);
        await this.writeState(app, { ...state, events: events.slice(-200), lastOperation: operation, updatedAt: event.at });
        return { ok: true, event, stateRevision: events.length };
      },
    };
  }

  private async readState(app: InstalledApp): Promise<Record<string, unknown>> {
    try { return JSON.parse(await this.deps.vfs.readFile(`${app.dataPath}/state.json`)) as Record<string, unknown>; }
    catch { return { schema: app.runtime.stateSchema, events: [] }; }
  }

  private async updateState(app: InstalledApp, patch: Record<string, unknown>): Promise<void> {
    await this.writeState(app, { ...(await this.readState(app)), ...patch });
  }

  private async writeState(app: InstalledApp, value: Record<string, unknown>): Promise<void> {
    await this.deps.vfs.writeFile(`${app.dataPath}/state.json`, JSON.stringify(value, null, 2));
  }
}

export class HostExecutionGateway {
  constructor(readonly rules: HostExecutionRule[] = []) {}

  async execute(computerId: string, appId: string, executable: string, args: string[], cwd: string): Promise<HostExecutionResult> {
    const resolvedExecutable = path.resolve(executable);
    const resolvedCwd = path.resolve(cwd);
    const rule = this.rules.find((candidate) => candidate.enabled &&
      (candidate.computerIds === '*' || candidate.computerIds.includes(computerId)) &&
      candidate.appIds.includes(appId) && candidate.executables.map((item) => path.resolve(item)).includes(resolvedExecutable) &&
      candidate.cwdRoots.some((root) => resolvedCwd === path.resolve(root) || resolvedCwd.startsWith(`${path.resolve(root)}${path.sep}`)));
    if (!rule) throw new Error(`host execution denied for ${appId}: ${resolvedExecutable}`);
    return new Promise((resolve, reject) => {
      const child = spawn(resolvedExecutable, args, { cwd: resolvedCwd, shell: false, stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = ''; let stderr = ''; let timedOut = false; let total = 0;
      const collect = (kind: 'stdout' | 'stderr', chunk: Buffer) => {
        total += chunk.byteLength;
        if (total > rule.maxOutputBytes) { child.kill('SIGKILL'); return; }
        if (kind === 'stdout') stdout += chunk.toString(); else stderr += chunk.toString();
      };
      child.stdout.on('data', (chunk: Buffer) => collect('stdout', chunk));
      child.stderr.on('data', (chunk: Buffer) => collect('stderr', chunk));
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, rule.timeoutMs);
      child.on('error', (error) => { clearTimeout(timer); reject(error); });
      child.on('close', (exitCode) => { clearTimeout(timer); resolve({ exitCode, stdout, stderr, timedOut }); });
    });
  }
}
