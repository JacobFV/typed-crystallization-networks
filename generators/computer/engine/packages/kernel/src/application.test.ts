import { describe, expect, it } from 'vitest';
import type { AppLaunchRequest, ComputerSpec, InstalledApp } from '@tcn-computer/protocol';
import { applicationPermissions, SeedApplicationRuntime, seedJavaScriptBundle, type ApplicationRuntimeDependencies } from './application.js';
import { PermissionManager, SecurityContext, globMatches } from './permissions.js';
import { ProcessManager } from './processes.js';
import { runSandboxedBundle, SandboxTimeoutError } from './sandbox.js';
import { parseSyscallRequest, SyscallRouter } from './syscalls.js';

const spec: ComputerSpec = {
  id: 'test-box', hostname: 'test-box', os: 'ubuntu', shell: 'bash', ipv4: '10.42.0.99',
  memoryBytes: 8 * 1024 ** 3, cpuCores: 4, disks: [], displays: [],
};

function installedApp(overrides: Partial<InstalledApp> = {}): InstalledApp {
  return {
    id: 'editor', name: 'Editor', version: '1.0.0', publisher: 'Seed', description: 'test app', icon: 'editor',
    supportedOS: ['ubuntu'], entrypoint: 'app://documents', packagePath: 'registry://seed/editor',
    capabilities: ['filesystem'], operations: ['open', 'edit', 'save', 'list', 'navigate', 'calculate'],
    serviceContracts: [], runtime: { kind: 'seed-js', apiVersion: 1, entryFile: 'main.seed.js', stateSchema: 'seed.app.editor.v1' },
    installedAt: new Date().toISOString(), installPath: '/opt/editor', dataPath: '/home/agent/.config/editor',
    receiptPath: '/var/lib/seed/apps/editor.json', registryHost: 'packages.seed.local', installState: 'installed',
    ...overrides,
  };
}

/** Minimal host stubs: enough surface for the runtime, none of the real simulation. */
function harness(app: InstalledApp, bundle: string) {
  const files = new Map<string, string>([[`${app.installPath}/${app.runtime.entryFile}`, bundle]]);
  const requests: string[] = [];
  const shellCommands: string[] = [];
  const processes = new ProcessManager(spec.id, { cpuCores: spec.cpuCores });
  processes.boot('systemd', '/home/agent', {});
  const deps = {
    spec,
    vfs: {
      readFile: async (filePath: string) => {
        const value = files.get(filePath);
        if (value === undefined) throw new Error(`no such file: ${filePath}`);
        return value;
      },
      writeFile: async (filePath: string, content: string) => { files.set(filePath, content); },
      list: (filePath: string) => [...files.keys()].filter((key) => key.startsWith(filePath)).map((key) => ({ name: key, path: key })),
    },
    processes,
    network: {
      request: async (_computerId: string, url: string) => { requests.push(url); return { status: 200, headers: {}, body: 'ok', traceId: 't-1' }; },
    },
    software: {
      gitCommand: async (argv: string[]) => ({ argv }),
      packageCommand: async (manager: string, argv: string[]) => ({ manager, argv }),
      supportedManagers: () => ['apt'],
    },
    executeShell: async (command: string) => { shellCommands.push(command); return { exitCode: 0, stdout: command }; },
    installedApp: (appId: string) => (appId === app.id ? app : undefined),
    serviceOperation: async (_app: InstalledApp, request: AppLaunchRequest) => ({ serviced: request.operation }),
    sandboxTimeoutMs: 1_000,
  } as unknown as ApplicationRuntimeDependencies;
  return { runtime: new SeedApplicationRuntime(deps), files, requests, shellCommands, processes };
}

const dispatchNothing = async () => ({ ok: true });

async function runBundle(source: string, request: unknown = {}, timeoutMs = 1_000) {
  return runSandboxedBundle({ source, filename: 'attack.seed.js', request, dispatch: dispatchNothing, timeoutMs });
}

describe('sandbox realm isolation', () => {
  it('runs a well-behaved bundle and returns its value', async () => {
    const outcome = await runSandboxedBundle({
      source: seedJavaScriptBundle(),
      filename: 'main.seed.js',
      request: { operation: 'list', payload: { path: '/home' } },
      dispatch: async (operation, payload) => ({ operation, payload }),
      timeoutMs: 1_000,
    });
    expect(outcome.value).toEqual({ operation: 'list', payload: { path: '/home' } });
    expect(outcome.syscalls).toBe(1);
  });

  // The confirmed escape: host intrinsics in the context made `Object.constructor` the host
  // `Function`, which `codeGeneration.strings: false` does not restrain.
  it.each([
    ['Object.constructor', `module.exports = async () => Object.constructor('return process')()`],
    ['Array.constructor', `module.exports = async () => Array.constructor('return process')().pid`],
    ['String.constructor', `module.exports = async () => String.constructor('return process')().pid`],
    ['Number.constructor', `module.exports = async () => Number.constructor('return process')().pid`],
    ['Promise.constructor', `module.exports = async () => Promise.constructor('return process')().pid`],
    ['JSON.constructor', `module.exports = async () => JSON.constructor('return process')().pid`],
    ['constructor.constructor', `module.exports = async () => ({}).constructor.constructor('return process')().pid`],
    ['sdk.constructor.constructor', `module.exports = async (seed) => seed.constructor.constructor('return process')().pid`],
    ['dispatch.constructor.constructor', `module.exports = async (seed) => seed.dispatch.constructor.constructor('return process')().pid`],
    ['request.constructor.constructor', `module.exports = async (seed, request) => request.constructor.constructor('return process')().pid`],
    ['array literal prototype', `module.exports = async () => [].constructor.constructor('return this.process')().pid`],
    ['eval', `module.exports = async () => eval('process.pid')`],
    ['generator constructor', `module.exports = async () => (function* () {}).constructor('return process')().pid`],
  ])('blocks the host realm through %s', async (_label, source) => {
    await expect(runBundle(source)).rejects.toThrow(/Code generation from strings disallowed|is not defined|not a function/);
  });

  it('blocks dynamic import of host modules', async () => {
    await expect(runBundle(`module.exports = async () => (await import('node:fs')).readFileSync('/etc/passwd', 'utf8')`))
      .rejects.toThrow(/dynamic import|not defined|not supported/i);
  });

  it('exposes no host ambient objects at all', async () => {
    const outcome = await runBundle(`module.exports = async () => [
      typeof process, typeof require, typeof globalThis.process, typeof Buffer,
      typeof module.constructor, typeof globalThis.__seedSettle === 'function'
    ].join(',')`);
    expect(outcome.value).toBe('undefined,undefined,undefined,undefined,function,true');
  });

  it('refuses to compile WebAssembly inside the sandbox realm', async () => {
    await expect(runBundle(`module.exports = async () => new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]))`))
      .rejects.toThrow(/[Ww]asm code generation disallowed/);
  });

  it('keeps host errors out of the sandbox realm', async () => {
    // A rejected host promise used to hand the bundle a host Error, and with it host `Function`.
    await expect(runSandboxedBundle({
      source: `module.exports = async (seed) => {
        try { await seed.dispatch('boom', {}); }
        catch (error) { return error.constructor.constructor('return process')().pid; }
      }`,
      filename: 'attack.seed.js',
      request: {},
      dispatch: async () => { throw new Error('kernel says no'); },
      timeoutMs: 1_000,
    })).rejects.toThrow(/Code generation from strings disallowed/);
  });

  it('propagates kernel errors as ordinary sandbox-realm errors', async () => {
    await expect(runSandboxedBundle({
      source: `module.exports = async (seed) => seed.dispatch('boom', {})`,
      filename: 'attack.seed.js', request: {}, timeoutMs: 1_000,
      dispatch: async () => { throw new Error('kernel says no'); },
    })).rejects.toThrow('kernel says no');
  });

  it('rejects a bundle that exports no function', async () => {
    await expect(runBundle('module.exports = 42;')).rejects.toThrow('attack.seed.js must export a function');
  });

  it('captures console output instead of writing to the host stdout', async () => {
    const outcome = await runBundle(`console.log('from', 'the sandbox'); module.exports = async () => 'done'`);
    expect(outcome.logs).toEqual(['log: from the sandbox']);
  });
});

describe('sandbox execution budget', () => {
  it('terminates a synchronous busy loop that outlives its budget', async () => {
    const started = Date.now();
    await expect(runBundle(`module.exports = async () => { const end = Date.now() + 1200; while (Date.now() < end); return 'finished'; }`, {}, 50))
      .rejects.toThrow(SandboxTimeoutError);
    // A Promise.race cannot interrupt this; terminating the worker can.
    expect(Date.now() - started).toBeLessThan(600);
  });

  it('terminates a busy loop at module evaluation time', async () => {
    await expect(runBundle(`const end = Date.now() + 1200; while (Date.now() < end); module.exports = async () => 1;`, {}, 50))
      .rejects.toThrow(/exceeded|Script execution timed out/);
  });

  it('terminates a bundle that never settles', async () => {
    await expect(runBundle(`module.exports = () => new Promise(() => {})`, {}, 60)).rejects.toThrow(SandboxTimeoutError);
  });

  it('does not spend the bundle budget on kernel syscall service time', async () => {
    const outcome = await runSandboxedBundle({
      source: `module.exports = async (seed) => { await seed.dispatch('slow', {}); await seed.dispatch('slow', {}); return 'ok'; }`,
      filename: 'main.seed.js', request: {}, timeoutMs: 80,
      dispatch: async () => { await new Promise((resolve) => setTimeout(resolve, 120)); return 1; },
    });
    expect(outcome.value).toBe('ok');
    expect(outcome.durationMs).toBeLessThan(80);
  });

  it('does not double-charge the budget for concurrent syscalls', async () => {
    const outcome = await runSandboxedBundle({
      source: `module.exports = async (seed) => (await Promise.all([seed.dispatch('a', {}), seed.dispatch('b', {}), seed.dispatch('c', {})])).length`,
      filename: 'main.seed.js', request: {}, timeoutMs: 150,
      dispatch: async () => { await new Promise((resolve) => setTimeout(resolve, 100)); return 1; },
    });
    expect(outcome.value).toBe(3);
    expect(outcome.syscalls).toBe(3);
  });

  it('enforces an absolute wall-clock ceiling even while syscalls are outstanding', async () => {
    await expect(runSandboxedBundle({
      source: `module.exports = async (seed) => { seed.dispatch('slow', {}); const end = Date.now() + 5000; while (Date.now() < end); }`,
      filename: 'main.seed.js', request: {}, timeoutMs: 400, hardTimeoutMs: 700,
      dispatch: async () => new Promise((resolve) => setTimeout(() => resolve(1), 10_000)),
    })).rejects.toThrow(/wall-clock/);
  });
});

describe('syscall boundary', () => {
  it('validates requests before anything else looks at them', () => {
    expect(() => parseSyscallRequest(null)).toThrow('syscall request must be an object');
    expect(() => parseSyscallRequest({ syscall: 'vfs.read' })).toThrow('integer pid');
    expect(() => parseSyscallRequest({ syscall: 'drop tables', pid: 1 })).toThrow('malformed syscall name');
    expect(() => parseSyscallRequest({ syscall: 'vfs.read', pid: 0 })).toThrow('integer pid');
    expect(() => parseSyscallRequest({ syscall: 'vfs.read', pid: 7, args: [] })).toThrow('plain object');
    expect(parseSyscallRequest({ syscall: 'vfs.read', pid: 7 })).toMatchObject({ pid: 7, syscall: 'vfs.read', args: {} });
  });

  it('denies calls from a pid with no security context', async () => {
    const permissions = new PermissionManager();
    const router = new SyscallRouter(permissions).register('vfs.read', async () => 'leaked');
    const response = await router.handle({ id: 'a', pid: 4242, syscall: 'vfs.read', args: {} });
    expect(response).toMatchObject({ success: false, error: 'no security context for pid 4242' });
  });

  it('checks the permission before the handler is ever reached', async () => {
    const permissions = new PermissionManager();
    permissions.setPermission(7, { label: 'sandboxed', allowedSyscalls: ['vfs.read'], fsRead: ['/tmp/**'], fsWrite: [], network: [] });
    let handlerRuns = 0;
    const router = new SyscallRouter(permissions)
      .register('vfs.read', async () => { handlerRuns += 1; return 'ok'; })
      .register('vfs.write', async () => { handlerRuns += 1; return 'ok'; });

    await expect(router.invoke({ pid: 7, syscall: 'vfs.write', args: {} })).rejects.toThrow('permission denied: sandboxed may not call vfs.write');
    expect(handlerRuns).toBe(0);
    expect(await router.invoke({ pid: 7, syscall: 'vfs.read', args: {} })).toBe('ok');
    expect(handlerRuns).toBe(1);
    expect(router.audit().map((entry) => entry.outcome)).toEqual(['denied', 'completed']);
  });

  it('scopes filesystem and network reach per process', () => {
    const context = new SecurityContext(9, {
      label: 'editor', allowedSyscalls: ['vfs.read'],
      fsRead: ['/home/agent/**'], fsWrite: ['/home/agent/Documents/**'], network: ['*.seed.local'],
    });
    expect(context.canAccessPath('/home/agent/Documents/notes.md', 'read')).toBe(true);
    expect(context.canAccessPath('/home/agent', 'read')).toBe(true);
    expect(context.canAccessPath('/etc/shadow', 'read')).toBe(false);
    expect(context.canAccessPath('/home/agent/Desktop/a.txt', 'write')).toBe(false);
    expect(context.canReachHost('intranet.seed.local')).toBe(true);
    expect(context.canReachHost('evil.example.com')).toBe(false);
    expect(globMatches('/tmp/a/b/c', '/tmp/**')).toBe(true);
    expect(globMatches('/tmp/a/b', '/tmp/*')).toBe(false);
  });

  it('derives a default-deny permission set from the manifest', () => {
    const local = applicationPermissions(installedApp({ capabilities: [] }), spec);
    expect(local.allowedSyscalls).not.toContain('vfs.read');
    expect(local.allowedSyscalls).not.toContain('net.request');
    expect(local.fsWrite).toEqual(['/home/agent/.config/editor/**']);

    const networked = applicationPermissions(installedApp({ capabilities: ['filesystem', 'network'] }), spec);
    expect(networked.allowedSyscalls).toEqual(expect.arrayContaining(['vfs.read', 'vfs.write', 'vfs.list', 'net.request']));
    expect(networked.allowedSyscalls).not.toContain('shell.exec');
    expect(networked.allowedSyscalls).not.toContain('git.command');

    const terminal = applicationPermissions(installedApp({ entrypoint: 'system://terminal', operations: ['execute'] }), spec);
    expect(terminal.allowedSyscalls).toContain('shell.exec');
  });
});

describe('application runtime end to end', () => {
  it('executes a bundle through the syscall boundary', async () => {
    const app = installedApp();
    const { runtime, files } = harness(app, seedJavaScriptBundle());
    const record = await runtime.execute('editor', { operation: 'edit', payload: { path: '/home/agent/Documents/proof.md', content: 'written by the app' } });
    expect(record.status).toBe('completed');
    expect(files.get('/home/agent/Documents/proof.md')).toBe('written by the app');
    expect(runtime.syscallAudit().some((entry) => entry.syscall === 'vfs.write' && entry.outcome === 'completed')).toBe(true);
  });

  it('reaps the application process and its permissions when execution ends', async () => {
    const app = installedApp();
    const { runtime, processes } = harness(app, seedJavaScriptBundle());
    const before = processes.list().length;
    await runtime.execute('editor', { operation: 'calculate', payload: { expression: '2*(3+4)' } });
    expect(processes.list()).toHaveLength(before);
    expect(processes.stats().zombies).toBe(0);
  });

  it('denies a syscall the manifest never granted', async () => {
    const app = installedApp({ capabilities: [], operations: ['open', 'edit', 'navigate'] });
    const { runtime } = harness(app, seedJavaScriptBundle());
    const record = await runtime.execute('editor', { operation: 'open', payload: { path: '/home/agent/Documents/proof.md' } });
    expect(record.status).toBe('failed');
    expect(record.error).toBe('permission denied: editor may not call vfs.read');
    expect(runtime.syscallAudit().at(-1)).toMatchObject({ syscall: 'vfs.read', outcome: 'denied' });
  });

  it('denies a granted syscall outside the process filesystem scope', async () => {
    const app = installedApp();
    const { runtime } = harness(app, seedJavaScriptBundle());
    const record = await runtime.execute('editor', { operation: 'edit', payload: { path: '/etc/shadow', content: 'root::0:0' } });
    expect(record.status).toBe('failed');
    expect(record.error).toBe('permission denied: editor may not write /etc/shadow');
  });

  it('denies network reach for an app without the network capability', async () => {
    const app = installedApp();
    const { runtime, requests } = harness(app, seedJavaScriptBundle());
    const record = await runtime.execute('editor', { operation: 'navigate', payload: { url: 'http://intranet.seed.local:8080/' } });
    expect(record.status).toBe('failed');
    expect(record.error).toContain('may not call net.request');
    expect(requests).toEqual([]);
  });

  it('records a failed execution when an installed bundle attempts a sandbox escape', async () => {
    const app = installedApp();
    const { runtime } = harness(app, `module.exports = async () => { const host = Object.constructor('return process')(); return host.pid; };`);
    const record = await runtime.execute('editor', { operation: 'open', payload: { path: '/x' } });
    expect(record.status).toBe('failed');
    expect(record.error).toContain('Code generation from strings disallowed');
    expect(record.result).toBeUndefined();
  });

  it('records a failed execution when an installed bundle burns its budget', async () => {
    const app = installedApp();
    const { runtime } = harness(app, `module.exports = async () => { const end = Date.now() + 4000; while (Date.now() < end); };`);
    const started = Date.now();
    const record = await runtime.execute('editor', { operation: 'open', payload: { path: '/x' } });
    expect(record.status).toBe('failed');
    expect(record.error).toContain('budget');
    expect(Date.now() - started).toBeLessThan(3_000);
  });
});
