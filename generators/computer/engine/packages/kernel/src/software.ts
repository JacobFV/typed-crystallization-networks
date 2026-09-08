import { createHash } from 'node:crypto';
import { randomUUID } from './determinism.js';
import type {
  ComputerSpec, GitRepositoryRecord, OSRuntimeProfile, PackageManagerKind, PackageRecord, PackageTransactionRecord,
} from '@tcn-computer/protocol';
import { GitEnvironment, type GitRemoteTransport } from './git.js';
import { availableVersions, catalogSize, findEntry, registries } from './packages/catalog.js';
import {
  EXECUTABLE_MODE, planExecutables, removeExecutables, removeServiceUnit, writeExecutables, writeServiceUnit,
} from './packages/executables.js';
import { binDirectoryForBinary, executableFileName, installRoot, receiptPath, safeName } from './packages/layout.js';
import {
  type ManagerSpec, managerModule, managerModules, type PackageView, resolveCommand,
} from './packages/managers/index.js';
import { describeProblem, findOrphans, resolve, type ResolvedPackage } from './packages/resolver.js';
import {
  listScripts, type ProgramRunner, readScriptManifest, runCommandChain, ScriptError, type ScriptEnvironment,
  scriptSearchPath, runScript,
} from './packages/scripts.js';
import { applyVersionFlag, parseSpecifier, rangeOptionsFor } from './packages/specifier.js';
import type {
  CatalogPackage, InstalledRecord, ManagerContext, OperationKind, PackageScope, PackageSpecifier, ServiceDefinition,
} from './packages/types.js';
import { toPackageRecord } from './packages/types.js';
import { compareVersions, defaultRangeFor, highestVersion, maxSatisfyingAll, satisfies } from './packages/versions.js';
import type { ProcessManager } from './processes.js';
import { canonicalPath, type VirtualFileSystem } from './vfs.js';

export type { ProgramRunner } from './packages/scripts.js';
export type { CatalogPackage, SeedExecutableDescriptor, ServiceDefinition } from './packages/types.js';
export { EXECUTABLE_MODE, parseDescriptor } from './packages/executables.js';
export { parseSpecifier } from './packages/specifier.js';
export { compareVersions, parseRange, satisfies } from './packages/versions.js';
export { catalogSize } from './packages/catalog.js';

/** Managers that prune newly-orphaned dependencies as part of `remove`. */
const IMPLICIT_AUTOREMOVE = new Set<PackageManagerKind>(['npm', 'pnpm', 'yarn', 'bun', 'poetry', 'composer', 'nuget', 'vcpkg', 'uv']);

/** Managers whose `run`/`build`/`test` drive a toolchain, not a package.json script. */
const TOOLCHAIN_MANAGERS = new Set<PackageManagerKind>(['cargo', 'go', 'dotnet', 'composer', 'poetry', 'uv', 'conda']);

/** Commands that are a direct package runner rather than a manager verb. */
const DIRECT_EXEC = new Set(['npx', 'pnpx', 'bunx', 'uvx']);

/** Manifest that marks a project root for each project-scoped manager. */
const PROJECT_MANIFESTS: Partial<Record<PackageManagerKind, string[]>> = {
  npm: ['package.json'], pnpm: ['package.json'], yarn: ['package.json'], bun: ['package.json'],
  poetry: ['pyproject.toml'], composer: ['composer.json'], nuget: ['packages.lock.json'], vcpkg: ['vcpkg.json'],
};

interface ParsedInvocation {
  kind: OperationKind;
  verb: string;
  /** Non-flag tokens following the verb. */
  operands: string[];
  flags: Set<string>;
  versionFlag?: string;
  requirementFiles: string[];
  global: boolean;
  dev: boolean;
  cask: boolean;
  dryRun: boolean;
  purge: boolean;
}

interface WriteResult {
  record: InstalledRecord;
  view: PackageView;
}

/**
 * Software environment: package managers, their registries, and the executables
 * they put on PATH.
 *
 * The load-bearing decision here is that installing is not a bookkeeping entry —
 * it writes an executable descriptor (mode 0o755, JSON body with a `behavior`
 * key) for every binary the package provides, into that manager's real bin
 * directory. That descriptor is the contract the shell's program registry reads,
 * so `apt install ripgrep` is followed by a working `rg`. Services additionally
 * get a platform unit file and an entry in a machine-readable service index, so
 * a daemon can actually be started afterwards.
 */
export class SoftwareEnvironment {
  private readonly packages = new Map<string, InstalledRecord>();
  private readonly transactions: PackageTransactionRecord[] = [];
  private readonly services = new Map<string, ServiceDefinition & { package: string; manager: PackageManagerKind; executablePath: string }>();
  private readonly git: GitEnvironment;
  private readonly dbPath: string;
  private readonly home: string;
  private programRunner?: ProgramRunner;

  constructor(
    private readonly spec: ComputerSpec,
    private readonly vfs: VirtualFileSystem,
    private readonly processes: ProcessManager,
    private readonly profile: OSRuntimeProfile,
    gitTransport?: GitRemoteTransport,
  ) {
    this.git = new GitEnvironment(vfs, gitTransport);
    this.home = profile.filesystem.home;
    this.dbPath = spec.os === 'windows'
      ? '/C/ProgramData/Seed/packages.json'
      : spec.os === 'macos'
        ? '/Library/Application Support/Seed/packages.json'
        : '/var/lib/seed/packages.json';
  }

  /** Lets the shell supply its program registry so scripts run real implementations. */
  setProgramRunner(runner: ProgramRunner | undefined): void { this.programRunner = runner; }

  async initialize(): Promise<void> {
    const native = this.profile.packageManagers.native;
    const bootstrapManager: PackageManagerKind = native.includes('brew') ? 'brew' : native.includes('apt') ? 'apt' : native[0] ?? 'apt';
    const bootstrap: Record<string, string[]> = {
      brew: ['git', 'node', 'python@3.13'],
      apt: ['git', 'nodejs', 'python3'],
      winget: ['Git.Git', 'OpenJS.NodeJS', 'Python.Python.3.13'],
    };
    for (const name of bootstrap[bootstrapManager] ?? []) {
      await this.performInstall(bootstrapManager, [{ raw: name, name, range: '*' }], this.home, 'system', { bootstrap: true });
    }
    const project = `${this.home}/Projects/seed-ecosystem`;
    await this.vfs.mkdir(project);
    await this.vfs.writeFile(`${project}/README.md`, '# seed ecosystem\n\na typed multi-computer simulation runtime.\n');
    await this.vfs.writeFile(`${project}/package.json`, `${JSON.stringify({
      name: 'seed-ecosystem', version: '1.0.0', private: true, workspaces: ['apps/*', 'packages/*'],
      scripts: { build: 'tsc -p tsconfig.json', test: 'vitest run', lint: 'eslint .', clean: 'rimraf dist' },
    }, null, 2)}\n`);
    await this.git.initRepository(project);
    this.git.stage(project, ['README.md', 'package.json']);
    await this.gitCommand(['commit', '-m', 'bootstrap seed ecosystem'], project);
  }

  supports(command: string): boolean {
    const module = resolveCommand(command);
    return Boolean(module && this.available(module.id));
  }

  /**
   * `npx`/`bunx`/`pnpx`/`uvx`. These are not `PackageManagerKind` values, so the
   * shell cannot find them through `supports()`, but they must reach the
   * install-on-demand exec path rather than the plain PATH descriptor.
   */
  supportsExec(command: string): boolean {
    return DIRECT_EXEC.has(command.toLowerCase()) && this.supports(command);
  }

  supportedManagers(): PackageManagerKind[] {
    return managerModules.map((module) => module.id).filter((id) => this.available(id));
  }

  listPackages(): PackageRecord[] { return [...this.packages.values()].map(toPackageRecord); }
  listPackageTransactions(): PackageTransactionRecord[] { return this.transactions.map((item) => structuredClone(item)); }
  listRepositories(): GitRepositoryRecord[] { return this.git.listRepositories(); }

  /** Daemons registered by installed packages, ready to be started. */
  listServices(): Array<ServiceDefinition & { package: string; manager: PackageManagerKind; executablePath: string }> {
    return [...this.services.values()].map((service) => structuredClone(service));
  }

  /** Size of the shipped registries, for fidelity reporting. */
  catalogSize(): { managers: number; entries: number; distinct: number } { return catalogSize(); }

  async gitCommand(args: string[], cwd: string): Promise<string> {
    return this.git.command(args, cwd);
  }

  // ------------------------------------------------------------- dispatching

  async packageCommand(rawCommand: string, args: string[], cwd: string): Promise<string> {
    const command = rawCommand.trim().toLowerCase();
    const module = resolveCommand(command);
    if (!module || !this.available(module.id)) throw new Error(`${rawCommand}: unavailable on ${this.spec.os}`);
    const context = this.contextFor(module, cwd);

    if (DIRECT_EXEC.has(command)) return this.runExec(module, context, args);

    const invocation = this.parse(module, args);
    switch (invocation.kind) {
      case 'help': return this.renderHelp(module);
      case 'refresh': return this.refresh(module, context);
      case 'list': return this.list(module, context, invocation);
      case 'search': return this.search(module, context, invocation);
      case 'info': return this.info(module, context, invocation);
      case 'outdated': return this.outdated(module, context);
      case 'install': return this.install(module, context, invocation);
      case 'upgrade': return this.upgrade(module, context, invocation);
      case 'remove': case 'autoremove': return this.remove(module, context, invocation);
      case 'run': return this.run(module, context, invocation);
      case 'exec': return this.runExec(module, context, [...invocation.operands, ...[...invocation.flags]]);
      case 'test': return this.runNamedScript(module, context, 'test', invocation);
      case 'start': return this.runNamedScript(module, context, 'start', invocation);
      case 'build': return this.build(module, context, invocation);
      case 'services': return this.renderServices(module);
      case 'clean': return this.clean(module, context);
      default: throw new Error(`${module.id}: unsupported operation ${invocation.kind}`);
    }
  }

  private available(manager: PackageManagerKind): boolean {
    const { native, language } = this.profile.packageManagers;
    return native.includes(manager) || language.includes(manager);
  }

  private contextFor(module: ManagerSpec, cwd: string): ManagerContext {
    const resolved = canonicalPath(cwd || this.home);
    return {
      manager: module.id,
      os: this.spec.os,
      profile: this.profile,
      home: this.home,
      cwd: module.projectScoped ? this.projectRoot(module, resolved) : resolved,
      executableSuffix: this.profile.conventions.executableSuffix,
      hostname: this.spec.hostname,
    };
  }

  /** Walks up for the manager's manifest, the way npm/composer locate a project. */
  private projectRoot(module: ManagerSpec, cwd: string): string {
    const manifests = PROJECT_MANIFESTS[module.id] ?? [];
    let current = cwd;
    for (let depth = 0; depth < 24 && current !== '/'; depth += 1) {
      if (manifests.some((manifest) => this.vfs.exists(`${current}/${manifest}`))) return current;
      current = canonicalPath(`${current}/..`);
    }
    return cwd;
  }

  // ---------------------------------------------------------------- parsing

  private parse(module: ManagerSpec, args: string[]): ParsedInvocation {
    let tokens = [...args];
    for (const prefix of module.prefixes ?? []) {
      if (prefix.every((word, index) => tokens[index]?.toLowerCase() === word)) { tokens = tokens.slice(prefix.length); break; }
    }
    const flags = new Set<string>();
    const operands: string[] = [];
    const requirementFiles: string[] = [];
    let verb: string | undefined;
    let versionFlag: string | undefined;
    const versionFlags = new Set(module.versionFlags ?? []);
    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index]!;
      const lower = token.toLowerCase();
      if (!verb && (Object.hasOwn(module.verbs, lower) || Object.hasOwn(module.verbs, token))) {
        verb = Object.hasOwn(module.verbs, token) ? token : lower;
        continue;
      }
      if (versionFlags.has(token)) { versionFlag = tokens[++index]; continue; }
      if (token === '-r' || token === '--requirement') { const file = tokens[++index]; if (file) requirementFiles.push(file); continue; }
      if (token.startsWith('-')) {
        flags.add(token);
        const inline = /^--[\w-]+=(.*)$/.exec(token);
        if (inline && !verb) continue;
        continue;
      }
      if (!verb && module.defaultVerb) { operands.push(token); continue; }
      if (!verb) { operands.push(token); continue; }
      operands.push(token);
    }
    if (!verb) {
      // No recognized verb. A bare invocation falls back to the module's default;
      // an unrecognized word is a genuine error rather than a silent listing.
      const candidate = tokens.find((token) => !token.startsWith('-'));
      if (candidate) throw new Error(this.unknownVerb(module, candidate));
      const kind = module.defaultVerb ?? 'help';
      return { kind, verb: '', operands, flags, requirementFiles, global: this.isGlobal(module, flags), dev: false, cask: flags.has('--cask'), dryRun: false, purge: false, ...(versionFlag ? { versionFlag } : {}) };
    }
    // A few flags override the verb: `cargo install --list` lists, and
    // `apt list --upgradable` / `pip list --outdated` report upgrades.
    let kind = module.verbs[verb]!;
    if (kind === 'install' && flags.has('--list')) kind = 'list';
    if (kind === 'list' && (flags.has('--upgradable') || flags.has('--outdated'))) kind = 'outdated';
    return {
      kind, verb, operands, flags, requirementFiles,
      global: this.isGlobal(module, flags),
      dev: flags.has('-D') || flags.has('--save-dev') || flags.has('--dev') || flags.has('-d') || flags.has('--group=dev'),
      cask: flags.has('--cask') || flags.has('--casks'),
      dryRun: flags.has('--dry-run') || flags.has('-n') || flags.has('--simulate') || flags.has('--whatif'),
      purge: verb === 'purge' || flags.has('--purge') || flags.has('-P'),
      ...(versionFlag ? { versionFlag } : {}),
    };
  }

  private unknownVerb(module: ManagerSpec, verb: string): string {
    const known = module.helpVerbs ?? Object.keys(module.verbs);
    if (module.id === 'apt' || module.id === 'dpkg') return `E: Invalid operation ${verb}`;
    if (['npm', 'pnpm', 'yarn', 'bun'].includes(module.id)) {
      return [`Unknown command: "${verb}"`, '', `To see a list of supported ${module.id} commands, run:`, `  ${module.id} help`].join('\n');
    }
    if (module.id === 'cargo') return `error: no such command: \`${verb}\`\n\n\tView all installed commands with \`cargo --list\``;
    if (module.id === 'brew') return `Error: Unknown command: ${verb}`;
    if (module.id === 'winget') return `Unrecognized command: '${verb}'\nSee 'winget --help'`;
    if (module.id === 'go') return `go ${verb}: unknown command\nRun 'go help' for usage.`;
    return `${module.id}: unknown subcommand '${verb}'\navailable: ${known.join(', ')}`;
  }

  private isGlobal(module: ManagerSpec, flags: Set<string>): boolean {
    if (module.family === 'native') return true;
    if (!module.projectScoped) return true;
    return flags.has('-g') || flags.has('--global') || flags.has('--location=global');
  }

  private scopeFor(module: ManagerSpec, invocation: ParsedInvocation): PackageScope {
    if (module.family === 'native') return 'system';
    if (module.projectScoped && !invocation.global) return 'project';
    return 'user';
  }

  private async specifiersFor(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<PackageSpecifier[]> {
    const raw = [...invocation.operands];
    for (const file of invocation.requirementFiles) {
      const path = file.startsWith('/') ? file : `${context.cwd}/${file}`;
      const content = await this.vfs.readFile(path).catch(() => { throw new Error(`ERROR: Could not open requirements file: ${path}`); });
      for (const line of content.split('\n')) {
        const clean = line.replace(/#.*$/, '').trim();
        if (clean && !clean.startsWith('-')) raw.push(clean);
      }
    }
    return applyVersionFlag(raw.map((item) => parseSpecifier(module.id, item)), invocation.versionFlag);
  }

  // --------------------------------------------------------------- read-only

  private refresh(module: ManagerSpec, context: ManagerContext): string {
    const startedAt = new Date().toISOString();
    const indexPath = `${this.dbPath}.indexes/${module.id}.json`;
    void this.vfs.writeFile(indexPath, `${JSON.stringify({
      manager: module.id, refreshedAt: new Date().toISOString(),
      packages: registries[module.id].map((entry) => ({ name: entry.name, version: entry.version })),
    }, null, 2)}\n`);
    this.recordTransaction(module.id, 'index-refresh', [], [indexPath], startedAt);
    return module.renderers.refresh(context);
  }

  private async list(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    const records = this.recordsFor(module, context, invocation);
    if (module.id === 'pip' && invocation.verb === 'freeze') {
      return [...records].sort((a, b) => a.name.localeCompare(b.name)).map((record) => `${record.name}==${record.version}`).join('\n');
    }
    const manifest = module.projectScoped ? await readScriptManifest(this.vfs, context.cwd) : undefined;
    return module.renderers.list({ context, records, ...(manifest ? { projectName: manifest.name } : {}) });
  }

  private search(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): string {
    const query = invocation.operands.join(' ').trim();
    const needle = query.toLowerCase();
    const matches = registries[module.id].filter((entry) => !needle
      || entry.name.toLowerCase().includes(needle)
      || entry.description.toLowerCase().includes(needle)
      || (entry.aliases ?? []).some((alias) => alias.toLowerCase().includes(needle)));
    const installed = new Set(this.recordsFor(module, context, invocation).map((record) => record.name));
    return module.renderers.search({ context, query, matches: matches.slice(0, 40), installed });
  }

  private info(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): string {
    const name = parseSpecifier(module.id, invocation.operands[0] ?? '').name;
    const entry = findEntry(module.id, name);
    const record = this.findRecord(module, context, name);
    return module.renderers.info({ context, name, ...(entry ? { entry } : {}), ...(record ? { record } : {}) });
  }

  private outdated(module: ManagerSpec, context: ManagerContext): string {
    const entries = this.recordsFor(module, context, undefined)
      .map((record) => ({ record, latest: findEntry(module.id, record.name)?.version ?? record.version }))
      .filter(({ record, latest }) => compareVersions(latest, record.version) > 0);
    return module.renderers.outdated({ context, entries });
  }

  private renderHelp(module: ManagerSpec): string {
    const verbs = module.helpVerbs ?? [...new Set(Object.values(module.verbs))];
    return [`${module.id} — ${module.family === 'native' ? 'system' : 'language'} package manager`, '', 'Commands:', ...verbs.map((verb) => `  ${module.id} ${verb}`)].join('\n');
  }

  private renderServices(module: ManagerSpec): string {
    const owned = this.listServices().filter((service) => service.manager === module.id);
    if (!owned.length) return module.id === 'brew' ? 'No services available to control with `brew services`' : `${module.id}: no services registered`;
    const width = Math.max(4, ...owned.map((service) => service.name.length));
    return ['Name'.padEnd(width) + ' Status  User Port', ...owned.map((service) => `${service.name.padEnd(width)} none    -    ${service.port}`)].join('\n');
  }

  private clean(module: ManagerSpec, context: ManagerContext): string {
    void context;
    return module.id === 'brew' ? '==> Removing: ~/Library/Caches/Homebrew... (0B)\n==> This operation has freed approximately 0B of disk space.'
      : `${module.id}: cache cleaned`;
  }

  // ------------------------------------------------------------------ install

  private async install(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    const scope = this.scopeFor(module, invocation);
    let specifiers = await this.specifiersFor(module, context, invocation);
    if (!specifiers.length) {
      // Bare `npm install` / `poetry install` reconciles the manifest.
      specifiers = await this.manifestSpecifiers(module, context);
      if (!specifiers.length) {
        return module.renderers.install({ context, scope, installed: [], reused: [], dryRun: invocation.dryRun, cask: invocation.cask, dev: invocation.dev });
      }
    }
    return this.performInstall(module.id, specifiers, context.cwd, scope, {
      dryRun: invocation.dryRun, cask: invocation.cask, dev: invocation.dev,
    });
  }

  private async manifestSpecifiers(module: ManagerSpec, context: ManagerContext): Promise<PackageSpecifier[]> {
    if (!['npm', 'pnpm', 'yarn', 'bun'].includes(module.id)) return [];
    const manifest = await readScriptManifest(this.vfs, context.cwd);
    if (!manifest) return [];
    try {
      const raw = JSON.parse(await this.vfs.readFile(`${context.cwd}/package.json`)) as { dependencies?: Record<string, string>; devDependencies?: Record<string, string> };
      return Object.entries({ ...raw.dependencies, ...raw.devDependencies }).map(([name, range]) => ({ raw: `${name}@${range}`, name, range }));
    } catch { return []; }
  }

  /**
   * Resolve → write → record. Shared by the command surface and by bootstrap so
   * a bootstrapped package is indistinguishable from a user-installed one.
   */
  private async performInstall(
    manager: PackageManagerKind,
    specifiers: readonly PackageSpecifier[],
    cwd: string,
    scope: PackageScope,
    options: { dryRun?: boolean; cask?: boolean; dev?: boolean; bootstrap?: boolean; refresh?: readonly string[] } = {},
  ): Promise<string> {
    const module = managerModule(manager);
    const context = this.contextFor(module, cwd);
    const startedAt = new Date().toISOString();
    const root = this.rootFor(scope, context);
    const installed = new Map<string, string>();
    // Upgrade targets are deliberately forgotten, so the resolver picks the
    // newest match instead of keeping the version already on disk.
    const refreshing = new Set(options.refresh ?? []);
    for (const record of this.scopedRecords(manager, scope, root)) {
      if (!refreshing.has(record.name)) installed.set(record.name, record.version);
    }

    const { plan, problems } = resolve({ manager, requests: specifiers, installed });
    if (problems.length) throw new Error(problems.map((problem) => describeProblem(manager, problem)).join('\n'));
    if (!plan.length) throw new Error(`${manager}: nothing to install`);

    const reused: PackageView[] = [];
    const fresh: ResolvedPackage[] = [];
    for (const item of plan) {
      if (item.satisfied) {
        const record = this.scopedRecords(manager, scope, root).find((candidate) => candidate.name === item.name);
        // Naming a package that arrived as a dependency makes it a direct one,
        // which is what keeps autoremove from taking it later.
        if (record && item.direct) record.dependencyType = 'direct';
        if (record) reused.push(this.viewOf(record, item.entry));
        continue;
      }
      fresh.push(item);
    }

    const written: WriteResult[] = [];
    if (!options.dryRun) {
      for (const item of fresh) written.push(await this.writePackage(module, context, item, scope, root, options.dev ?? false));
    }
    const views = options.dryRun
      ? fresh.map((item) => ({
        name: item.name, version: item.version, direct: item.direct, entry: item.entry,
        installPath: installRoot(context, item.entry, item.version, scope), binaries: [],
      }))
      : written.map((result) => result.view);

    if (!options.dryRun) {
      await this.updateProjectMetadata(module, context, scope, root, options.dev ?? false);
      await this.updateNativeDatabase(module, context);
      await this.persist();
      this.recordTransaction(manager, 'install', views.filter((view) => view.direct).map((view) => view.name), written.flatMap((result) => result.record.files), startedAt);
    }

    const transcript = module.renderers.install({ context, scope, installed: views, reused, dryRun: options.dryRun ?? false, cask: options.cask ?? false, dev: options.dev ?? false });
    return this.withSummary(transcript, manager, 'installed', views.filter((view) => view.direct));
  }

  /**
   * Transcripts are the manager's authentic output. Callers that need to check
   * an outcome should assert on observable state — `listPackages()`, the
   * receipt, or the executable descriptor on PATH — rather than on prose, which
   * is why no synthetic summary line is appended here.
   */
  private withSummary(transcript: string, _manager: PackageManagerKind, _verb: string, _views: readonly PackageView[]): string {
    return transcript;
  }

  private viewOf(record: InstalledRecord, entry: CatalogPackage): PackageView {
    return {
      name: record.name, version: record.version, direct: record.dependencyType === 'direct', entry,
      installPath: record.installPath, binaries: [...record.binaries],
    };
  }

  private async writePackage(
    module: ManagerSpec,
    context: ManagerContext,
    item: ResolvedPackage,
    scope: PackageScope,
    root: string,
    dev: boolean,
  ): Promise<WriteResult> {
    const key = this.packageKey(module.id, item.name, scope, root);
    const previous = this.packages.get(key);
    if (previous) await this.deleteFiles(module, context, previous);
    const path = installRoot(context, item.entry, item.version, scope);
    const files: string[] = [];

    files.push(...await this.writePayload(module, context, item, path, scope));
    const plans = planExecutables(context, item.entry, item.version, scope);
    const binaries = await writeExecutables(this.vfs, plans);
    files.push(...binaries);
    if (item.entry.service) {
      files.push(...await writeServiceUnit(this.vfs, context, item.entry.service, item.entry.name, item.version));
      this.services.set(item.entry.service.name, {
        ...item.entry.service, package: item.entry.name, manager: module.id,
        executablePath: `${binDirectoryForBinary(context, 'system', item.entry, item.entry.service.executable)}/${executableFileName(context, item.entry.service.executable)}`,
      });
    }
    const receipt = receiptPath(this.profile, context, item.entry, item.version, path);
    const record: InstalledRecord = {
      id: previous?.id ?? randomUUID(),
      name: item.name,
      displayName: item.entry.name,
      version: item.version,
      manager: module.id,
      scope,
      installPath: path,
      installedAt: new Date().toISOString(),
      files: [...files, receipt],
      source: `registry://${module.id}/${item.name}@${item.version}`,
      integrity: createHash('sha256').update(`${module.id}:${item.name}:${item.version}`).digest('hex'),
      dependencies: [...item.dependencies],
      dependencyType: item.direct ? 'direct' : previous?.dependencyType ?? 'transitive',
      aliases: [...(item.entry.aliases ?? [])],
      binaries,
      provides: [...(item.entry.binaries ?? [])],
      requestedRange: item.requestedRange,
      dependencyRanges: { ...item.dependencyRanges },
      root,
      ...(item.entry.service ? { service: item.entry.service } : {}),
      ...(item.entry.cask ? { cask: true } : {}),
      ...(item.entry.section ? { section: item.entry.section } : {}),
      description: item.entry.description,
    };
    await this.vfs.writeFile(receipt, `${JSON.stringify({
      manager: module.id, package: item.name, version: item.version, scope, installPath: path,
      installedAt: record.installedAt, integrity: record.integrity, requestedRange: item.requestedRange,
      dependencies: item.dependencyRanges, binaries, dev, source: record.source,
    }, null, 2)}\n`);
    this.packages.set(key, record);
    const view: PackageView = {
      name: item.name, version: item.version, direct: item.direct, entry: item.entry, installPath: path, binaries,
      ...(previous && previous.version !== item.version ? { previousVersion: previous.version } : {}),
    };
    return { record, view };
  }

  /** The package payload, in whatever shape the ecosystem expects to find it. */
  private async writePayload(module: ManagerSpec, context: ManagerContext, item: ResolvedPackage, path: string, scope: PackageScope): Promise<string[]> {
    const { entry, version } = item;
    const files: string[] = [];
    const write = async (file: string, content: string): Promise<void> => { await this.vfs.writeFile(file, content); files.push(file); };
    if (['npm', 'pnpm', 'yarn', 'bun'].includes(module.id)) {
      const bin = Object.fromEntries((entry.binaries ?? []).map((binary) => [binary, './bin/cli.js']));
      await write(`${path}/package.json`, `${JSON.stringify({
        name: entry.name, version, description: entry.description, license: entry.license ?? 'MIT',
        main: 'index.js', ...(Object.keys(bin).length ? { bin } : {}), dependencies: entry.dependencies ?? {},
      }, null, 2)}\n`);
      await write(`${path}/index.js`, `// ${entry.name}@${version}\nmodule.exports = {};\n`);
      if (module.id === 'pnpm' && scope === 'project') {
        // pnpm's store layout links the flat name at the symlink position.
        const link = `${context.cwd}/node_modules/${safeName(module.id, entry.name)}`;
        if (!this.vfs.exists(link)) {
          // Scoped names nest one level deeper, so the scope directory has to exist first.
          await this.vfs.mkdir(link.slice(0, link.lastIndexOf('/')));
          await this.vfs.symlink(path, link);
          files.push(link);
        }
      }
      return files;
    }
    if (['pip', 'pipx', 'uv', 'poetry', 'conda'].includes(module.id)) {
      await write(`${path}/__init__.py`, `__version__ = "${version}"\n`);
      const site = path.slice(0, path.lastIndexOf('/'));
      const distInfo = `${site}/${safeName(module.id, entry.name).replaceAll('-', '_')}-${version}.dist-info`;
      await write(`${distInfo}/METADATA`, [
        'Metadata-Version: 2.3', `Name: ${entry.name}`, `Version: ${version}`, `Summary: ${entry.description}`,
        ...Object.entries(entry.dependencies ?? {}).map(([dependency, range]) => `Requires-Dist: ${dependency}${range === '*' ? '' : ` (${range})`}`),
        '',
      ].join('\n'));
      await write(`${distInfo}/RECORD`, `${entry.name}/__init__.py,sha256=${createHash('sha256').update(version).digest('hex').slice(0, 43)},${version.length}\n`);
      await write(`${distInfo}/WHEEL`, 'Wheel-Version: 1.0\nGenerator: seed\nRoot-Is-Purelib: true\nTag: py3-none-any\n');
      return files;
    }
    if (module.id === 'cargo') {
      await write(`${path}/Cargo.toml`, [
        '[package]', `name = "${entry.name}"`, `version = "${version}"`, 'edition = "2021"',
        `description = "${entry.description}"`, '', '[dependencies]',
        ...Object.entries(entry.dependencies ?? {}).map(([dependency, range]) => `${dependency} = "${range.replace(/^\^/, '')}"`),
        '',
      ].join('\n'));
      return files;
    }
    if (module.id === 'go') {
      await write(`${path}/go.mod`, `module ${entry.name}\n\ngo 1.25\n`);
      return files;
    }
    if (module.id === 'gem') {
      await write(`${path}/${entry.name}.gemspec`, [
        'Gem::Specification.new do |s|', `  s.name    = "${entry.name}"`, `  s.version = "${version}"`,
        `  s.summary = "${entry.description}"`, 'end', '',
      ].join('\n'));
      return files;
    }
    if (module.id === 'composer') {
      await write(`${path}/composer.json`, `${JSON.stringify({ name: entry.name, description: entry.description, version, require: entry.dependencies ?? {} }, null, 2)}\n`);
      return files;
    }
    if (module.id === 'dpkg' || module.id === 'apt') {
      await write(`${path}/seed-package.json`, `${JSON.stringify({ manager: module.id, package: entry.name, version, architecture: 'amd64', section: entry.section ?? 'misc' }, null, 2)}\n`);
      return files;
    }
    await write(`${path}/seed-package.json`, `${JSON.stringify({
      manager: module.id, package: entry.name, version, description: entry.description,
      binaries: entry.binaries ?? [], ...(entry.publisher ? { publisher: entry.publisher } : {}),
    }, null, 2)}\n`);
    return files;
  }

  // ------------------------------------------------------------------ upgrade

  private async upgrade(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    const scope = this.scopeFor(module, invocation);
    const root = this.rootFor(scope, context);
    const named = new Set((await this.specifiersFor(module, context, invocation)).map((specifier) => specifier.name.toLowerCase()));
    const candidates = this.scopedRecords(module.id, scope, root)
      .filter((record) => !named.size || named.has(record.name.toLowerCase()) || record.aliases.some((alias) => named.has(alias.toLowerCase())));
    const upgradable = candidates.filter((record) => {
      const entry = findEntry(module.id, record.name);
      if (!entry) return false;
      const best = maxSatisfyingAll(availableVersions(entry), record.dependencyType === 'direct' ? ['*'] : [record.requestedRange], rangeOptionsFor(module.id));
      return Boolean(best && compareVersions(best, record.version) > 0);
    });
    if (!upgradable.length) {
      return module.renderers.install({ context, scope, installed: [], reused: candidates.map((record) => this.viewOf(record, findEntry(module.id, record.name)!)), dryRun: false, cask: false, dev: false });
    }
    const startedAt = new Date().toISOString();
    const specifiers = upgradable.map((record) => ({ raw: record.name, name: record.name, range: '*' }));
    const output = await this.performInstall(module.id, specifiers, context.cwd, scope, {
      dryRun: invocation.dryRun, refresh: upgradable.map((record) => record.name),
    });
    this.recordTransaction(module.id, 'upgrade', upgradable.map((record) => record.name), upgradable.flatMap((record) => record.files), startedAt);
    return output;
  }

  // ------------------------------------------------------------------- remove

  private async remove(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    const scope = this.scopeFor(module, invocation);
    const root = this.rootFor(scope, context);
    const startedAt = new Date().toISOString();
    const autoremoveOnly = invocation.kind === 'autoremove';
    const targets: InstalledRecord[] = [];
    if (!autoremoveOnly) {
      for (const specifier of await this.specifiersFor(module, context, invocation)) {
        const record = this.findRecord(module, context, specifier.name, scope, root);
        if (record) targets.push(record);
      }
    }

    const removedViews: PackageView[] = [];
    for (const record of targets) {
      await this.deleteFiles(module, context, record);
      this.packages.delete(this.packageKey(module.id, record.name, record.scope, record.root));
      removedViews.push(this.viewOf(record, findEntry(module.id, record.name) ?? this.syntheticEntry(record)));
    }

    // Reference counting: what is no longer reachable from a direct package.
    const remaining = this.scopedRecords(module.id, scope, root);
    const orphanNames = findOrphans(remaining.map((record) => ({ name: record.name, dependencies: record.dependencies, dependencyType: record.dependencyType })));
    const autoremoved: PackageView[] = [];
    if (autoremoveOnly || IMPLICIT_AUTOREMOVE.has(module.id)) {
      for (const name of orphanNames) {
        const record = remaining.find((candidate) => candidate.name === name);
        if (!record) continue;
        await this.deleteFiles(module, context, record);
        this.packages.delete(this.packageKey(module.id, record.name, record.scope, record.root));
        autoremoved.push(this.viewOf(record, findEntry(module.id, record.name) ?? this.syntheticEntry(record)));
      }
    }

    await this.updateProjectMetadata(module, context, scope, root, false);
    await this.updateNativeDatabase(module, context);
    await this.persist();
    const all = [...removedViews, ...autoremoved];
    if (all.length) this.recordTransaction(module.id, 'remove', all.map((view) => view.name), all.flatMap((view) => view.binaries), startedAt);
    const transcript = module.renderers.remove({
      context, removed: removedViews, autoremoved,
      orphans: autoremoved.length ? [] : orphanNames,
      purge: invocation.purge,
    });
    return this.withSummary(transcript, module.id, 'removed', all);
  }

  private syntheticEntry(record: InstalledRecord): CatalogPackage {
    return {
      name: record.displayName, version: record.version, description: record.description,
      binaries: record.provides, ...(record.section ? { section: record.section } : {}),
      ...(record.cask ? { cask: true } : {}), aliases: record.aliases,
    };
  }

  private async deleteFiles(module: ManagerSpec, context: ManagerContext, record: InstalledRecord): Promise<void> {
    await removeExecutables(this.vfs, record.binaries, { package: record.displayName, manager: module.id });
    for (const file of record.files) {
      if (record.binaries.includes(file)) continue;
      await this.vfs.remove(file);
    }
    await this.vfs.remove(record.installPath, { recursive: true });
    if (record.service) {
      await removeServiceUnit(this.vfs, context, record.service);
      this.services.delete(record.service.name);
    }
  }

  // -------------------------------------------------------------- run / exec

  private scriptEnvironment(module: ManagerSpec, context: ManagerContext): ScriptEnvironment {
    return {
      vfs: this.vfs, context,
      searchPath: scriptSearchPath(context, module.id),
      ...(this.programRunner ? { runner: this.programRunner } : {}),
    };
  }

  private async run(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    if (TOOLCHAIN_MANAGERS.has(module.id)) return this.runToolchain(module, context, invocation.verb || 'run', invocation);
    const [script, ...rest] = invocation.operands;
    if (!script) {
      const manifest = await readScriptManifest(this.vfs, context.cwd);
      if (!manifest) throw new Error(`${module.id} error code ENOENT\n${module.id} error Could not read package.json in ${context.cwd}`);
      return listScripts(module.id, manifest);
    }
    return this.runNamedScript(module, context, script, { ...invocation, operands: rest });
  }

  private async runNamedScript(module: ManagerSpec, context: ManagerContext, script: string, invocation: ParsedInvocation): Promise<string> {
    const environment = this.scriptEnvironment(module, context);
    if (TOOLCHAIN_MANAGERS.has(module.id)) return this.runToolchain(module, context, script, invocation);
    const pid = this.processes.spawn({ executable: module.id, argv: ['run', script], cwd: context.cwd, env: {}, memoryBytes: 32 * 1024 * 1024 }).pid;
    try {
      return await runScript(environment, module.id, script, invocation.operands);
    } catch (error) {
      if (error instanceof ScriptError) throw new Error(error.message);
      throw error;
    } finally {
      this.processes.kill(pid);
    }
  }

  /** `cargo build`, `go test`, `dotnet run`, `poetry run …`: a toolchain, not a package.json script. */
  private async runToolchain(module: ManagerSpec, context: ManagerContext, verbOrCommand: string, invocation: ParsedInvocation): Promise<string> {
    const environment = this.scriptEnvironment(module, context);
    if (['poetry', 'uv', 'conda', 'composer'].includes(module.id)) {
      const command = [verbOrCommand, ...invocation.operands].join(' ').trim();
      if (!command) throw new Error(`${module.id}: run requires a command`);
      const outcome = await runCommandChain(environment, command);
      if (outcome.exitCode !== 0) throw new Error(outcome.stdout || `${module.id}: command failed with exit code ${outcome.exitCode}`);
      return outcome.stdout;
    }
    return this.build(module, context, { ...invocation, verb: verbOrCommand, kind: verbOrCommand === 'test' ? 'test' : verbOrCommand === 'run' ? 'run' : 'build' });
  }

  private async build(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation): Promise<string> {
    const verb = invocation.verb || 'build';
    if (module.id === 'cargo') return this.cargoBuild(context, verb, invocation);
    if (module.id === 'go') return this.goBuild(context, verb, invocation);
    if (module.id === 'dotnet') return this.dotnetBuild(context, verb);
    if (['npm', 'pnpm', 'yarn', 'bun'].includes(module.id)) {
      if (verb === 'init') {
        await this.vfs.writeFile(`${context.cwd}/package.json`, `${JSON.stringify({ name: context.cwd.split('/').at(-1) ?? 'seed-project', version: '1.0.0', private: true, scripts: { test: 'echo "Error: no test specified" && exit 1' } }, null, 2)}\n`);
        return `Wrote to ${context.cwd}/package.json`;
      }
      return this.runNamedScript(module, context, verb, invocation);
    }
    return `${module.id} ${verb}: nothing to do`;
  }

  private async cargoBuild(context: ManagerContext, verb: string, invocation: ParsedInvocation): Promise<string> {
    const manifest = `${context.cwd}/Cargo.toml`;
    if (verb === 'new' || verb === 'init') {
      const name = invocation.operands[0] ?? context.cwd.split('/').at(-1) ?? 'seed-crate';
      const root = verb === 'new' ? `${context.cwd}/${name}` : context.cwd;
      await this.vfs.writeFile(`${root}/Cargo.toml`, `[package]\nname = "${name}"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n`);
      await this.vfs.writeFile(`${root}/src/main.rs`, 'fn main() {\n    println!("Hello, world!");\n}\n');
      return `    Creating binary (application) \`${name}\` package\nnote: see more \`Cargo.toml\` keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html`;
    }
    if (!this.vfs.exists(manifest)) {
      throw new Error(`error: could not find \`Cargo.toml\` in \`${context.cwd}\` or any parent directory`);
    }
    const source = await this.vfs.readFile(manifest);
    const name = /name\s*=\s*"([^"]+)"/.exec(source)?.[1] ?? 'seed-crate';
    const version = /version\s*=\s*"([^"]+)"/.exec(source)?.[1] ?? '0.1.0';
    const release = invocation.flags.has('--release');
    const profile = release ? 'release' : 'dev';
    const target = `${context.cwd}/target/${release ? 'release' : 'debug'}/${executableFileName(context, name)}`;
    if (verb === 'clean') { await this.vfs.remove(`${context.cwd}/target`, { recursive: true }); return '     Removed 0 files'; }
    const lines = [`   Compiling ${name} v${version} (${context.cwd})`];
    if (verb === 'check') { lines.push(`    Finished \`${profile}\` profile [unoptimized + debuginfo] target(s) in 0.42s`); return lines.join('\n'); }
    await this.vfs.writeFile(target, `${JSON.stringify({ seedExecutable: 1, name, package: name, manager: 'cargo', version, provides: [name], behavior: name }, null, 2)}\n`);
    await this.vfs.chmod(target, EXECUTABLE_MODE);
    lines.push(`    Finished \`${profile}\` profile [${release ? 'optimized' : 'unoptimized + debuginfo'}] target(s) in 1.21s`);
    if (verb === 'test') {
      lines.push(`     Running unittests src/main.rs (target/debug/deps/${name})`);
      lines.push('');
      lines.push('running 2 tests');
      lines.push('test tests::it_works ... ok');
      lines.push('test tests::it_adds ... ok');
      lines.push('');
      lines.push('test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s');
      return lines.join('\n');
    }
    if (verb === 'run' || verb === 'r') {
      lines.push(`     Running \`target/${release ? 'release' : 'debug'}/${name}\``);
      lines.push('Hello, world!');
    }
    return lines.join('\n');
  }

  private async goBuild(context: ManagerContext, verb: string, invocation: ParsedInvocation): Promise<string> {
    const manifest = `${context.cwd}/go.mod`;
    if (verb === 'mod') {
      const sub = invocation.operands[0];
      if (sub === 'init') {
        const name = invocation.operands[1] ?? `example.com/${context.cwd.split('/').at(-1) ?? 'seed'}`;
        await this.vfs.writeFile(manifest, `module ${name}\n\ngo 1.25\n`);
        return `go: creating new go.mod: module ${name}`;
      }
      if (sub === 'tidy') return 'go: downloading modules\ngo: finding module dependencies';
      throw new Error(`go mod ${sub ?? ''}: unknown subcommand`);
    }
    if (!this.vfs.exists(manifest)) throw new Error(`go: ${context.cwd}: go.mod file not found in current directory or any parent directory`);
    const module = /module\s+(\S+)/.exec(await this.vfs.readFile(manifest))?.[1] ?? 'seed';
    const name = module.split('/').at(-1) ?? 'seed';
    if (verb === 'test') {
      return ['ok  \t' + module + '\t0.004s'].join('\n');
    }
    if (verb === 'run') return 'Hello, world!';
    if (verb === 'vet' || verb === 'fmt' || verb === 'generate') return '';
    if (verb === 'clean') { await this.vfs.remove(`${context.cwd}/${name}`); return ''; }
    const target = `${context.cwd}/${executableFileName(context, name)}`;
    await this.vfs.writeFile(target, `${JSON.stringify({ seedExecutable: 1, name, package: module, manager: 'go', version: 'devel', provides: [name], behavior: name }, null, 2)}\n`);
    await this.vfs.chmod(target, EXECUTABLE_MODE);
    return '';
  }

  private dotnetBuild(context: ManagerContext, verb: string): string {
    if (verb === 'test') {
      return ['  Determining projects to restore...', `  Restored ${context.cwd}/project.csproj (in 210 ms).`, '', 'Passed!  - Failed: 0, Passed: 2, Skipped: 0, Total: 2, Duration: 12 ms'].join('\n');
    }
    if (verb === 'run') return 'Hello, World!';
    return ['  Determining projects to restore...', `  Restored ${context.cwd}/project.csproj (in 210 ms).`, `  project -> ${context.cwd}/bin/Debug/net9.0/project.dll`, '', 'Build succeeded.', '    0 Warning(s)', '    0 Error(s)'].join('\n');
  }

  /** `npx <tool>` / `pnpm dlx` / `bunx`: install on demand, then run. */
  private async runExec(module: ManagerSpec, context: ManagerContext, args: string[]): Promise<string> {
    const operands = args.filter((argument) => !argument.startsWith('-'));
    const [tool, ...rest] = operands;
    if (!tool) throw new Error(`${module.id}: exec requires a command`);
    const environment = this.scriptEnvironment(module, context);
    const specifier = parseSpecifier(module.id, tool);
    const binary = specifier.name.includes('/') ? specifier.name.split('/').at(-1)! : specifier.name;
    let program = await import('./packages/scripts.js').then((scripts) => scripts.resolveProgram(environment, binary));
    if (!program && findEntry(module.id, specifier.name)) {
      // dlx semantics: fetch into the project, then run it.
      await this.performInstall(module.id, [specifier], context.cwd, this.scopeFor(module, { global: false } as ParsedInvocation), {});
      program = await import('./packages/scripts.js').then((scripts) => scripts.resolveProgram(environment, binary));
    }
    if (!program) throw new Error(`${module.id}: command not found: ${binary}`);
    const outcome = await runCommandChain(environment, [binary, ...rest].join(' '));
    if (outcome.exitCode !== 0) throw new Error(outcome.stdout || `${binary} exited with code ${outcome.exitCode}`);
    return outcome.stdout;
  }

  // ------------------------------------------------------------- bookkeeping

  private rootFor(scope: PackageScope, context: ManagerContext): string {
    return scope === 'project' ? canonicalPath(context.cwd) : this.home;
  }

  private packageKey(manager: PackageManagerKind, name: string, scope: PackageScope, root: string): string {
    return `${manager}:${scope}:${root}:${name}`;
  }

  private scopedRecords(manager: PackageManagerKind, scope: PackageScope, root: string): InstalledRecord[] {
    // apt and dpkg are two front ends over one Debian database, so each sees
    // what the other installed — that is what makes `dpkg -l` honest after `apt install`.
    const family = manager === 'apt' || manager === 'dpkg' ? new Set(['apt', 'dpkg']) : new Set([manager]);
    return [...this.packages.values()].filter((record) => family.has(record.manager) && record.scope === scope && record.root === root);
  }

  private recordsFor(module: ManagerSpec, context: ManagerContext, invocation: ParsedInvocation | undefined): InstalledRecord[] {
    const scope = invocation ? this.scopeFor(module, invocation) : module.family === 'native' ? 'system' : module.projectScoped ? 'project' : 'user';
    const scoped = this.scopedRecords(module.id, scope, this.rootFor(scope, context));
    if (scoped.length || scope !== 'project') return scoped;
    return this.scopedRecords(module.id, 'user', this.home);
  }

  private findRecord(module: ManagerSpec, context: ManagerContext, name: string, scope?: PackageScope, root?: string): InstalledRecord | undefined {
    const needle = name.trim().toLowerCase();
    const entry = findEntry(module.id, name);
    const matches = [...this.packages.values()].filter((record) => record.manager === module.id
      && (record.name.toLowerCase() === needle
        || record.displayName.toLowerCase() === needle
        || record.aliases.some((alias) => alias.toLowerCase() === needle)
        || (entry ? record.name === entry.name : false)));
    if (scope && root) return matches.find((record) => record.scope === scope && record.root === root) ?? matches.find((record) => record.scope === scope);
    const cwd = canonicalPath(context.cwd);
    return matches.find((record) => record.scope === 'project' && record.root === cwd) ?? matches[0];
  }

  private async persist(): Promise<void> {
    await this.vfs.writeFile(this.dbPath, `${JSON.stringify(this.listPackages(), null, 2)}\n`);
  }

  private recordTransaction(manager: PackageManagerKind, operation: PackageTransactionRecord['operation'], packages: string[], receiptPaths: string[], startedAt: string): void {
    this.transactions.push({ id: randomUUID(), manager, operation, packages, startedAt, completedAt: new Date().toISOString(), status: 'committed', receiptPaths });
    if (this.transactions.length > 300) this.transactions.shift();
  }

  /** Debian keeps `/var/lib/dpkg/status`; keeping it current is what makes `dpkg -s` honest. */
  private async updateNativeDatabase(module: ManagerSpec, context: ManagerContext): Promise<void> {
    if (module.id !== 'apt' && module.id !== 'dpkg') return;
    const records = [...this.packages.values()].filter((record) => record.manager === 'apt' || record.manager === 'dpkg');
    const stanzas = records.map((record) => [
      `Package: ${record.aliases[0] ?? record.name}`,
      'Status: install ok installed',
      'Priority: optional',
      `Section: ${record.section ?? 'misc'}`,
      `Installed-Size: ${Math.max(1, Math.round(record.files.length * 64))}`,
      'Architecture: amd64',
      `Version: ${record.version}`,
      ...(Object.keys(record.dependencyRanges).length ? [`Depends: ${Object.entries(record.dependencyRanges).map(([name, range]) => `${name}${range === '*' ? '' : ` (${range})`}`).join(', ')}`] : []),
      `Description: ${record.description}`,
      '',
    ].join('\n'));
    void context;
    await this.vfs.writeFile('/var/lib/dpkg/status', stanzas.join('\n'));
  }

  /** Manifests and lockfiles the project-scoped managers own. */
  private async updateProjectMetadata(module: ManagerSpec, context: ManagerContext, scope: PackageScope, root: string, dev: boolean): Promise<void> {
    if (module.id === 'pip' || module.id === 'uv') {
      const requirements = `${context.cwd}/requirements.txt`;
      if (this.vfs.exists(requirements)) {
        const records = this.scopedRecords(module.id, scope, root).filter((record) => record.dependencyType === 'direct');
        await this.vfs.writeFile(requirements, `${records.map((record) => `${record.name}==${record.version}`).sort().join('\n')}\n`);
      }
      return;
    }
    if (!module.projectScoped || scope !== 'project') return;
    const records = this.scopedRecords(module.id, scope, root);
    const direct = records.filter((record) => record.dependencyType === 'direct');
    const dependencies = Object.fromEntries(direct.map((record) => [record.name, defaultRangeFor(module.id, record.version)]));

    if (['npm', 'pnpm', 'yarn', 'bun'].includes(module.id)) {
      let manifest: Record<string, unknown> = { name: 'seed-project', version: '1.0.0', private: true };
      try { manifest = JSON.parse(await this.vfs.readFile(`${root}/package.json`)) as Record<string, unknown>; } catch { /* create it */ }
      const key = dev ? 'devDependencies' : 'dependencies';
      const existingOther = (manifest[dev ? 'dependencies' : 'devDependencies'] ?? {}) as Record<string, string>;
      const merged = dev
        ? { ...manifest, dependencies: existingOther, devDependencies: dependencies }
        : { ...manifest, dependencies, ...(Object.keys(existingOther).length ? { devDependencies: existingOther } : {}) };
      void key;
      await this.vfs.writeFile(`${root}/package.json`, `${JSON.stringify(merged, null, 2)}\n`);
      const lockPath = module.id === 'npm' ? `${root}/package-lock.json`
        : module.id === 'pnpm' ? `${root}/pnpm-lock.yaml`
          : module.id === 'bun' ? `${root}/bun.lock` : `${root}/yarn.lock`;
      await this.vfs.writeFile(lockPath, this.renderJsLock(module.id, root, records, direct));
      return;
    }
    if (module.id === 'poetry') {
      await this.vfs.writeFile(`${root}/pyproject.toml`, [
        '[tool.poetry]', 'name = "seed-project"', 'version = "0.1.0"', 'description = ""', '',
        '[tool.poetry.dependencies]', 'python = "^3.13"',
        ...direct.map((record) => `${record.name} = "^${record.version}"`),
        '', '[build-system]', 'requires = ["poetry-core"]', 'build-backend = "poetry.core.masonry.api"', '',
      ].join('\n'));
      await this.vfs.writeFile(`${root}/poetry.lock`, [
        ...records.map((record) => [
          '[[package]]', `name = "${record.name}"`, `version = "${record.version}"`,
          `description = "${record.description}"`, 'optional = false', 'python-versions = ">=3.9"', '',
        ].join('\n')),
        '[metadata]', 'lock-version = "2.1"', 'python-versions = "^3.13"', '',
      ].join('\n'));
      return;
    }
    if (module.id === 'composer') {
      let manifest: Record<string, unknown> = { name: 'seed/project', type: 'project' };
      try { manifest = JSON.parse(await this.vfs.readFile(`${root}/composer.json`)) as Record<string, unknown>; } catch { /* create it */ }
      await this.vfs.writeFile(`${root}/composer.json`, `${JSON.stringify({ ...manifest, require: dependencies }, null, 2)}\n`);
      await this.vfs.writeFile(`${root}/composer.lock`, `${JSON.stringify({
        _readme: ['This file locks the dependencies of your project to a known state'],
        'content-hash': createHash('sha256').update(JSON.stringify(dependencies)).digest('hex').slice(0, 32),
        packages: records.map((record) => ({ name: record.name, version: record.version, type: 'library', description: record.description })),
        'packages-dev': [], 'plugin-api-version': '2.6.0',
      }, null, 2)}\n`);
      return;
    }
    if (module.id === 'nuget') {
      await this.vfs.writeFile(`${root}/packages.lock.json`, `${JSON.stringify({
        version: 1,
        dependencies: { 'net9.0': Object.fromEntries(records.map((record) => [record.name, { type: record.dependencyType === 'direct' ? 'Direct' : 'Transitive', resolved: record.version, contentHash: record.integrity }])) },
      }, null, 2)}\n`);
      return;
    }
    if (module.id === 'vcpkg') {
      await this.vfs.writeFile(`${root}/vcpkg.json`, `${JSON.stringify({
        name: 'seed-project', version: '0.1.0',
        dependencies: direct.map((record) => record.name),
      }, null, 2)}\n`);
    }
  }

  private renderJsLock(manager: PackageManagerKind, root: string, records: readonly InstalledRecord[], direct: readonly InstalledRecord[]): string {
    if (manager === 'pnpm') {
      return [
        "lockfileVersion: '9.0'",
        '',
        'settings:',
        '  autoInstallPeers: true',
        '  excludeLinksFromLockfile: false',
        '',
        'importers:',
        '',
        '  .:',
        ...(direct.length ? ['    dependencies:', ...direct.flatMap((record) => [`      ${record.name}:`, `        specifier: ${defaultRangeFor(manager, record.version)}`, `        version: ${record.version}`])] : []),
        '',
        'packages:',
        '',
        ...records.flatMap((record) => [`  ${record.name}@${record.version}:`, `    resolution: {integrity: sha512-${record.integrity.slice(0, 40)}}`, '']),
      ].join('\n');
    }
    if (manager === 'yarn') {
      return [
        '# THIS IS AN AUTOGENERATED FILE. DO NOT EDIT THIS FILE DIRECTLY.',
        '# yarn lockfile v1',
        '',
        ...records.flatMap((record) => [
          `"${record.name}@${defaultRangeFor(manager, record.version)}":`,
          `  version "${record.version}"`,
          `  resolved "https://registry.seed.local/${record.name}/-/${record.name.split('/').at(-1)}-${record.version}.tgz#${record.integrity.slice(0, 40)}"`,
          `  integrity sha512-${record.integrity.slice(0, 40)}`,
          ...(Object.keys(record.dependencyRanges).length ? ['  dependencies:', ...Object.entries(record.dependencyRanges).map(([name, range]) => `    ${name} "${range}"`)] : []),
          '',
        ]),
      ].join('\n');
    }
    if (manager === 'bun') {
      return `${JSON.stringify({
        lockfileVersion: 1,
        workspaces: { '': { name: 'seed-project', dependencies: Object.fromEntries(direct.map((record) => [record.name, defaultRangeFor(manager, record.version)])) } },
        packages: Object.fromEntries(records.map((record) => [record.name, [`${record.name}@${record.version}`, {}, record.dependencyRanges, `sha512-${record.integrity.slice(0, 40)}`]])),
      }, null, 2)}\n`;
    }
    return `${JSON.stringify({
      name: root.split('/').at(-1) ?? 'seed-project',
      version: '1.0.0',
      lockfileVersion: 3,
      requires: true,
      packages: {
        '': { name: root.split('/').at(-1) ?? 'seed-project', version: '1.0.0', dependencies: Object.fromEntries(direct.map((record) => [record.name, defaultRangeFor(manager, record.version)])) },
        ...Object.fromEntries(records.map((record) => [`node_modules/${record.name}`, {
          version: record.version,
          resolved: `https://registry.seed.local/${record.name}/-/${record.name.split('/').at(-1)}-${record.version}.tgz`,
          integrity: `sha512-${record.integrity.slice(0, 40)}`,
          ...(Object.keys(record.dependencyRanges).length ? { dependencies: record.dependencyRanges } : {}),
          ...(record.provides.length ? { bin: Object.fromEntries(record.provides.map((binary) => [binary, 'bin/cli.js'])) } : {}),
        }])),
      },
    }, null, 2)}\n`;
  }
}

/** Highest catalog version satisfying a range — exported for callers doing their own planning. */
export function resolveVersion(manager: PackageManagerKind, name: string, range: string): string | undefined {
  const entry = findEntry(manager, name);
  if (!entry) return undefined;
  const versions = availableVersions(entry);
  return maxSatisfyingAll(versions, [range], rangeOptionsFor(manager))
    ?? (satisfies(entry.version, range) ? entry.version : highestVersion(versions));
}
