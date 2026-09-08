import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import type { ComputerSpec, OSRuntimeProfile } from '@tcn-computer/protocol';
import { availableVersions, catalogSize, findEntry, registries } from './packages/catalog.js';
import { parseDescriptor } from './packages/executables.js';
import { describeProblem, findOrphans, resolve } from './packages/resolver.js';
import { parseSpecifier } from './packages/specifier.js';
import type { SeedExecutableDescriptor } from './packages/types.js';
import { compareVersions, highestVersion, maxSatisfyingAll, parseVersion, satisfies } from './packages/versions.js';
import { ProcessManager } from './processes.js';
import { SoftwareEnvironment } from './software.js';
import { VirtualFileSystem } from './vfs.js';

/* -------------------------------------------------------------------------- *
 * Harness: a real VFS and process table behind one software environment. The
 * profiles mirror the shipped OS packages, including their receipt roots, so
 * path assertions here are assertions about the real layout.
 * -------------------------------------------------------------------------- */

const macProfile: OSRuntimeProfile = {
  shell: { default: 'zsh', executable: '/bin/zsh', promptDialect: 'posix', startupFiles: ['~/.zshrc'] },
  filesystem: { root: '/', home: '/Users/agent', applications: '/Applications', userData: '/Users/agent/Library', temporary: '/private/tmp', caseSensitive: false, pathSeparator: '/' },
  packageManagers: {
    native: ['mas', 'brew'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'gem', 'composer', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/var/db/receipts', '/opt/homebrew/Cellar', '/Applications'],
  },
  bootServices: [{ id: 'launchd', executable: '/sbin/launchd', role: 'init', parent: null, required: true }],
  conventions: { executableSuffix: '', sharedLibrarySuffix: '.dylib', environmentPathKey: 'PATH', localhostNames: ['localhost'] },
};

const ubuntuProfile: OSRuntimeProfile = {
  ...macProfile,
  shell: { default: 'bash', executable: '/usr/bin/bash', promptDialect: 'posix', startupFiles: ['~/.bashrc'] },
  filesystem: { root: '/', home: '/home/agent', applications: '/opt', userData: '/home/agent/.config', temporary: '/tmp', caseSensitive: true, pathSeparator: '/' },
  packageManagers: {
    native: ['apt', 'dpkg', 'snap', 'flatpak'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'gem', 'composer', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/var/lib/dpkg', '/var/lib/snapd', '/var/lib/flatpak', '/var/lib/seed/apps'],
  },
};

const windowsProfile: OSRuntimeProfile = {
  shell: { default: 'powershell', executable: 'C:\\Program Files\\PowerShell\\7\\pwsh.exe', promptDialect: 'powershell', startupFiles: ['$PROFILE'] },
  filesystem: { root: '/C', home: '/C/Users/agent', applications: '/C/Program Files', userData: '/C/Users/agent/AppData/Roaming', temporary: '/C/Users/agent/AppData/Local/Temp', caseSensitive: false, pathSeparator: '\\' },
  packageManagers: {
    native: ['winget', 'choco', 'scoop'],
    language: ['npm', 'pnpm', 'yarn', 'bun', 'pip', 'pipx', 'poetry', 'uv', 'cargo', 'go', 'dotnet', 'nuget', 'vcpkg', 'conda'],
    receiptRoots: ['/C/ProgramData/Seed/AppRepository', '/C/ProgramData/chocolatey', '/C/Users/agent/scoop/apps'],
  },
  bootServices: [{ id: 'smss', executable: 'C:\\Windows\\System32\\smss.exe', role: 'init', parent: null, required: true }],
  conventions: { executableSuffix: '.exe', sharedLibrarySuffix: '.dll', environmentPathKey: 'Path', localhostNames: ['localhost'] },
};

function specFor(os: ComputerSpec['os'], id: string): ComputerSpec {
  return {
    id, hostname: id, os, shell: os === 'windows' ? 'powershell' : os === 'macos' ? 'zsh' : 'bash',
    ipv4: '10.42.0.30', memoryBytes: 16 * 1024 ** 3, cpuCores: 8,
    disks: [{ id: 'disk0', label: 'Seed', mount: os === 'windows' ? 'C:' : '/', capacityBytes: 512 * 1024 ** 3 }],
    displays: [],
  };
}

interface Box {
  software: SoftwareEnvironment;
  vfs: VirtualFileSystem;
  home: string;
  profile: OSRuntimeProfile;
  run(command: string, cwd?: string): Promise<string>;
  fails(command: string, cwd?: string): Promise<string>;
}

let stateRoot = '';
let counter = 0;

beforeAll(async () => { stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-software-')); });

async function box(os: ComputerSpec['os'] = 'ubuntu', options: { bootstrap?: boolean } = {}): Promise<Box> {
  const profile = os === 'windows' ? windowsProfile : os === 'macos' ? macProfile : ubuntuProfile;
  const spec = specFor(os, `${os}-${++counter}`);
  const vfs = new VirtualFileSystem(stateRoot, `run-${counter}`, spec, { caseSensitive: profile.filesystem.caseSensitive });
  await vfs.initialize();
  await vfs.mkdir(profile.filesystem.home);
  const processes = new ProcessManager(spec.id);
  processes.boot('init', profile.filesystem.home, {});
  const software = new SoftwareEnvironment(spec, vfs, processes, profile);
  if (options.bootstrap) await software.initialize();
  const split = (command: string): [string, string[]] => {
    const tokens = command.match(/"[^"]*"|'[^']*'|\S+/g)?.map((token) => token.replace(/^["']|["']$/g, '')) ?? [];
    return [tokens[0] ?? '', tokens.slice(1)];
  };
  return {
    software, vfs, home: profile.filesystem.home, profile,
    async run(command, cwd = profile.filesystem.home) {
      const [manager, args] = split(command);
      return software.packageCommand(manager, args, cwd);
    },
    async fails(command, cwd = profile.filesystem.home) {
      const [manager, args] = split(command);
      try {
        await software.packageCommand(manager, args, cwd);
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
      throw new Error(`expected \`${command}\` to fail`);
    },
  };
}

async function descriptorAt(vfs: VirtualFileSystem, file: string): Promise<SeedExecutableDescriptor> {
  const inode = vfs.statSync(file);
  expect(inode, `expected an executable at ${file}`).toBeDefined();
  expect(inode!.kind).toBe('file');
  expect(inode!.mode & 0o7777).toBe(0o755);
  const descriptor = parseDescriptor(await vfs.readFile(file));
  expect(descriptor, `expected a seed executable descriptor at ${file}`).toBeDefined();
  return descriptor!;
}

/* ------------------------------------------------------------------ versions */

describe('version algebra', () => {
  it('orders semver, prereleases, Debian epochs and packaging revisions', () => {
    expect(compareVersions('1.2.3', '1.10.0')).toBe(-1);
    expect(compareVersions('2.0.0', '2.0.0')).toBe(0);
    expect(compareVersions('1.0.0-beta.2', '1.0.0')).toBe(-1);
    expect(compareVersions('1.0.0-alpha', '1.0.0-beta')).toBe(-1);
    expect(compareVersions('1.0.0-rc.1', '1.0.0-rc.2')).toBe(-1);
    // A Debian epoch outranks any upstream version.
    expect(compareVersions('1:1.0', '9.9.9')).toBe(1);
    // A packaging revision sorts above the plain upstream version.
    expect(compareVersions('1.26.3-1ubuntu1', '1.26.3')).toBe(1);
    expect(compareVersions('1.26.3-1ubuntu1', '1.26.3-2ubuntu1')).toBe(-1);
    // Homebrew bottle revision.
    expect(compareVersions('2.48.1_1', '2.48.1')).toBe(1);
    // PEP 440 glues the prerelease onto the release number.
    expect(compareVersions('2.2.0rc1', '2.2.0')).toBe(-1);
    expect(highestVersion(['1.0.0', '1.10.0', '1.9.0'])).toBe('1.10.0');
  });

  it('parses epochs, prereleases and revisions into their own fields', () => {
    expect(parseVersion('1:1.24.0-2ubuntu7')).toMatchObject({ epoch: 1, parts: [1, 24, 0], pre: [], revision: '2ubuntu7' });
    expect(parseVersion('v1.0.0-beta.3')).toMatchObject({ epoch: 0, parts: [1, 0, 0], pre: ['beta', 3] });
  });

  it('satisfies caret, tilde, wildcard, comparator, hyphen and union ranges', () => {
    expect(satisfies('1.2.9', '^1.2.3')).toBe(true);
    expect(satisfies('2.0.0', '^1.2.3')).toBe(false);
    expect(satisfies('0.2.9', '^0.2.3')).toBe(true);
    expect(satisfies('0.3.0', '^0.2.3')).toBe(false);
    expect(satisfies('1.2.9', '~1.2.3')).toBe(true);
    expect(satisfies('1.3.0', '~1.2.3')).toBe(false);
    expect(satisfies('1.2.9', '1.2.x')).toBe(true);
    expect(satisfies('1.3.0', '1.2.*')).toBe(false);
    expect(satisfies('2.5.0', '>=2.0.0 <3.0.0')).toBe(true);
    expect(satisfies('3.0.0', '>=2.0.0 <3.0.0')).toBe(false);
    expect(satisfies('1.5.0', '1.0.0 - 2.0.0')).toBe(true);
    expect(satisfies('7.1.5', '^5.0.0 || ^6.0.0 || ^7.0.0')).toBe(true);
    expect(satisfies('4.0.0', '^5.0.0 || ^6.0.0 || ^7.0.0')).toBe(false);
    expect(satisfies('1.2.3', '*')).toBe(true);
    expect(satisfies('1.2.3', '!=1.2.3')).toBe(false);
  });

  it('reads the tilde the way each ecosystem does', () => {
    // npm: `~1.2` pins the minor.
    expect(satisfies('1.3.0', '~1.2')).toBe(false);
    // Composer/RubyGems: the last named component is the one allowed to vary.
    expect(satisfies('1.3.0', '~1.2', { tildeCompatible: true })).toBe(true);
    expect(satisfies('2.0.0', '~1.2', { tildeCompatible: true })).toBe(false);
    // RubyGems' pessimistic operator means the same thing everywhere.
    expect(satisfies('1.9.0', '~> 1.0')).toBe(true);
    expect(satisfies('2.0.0', '~> 1.0')).toBe(false);
    // PEP 440.
    expect(satisfies('1.2.9', '~=1.2.3')).toBe(true);
    expect(satisfies('1.3.0', '~=1.2.3')).toBe(false);
  });

  it('reads a bare version as an exact pin, except where the manager means caret', () => {
    expect(satisfies('1.2.4', '1.2.3')).toBe(false);
    expect(satisfies('1.2.4', '1.2.3', { bareCaret: true })).toBe(true);
    expect(satisfies('2.0.0', '1.2.3', { bareCaret: true })).toBe(false);
  });

  it('picks the highest version satisfying every range at once', () => {
    const versions = ['1.0.0', '1.5.0', '2.0.0', '2.5.0', '3.0.0'];
    expect(maxSatisfyingAll(versions, ['>=1.0.0'])).toBe('3.0.0');
    expect(maxSatisfyingAll(versions, ['>=1.0.0', '<3.0.0'])).toBe('2.5.0');
    expect(maxSatisfyingAll(versions, ['^1.0.0', '>=2.0.0'])).toBeUndefined();
  });
});

/* ---------------------------------------------------------------- specifiers */

describe('specifier parsing', () => {
  it('separates an npm name from its range, including scoped names', () => {
    expect(parseSpecifier('npm', 'react')).toMatchObject({ name: 'react', range: '*' });
    expect(parseSpecifier('npm', 'react@18.2.0')).toMatchObject({ name: 'react', range: '18.2.0' });
    expect(parseSpecifier('npm', 'react@^18.0.0')).toMatchObject({ name: 'react', range: '^18.0.0' });
    expect(parseSpecifier('npm', '@types/node')).toMatchObject({ name: '@types/node', range: '*' });
    expect(parseSpecifier('npm', '@types/node@^24.0.0')).toMatchObject({ name: '@types/node', range: '^24.0.0' });
    expect(parseSpecifier('pnpm', '@typescript-eslint/parser@8.44.0')).toMatchObject({ name: '@typescript-eslint/parser', range: '8.44.0' });
    expect(parseSpecifier('npm', 'vite@latest')).toMatchObject({ name: 'vite', range: '*', tag: 'latest' });
    expect(parseSpecifier('npm', 'vite@next')).toMatchObject({ name: 'vite', tag: 'next' });
  });

  it('recognizes non-registry npm sources instead of inventing a package name', () => {
    expect(parseSpecifier('npm', 'file:../local-lib')).toMatchObject({ name: 'local-lib', source: { kind: 'file' } });
    expect(parseSpecifier('npm', 'github:seed/tool')).toMatchObject({ name: 'tool', source: { kind: 'git' } });
    expect(parseSpecifier('npm', 'https://registry.seed.local/x/-/x-1.0.0.tgz')).toMatchObject({ source: { kind: 'url' } });
  });

  it('parses PEP 508 operators and extras', () => {
    expect(parseSpecifier('pip', 'numpy==2.3.3')).toMatchObject({ name: 'numpy', range: '==2.3.3' });
    expect(parseSpecifier('pip', 'requests>=2.30')).toMatchObject({ name: 'requests', range: '>=2.30' });
    expect(parseSpecifier('pip', 'django~=5.2')).toMatchObject({ name: 'django', range: '~=5.2' });
    expect(parseSpecifier('pip', 'requests>=2.30,<3')).toMatchObject({ name: 'requests', range: '>=2.30,<3' });
    expect(parseSpecifier('pip', 'fastapi[all]==0.116.1')).toMatchObject({ name: 'fastapi', range: '==0.116.1', extras: ['all'] });
    expect(parseSpecifier('conda', 'numpy=2.3.3')).toMatchObject({ name: 'numpy', range: '==2.3.3' });
  });

  it('parses Debian pins, architecture qualifiers and release suffixes', () => {
    expect(parseSpecifier('apt', 'nginx=1.26.3-1ubuntu1')).toMatchObject({ name: 'nginx', range: '=1.26.3-1ubuntu1' });
    expect(parseSpecifier('apt', 'nginx:amd64')).toMatchObject({ name: 'nginx', architecture: 'amd64' });
    expect(parseSpecifier('apt', 'nginx/noble')).toMatchObject({ name: 'nginx', range: '*' });
    expect(parseSpecifier('dpkg', 'seed-agent_1.0_amd64.deb')).toMatchObject({ name: 'seed-agent_1.0_amd64.deb' });
  });

  it('keeps a versioned Homebrew formula name intact', () => {
    // `python@3.13` is the formula's name, not `python` pinned to 3.13.
    expect(parseSpecifier('brew', 'python@3.13')).toMatchObject({ name: 'python@3.13', range: '*' });
    expect(parseSpecifier('brew', 'postgresql@17')).toMatchObject({ name: 'postgresql@17' });
    expect(parseSpecifier('brew', 'node@22')).toMatchObject({ name: 'node@22' });
    // An unknown formula with a version-shaped suffix does split.
    expect(parseSpecifier('brew', 'someformula@1.2.3')).toMatchObject({ name: 'someformula', range: '1.2.3' });
  });

  it('parses cargo, go, composer, gem and nuget syntaxes', () => {
    expect(parseSpecifier('cargo', 'ripgrep@14.1.1')).toMatchObject({ name: 'ripgrep', range: '14.1.1' });
    expect(parseSpecifier('go', 'github.com/junegunn/fzf@v0.65.0')).toMatchObject({ name: 'github.com/junegunn/fzf', range: 'v0.65.0' });
    expect(parseSpecifier('go', 'golang.org/x/tools/gopls')).toMatchObject({ name: 'golang.org/x/tools/gopls', range: '*' });
    expect(parseSpecifier('go', 'example.com/unlisted/tool')).toMatchObject({ name: 'example.com/unlisted/tool', tag: 'latest' });
    expect(parseSpecifier('composer', 'symfony/console:^7.2')).toMatchObject({ name: 'symfony/console', range: '^7.2' });
    expect(parseSpecifier('composer', 'laravel/framework')).toMatchObject({ name: 'laravel/framework', range: '*' });
    expect(parseSpecifier('gem', 'rails:8.0.2.1')).toMatchObject({ name: 'rails', range: '8.0.2.1' });
    expect(parseSpecifier('nuget', 'Serilog@4.3.0')).toMatchObject({ name: 'Serilog', range: '4.3.0' });
  });
});

/* ------------------------------------------------------------------ resolver */

describe('dependency resolution', () => {
  it('computes a transitive closure ordered dependencies-first', () => {
    const { plan, problems } = resolve({ manager: 'npm', requests: [{ raw: 'vite', name: 'vite', range: '*' }] });
    expect(problems).toEqual([]);
    const names = plan.map((item) => item.name);
    expect(names).toContain('vite');
    expect(names).toEqual(expect.arrayContaining(['esbuild', 'rollup', 'postcss']));
    // postcss pulls nanoid, which is two levels down from vite.
    expect(names).toContain('nanoid');
    expect(names.indexOf('esbuild')).toBeLessThan(names.indexOf('vite'));
    expect(plan.find((item) => item.name === 'vite')?.direct).toBe(true);
    expect(plan.find((item) => item.name === 'esbuild')?.direct).toBe(false);
  });

  it('intersects competing ranges rather than taking the newest of each', () => {
    // postcss requires nanoid ^3.3.8 even though nanoid 5.x exists.
    const { plan } = resolve({ manager: 'npm', requests: [{ raw: 'postcss', name: 'postcss', range: '*' }] });
    const nanoid = plan.find((item) => item.name === 'nanoid');
    expect(nanoid?.version).toBe('3.3.11');
    expect(satisfies(nanoid!.version, '^3.3.8')).toBe(true);
  });

  it('honours an explicit version request over the newest release', () => {
    const { plan } = resolve({ manager: 'npm', requests: [{ raw: 'react@18.2.0', name: 'react', range: '18.2.0' }] });
    expect(plan.find((item) => item.name === 'react')?.version).toBe('18.2.0');
  });

  it('keeps an already-installed version when it still satisfies everything', () => {
    const installed = new Map([['typescript', '5.6.3']]);
    const { plan } = resolve({ manager: 'npm', requests: [{ raw: 'typescript@^5.0.0', name: 'typescript', range: '^5.0.0' }], installed });
    expect(plan.find((item) => item.name === 'typescript')).toMatchObject({ version: '5.6.3', satisfied: true });
  });

  it('reports an unsatisfiable requirement with every contributing constraint', () => {
    const { problems } = resolve({ manager: 'npm', requests: [{ raw: 'typescript@^9.0.0', name: 'typescript', range: '^9.0.0' }] });
    expect(problems).toHaveLength(1);
    const message = describeProblem('npm', problems[0]!);
    expect(message).toContain('could not resolve dependencies for typescript');
    expect(message).toContain('requested requires typescript@^9.0.0');
    expect(message).toContain('available versions:');
  });

  it('reports an unknown package instead of installing a placeholder', () => {
    const { plan, problems } = resolve({ manager: 'apt', requests: [{ raw: 'not-a-package', name: 'not-a-package', range: '*' }] });
    expect(plan).toEqual([]);
    expect(describeProblem('apt', problems[0]!)).toBe('apt: unable to locate package not-a-package');
  });

  it('detects a declared conflict between two co-installed packages', () => {
    const { problems } = resolve({
      manager: 'npm',
      requests: [{ raw: 'jest', name: 'jest', range: '*' }, { raw: 'vitest', name: 'vitest', range: '*' }],
    });
    const conflict = problems.find((problem) => problem.kind === 'conflict');
    expect(conflict).toBeDefined();
    expect(describeProblem('npm', conflict!)).toContain('conflicts with vitest');
  });

  it('reference-counts transitive packages to find orphans', () => {
    const packages = [
      { name: 'vite', dependencies: ['esbuild', 'rollup'], dependencyType: 'direct' as const },
      { name: 'esbuild', dependencies: [], dependencyType: 'transitive' as const },
      { name: 'rollup', dependencies: ['@types/estree'], dependencyType: 'transitive' as const },
      { name: '@types/estree', dependencies: [], dependencyType: 'transitive' as const },
      { name: 'orphaned', dependencies: [], dependencyType: 'transitive' as const },
    ];
    expect(findOrphans(packages)).toEqual(['orphaned']);
    expect(findOrphans(packages.filter((item) => item.name !== 'vite'))).toEqual(expect.arrayContaining(['esbuild', 'rollup', '@types/estree', 'orphaned']));
  });
});

/* --------------------------------------------------------- executable contract */

describe('the executable descriptor contract', () => {
  it('writes a 0o755 JSON descriptor for every binary a package provides', async () => {
    const machine = await box();
    await machine.run('apt install ripgrep');
    const descriptor = await descriptorAt(machine.vfs, '/usr/bin/rg');
    expect(descriptor).toMatchObject({
      seedExecutable: 1, name: 'rg', package: 'ripgrep', manager: 'apt', behavior: 'rg', provides: ['rg'],
    });
    expect(descriptor.version).toBe('14.1.1-1');
  });

  it('writes one descriptor per binary and lists the package siblings in provides', async () => {
    const machine = await box('macos');
    await machine.run('brew install node');
    for (const binary of ['node', 'npm', 'npx', 'corepack']) {
      const descriptor = await descriptorAt(machine.vfs, `/opt/homebrew/bin/${binary}`);
      expect(descriptor.name).toBe(binary);
      expect(descriptor.package).toBe('node');
      expect(descriptor.provides).toEqual(['node', 'npm', 'npx', 'corepack']);
    }
  });

  it('keeps the behavior key canonical when the distro renames the binary', async () => {
    const machine = await box();
    await machine.run('apt install fd-find bat');
    // Debian ships these as fdfind/batcat; the behavior the shell implements is fd/bat.
    expect((await descriptorAt(machine.vfs, '/usr/bin/fdfind')).behavior).toBe('fd');
    expect((await descriptorAt(machine.vfs, '/usr/bin/batcat')).behavior).toBe('bat');
  });

  it('places each manager\'s binaries in its real bin directory', async () => {
    const linux = await box();
    await linux.run('cargo install just');
    await linux.run('pipx install ruff');
    await linux.run('go install github.com/junegunn/fzf@latest');
    await linux.run('snap install gh');
    expect((await descriptorAt(linux.vfs, '/home/agent/.cargo/bin/just')).manager).toBe('cargo');
    expect((await descriptorAt(linux.vfs, '/home/agent/.local/bin/ruff')).manager).toBe('pipx');
    expect((await descriptorAt(linux.vfs, '/home/agent/go/bin/fzf')).manager).toBe('go');
    expect((await descriptorAt(linux.vfs, '/snap/bin/gh')).manager).toBe('snap');

    const mac = await box('macos');
    await mac.run('brew install ripgrep');
    expect((await descriptorAt(mac.vfs, '/opt/homebrew/bin/rg')).package).toBe('ripgrep');
  });

  it('appends conventions.executableSuffix on Windows', async () => {
    const machine = await box('windows');
    await machine.run('winget install GitHub.cli');
    await machine.run('scoop install jq');
    await machine.run('choco install ripgrep');
    expect((await descriptorAt(machine.vfs, '/C/Users/agent/AppData/Local/Microsoft/WinGet/Links/gh.exe')).behavior).toBe('gh');
    expect((await descriptorAt(machine.vfs, '/C/Users/agent/scoop/shims/jq.exe')).behavior).toBe('jq');
    expect((await descriptorAt(machine.vfs, '/C/ProgramData/chocolatey/bin/rg.exe')).behavior).toBe('rg');
    expect(machine.vfs.statSync('/C/Users/agent/scoop/shims/jq')).toBeUndefined();
  });

  it('installs project-scoped JS tools into node_modules/.bin', async () => {
    const machine = await box();
    const project = '/home/agent/app';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'app', version: '1.0.0' }));
    await machine.run('npm install typescript', project);
    const descriptor = await descriptorAt(machine.vfs, `${project}/node_modules/.bin/tsc`);
    expect(descriptor).toMatchObject({ package: 'typescript', manager: 'npm', behavior: 'tsc' });
    // Not a global install.
    expect(machine.vfs.statSync('/home/agent/.local/bin/tsc')).toBeUndefined();
  });

  it('removes the descriptors it wrote when the package is removed', async () => {
    const machine = await box();
    await machine.run('apt install ripgrep tree');
    expect(machine.vfs.statSync('/usr/bin/rg')).toBeDefined();
    await machine.run('apt remove ripgrep');
    expect(machine.vfs.statSync('/usr/bin/rg')).toBeUndefined();
    // A package that was not removed keeps its binary.
    expect(machine.vfs.statSync('/usr/bin/tree')).toBeDefined();
  });

  it('leaves a descriptor another package owns alone', async () => {
    const machine = await box('macos');
    await machine.run('brew install docker');
    await machine.run('brew install --cask orbstack');
    // Both ship a `docker` binary; orbstack wrote last and owns the descriptor.
    expect((await descriptorAt(machine.vfs, '/opt/homebrew/bin/docker')).package).toBe('orbstack');
    await machine.run('brew uninstall docker');
    expect((await descriptorAt(machine.vfs, '/opt/homebrew/bin/docker')).package).toBe('orbstack');
  });
});

/* ------------------------------------------------------------------ services */

describe('service registration', () => {
  it('registers a startable daemon with a unit file and an index entry', async () => {
    const machine = await box();
    await machine.run('apt install nginx');
    const descriptor = await descriptorAt(machine.vfs, '/usr/sbin/nginx');
    expect(descriptor.service).toMatchObject({ name: 'nginx', port: 80, protocol: 'http' });
    const unit = await machine.vfs.readFile('/lib/systemd/system/nginx.service');
    expect(unit).toContain('Description=nginx HTTP server');
    expect(unit).toContain('ExecStart=/usr/sbin/nginx');
    const index = JSON.parse(await machine.vfs.readFile('/var/lib/seed/services.json')) as Record<string, { executablePath: string; port: number }>;
    expect(index.nginx).toMatchObject({ executablePath: '/usr/sbin/nginx', port: 80 });
    expect(machine.software.listServices().map((service) => service.name)).toContain('nginx');
  });

  it('registers postgres and redis daemons and drops them on removal', async () => {
    const machine = await box();
    await machine.run('apt install postgresql-17 redis-server');
    expect(machine.software.listServices().map((service) => service.name).sort()).toEqual(['postgresql', 'redis']);
    await machine.run('apt remove redis-server');
    expect(machine.software.listServices().map((service) => service.name)).toEqual(['postgresql']);
    expect(machine.vfs.statSync('/lib/systemd/system/redis.service')).toBeUndefined();
  });

  it('writes a launchd plist on macOS instead of a systemd unit', async () => {
    const machine = await box('macos');
    await machine.run('brew install redis');
    const plist = await machine.vfs.readFile('/opt/homebrew/opt/redis/homebrew.mxcl.redis.plist');
    expect(plist).toContain('homebrew.mxcl.redis');
    expect(plist).toContain('/opt/homebrew/bin/redis-server');
  });
});

/* -------------------------------------------------- transitive + autoremove */

describe('transitive installs and autoremove', () => {
  it('installs the whole closure and records dependency edges', async () => {
    const machine = await box();
    await machine.run('apt install nginx');
    const packages = machine.software.listPackages();
    const names = packages.map((item) => item.name);
    expect(names).toEqual(expect.arrayContaining(['nginx', 'nginx-common', 'libc6', 'libpcre2-8-0', 'libssl3t64', 'zlib1g']));
    const nginx = packages.find((item) => item.name === 'nginx')!;
    expect(nginx.dependencyType).toBe('direct');
    expect(nginx.dependencies).toEqual(expect.arrayContaining(['nginx-common', 'libc6']));
    expect(packages.find((item) => item.name === 'libpcre2-8-0')?.dependencyType).toBe('transitive');
  });

  it('leaves orphans behind on apt remove and names them, then autoremove collects them', async () => {
    const machine = await box();
    await machine.run('apt install nginx');
    const removed = await machine.run('apt remove nginx');
    expect(removed).toContain('The following packages were automatically installed and are no longer required:');
    expect(removed).toContain("Use 'apt autoremove' to remove them.");
    // Still installed: apt does not prune on remove.
    expect(machine.software.listPackages().map((item) => item.name)).toContain('nginx-common');
    const collected = await machine.run('apt autoremove');
    expect(collected).toContain('Removing nginx-common');
    expect(machine.software.listPackages().map((item) => item.name)).not.toContain('nginx-common');
    expect(machine.vfs.statSync('/usr/bin/nginx')).toBeUndefined();
  });

  it('keeps a dependency that another installed package still needs', async () => {
    const machine = await box();
    await machine.run('apt install nginx curl');
    await machine.run('apt autoremove');
    // curl and nginx both need libc6, so nothing is orphaned.
    expect(machine.software.listPackages().map((item) => item.name)).toContain('libc6');
    await machine.run('apt remove nginx');
    await machine.run('apt autoremove');
    const remaining = machine.software.listPackages().map((item) => item.name);
    expect(remaining).toContain('libc6');
    expect(remaining).not.toContain('libpcre2-8-0');
  });

  it('prunes orphans immediately for npm, the way npm actually behaves', async () => {
    const machine = await box();
    const project = '/home/agent/app';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'app', version: '1.0.0' }));
    await machine.run('npm install vite', project);
    expect(machine.software.listPackages().map((item) => item.name)).toContain('esbuild');
    await machine.run('npm uninstall vite', project);
    const remaining = machine.software.listPackages().map((item) => item.name);
    expect(remaining).not.toContain('vite');
    expect(remaining).not.toContain('esbuild');
    expect(remaining).not.toContain('nanoid');
  });

  it('promotes a transitive package to direct when it is asked for by name', async () => {
    const machine = await box();
    const project = '/home/agent/app2';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'app2', version: '1.0.0' }));
    await machine.run('npm install vite', project);
    expect(machine.software.listPackages().find((item) => item.name === 'esbuild')?.dependencyType).toBe('transitive');
    await machine.run('npm install esbuild', project);
    expect(machine.software.listPackages().find((item) => item.name === 'esbuild')?.dependencyType).toBe('direct');
    await machine.run('npm uninstall vite', project);
    // esbuild was requested directly, so pruning vite must not take it.
    expect(machine.software.listPackages().map((item) => item.name)).toContain('esbuild');
  });
});

/* ---------------------------------------------------------- specifier install */

describe('installing a specific version', () => {
  it('installs name@version as the package, not as a package called name@version', async () => {
    const machine = await box();
    const project = '/home/agent/pinned';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'pinned', version: '1.0.0' }));
    await machine.run('npm install react@18.2.0', project);
    const react = machine.software.listPackages().find((item) => item.manager === 'npm' && item.name === 'react');
    expect(react).toBeDefined();
    expect(react!.version).toBe('18.2.0');
    expect(react!.installPath).toBe(`${project}/node_modules/react`);
    expect(machine.software.listPackages().some((item) => item.name.includes('@18.2.0'))).toBe(false);
    expect(machine.vfs.statSync(`${project}/node_modules/react@18.2.0`)).toBeUndefined();
    const manifest = JSON.parse(await machine.vfs.readFile(`${project}/node_modules/react/package.json`)) as { version: string };
    expect(manifest.version).toBe('18.2.0');
  });

  it('nests a scoped package under its scope directory', async () => {
    const machine = await box();
    const project = '/home/agent/scoped';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'scoped', version: '1.0.0' }));
    await machine.run('npm install @types/node@^24.0.0', project);
    const record = machine.software.listPackages().find((item) => item.name === '@types/node')!;
    expect(record.installPath).toBe(`${project}/node_modules/@types/node`);
    expect(satisfies(record.version, '^24.0.0')).toBe(true);
  });

  it('resolves a range to the newest matching release', async () => {
    const machine = await box();
    const project = '/home/agent/ranged';
    await machine.vfs.writeFile(`${project}/package.json`, JSON.stringify({ name: 'ranged', version: '1.0.0' }));
    await machine.run('npm install typescript@~5.6.0', project);
    expect(machine.software.listPackages().find((item) => item.name === 'typescript')?.version).toBe('5.6.3');
  });

  it('honours a pip pin and an apt pin', async () => {
    const machine = await box();
    await machine.run('pip install pandas==2.2.3');
    expect(machine.software.listPackages().find((item) => item.name === 'pandas')?.version).toBe('2.2.3');
    await machine.run('apt install nginx=1.24.0-2ubuntu7');
    expect(machine.software.listPackages().find((item) => item.name === 'nginx')?.version).toBe('1.24.0-2ubuntu7');
  });

  it('honours an out-of-band --version flag', async () => {
    const machine = await box('windows');
    await machine.run('winget install OpenJS.NodeJS --version 24.8.0');
    expect(machine.software.listPackages().find((item) => item.name === 'OpenJS.NodeJS')?.version).toBe('24.8.0');
  });

  it('fails loudly on an unsatisfiable pin instead of inventing a version', async () => {
    const machine = await box();
    const message = await machine.fails('pip install numpy==99.0.0');
    expect(message).toContain('could not resolve dependencies for numpy');
    expect(machine.software.listPackages()).toEqual([]);
  });
});

/* ------------------------------------------------------------- script running */

describe('script running', () => {
  async function project(machine: Box, scripts: Record<string, string>): Promise<string> {
    const root = `/home/agent/scripted-${++counter}`;
    await machine.vfs.writeFile(`${root}/package.json`, JSON.stringify({ name: 'scripted', version: '2.1.0', scripts }, null, 2));
    return root;
  }

  it('runs a script through the tools installed into node_modules/.bin', async () => {
    const machine = await box();
    const root = await project(machine, { build: 'tsc -p tsconfig.json' });
    await machine.run('npm install typescript', root);
    const output = await machine.run('npm run build', root);
    expect(output).toContain('> scripted@2.1.0 build');
    expect(output).toContain('> tsc -p tsconfig.json');
    // tsc really wrote its build info into the project.
    expect(machine.vfs.statSync(`${root}/dist/.tsbuildinfo`)).toBeDefined();
  });

  it('produces real build artifacts and test output', async () => {
    const machine = await box();
    const root = await project(machine, { build: 'vite build', test: 'vitest run' });
    await machine.run('npm install vite vitest', root);
    const build = await machine.run('npm run build', root);
    expect(build).toContain('building for production');
    expect(await machine.vfs.readFile(`${root}/dist/index.html`)).toContain('<!doctype html>');
    const test = await machine.run('npm test', root);
    expect(test).toContain('Test Files  1 passed');
  });

  it('runs pre and post hooks around the named script', async () => {
    const machine = await box();
    const root = await project(machine, { prebuild: 'rimraf dist', build: 'tsc', postbuild: 'tsc --noEmit' });
    await machine.run('npm install typescript rimraf', root);
    const output = await machine.run('npm run build', root);
    expect(output).toContain('> scripted@2.1.0 prebuild');
    expect(output).toContain('> scripted@2.1.0 build');
    expect(output).toContain('> scripted@2.1.0 postbuild');
    expect(output.indexOf('prebuild')).toBeLessThan(output.indexOf('postbuild'));
  });

  it('chains commands with && and stops at the first failure', async () => {
    const machine = await box();
    const root = await project(machine, { check: 'tsc --noEmit && not-installed --flag && tsc' });
    await machine.run('npm install typescript', root);
    const message = await machine.fails('npm run check', root);
    expect(message).toContain('sh: 1: not-installed: not found');
    expect(message).toContain('Lifecycle script `check` failed');
    expect(message).toContain('npm error code 127');
  });

  it('fails honestly when the script calls a tool that was never installed', async () => {
    const machine = await box();
    const root = await project(machine, { build: 'tsc -p tsconfig.json' });
    const message = await machine.fails('npm run build', root);
    expect(message).toContain('sh: 1: tsc: not found');
    expect(message).not.toContain('no such file');
  });

  it('recurses into a nested manager run and bounds the recursion', async () => {
    const machine = await box();
    const root = await project(machine, { all: 'npm run compile', compile: 'tsc' });
    await machine.run('npm install typescript', root);
    const output = await machine.run('npm run all', root);
    expect(output).toContain('> scripted@2.1.0 all');
    expect(output).toContain('> scripted@2.1.0 compile');

    const loop = await project(machine, { spin: 'npm run spin' });
    expect(await machine.fails('npm run spin', loop)).toContain('Maximum script recursion depth exceeded');
  });

  it('lists the available scripts when run is called with no name', async () => {
    const machine = await box();
    const root = await project(machine, { build: 'tsc', test: 'vitest run' });
    const output = await machine.run('npm run', root);
    expect(output).toContain('Scripts available in scripted@2.1.0');
    expect(output).toContain('  build');
    expect(output).toContain('    tsc');
  });

  it('reports a missing script with the names that do exist', async () => {
    const machine = await box();
    const root = await project(machine, { build: 'tsc' });
    const message = await machine.fails('npm run frobnicate', root);
    expect(message).toContain('Missing script: "frobnicate"');
    expect(message).toContain('npm run build');
  });

  it('runs scripts through pnpm, yarn and bun as well', async () => {
    for (const manager of ['pnpm', 'yarn', 'bun'] as const) {
      const machine = await box();
      const root = await project(machine, { build: 'tsc' });
      await machine.run(`${manager} add typescript`, root);
      expect(await machine.run(`${manager} run build`, root)).toContain('> scripted@2.1.0 build');
    }
  });

  it('builds, tests and runs a cargo crate against a real Cargo.toml', async () => {
    const machine = await box();
    const root = '/home/agent/crate';
    await machine.run('cargo new crate', '/home/agent');
    expect(await machine.vfs.readFile(`${root}/Cargo.toml`)).toContain('name = "crate"');
    const built = await machine.run('cargo build', root);
    expect(built).toContain('Compiling crate v0.1.0');
    expect(built).toContain('Finished `dev` profile');
    // The build produced a runnable executable descriptor.
    expect((await descriptorAt(machine.vfs, `${root}/target/debug/crate`)).behavior).toBe('crate');
    expect(await machine.run('cargo test', root)).toContain('test result: ok. 2 passed');
    expect(await machine.run('cargo run', root)).toContain('Hello, world!');
    expect(await machine.fails('cargo build', '/home/agent')).toContain('could not find `Cargo.toml`');
  });

  it('initializes and builds a go module', async () => {
    const machine = await box();
    const root = '/home/agent/gomod';
    await machine.vfs.mkdir(root);
    expect(await machine.run('go mod init example.com/demo', root)).toContain('creating new go.mod');
    await machine.run('go build', root);
    expect((await descriptorAt(machine.vfs, `${root}/demo`)).manager).toBe('go');
    expect(await machine.run('go test', root)).toContain('ok  \texample.com/demo');
  });
});

/* -------------------------------------------------------- unknown subcommands */

describe('unknown verbs', () => {
  it('rejects an unrecognized npm subcommand instead of listing packages', async () => {
    const machine = await box();
    const message = await machine.fails('npm frobnicate');
    expect(message).toContain('Unknown command: "frobnicate"');
    expect(message).toContain('npm help');
  });

  it('rejects unknown verbs in each family\'s own voice', async () => {
    const linux = await box();
    expect(await linux.fails('apt frobnicate')).toBe('E: Invalid operation frobnicate');
    expect(await linux.fails('cargo frobnicate')).toContain('no such command: `frobnicate`');
    expect(await linux.fails('go frobnicate')).toContain('go frobnicate: unknown command');
    expect(await linux.fails('pip frobnicate')).toContain("unknown subcommand 'frobnicate'");
    const mac = await box('macos');
    expect(await mac.fails('brew frobnicate')).toBe('Error: Unknown command: frobnicate');
    const windows = await box('windows');
    expect(await windows.fails('winget frobnicate')).toContain("Unrecognized command: 'frobnicate'");
  });

  it('refuses a manager the operating system does not ship', async () => {
    const mac = await box('macos');
    expect(await mac.fails('apt install nginx')).toContain('unavailable on macos');
    const windows = await box('windows');
    expect(await windows.fails('brew install ripgrep')).toContain('unavailable on windows');
    expect(mac.software.supports('brew')).toBe(true);
    expect(mac.software.supports('apt')).toBe(false);
    expect(mac.software.supportedManagers()).toContain('brew');
    expect(mac.software.supportedManagers()).not.toContain('winget');
  });
});

/* --------------------------------------------------------- per-manager output */

describe('per-manager command surfaces', () => {
  it('renders apt\'s install transcript', async () => {
    const machine = await box();
    const output = await machine.run('apt install nginx');
    expect(output).toContain('Reading package lists... Done');
    expect(output).toContain('Building dependency tree... Done');
    expect(output).toContain('The following additional packages will be installed:');
    expect(output).toContain('The following NEW packages will be installed:');
    expect(output).toMatch(/0 upgraded, \d+ newly installed, 0 to remove and 0 not upgraded\./);
    expect(output).toMatch(/Need to get .+ of archives\./);
    expect(output).toMatch(/After this operation, .+ of additional disk space will be used\./);
    expect(output).toContain('Setting up nginx (1.26.3-1ubuntu1) ...');
  });

  it('maintains /var/lib/dpkg/status and renders dpkg -l as a status table', async () => {
    const machine = await box();
    await machine.run('apt install jq');
    const status = await machine.vfs.readFile('/var/lib/dpkg/status');
    expect(status).toContain('Package: jq');
    expect(status).toContain('Status: install ok installed');
    expect(status).toContain('Architecture: amd64');
    const listing = await machine.run('dpkg -l');
    expect(listing).toContain('Desired=Unknown/Install/Remove/Purge/Hold');
    expect(listing).toContain('||/ Name');
    expect(listing).toMatch(/^ii {2}jq/m);
  });

  it('installs a .deb through dpkg -i and removes it again', async () => {
    const machine = await box();
    const installed = await machine.run('dpkg -i seed-agent_1.0_amd64.deb');
    expect(installed).toContain('Selecting previously unselected package seed-agent.');
    expect(installed).toContain('Unpacking seed-agent (1.0) ...');
    expect(machine.vfs.statSync('/var/lib/dpkg/info/seed-agent_1.0_amd64.deb.list')?.kind).toBe('file');
    // Removable by the short package name as well as the archive name.
    expect(await machine.run('dpkg -r seed-agent')).toContain('Removing seed-agent (1.0) ...');
    expect(machine.vfs.statSync('/var/lib/dpkg/info/seed-agent_1.0_amd64.deb.list')).toBeUndefined();
  });

  it('renders brew\'s Cellar transcript, cask installs and services', async () => {
    const machine = await box('macos');
    const output = await machine.run('brew install ripgrep');
    expect(output).toContain('==> Fetching ripgrep');
    expect(output).toContain('==> Pouring ripgrep--14.1.1.arm64_sequoia.bottle.tar.gz');
    expect(output).toContain('🍺  /opt/homebrew/Cellar/ripgrep/14.1.1:');
    expect(output).toContain('==> Running `brew cleanup ripgrep`...');

    const cask = await machine.run('brew install --cask visual-studio-code');
    expect(cask).toContain('==> Installing Cask visual-studio-code');
    expect(cask).toContain("==> Moving App 'Visual Studio Code.app' to '/Applications/Visual Studio Code.app'");
    expect(machine.software.listPackages().find((item) => item.name === 'visual-studio-code')?.installPath)
      .toBe('/opt/homebrew/Caskroom/visual-studio-code/1.104.0');

    const listing = await machine.run('brew list');
    expect(listing).toContain('==> Formulae');
    expect(listing).toContain('ripgrep 14.1.1');
    expect(listing).toContain('==> Casks');

    await machine.run('brew install redis');
    expect(await machine.run('brew services')).toContain('redis');
  });

  it('reports a plausible current version rather than a hash-derived one', async () => {
    const machine = await box('macos');
    const info = await machine.run('brew info ffmpeg');
    expect(info).toContain('==> ffmpeg: stable 8.0 (bottled)');
    expect(info).not.toMatch(/stable \d{1,2}\.\d{1,2}\.\d{1,2} \(bottled\)\n.*7\.20\.6/);
    // Nothing in the catalog is bounded by the old 12.22.10 hash ceiling.
    const versions = registries.brew.map((entry) => entry.version);
    expect(versions).toContain('8.0');
    expect(findEntry('npm', 'typescript')?.version).toBe('5.9.2');
    expect(findEntry('apt', 'nginx')?.version).toBe('1.26.3-1ubuntu1');
  });

  it('renders winget\'s ruled table listing', async () => {
    const machine = await box('windows');
    await machine.run('winget install Docker.DockerDesktop');
    const listing = await machine.run('winget list');
    const lines = listing.split('\n');
    expect(lines[0]).toMatch(/^Name\s+Id\s+Version\s+Available\s+Source$/);
    expect(lines[1]).toMatch(/^-+$/);
    expect(listing).toContain('Docker Desktop');
    expect(listing).toContain('Docker.DockerDesktop');
    // The VCRedist dependency came along.
    expect(listing).toContain('Microsoft.VCRedist.2015+.x64');
    expect(await machine.run('winget show GitHub.cli')).toContain('Publisher: GitHub, Inc.');
  });

  it('renders chocolatey and scoop in their own formats', async () => {
    const machine = await box('windows');
    const choco = await machine.run('choco install ripgrep');
    expect(choco).toContain('Chocolatey v2.5.1');
    expect(choco).toContain('ripgrep v14.1.1 [Approved]');
    expect(choco).toContain('Chocolatey installed 1/1 package.');
    const scoop = await machine.run('scoop install jq');
    expect(scoop).toContain("Installing 'jq' (1.8.1) [64bit] from 'main' bucket");
    expect(scoop).toContain("Creating shim for 'jq'.");
    expect(await machine.run('scoop list')).toContain('Installed apps:');
  });

  it('renders cargo\'s registry transcript and installed listing', async () => {
    const machine = await box();
    const output = await machine.run('cargo install ripgrep');
    expect(output).toContain('    Updating crates.io index');
    expect(output).toContain('  Downloaded ripgrep v14.1.1');
    expect(output).toContain('   Compiling ripgrep v14.1.1');
    expect(output).toContain('    Finished `release` profile [optimized] target(s)');
    expect(output).toContain('   Installed package `ripgrep v14.1.1` (executable `rg`)');
    // The crate source landed in the registry cache.
    expect(await machine.vfs.readFile('/home/agent/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ripgrep-14.1.1/Cargo.toml'))
      .toContain('name = "ripgrep"');
    expect(await machine.run('cargo install --list')).toContain('ripgrep v14.1.1:');
  });

  it('renders pip wheels, freeze output and requirements files', async () => {
    const machine = await box();
    const output = await machine.run('pip install fastapi');
    expect(output).toContain('Collecting fastapi');
    expect(output).toMatch(/Downloading fastapi-0\.116\.1-py3-none-any\.whl \(\d+ kB\)/);
    expect(output).toContain('Collecting starlette (from fastapi)');
    expect(output).toContain('Installing collected packages:');
    expect(output).toContain('Successfully installed');
    // A wheel really unpacked, dist-info and all.
    expect(await machine.vfs.readFile('/home/agent/.local/lib/python3.13/site-packages/fastapi-0.116.1.dist-info/METADATA'))
      .toContain('Name: fastapi');
    expect(await machine.run('pip freeze')).toContain('fastapi==0.116.1');

    // requirements.txt is kept in step when the project keeps one.
    const root = '/home/agent/pyapp';
    await machine.vfs.writeFile(`${root}/requirements.txt`, '');
    await machine.run('pip install ruff', root);
    expect(await machine.vfs.readFile(`${root}/requirements.txt`)).toContain('ruff==0.13.0');
    // ...and it can be read back in.
    const other = await box();
    await other.vfs.writeFile('/home/agent/reqs.txt', '# comment\nruff==0.13.0\npytest\n');
    expect(await other.run('pip install -r reqs.txt')).toContain('Successfully installed');
    expect(other.software.listPackages().map((item) => item.name)).toEqual(expect.arrayContaining(['ruff', 'pytest']));
  });

  it('renders pnpm, yarn and bun install output distinctly', async () => {
    const machine = await box();
    const root = '/home/agent/js';
    await machine.vfs.writeFile(`${root}/package.json`, JSON.stringify({ name: 'js', version: '1.0.0' }));
    expect(await machine.run('pnpm add vite', root)).toContain('Done in 1.2s using pnpm v10.15.1');

    const yarnBox = await box();
    await yarnBox.vfs.writeFile('/home/agent/y/package.json', JSON.stringify({ name: 'y', version: '1.0.0' }));
    const yarnOut = await yarnBox.run('yarn add react', '/home/agent/y');
    expect(yarnOut).toContain('yarn add v1.22.22');
    expect(yarnOut).toContain('[4/4] Building fresh packages...');
    expect(yarnOut).toContain('✨  Done in 1.23s.');

    const bunBox = await box();
    await bunBox.vfs.writeFile('/home/agent/b/package.json', JSON.stringify({ name: 'b', version: '1.0.0' }));
    const bunOut = await bunBox.run('bun add hono', '/home/agent/b');
    expect(bunOut).toContain('bun add v1.2.21');
    expect(bunOut).toContain('installed hono@4.9.6');
  });

  it('renders snap, flatpak, gem, composer, conda and vcpkg surfaces', async () => {
    const linux = await box();
    expect(await linux.run('snap install code')).toContain('code 1.104.0 from vscode✓ installed');
    expect(await linux.run('flatpak install org.gimp.GIMP')).toContain('Installation complete.');
    expect(await linux.run('gem install rake')).toContain('1 gem installed');
    expect(await linux.run('conda install numpy')).toContain('The following NEW packages will be INSTALLED:');
    const composerRoot = '/home/agent/php';
    await linux.vfs.writeFile(`${composerRoot}/composer.json`, JSON.stringify({ name: 'seed/php' }));
    expect(await linux.run('composer require monolog/monolog', composerRoot)).toContain('Generating autoload files');
    const vcpkgRoot = '/home/agent/cpp';
    await linux.vfs.writeFile(`${vcpkgRoot}/vcpkg.json`, JSON.stringify({ name: 'cpp', version: '0.1.0' }));
    expect(await linux.run('vcpkg install fmt', vcpkgRoot)).toContain('Computing installation plan...');
  });

  it('installs a dotnet tool through the `tool` prefix', async () => {
    const machine = await box('windows');
    const output = await machine.run('dotnet tool install --global dotnet-ef');
    expect(output).toContain("Tool 'dotnet-ef' (version '9.0.9') was successfully installed.");
    expect((await descriptorAt(machine.vfs, '/C/Users/agent/.dotnet/tools/dotnet-ef.exe')).manager).toBe('dotnet');
  });

  it('accepts uv\'s pip and tool prefixes', async () => {
    const machine = await box();
    expect(await machine.run('uv pip install ruff')).toContain('+ ruff==0.13.0');
    expect(await machine.run('uv tool install pytest')).toContain('+ pytest==8.4.2');
  });

  it('searches, shows and refreshes each index', async () => {
    const machine = await box();
    const search = await machine.run('apt search json');
    expect(search).toContain('Sorting... Done');
    expect(search).toContain('jq/noble 1.8.1-1 amd64');
    const show = await machine.run('apt show ripgrep');
    expect(show).toContain('Package: ripgrep');
    expect(show).toContain('Section: utils');
    expect(await machine.run('apt update')).toContain('Reading package lists... Done');
    expect(await machine.run('apt install not-a-real-package').catch((error: Error) => error.message))
      .toContain('unable to locate package not-a-real-package');
  });

  it('reports outdated packages against the catalog', async () => {
    const machine = await box();
    const root = '/home/agent/old';
    await machine.vfs.writeFile(`${root}/package.json`, JSON.stringify({ name: 'old', version: '1.0.0' }));
    await machine.run('npm install typescript@5.4.5', root);
    const outdated = await machine.run('npm outdated', root);
    expect(outdated).toContain('typescript');
    expect(outdated).toContain('5.4.5');
    expect(outdated).toContain('5.9.2');
    await machine.run('npm update typescript', root);
    expect(machine.software.listPackages().find((item) => item.name === 'typescript')?.version).toBe('5.9.2');
  });
});

/* ------------------------------------------------- receipts, locks, manifests */

describe('receipts, lockfiles and transactions', () => {
  it('writes every receipt under profile.packageManagers.receiptRoots', async () => {
    const linux = await box();
    await linux.run('apt install jq');
    await linux.run('cargo install just');
    await linux.run('snap install gh');
    const roots = linux.profile.packageManagers.receiptRoots;
    for (const record of linux.software.listPackages()) {
      const receipt = record.files.find((file) => roots.some((root) => file.startsWith(`${root}/`)));
      expect(receipt, `no receipt under a receiptRoot for ${record.manager}/${record.name}`).toBeDefined();
      expect(linux.vfs.statSync(receipt!)?.kind).toBe('file');
    }
    // apt/dpkg keep the Debian file list; brew keeps its INSTALL_RECEIPT in the keg.
    expect(linux.vfs.statSync('/var/lib/dpkg/info/jq.list')?.kind).toBe('file');
    const mac = await box('macos');
    await mac.run('brew install jq');
    const receipt = JSON.parse(await mac.vfs.readFile('/opt/homebrew/Cellar/jq/1.8.1/INSTALL_RECEIPT.json')) as { manager: string; package: string };
    expect(receipt).toMatchObject({ manager: 'brew', package: 'jq' });
  });

  it('writes a manifest and a lockfile per project manager', async () => {
    const machine = await box();
    const root = '/home/agent/locked';
    await machine.vfs.writeFile(`${root}/package.json`, JSON.stringify({ name: 'locked', version: '1.0.0', scripts: { build: 'tsc' } }));
    await machine.run('npm install vite', root);
    const manifest = JSON.parse(await machine.vfs.readFile(`${root}/package.json`)) as { dependencies: Record<string, string>; scripts: Record<string, string> };
    expect(manifest.dependencies.vite).toMatch(/^\^7\./);
    // Installing must not eat the scripts block.
    expect(manifest.scripts.build).toBe('tsc');
    const lock = JSON.parse(await machine.vfs.readFile(`${root}/package-lock.json`)) as { lockfileVersion: number; packages: Record<string, { version: string }> };
    expect(lock.lockfileVersion).toBe(3);
    expect(lock.packages['node_modules/vite']?.version).toMatch(/^7\./);
    expect(lock.packages['node_modules/esbuild']).toBeDefined();

    const pnpmBox = await box();
    await pnpmBox.vfs.writeFile('/home/agent/p/package.json', JSON.stringify({ name: 'p', version: '1.0.0' }));
    await pnpmBox.run('pnpm add vite', '/home/agent/p');
    const pnpmLock = await pnpmBox.vfs.readFile('/home/agent/p/pnpm-lock.yaml');
    expect(pnpmLock).toContain("lockfileVersion: '9.0'");
    expect(pnpmLock).toContain('vite@');
  });

  it('writes poetry, composer, nuget and vcpkg manifests', async () => {
    const machine = await box();
    const poetryRoot = '/home/agent/poetry-app';
    await machine.vfs.writeFile(`${poetryRoot}/pyproject.toml`, '[tool.poetry]\nname = "poetry-app"\n');
    await machine.run('poetry add fastapi', poetryRoot);
    expect(await machine.vfs.readFile(`${poetryRoot}/pyproject.toml`)).toContain('fastapi = "^0.116.1"');
    expect(await machine.vfs.readFile(`${poetryRoot}/poetry.lock`)).toContain('name = "fastapi"');

    const composerRoot = '/home/agent/composer-app';
    await machine.vfs.writeFile(`${composerRoot}/composer.json`, JSON.stringify({ name: 'seed/app' }));
    await machine.run('composer require guzzlehttp/guzzle', composerRoot);
    const composerLock = JSON.parse(await machine.vfs.readFile(`${composerRoot}/composer.lock`)) as { packages: { name: string }[] };
    expect(composerLock.packages.map((item) => item.name)).toContain('guzzlehttp/guzzle');

    const nugetRoot = '/home/agent/dotnet-app';
    await machine.vfs.writeFile(`${nugetRoot}/packages.lock.json`, '{}');
    await machine.run('nuget install Serilog', nugetRoot);
    expect(await machine.vfs.readFile(`${nugetRoot}/packages.lock.json`)).toContain('Serilog');

    const vcpkgRoot = '/home/agent/vcpkg-app';
    await machine.vfs.writeFile(`${vcpkgRoot}/vcpkg.json`, JSON.stringify({ name: 'app', version: '0.1.0' }));
    await machine.run('vcpkg install spdlog', vcpkgRoot);
    expect(await machine.vfs.readFile(`${vcpkgRoot}/vcpkg.json`)).toContain('spdlog');
  });

  it('commits a transaction per operation with its receipt paths', async () => {
    const machine = await box();
    await machine.run('apt update');
    await machine.run('apt install jq');
    await machine.run('apt remove jq');
    const transactions = machine.software.listPackageTransactions();
    expect(transactions.map((item) => item.operation)).toEqual(expect.arrayContaining(['index-refresh', 'install', 'remove']));
    expect(transactions.every((item) => item.status === 'committed')).toBe(true);
    const install = transactions.find((item) => item.operation === 'install')!;
    expect(install.manager).toBe('apt');
    expect(install.packages).toContain('jq');
    expect(install.receiptPaths.some((file) => file.startsWith('/var/lib/dpkg/'))).toBe(true);
    expect(Date.parse(install.completedAt)).toBeGreaterThanOrEqual(Date.parse(install.startedAt));
  });

  it('records a 64-character integrity digest and a resolvable source for every package', async () => {
    const machine = await box();
    await machine.run('apt install jq');
    for (const record of machine.software.listPackages()) {
      expect(record.integrity).toHaveLength(64);
      expect(record.source).toMatch(/^registry:\/\/apt\/.+@.+$/);
      expect(record.files.length).toBeGreaterThan(0);
      expect(machine.vfs.statSync(record.installPath)).toBeDefined();
    }
  });

  it('persists the package database to disk', async () => {
    const machine = await box();
    await machine.run('apt install tree');
    const database = JSON.parse(await machine.vfs.readFile('/var/lib/seed/packages.json')) as { name: string }[];
    expect(database.map((item) => item.name)).toContain('tree');
  });
});

/* -------------------------------------------------------------- bootstrapping */

describe('bootstrap and catalog', () => {
  it('bootstraps a usable toolchain and a git-tracked project', async () => {
    const machine = await box('ubuntu', { bootstrap: true });
    const names = machine.software.listPackages().map((item) => item.name);
    expect(names).toEqual(expect.arrayContaining(['git', 'nodejs', 'python3']));
    expect((await descriptorAt(machine.vfs, '/usr/bin/git')).behavior).toBe('git');
    expect((await descriptorAt(machine.vfs, '/usr/bin/node')).behavior).toBe('node');
    expect((await descriptorAt(machine.vfs, '/usr/bin/python3')).behavior).toBe('python3');
    expect(machine.software.listRepositories().map((repository) => repository.root)).toContain('/home/agent/Projects/seed-ecosystem');
  });

  it('bootstraps through the manager the OS profile declares', async () => {
    const mac = await box('macos', { bootstrap: true });
    expect(mac.software.listPackages().every((item) => item.manager === 'brew')).toBe(true);
    expect((await descriptorAt(mac.vfs, '/opt/homebrew/bin/git')).manager).toBe('brew');
    const windows = await box('windows', { bootstrap: true });
    expect(windows.software.listPackages().every((item) => item.manager === 'winget')).toBe(true);
  });

  it('ships a registry per manager that is large enough to search', () => {
    const size = catalogSize();
    expect(size.managers).toBe(25);
    expect(size.entries).toBeGreaterThan(700);
    for (const [manager, entries] of Object.entries(registries)) {
      expect(entries.length, `${manager} registry is too small`).toBeGreaterThanOrEqual(12);
      for (const entry of entries) {
        expect(entry.version, `${manager}/${entry.name} has no version`).toBeTruthy();
        expect(availableVersions(entry)).toContain(entry.version);
        expect(entry.description, `${manager}/${entry.name} has no description`).toBeTruthy();
      }
    }
  });

  it('resolves every catalog entry without a dangling or unsatisfiable dependency', () => {
    const failures: string[] = [];
    for (const manager of Object.keys(registries) as (keyof typeof registries)[]) {
      for (const entry of registries[manager]) {
        const { problems } = resolve({ manager, requests: [{ raw: entry.name, name: entry.name, range: '*' }] });
        // A declared conflict is a fact about the catalog, not a broken entry.
        for (const problem of problems.filter((item) => item.kind !== 'conflict')) {
          failures.push(`${manager}/${entry.name}: ${describeProblem(manager, problem).split('\n')[0]}`);
        }
      }
    }
    expect(failures).toEqual([]);
  });
});
