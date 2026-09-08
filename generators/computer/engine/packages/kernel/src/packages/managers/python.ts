import type { OperationKind } from '../types.js';
import { defineManager, type ManagerSpec, table } from './shared.js';

const PYTHON = '3.13.7';
const SITE = 'python3.13/site-packages';

/** Wheel filename pip echoes while collecting. */
const wheel = (name: string, version: string): string => `${name.replaceAll('-', '_')}-${version}-py3-none-any.whl`;

const pipVerbs: Readonly<Record<string, OperationKind>> = {
  install: 'install', uninstall: 'remove', remove: 'remove', download: 'install',
  list: 'list', freeze: 'list', show: 'info', search: 'search', check: 'info',
  cache: 'clean', config: 'info', wheel: 'build', help: 'help', index: 'search', inspect: 'info',
};

export const pipManager = defineManager({
  id: 'pip',
  commands: ['pip', 'pip3'],
  family: 'language',
  projectScoped: false,
  verbs: pipVerbs,
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'list', 'freeze', 'show', 'search', 'check', 'download', 'wheel', 'cache', 'config'],
  renderers: {
    install({ installed, reused, context }) {
      if (!installed.length) {
        return reused.map((view) => `Requirement already satisfied: ${view.entry.name} in ${context.home}/.local/lib/${SITE} (${view.version})`).join('\n');
      }
      const lines: string[] = [];
      // pip collects the named requirement first, then walks into its dependencies.
      const dependentOf = (name: string): string | undefined =>
        installed.find((candidate) => Object.hasOwn(candidate.entry.dependencies ?? {}, name))?.entry.name;
      for (const view of [...installed].sort((a, b) => Number(b.direct) - Number(a.direct))) {
        const parent = view.direct ? undefined : dependentOf(view.entry.name);
        lines.push(`Collecting ${view.entry.name}${parent ? ` (from ${parent})` : ''}`);
        lines.push(`  Downloading ${wheel(view.entry.name, view.version)} (${Math.max(8, Math.round((view.entry.sizeBytes ?? 131_072) / 1024))} kB)`);
      }
      for (const view of reused) lines.push(`Requirement already satisfied: ${view.entry.name} in ${context.home}/.local/lib/${SITE} (${view.version})`);
      lines.push(`Installing collected packages: ${installed.map((view) => view.entry.name).join(', ')}`);
      lines.push(`Successfully installed ${installed.map((view) => `${view.entry.name}-${view.version}`).join(' ')}`);
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'WARNING: Skipping as it is not installed.';
      return all.flatMap((view) => [
        `Found existing installation: ${view.entry.name} ${view.version}`,
        `Uninstalling ${view.entry.name}-${view.version}:`,
        `  Successfully uninstalled ${view.entry.name}-${view.version}`,
      ]).join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      return table(['Package', 'Version'], [...records].sort((a, b) => a.name.localeCompare(b.name)).map((record) => [record.name, record.version]), '-');
    },
    search({ query, matches }) {
      if (!matches.length) return `ERROR: No matches found for '${query}'`;
      return matches.map((entry) => `${entry.name} (${entry.version})  - ${entry.description}`).join('\n');
    },
    info({ name, entry, record, context }) {
      if (!entry) return `WARNING: Package(s) not found: ${name}`;
      const requires = Object.keys(entry.dependencies ?? {});
      return [
        `Name: ${entry.name}`,
        `Version: ${record?.version ?? entry.version}`,
        `Summary: ${entry.description}`,
        `Home-page: https://pypi.seed.local/project/${entry.name}/`,
        'Author: seed',
        `License: ${entry.license ?? 'MIT'}`,
        `Location: ${record?.installPath.replace(/\/[^/]+$/, '') ?? `${context.home}/.local/lib/${SITE}`}`,
        `Requires: ${requires.join(', ')}`,
        'Required-by: ',
      ].join('\n');
    },
    refresh() { return 'Looking in indexes: https://pypi.seed.local/simple'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      return table(['Package', 'Version', 'Latest', 'Type'], entries.map(({ record, latest }) => [record.name, record.version, latest, 'wheel']), '-');
    },
  },
});

export const pipxManager = defineManager({
  id: 'pipx',
  commands: ['pipx'],
  family: 'language',
  projectScoped: false,
  verbs: {
    install: 'install', uninstall: 'remove', 'uninstall-all': 'remove', upgrade: 'upgrade', 'upgrade-all': 'upgrade',
    list: 'list', run: 'exec', runpip: 'exec', inject: 'install', reinstall: 'install',
    ensurepath: 'info', environment: 'info', help: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'upgrade', 'upgrade-all', 'list', 'run', 'inject', 'reinstall', 'ensurepath', 'environment'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `'${view.entry.name}' already seems to be installed. Not modifying existing installation.`).join('\n');
      const lines: string[] = [];
      for (const view of installed.filter((item) => item.direct)) {
        lines.push(`  installed package ${view.entry.name} ${view.version}, installed using Python ${PYTHON}`);
        lines.push('  These apps are now globally available');
        for (const binary of view.entry.binaries ?? []) lines.push(`    - ${binary}`);
      }
      lines.push('done! ✨ 🌟 ✨');
      return lines.join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'Nothing to uninstall.';
      return [...removed.map((view) => `uninstalled ${view.entry.name}! ✨ 🌟 ✨`)].join('\n');
    },
    list({ records, context }) {
      if (!records.length) return 'nothing has been installed with pipx 😴';
      const lines = [`venvs are in ${context.home}/.local/share/pipx/venvs`, `apps are exposed on your $PATH at ${context.home}/.local/bin`, `manual pip install ${records.length}`];
      for (const record of records) {
        lines.push(`   package ${record.name} ${record.version}, installed using Python ${PYTHON}`);
        for (const binary of record.provides) lines.push(`    - ${binary}`);
      }
      return lines.join('\n');
    },
    search: pipManager.renderers.search,
    info: pipManager.renderers.info,
    refresh() { return `pipx environment variables:\nPIPX_HOME=~/.local/share/pipx\nPIPX_BIN_DIR=~/.local/bin`; },
    outdated({ entries }) {
      if (!entries.length) return 'Versions did not change after running pipx upgrade-all.';
      return entries.map(({ record, latest }) => `upgraded package ${record.name} from ${record.version} to ${latest}`).join('\n');
    },
  },
});

export const uvManager = defineManager({
  id: 'uv',
  commands: ['uv', 'uvx'],
  family: 'language',
  projectScoped: false,
  prefixes: [['pip'], ['tool'], ['python']],
  verbs: {
    install: 'install', add: 'install', sync: 'install', uninstall: 'remove', remove: 'remove',
    list: 'list', freeze: 'list', show: 'info', tree: 'list', upgrade: 'upgrade', lock: 'build',
    run: 'run', 'tool-run': 'exec', venv: 'build', build: 'build', init: 'build', cache: 'clean',
    help: 'help', search: 'search', export: 'list',
  },
  defaultVerb: 'help',
  helpVerbs: ['add', 'remove', 'sync', 'lock', 'run', 'tool install', 'pip install', 'venv', 'tree', 'build', 'init', 'cache'],
  renderers: {
    install({ installed, reused }) {
      const resolved = installed.length + reused.length;
      if (!installed.length) return `Resolved ${resolved} ${resolved === 1 ? 'package' : 'packages'} in 8ms\nAudited ${resolved} ${resolved === 1 ? 'package' : 'packages'} in 0.1ms`;
      return [
        `Resolved ${resolved} ${resolved === 1 ? 'package' : 'packages'} in 12ms`,
        `Prepared ${installed.length} ${installed.length === 1 ? 'package' : 'packages'} in 45ms`,
        `Installed ${installed.length} ${installed.length === 1 ? 'package' : 'packages'} in 8ms`,
        ...installed.map((view) => ` + ${view.entry.name}==${view.version}`),
      ].join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'Audited 0 packages in 0.1ms';
      return [`Uninstalled ${all.length} ${all.length === 1 ? 'package' : 'packages'} in 3ms`, ...all.map((view) => ` - ${view.entry.name}==${view.version}`)].join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      return [...records].sort((a, b) => a.name.localeCompare(b.name)).map((record) => `${record.name}==${record.version}`).join('\n');
    },
    search: pipManager.renderers.search,
    info: pipManager.renderers.info,
    refresh() { return 'Using index https://pypi.seed.local/simple'; },
    outdated({ entries }) {
      if (!entries.length) return 'All packages up to date';
      return entries.map(({ record, latest }) => ` ~ ${record.name}==${record.version} -> ${latest}`).join('\n');
    },
  },
});

export const poetryManager = defineManager({
  id: 'poetry',
  commands: ['poetry'],
  family: 'language',
  projectScoped: true,
  verbs: {
    add: 'install', install: 'install', remove: 'remove', update: 'upgrade', lock: 'build',
    show: 'list', list: 'list', search: 'search', info: 'info', run: 'run', shell: 'run',
    build: 'build', publish: 'build', check: 'info', init: 'build', new: 'build', env: 'info', help: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['add', 'install', 'remove', 'update', 'lock', 'show', 'run', 'build', 'publish', 'check', 'init', 'new', 'env'],
  renderers: {
    install({ installed, reused }) {
      const direct = installed.filter((view) => view.direct);
      const lines: string[] = [];
      for (const view of direct) lines.push(`Using version ^${view.version} for ${view.entry.name}`);
      lines.push('');
      lines.push('Updating dependencies');
      lines.push('Resolving dependencies... (0.3s)');
      lines.push('');
      if (!installed.length) {
        lines.push('No dependencies to install or update');
        void reused;
        return lines.join('\n');
      }
      lines.push(`Package operations: ${installed.length} ${installed.length === 1 ? 'install' : 'installs'}, 0 updates, 0 removals`);
      lines.push('');
      for (const view of installed) lines.push(`  - Installing ${view.entry.name} (${view.version})`);
      lines.push('');
      lines.push('Writing lock file');
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'Updating dependencies\nResolving dependencies... (0.1s)\n\nNo dependencies to install or update';
      return [
        'Updating dependencies',
        'Resolving dependencies... (0.2s)',
        '',
        `Package operations: 0 installs, 0 updates, ${all.length} ${all.length === 1 ? 'removal' : 'removals'}`,
        '',
        ...all.map((view) => `  - Removing ${view.entry.name} (${view.version})`),
        '',
        'Writing lock file',
      ].join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      const width = Math.max(...records.map((record) => record.name.length)) + 2;
      return [...records].sort((a, b) => a.name.localeCompare(b.name))
        .map((record) => `${record.name.padEnd(width)}${record.version.padEnd(10)} ${record.description}`).join('\n');
    },
    search: pipManager.renderers.search,
    info: pipManager.renderers.info,
    refresh() { return 'Poetry (version 2.1.4)'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      const width = Math.max(8, ...entries.map(({ record }) => record.name.length)) + 2;
      return entries.map(({ record, latest }) => `${record.name.padEnd(width)}${record.version.padEnd(10)} ${latest.padEnd(10)} ${record.description}`).join('\n');
    },
  },
});

export const condaManager = defineManager({
  id: 'conda',
  commands: ['conda', 'mamba', 'micromamba'],
  family: 'language',
  projectScoped: false,
  verbs: {
    install: 'install', create: 'install', remove: 'remove', uninstall: 'remove', update: 'upgrade',
    upgrade: 'upgrade', list: 'list', search: 'search', info: 'info', clean: 'clean',
    activate: 'run', deactivate: 'run', env: 'list', config: 'info', run: 'run', help: 'help', init: 'build',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'create', 'remove', 'update', 'list', 'search', 'info', 'clean', 'env', 'config', 'run', 'activate'],
  renderers: {
    install({ installed, reused, context }) {
      const lines = [
        'Channels:',
        ' - defaults',
        ' - conda-forge',
        `Platform: ${context.os === 'windows' ? 'win-64' : context.os === 'macos' ? 'osx-arm64' : 'linux-64'}`,
        'Collecting package metadata (repodata.json): done',
        'Solving environment: done',
        '',
      ];
      if (!installed.length) {
        lines.push('# All requested packages already installed.');
        void reused;
        return lines.join('\n');
      }
      const platform = context.os === 'windows' ? 'win-64' : context.os === 'macos' ? 'osx-arm64' : 'linux-64';
      lines.push('## Package Plan ##');
      lines.push('');
      lines.push(`  environment location: ${context.home}/miniconda3`);
      lines.push('');
      lines.push('  added / updated specs:');
      for (const view of installed.filter((item) => item.direct)) lines.push(`    - ${view.entry.name}`);
      lines.push('');
      lines.push('The following NEW packages will be INSTALLED:');
      lines.push('');
      const width = Math.max(...installed.map((view) => view.entry.name.length)) + 2;
      for (const view of installed) {
        lines.push(`  ${view.entry.name.padEnd(width)} ${view.entry.section ?? 'pkgs/main'}/${platform}::${view.entry.name}-${view.version}-py313_0`);
      }
      lines.push('');
      lines.push('Preparing transaction: done');
      lines.push('Verifying transaction: done');
      lines.push('Executing transaction: done');
      return lines.join('\n');
    },
    remove({ removed, autoremoved, context }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'Collecting package metadata (repodata.json): done\nSolving environment: done\n\n# All requested packages already removed.';
      const width = Math.max(...all.map((view) => view.entry.name.length)) + 2;
      return [
        'Collecting package metadata (repodata.json): done',
        'Solving environment: done',
        '',
        '## Package Plan ##',
        '',
        `  environment location: ${context.home}/miniconda3`,
        '',
        '  removed specs:',
        ...all.map((view) => `    - ${view.entry.name}`),
        '',
        'The following packages will be REMOVED:',
        '',
        ...all.map((view) => `  ${view.entry.name.padEnd(width)} ${view.version}-py313_0`),
        '',
        'Preparing transaction: done',
        'Verifying transaction: done',
        'Executing transaction: done',
      ].join('\n');
    },
    list({ records, context }) {
      const lines = [`# packages in environment at ${context.home}/miniconda3:`, '#', `# ${'Name'.padEnd(24)}${'Version'.padEnd(16)}${'Build'.padEnd(16)}Channel`];
      for (const record of [...records].sort((a, b) => a.name.localeCompare(b.name))) {
        lines.push(`${record.name.padEnd(26)}${record.version.padEnd(16)}${'py313_0'.padEnd(16)}${record.section ?? 'pkgs/main'}`);
      }
      return lines.join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `PackagesNotFoundError: The following packages are not available from current channels:\n\n  - ${query}`;
      return ['Loading channels: done', `# Name${' '.repeat(20)}Version${' '.repeat(11)}Build  Channel`,
        ...matches.map((entry) => `${entry.name.padEnd(26)}${entry.version.padEnd(18)}py313_0  ${entry.section ?? 'pkgs/main'}`)].join('\n');
    },
    info({ name, entry, record, context }) {
      if (!entry) return `PackagesNotFoundError: The following packages are not available from current channels:\n\n  - ${name}`;
      return [
        `${entry.name} ${record?.version ?? entry.version} py313_0`,
        '-'.repeat(40),
        `file name   : ${entry.name}-${entry.version}-py313_0.conda`,
        `name        : ${entry.name}`,
        `version     : ${record?.version ?? entry.version}`,
        `channel     : ${entry.section ?? 'pkgs/main'}`,
        `subdir      : ${context.os === 'windows' ? 'win-64' : context.os === 'macos' ? 'osx-arm64' : 'linux-64'}`,
        `summary     : ${entry.description}`,
        `dependencies: ${Object.entries(entry.dependencies ?? {}).map(([dependency, range]) => `${dependency} ${range}`).join(', ') || 'none'}`,
      ].join('\n');
    },
    refresh() { return 'Collecting package metadata (repodata.json): done\nAll requested packages already installed.'; },
    outdated({ entries }) {
      if (!entries.length) return '# All packages are up to date.';
      return entries.map(({ record, latest }) => `${record.name.padEnd(26)}${record.version.padEnd(16)}-> ${latest}`).join('\n');
    },
  },
});

export const pythonManagers: readonly ManagerSpec[] = [pipManager, pipxManager, uvManager, poetryManager, condaManager];
