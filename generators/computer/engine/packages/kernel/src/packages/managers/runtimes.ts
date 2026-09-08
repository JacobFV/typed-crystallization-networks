import { defineManager, type ManagerSpec, packageWord, table } from './shared.js';

export const gemManager = defineManager({
  id: 'gem',
  commands: ['gem'],
  family: 'language',
  projectScoped: false,
  verbs: {
    install: 'install', i: 'install', uninstall: 'remove', remove: 'remove', update: 'upgrade',
    list: 'list', query: 'list', search: 'search', info: 'info', 'specification': 'info',
    outdated: 'outdated', cleanup: 'clean', build: 'build', push: 'build', which: 'info',
    contents: 'info', environment: 'info', help: 'help', pristine: 'install',
  },
  defaultVerb: 'help',
  versionFlags: ['-v', '--version'],
  helpVerbs: ['install', 'uninstall', 'update', 'list', 'search', 'info', 'outdated', 'cleanup', 'build', 'push', 'contents', 'environment'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `Successfully installed ${view.entry.name}-${view.version}`).join('\n') || 'Nothing to install';
      const lines: string[] = [];
      for (const view of installed) lines.push(`Fetching ${view.entry.name}-${view.version}.gem`);
      for (const view of installed) {
        lines.push(`Successfully installed ${view.entry.name}-${view.version}`);
        for (const binary of view.entry.binaries ?? []) lines.push(`  Installing executable ${binary}`);
      }
      for (const view of installed) {
        lines.push(`Parsing documentation for ${view.entry.name}-${view.version}`);
        lines.push(`Installing ri documentation for ${view.entry.name}-${view.version}`);
      }
      lines.push(`Done installing documentation for ${installed.map((view) => view.entry.name).join(', ')} after 1 seconds`);
      lines.push(`${installed.length} ${installed.length === 1 ? 'gem' : 'gems'} installed`);
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'ERROR:  While executing gem ... (Gem::InstallError)\n    is not installed';
      return all.flatMap((view) => [
        `Removing ${(view.entry.binaries ?? [])[0] ?? view.entry.name}`,
        `Successfully uninstalled ${view.entry.name}-${view.version}`,
      ]).join('\n');
    },
    list({ records }) {
      const lines = ['', '*** LOCAL GEMS ***', ''];
      for (const record of [...records].sort((a, b) => a.name.localeCompare(b.name))) lines.push(`${record.name} (${record.version})`);
      return lines.join('\n');
    },
    search({ query, matches }) {
      const lines = ['', '*** REMOTE GEMS ***', ''];
      if (!matches.length) return `${lines.join('\n')}\nNo gems found matching ${query}`;
      for (const entry of matches) lines.push(`${entry.name} (${entry.version})`);
      return lines.join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `ERROR:  Could not find a valid gem '${name}' (>= 0) in any repository`;
      return [
        '',
        `${entry.name} (${record?.version ?? entry.version})`,
        `  Authors: seed`,
        `  Homepage: https://rubygems.seed.local/gems/${entry.name}`,
        `  License: ${entry.license ?? 'MIT'}`,
        ...(record ? [`  Installed at: ${record.installPath}`] : []),
        '',
        `  ${entry.description}`,
        '',
      ].join('\n');
    },
    refresh() { return 'RubyGems source https://rubygems.seed.local/ is reachable'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      return entries.map(({ record, latest }) => `${record.name} (${record.version} < ${latest})`).join('\n');
    },
  },
});

export const composerManager = defineManager({
  id: 'composer',
  commands: ['composer'],
  family: 'language',
  projectScoped: true,
  bareCaret: true,
  verbs: {
    require: 'install', install: 'install', 'create-project': 'install', update: 'upgrade', u: 'upgrade',
    remove: 'remove', show: 'list', list: 'list', search: 'search', info: 'info', outdated: 'outdated',
    'dump-autoload': 'build', dumpautoload: 'build', validate: 'info', 'clear-cache': 'clean',
    run: 'run', 'run-script': 'run', exec: 'exec', why: 'info', licenses: 'list', help: 'help', init: 'build',
  },
  defaultVerb: 'help',
  helpVerbs: ['require', 'install', 'update', 'remove', 'show', 'search', 'outdated', 'dump-autoload', 'validate', 'run-script', 'exec', 'why', 'licenses', 'init', 'create-project'],
  renderers: {
    install({ installed, reused }) {
      const lines = ['./composer.json has been updated'];
      lines.push('Running composer update ' + installed.filter((view) => view.direct).map((view) => view.entry.name).join(' '));
      lines.push('Loading composer repositories with package information');
      lines.push('Updating dependencies');
      if (!installed.length) {
        lines.push('Nothing to modify in lock file');
        lines.push('Installing dependencies from lock file (including require-dev)');
        lines.push(`Nothing to install, update or remove (${reused.length} ${packageWord(reused.length)} already present)`);
        lines.push('Generating autoload files');
        return lines.join('\n');
      }
      lines.push(`Lock file operations: ${installed.length} ${installed.length === 1 ? 'install' : 'installs'}, 0 updates, 0 removals`);
      for (const view of installed) lines.push(`  - Locking ${view.entry.name} (${view.version})`);
      lines.push('Writing lock file');
      lines.push('Installing dependencies from lock file (including require-dev)');
      lines.push(`Package operations: ${installed.length} ${installed.length === 1 ? 'install' : 'installs'}, 0 updates, 0 removals`);
      for (const view of installed) lines.push(`  - Installing ${view.entry.name} (${view.version}): Extracting archive`);
      lines.push('Generating autoload files');
      lines.push(`${installed.length} ${packageWord(installed.length)} you are using ${installed.length === 1 ? 'is' : 'are'} looking for funding.`);
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      const lines = ['./composer.json has been updated', 'Loading composer repositories with package information', 'Updating dependencies'];
      if (!all.length) {
        lines.push('Nothing to modify in lock file');
        return lines.join('\n');
      }
      lines.push(`Lock file operations: 0 installs, 0 updates, ${all.length} ${all.length === 1 ? 'removal' : 'removals'}`);
      for (const view of all) lines.push(`  - Removing ${view.entry.name} (${view.version})`);
      lines.push('Writing lock file');
      lines.push('Installing dependencies from lock file (including require-dev)');
      lines.push(`Package operations: 0 installs, 0 updates, ${all.length} ${all.length === 1 ? 'removal' : 'removals'}`);
      for (const view of all) lines.push(`  - Removing ${view.entry.name} (${view.version})`);
      lines.push('Generating autoload files');
      return lines.join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      const width = Math.max(...records.map((record) => record.name.length)) + 2;
      return [...records].sort((a, b) => a.name.localeCompare(b.name))
        .map((record) => `${record.name.padEnd(width)}${record.version.padEnd(12)} ${record.description}`).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `No results found for "${query}"`;
      return matches.map((entry) => `${entry.name} ${entry.description}`).join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `Package ${name} not found`;
      const requires = Object.entries(entry.dependencies ?? {});
      return [
        `name     : ${entry.name}`,
        `descrip. : ${entry.description}`,
        `keywords : ${(entry.keywords ?? ['php']).join(', ')}`,
        `versions : * ${record?.version ?? entry.version}`,
        `type     : library`,
        `license  : ${entry.license ?? 'MIT License (MIT)'}`,
        `source   : [git] https://github.com/${entry.name}.git`,
        ...(record ? [`path     : ${record.installPath}`] : []),
        ...(requires.length ? ['', 'requires', ...requires.map(([dependency, range]) => `${dependency} ${range}`)] : []),
      ].join('\n');
    },
    refresh() { return 'Loading composer repositories with package information\nCache cleared.'; },
    outdated({ entries }) {
      if (!entries.length) return 'All packages are up to date.';
      const width = Math.max(...entries.map(({ record }) => record.name.length)) + 2;
      return entries.map(({ record, latest }) => `${record.name.padEnd(width)}${record.version.padEnd(12)} ! ${latest.padEnd(12)} ${record.description}`).join('\n');
    },
  },
});

export const dotnetManager = defineManager({
  id: 'dotnet',
  commands: ['dotnet'],
  family: 'language',
  projectScoped: false,
  prefixes: [['tool'], ['workload']],
  verbs: {
    install: 'install', add: 'install', uninstall: 'remove', remove: 'remove', update: 'upgrade',
    list: 'list', search: 'search', info: 'info', restore: 'install', build: 'build', test: 'test',
    run: 'run', publish: 'build', clean: 'clean', new: 'build', pack: 'build', help: 'help', nuget: 'refresh',
  },
  defaultVerb: 'help',
  versionFlags: ['--version'],
  helpVerbs: ['tool install', 'tool uninstall', 'tool list', 'tool update', 'add package', 'restore', 'build', 'test', 'run', 'publish', 'clean', 'new', 'pack', 'nuget'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) {
        return reused.map((view) => `Tool '${view.entry.name}' is already installed.`).join('\n');
      }
      return installed.filter((view) => view.direct).flatMap((view) => [
        ...(view.entry.binaries ?? []).map((binary) => `You can invoke the tool using the following command: ${binary}`),
        `Tool '${view.entry.name}' (version '${view.version}') was successfully installed.`,
      ]).join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return "A tool with the specified name is not installed.";
      return all.map((view) => `Tool '${view.entry.name}' (version '${view.version}') was successfully uninstalled.`).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'Package Id      Version      Commands\n---------------------------------------';
      return table(['Package Id', 'Version', 'Commands'], records.map((record) => [record.name, record.version, record.provides.join(', ')]), '-');
    },
    search({ query, matches }) {
      if (!matches.length) return `No results found for '${query}'.`;
      return table(['Package ID', 'Latest Version', 'Authors', 'Downloads'], matches.map((entry) => [entry.name, entry.version, entry.publisher ?? 'seed', '1,000,000']), '-');
    },
    info({ name, entry, record }) {
      if (!entry) return `Error: no package found matching '${name}'.`;
      return [
        `${entry.name}`,
        `  Latest Version: ${record?.version ?? entry.version}`,
        `  Authors: ${entry.publisher ?? 'seed'}`,
        `  Description: ${entry.description}`,
        ...((entry.binaries ?? []).length ? [`  Commands: ${(entry.binaries ?? []).join(', ')}`] : []),
        ...(record ? [`  Installed: ${record.installPath}`] : []),
      ].join('\n');
    },
    refresh() { return 'Registered Sources:\n  1.  nuget.org [Enabled]\n      https://api.nuget.seed.local/v3/index.json'; },
    outdated({ entries }) {
      if (!entries.length) return 'All tools are up to date.';
      return entries.map(({ record, latest }) => `Tool '${record.name}' was updated from version '${record.version}' to version '${latest}'.`).join('\n');
    },
  },
});

export const nugetManager = defineManager({
  id: 'nuget',
  commands: ['nuget'],
  family: 'language',
  projectScoped: true,
  verbs: {
    install: 'install', add: 'install', restore: 'install', uninstall: 'remove', remove: 'remove',
    update: 'upgrade', list: 'list', search: 'search', info: 'info', spec: 'build', pack: 'build',
    push: 'build', sources: 'refresh', locals: 'clean', help: 'help', config: 'info', verify: 'info',
  },
  defaultVerb: 'help',
  versionFlags: ['-Version', '--version'],
  helpVerbs: ['install', 'restore', 'update', 'list', 'search', 'spec', 'pack', 'push', 'sources', 'locals', 'config', 'verify'],
  renderers: {
    install({ installed, reused, context }) {
      if (!installed.length) return reused.map((view) => `  GET https://api.nuget.seed.local/v3/${view.entry.name}\n  Package "${view.entry.name}.${view.version}" is already installed.`).join('\n');
      const lines: string[] = [];
      for (const view of installed) {
        lines.push(`info : Adding PackageReference for package '${view.entry.name}' into project '${context.cwd}/project.csproj'.`);
        lines.push(`info :   GET https://api.nuget.seed.local/v3/flatcontainer/${view.entry.name.toLowerCase()}/${view.version}/${view.entry.name.toLowerCase()}.${view.version}.nupkg`);
        lines.push(`info : Installing ${view.entry.name} ${view.version}.`);
      }
      lines.push(`log  : Restored ${context.cwd}/project.csproj (in 421 ms).`);
      return lines.join('\n');
    },
    remove({ removed, autoremoved, context }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'info : Package reference not found.';
      return all.map((view) => `info : Removing PackageReference for package '${view.entry.name}' from project '${context.cwd}/project.csproj'.`).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'No packages installed.';
      return records.map((record) => `${record.name} ${record.version}`).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `No results found for '${query}'.`;
      return matches.flatMap((entry) => [
        `> ${entry.name} | ${entry.version} | Downloads: 1,000,000`,
        `  ${entry.description}`,
        '',
      ]).join('\n').trimEnd();
    },
    info({ name, entry, record }) {
      if (!entry) return `No packages found matching '${name}'.`;
      return [
        `> ${entry.name} | ${record?.version ?? entry.version}`,
        `  ${entry.description}`,
        `  Dependencies: ${Object.entries(entry.dependencies ?? {}).map(([dependency, range]) => `${dependency} ${range}`).join(', ') || 'none'}`,
        ...(record ? [`  Path: ${record.installPath}`] : []),
      ].join('\n');
    },
    refresh() { return 'Registered Sources:\n  1.  nuget.org [Enabled]\n      https://api.nuget.seed.local/v3/index.json'; },
    outdated({ entries }) {
      if (!entries.length) return 'All packages are up to date.';
      return entries.map(({ record, latest }) => `   > ${record.name}   ${record.version} -> ${latest}`).join('\n');
    },
  },
});

export const vcpkgManager = defineManager({
  id: 'vcpkg',
  commands: ['vcpkg'],
  family: 'language',
  projectScoped: true,
  verbs: {
    install: 'install', add: 'install', remove: 'remove', upgrade: 'upgrade', update: 'outdated',
    list: 'list', search: 'search', info: 'info', 'depend-info': 'info', integrate: 'build',
    export: 'build', 'x-update-baseline': 'refresh', help: 'help', version: 'info', hash: 'info',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'remove', 'upgrade', 'update', 'list', 'search', 'depend-info', 'integrate', 'export', 'x-update-baseline', 'version'],
  renderers: {
    install({ installed, reused, context }) {
      const triplet = context.os === 'windows' ? 'x64-windows' : context.os === 'macos' ? 'arm64-osx' : 'x64-linux';
      if (!installed.length) {
        return reused.map((view) => `The following packages are already installed:\n    ${view.entry.name}:${triplet}`).join('\n');
      }
      const lines = ['Computing installation plan...', 'The following packages will be built and installed:'];
      for (const view of installed) lines.push(`    ${view.direct ? '' : '  * '}${view.entry.name}:${triplet} -> ${view.version}`);
      lines.push(`Detecting compiler hash for triplet ${triplet}...`);
      installed.forEach((view, index) => {
        lines.push(`Building ${view.entry.name}:${triplet}...`);
        lines.push(`Installing ${index + 1}/${installed.length} ${view.entry.name}:${triplet}...`);
        lines.push(`Elapsed time to handle ${view.entry.name}:${triplet}: 12 s`);
      });
      lines.push('Total install time: 42 s');
      for (const view of installed.filter((item) => item.direct)) {
        lines.push(`${view.entry.name} provides CMake targets:`);
        lines.push('');
        lines.push(`    find_package(${view.entry.name} CONFIG REQUIRED)`);
        lines.push(`    target_link_libraries(main PRIVATE ${view.entry.name}::${view.entry.name})`);
        lines.push('');
      }
      return lines.join('\n');
    },
    remove({ removed, autoremoved, context }) {
      const triplet = context.os === 'windows' ? 'x64-windows' : context.os === 'macos' ? 'arm64-osx' : 'x64-linux';
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'The following packages are not installed, so not removed.';
      return ['The following packages will be removed:', ...all.map((view) => `    ${view.entry.name}:${triplet}`),
        ...all.map((view) => `Removing ${view.entry.name}:${triplet}...`)].join('\n');
    },
    list({ records, context }) {
      const triplet = context.os === 'windows' ? 'x64-windows' : context.os === 'macos' ? 'arm64-osx' : 'x64-linux';
      if (!records.length) return 'No packages are installed. Did you mean `search`?';
      return records.map((record) => `${`${record.name}:${triplet}`.padEnd(37)}${record.version.padEnd(16)}${record.description}`).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `error: no packages match the query "${query}"`;
      return matches.map((entry) => `${entry.name.padEnd(24)}${entry.version.padEnd(16)}${entry.description}`).join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `error: ${name} does not exist in the port registry`;
      return [
        `${entry.name}: ${Object.keys(entry.dependencies ?? {}).join(', ') || '(no dependencies)'}`,
        `  version: ${record?.version ?? entry.version}`,
        `  description: ${entry.description}`,
        ...(record ? [`  installed: ${record.installPath}`] : []),
      ].join('\n');
    },
    refresh() { return 'Updated the baseline to the current commit of the registry.'; },
    outdated({ entries }) {
      if (!entries.length) return 'No packages need updating.';
      return ['The following packages differ from their port versions:', ...entries.map(({ record, latest }) => `    ${record.name.padEnd(32)}${record.version} -> ${latest}`)].join('\n');
    },
  },
});

export const runtimeManagers: readonly ManagerSpec[] = [gemManager, composerManager, dotnetManager, nugetManager, vcpkgManager];
