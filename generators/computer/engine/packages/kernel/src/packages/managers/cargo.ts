import { defineManager, humanBytes } from './shared.js';

export const cargoManager = defineManager({
  id: 'cargo',
  commands: ['cargo'],
  family: 'language',
  projectScoped: false,
  bareCaret: true,
  verbs: {
    install: 'install', add: 'install', uninstall: 'remove', remove: 'remove', rm: 'remove',
    update: 'upgrade', list: 'list', search: 'search', info: 'info', tree: 'list',
    build: 'build', b: 'build', check: 'build', test: 'test', t: 'test', run: 'run', r: 'run',
    bench: 'test', clean: 'clean', doc: 'build', fmt: 'build', clippy: 'build', new: 'build', init: 'build',
    publish: 'build', fetch: 'refresh', help: 'help', metadata: 'info', vendor: 'build',
  },
  defaultVerb: 'help',
  versionFlags: ['--version', '--vers'],
  helpVerbs: ['build', 'check', 'test', 'run', 'bench', 'add', 'remove', 'install', 'uninstall', 'update', 'search', 'tree', 'clean', 'doc', 'fmt', 'clippy', 'new', 'init', 'publish'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) {
        return reused.map((view) => [
          `     Ignored package \`${view.entry.name} v${view.version}\` is already installed, use --force to override`,
        ].join('\n')).join('\n');
      }
      const lines = ['    Updating crates.io index'];
      const downloadBytes = installed.reduce((sum, view) => sum + (view.entry.sizeBytes ?? 1_048_576), 0);
      for (const view of installed) lines.push(`  Downloaded ${view.entry.name} v${view.version}`);
      lines.push(`  Downloaded ${installed.length} ${installed.length === 1 ? 'crate' : 'crates'} (${humanBytes(downloadBytes)}) in 0.35s`);
      for (const view of installed.filter((item) => item.direct)) lines.push(`  Installing ${view.entry.name} v${view.version}`);
      for (const view of [...installed].reverse()) lines.push(`   Compiling ${view.entry.name} v${view.version}`);
      lines.push('    Finished `release` profile [optimized] target(s) in 42.15s');
      for (const view of installed) {
        for (const binary of view.binaries) lines.push(`  Installing ${binary}`);
      }
      for (const view of installed.filter((item) => item.direct)) {
        const binaries = (view.entry.binaries ?? []).map((binary) => `\`${binary}\``).join(', ');
        lines.push(`   Installed package \`${view.entry.name} v${view.version}\`${binaries ? ` (${(view.entry.binaries ?? []).length === 1 ? 'executable' : 'executables'} ${binaries})` : ''}`);
      }
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'error: package ID specification did not match any packages';
      return all.flatMap((view) => view.binaries.map((binary) => `    Removing ${binary}`)).join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      return [...records].sort((a, b) => a.name.localeCompare(b.name))
        .flatMap((record) => [`${record.name} v${record.version}:`, ...record.provides.map((binary) => `    ${binary}`)]).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `note: no crates matched "${query}"`;
      const width = Math.max(...matches.map((entry) => `${entry.name} = "${entry.version}"`.length));
      return [
        ...matches.map((entry) => `${`${entry.name} = "${entry.version}"`.padEnd(width)}    # ${entry.description}`),
        `... and more crates found. Use --limit N to see more.`,
      ].join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `error: could not find \`${name}\` in registry \`crates-io\``;
      const dependencies = Object.entries(entry.dependencies ?? {});
      return [
        `${entry.name} #${entry.keywords?.[0] ?? 'cli'}`,
        entry.description,
        `version: ${record?.version ?? entry.version}`,
        `license: ${entry.license ?? 'MIT OR Apache-2.0'}`,
        `rust-version: 1.80`,
        `documentation: https://docs.seed.local/${entry.name}`,
        `repository: https://github.com/seed/${entry.name}`,
        ...(dependencies.length ? ['dependencies:', ...dependencies.map(([dependency, range]) => `  ${dependency} ${range}`)] : []),
        ...((entry.binaries ?? []).length ? [`binaries: ${(entry.binaries ?? []).join(', ')}`] : []),
        ...(record ? [`installed: ${record.installPath}`] : []),
      ].join('\n');
    },
    refresh() { return '    Updating crates.io index'; },
    outdated({ entries }) {
      if (!entries.length) return '    Updating crates.io index\n    Everything is up-to-date';
      return ['    Updating crates.io index', ...entries.map(({ record, latest }) => `    Updating ${record.name} v${record.version} -> v${latest}`)].join('\n');
    },
  },
});

export const goManager = defineManager({
  id: 'go',
  commands: ['go'],
  family: 'language',
  projectScoped: false,
  verbs: {
    install: 'install', get: 'install', build: 'build', test: 'test', run: 'run', vet: 'build',
    fmt: 'build', list: 'list', clean: 'clean', mod: 'build', version: 'info', doc: 'info',
    tool: 'exec', work: 'build', generate: 'build', env: 'info', help: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['build', 'install', 'get', 'run', 'test', 'vet', 'fmt', 'list', 'clean', 'mod', 'work', 'generate', 'doc', 'env', 'version'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.length ? '' : 'go: no packages to install';
      const lines: string[] = [];
      for (const view of installed) lines.push(`go: downloading ${view.entry.name} ${view.version}`);
      for (const view of installed.filter((item) => item.direct)) {
        lines.push(`go: installed ${view.entry.name}@${view.version}`);
        for (const binary of view.binaries) lines.push(`go: wrote ${binary}`);
      }
      return lines.join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'go: nothing to remove (use `go clean -modcache` for the module cache)';
      return removed.flatMap((view) => view.binaries.map((binary) => `go: removed ${binary}`)).join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      return [...records].sort((a, b) => a.name.localeCompare(b.name)).map((record) => `${record.name} ${record.version}`).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `go: no modules matched ${query}`;
      return matches.map((entry) => `${entry.name} ${entry.version}\n    ${entry.description}`).join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `go: module ${name}: not found`;
      return [
        `module ${entry.name}`,
        `    ${entry.description}`,
        `go: ${record?.version ?? entry.version}`,
        ...(record ? [`installed: ${record.installPath}`, `binaries: ${record.provides.join(', ')}`] : []),
      ].join('\n');
    },
    refresh() { return 'go: module cache verified'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      return entries.map(({ record, latest }) => `${record.name} ${record.version} [${latest}]`).join('\n');
    },
  },
});
