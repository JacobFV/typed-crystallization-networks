import type { OperationKind } from '../types.js';
import { defineManager, type ManagerSpec, packageWord, type Renderers, table } from './shared.js';

/**
 * npm, pnpm, yarn and bun resolve against the same registry but disagree about
 * almost everything they print, and all four have a script runner — the surface
 * that makes a project buildable at all.
 */

const sharedVerbs: Readonly<Record<string, OperationKind>> = {
  install: 'install', i: 'install', add: 'install', ci: 'install',
  uninstall: 'remove', remove: 'remove', rm: 'remove', un: 'remove',
  update: 'upgrade', up: 'upgrade', upgrade: 'upgrade',
  list: 'list', ls: 'list', outdated: 'outdated',
  view: 'info', info: 'info', show: 'info', search: 'search',
  run: 'run', 'run-script': 'run', exec: 'exec', test: 'test', start: 'start', build: 'build',
  prune: 'autoremove', dedupe: 'autoremove', cache: 'clean', help: 'help', why: 'info', link: 'install',
};

/** `node_modules` layout for the `ls` tree. */
function tree(records: readonly { name: string; version: string; dependencyType: string }[]): string[] {
  const direct = records.filter((record) => record.dependencyType === 'direct').sort((a, b) => a.name.localeCompare(b.name));
  return direct.map((record, index) => `${index === direct.length - 1 ? '└──' : '├──'} ${record.name}@${record.version}`);
}

const registryInfo: Renderers['info'] = ({ name, entry, record }) => {
  if (!entry) return `npm error code E404\nnpm error 404 Not Found - GET https://registry.seed.local/${name} - Not found`;
  const dependencies = Object.entries(entry.dependencies ?? {});
  const versions = entry.versions ?? [entry.version];
  return [
    `${entry.name}@${record?.version ?? entry.version} | ${entry.license ?? 'MIT'} | deps: ${dependencies.length || 'none'} | versions: ${versions.length}`,
    entry.description,
    `https://registry.seed.local/${entry.name}`,
    '',
    ...(dependencies.length ? ['dependencies:', ...dependencies.map(([dependency, range]) => `${dependency}: ${range}`), ''] : []),
    `dist-tags:`,
    `latest: ${entry.version}`,
    '',
    `published by seed-registry`,
    ...(record ? [`installed: ${record.version} at ${record.installPath}`] : []),
  ].join('\n');
};

const registrySearch: Renderers['search'] = ({ query, matches }) => {
  if (!matches.length) return `No matches found for "${query}"`;
  return table(
    ['NAME', 'DESCRIPTION', 'AUTHOR', 'DATE', 'VERSION'],
    matches.map((entry) => [entry.name, entry.description.slice(0, 48), '=seed', '2026-06-01', entry.version]),
  );
};

const registryOutdated: Renderers['outdated'] = ({ entries }) => {
  if (!entries.length) return '';
  return table(
    ['Package', 'Current', 'Wanted', 'Latest', 'Location', 'Depended by'],
    entries.map(({ record, latest }) => [record.name, record.version, latest, latest, `node_modules/${record.name}`, 'seed-project']),
  );
};

export const npmManager = defineManager({
  id: 'npm',
  commands: ['npm', 'npx'],
  family: 'language',
  projectScoped: true,
  verbs: { ...sharedVerbs, init: 'build', publish: 'build', pack: 'build', audit: 'info', fund: 'info', ping: 'refresh', 'dist-tag': 'info' },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'update', 'run', 'exec', 'test', 'start', 'ls', 'outdated', 'view', 'search', 'prune', 'dedupe', 'audit', 'init', 'publish'],
  renderers: {
    install({ installed, reused, dev }) {
      const total = installed.length + reused.length;
      if (!installed.length) return `up to date, audited ${total || 1} ${packageWord(total || 1)} in 0s\n\nfound 0 vulnerabilities`;
      const funding = Math.min(2, installed.length);
      return [
        `added ${installed.length} ${packageWord(installed.length)}, and audited ${total + 1} ${packageWord(total + 1)} in 1s`,
        '',
        `${funding} ${packageWord(funding)} ${funding === 1 ? 'is' : 'are'} looking for funding`,
        '  run `npm fund` for details',
        '',
        'found 0 vulnerabilities',
        ...(dev ? ['', `saved to devDependencies`] : []),
      ].join('\n');
    },
    remove({ removed, autoremoved }) {
      const total = removed.length + autoremoved.length;
      if (!total) return 'up to date, audited 1 package in 0s\n\nfound 0 vulnerabilities';
      return [
        `removed ${total} ${packageWord(total)} in 0s`,
        '',
        'found 0 vulnerabilities',
      ].join('\n');
    },
    list({ records, context, projectName }) {
      const header = `${projectName ?? 'seed-project'}@1.0.0 ${context.cwd}`;
      return records.length ? [header, ...tree(records)].join('\n') : `${header}\n└── (empty)`;
    },
    search: registrySearch,
    info: registryInfo,
    refresh() { return 'npm notice PING https://registry.seed.local/\nnpm notice PONG 12ms'; },
    outdated: registryOutdated,
  },
});

export const pnpmManager = defineManager({
  id: 'pnpm',
  commands: ['pnpm', 'pnpx'],
  family: 'language',
  projectScoped: true,
  verbs: { ...sharedVerbs, dlx: 'exec', why: 'info', store: 'clean', patch: 'build', publish: 'build', init: 'build' },
  defaultVerb: 'help',
  helpVerbs: ['add', 'install', 'remove', 'update', 'run', 'exec', 'dlx', 'test', 'start', 'list', 'outdated', 'why', 'prune', 'store', 'publish'],
  renderers: {
    install({ installed, reused, dev }) {
      if (!installed.length) {
        return ['Already up to date', '', `Done in 0.3s using pnpm v10.15.1`].join('\n');
      }
      const direct = installed.filter((view) => view.direct);
      return [
        `Packages: +${installed.length}`,
        '+'.repeat(installed.length),
        `Progress: resolved ${installed.length + reused.length}, reused ${reused.length}, downloaded ${installed.length}, added ${installed.length}, done`,
        '',
        `${dev ? 'devDependencies' : 'dependencies'}:`,
        ...direct.map((view) => `+ ${view.entry.name} ${view.version}`),
        '',
        'Done in 1.2s using pnpm v10.15.1',
      ].join('\n');
    },
    remove({ removed, autoremoved }) {
      const total = removed.length + autoremoved.length;
      if (!total) return 'Already up to date\n\nDone in 0.2s using pnpm v10.15.1';
      return [
        `Packages: -${total}`,
        '-'.repeat(total),
        '',
        'dependencies:',
        ...removed.map((view) => `- ${view.entry.name} ${view.version}`),
        '',
        'Done in 0.4s using pnpm v10.15.1',
      ].join('\n');
    },
    list({ records, context, projectName }) {
      const direct = records.filter((record) => record.dependencyType === 'direct');
      const header = `${projectName ?? 'seed-project'}@1.0.0 ${context.cwd}`;
      if (!direct.length) return `Legend: production dependency, optional only, dev only\n\n${header}\n\n(no dependencies)`;
      return [
        'Legend: production dependency, optional only, dev only',
        '',
        header,
        '',
        'dependencies:',
        ...direct.map((record) => `${record.name} ${record.version}`),
      ].join('\n');
    },
    search: registrySearch,
    info: registryInfo,
    refresh() { return 'pnpm: registry https://registry.seed.local/ reachable'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      return table(['Package', 'Current', 'Latest'], entries.map(({ record, latest }) => [record.name, record.version, latest]), '─');
    },
  },
});

export const yarnManager = defineManager({
  id: 'yarn',
  commands: ['yarn', 'yarnpkg'],
  family: 'language',
  projectScoped: true,
  verbs: { ...sharedVerbs, dlx: 'exec', why: 'info', workspaces: 'list', 'set': 'build', init: 'build', publish: 'build' },
  defaultVerb: 'install',
  helpVerbs: ['add', 'install', 'remove', 'up', 'run', 'exec', 'dlx', 'test', 'list', 'outdated', 'why', 'workspaces', 'init', 'publish'],
  renderers: {
    install({ installed, reused }) {
      const lines = [
        'yarn add v1.22.22',
        '[1/4] Resolving packages...',
        '[2/4] Fetching packages...',
        '[3/4] Linking dependencies...',
        '[4/4] Building fresh packages...',
      ];
      if (!installed.length) {
        lines.push('success Already up-to-date.');
        lines.push('✨  Done in 0.31s.');
        return lines.join('\n');
      }
      const direct = installed.filter((view) => view.direct);
      lines.push(`success Saved ${installed.length} new ${installed.length === 1 ? 'dependency' : 'dependencies'}.`);
      lines.push('info Direct dependencies');
      direct.forEach((view, index) => lines.push(`${index === direct.length - 1 ? '└─' : '├─'} ${view.entry.name}@${view.version}`));
      lines.push('info All dependencies');
      const all = [...installed, ...reused].sort((a, b) => a.entry.name.localeCompare(b.entry.name));
      all.forEach((view, index) => lines.push(`${index === all.length - 1 ? '└─' : '├─'} ${view.entry.name}@${view.version}`));
      lines.push('✨  Done in 1.23s.');
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const total = removed.length + autoremoved.length;
      return [
        'yarn remove v1.22.22',
        '[1/2] Removing module ' + (removed.map((view) => view.entry.name).join(', ') || 'nothing') + '...',
        '[2/2] Regenerating lockfile and installing missing dependencies...',
        total ? `success Uninstalled ${total} ${packageWord(total)}.` : 'success Nothing to remove.',
        '✨  Done in 0.52s.',
      ].join('\n');
    },
    list({ records }) {
      const lines = ['yarn list v1.22.22'];
      const direct = records.filter((record) => record.dependencyType === 'direct').sort((a, b) => a.name.localeCompare(b.name));
      direct.forEach((record, index) => lines.push(`${index === direct.length - 1 ? '└─' : '├─'} ${record.name}@${record.version}`));
      lines.push('✨  Done in 0.10s.');
      return lines.join('\n');
    },
    search: registrySearch,
    info: registryInfo,
    refresh() { return 'yarn config v1.22.22\nregistry https://registry.seed.local/\n✨  Done in 0.05s.'; },
    outdated({ entries }) {
      if (!entries.length) return 'yarn outdated v1.22.22\n✨  Done in 0.11s.';
      return ['yarn outdated v1.22.22', 'info Color legend :', ' "<red>"    : Major Update backward-incompatible updates', table(
        ['Package', 'Current', 'Wanted', 'Latest', 'Package Type', 'URL'],
        entries.map(({ record, latest }) => [record.name, record.version, latest, latest, 'dependencies', `https://registry.seed.local/${record.name}`]),
      ), '✨  Done in 0.35s.'].join('\n');
    },
  },
});

export const bunManager = defineManager({
  id: 'bun',
  commands: ['bun', 'bunx'],
  family: 'language',
  projectScoped: true,
  verbs: { ...sharedVerbs, x: 'exec', create: 'build', init: 'build', publish: 'build', pm: 'list', repl: 'run' },
  defaultVerb: 'help',
  helpVerbs: ['add', 'install', 'remove', 'update', 'run', 'x', 'test', 'build', 'pm', 'outdated', 'init', 'create', 'publish'],
  renderers: {
    install({ installed, reused, dev }) {
      const lines = ['bun add v1.2.21', ''];
      if (!installed.length) {
        lines.push(`Checked ${reused.length || 1} ${packageWord(reused.length || 1)} (no changes) [3.00ms]`);
        return lines.join('\n');
      }
      for (const view of installed.filter((item) => item.direct)) {
        lines.push(`installed ${view.entry.name}@${view.version}${dev ? ' (dev)' : ''}`);
        for (const binary of view.entry.binaries ?? []) lines.push(`  ${binary}`);
      }
      lines.push('');
      lines.push(`${installed.length} ${packageWord(installed.length)} installed [123.00ms]`);
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const total = removed.length + autoremoved.length;
      return ['bun remove v1.2.21', '', total ? `${total} ${packageWord(total)} removed [12.00ms]` : 'no packages removed [1.00ms]'].join('\n');
    },
    list({ records, context }) {
      if (!records.length) return `${context.cwd} node_modules (0)`;
      return [`${context.cwd} node_modules (${records.length})`, ...tree(records)].join('\n');
    },
    search: registrySearch,
    info: registryInfo,
    refresh() { return 'bun: registry https://registry.seed.local/ ok'; },
    outdated: registryOutdated,
  },
});

export const javascriptManagers: readonly ManagerSpec[] = [npmManager, pnpmManager, yarnManager, bunManager];
