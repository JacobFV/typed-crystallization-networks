import { defineManager, humanBytes, table } from './shared.js';

export const snapManager = defineManager({
  id: 'snap',
  commands: ['snap'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', remove: 'remove', refresh: 'upgrade', list: 'list', find: 'search', search: 'search',
    info: 'info', services: 'services', start: 'start', stop: 'start', enable: 'start', disable: 'start',
    changes: 'list', 'known': 'info', help: 'help', version: 'info', connections: 'list',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'remove', 'refresh', 'list', 'find', 'info', 'services', 'start', 'stop', 'connections'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `snap "${view.entry.name}" is already installed, see 'snap help refresh'`).join('\n');
      return installed.map((view) => [
        `Download snap "${view.entry.name}" (${Math.round((view.entry.sizeBytes ?? 67_108_864) / 1_048_576)}) from channel "stable"`,
        `Fetch and check assertions for snap "${view.entry.name}" (${view.version})`,
        `Mount snap "${view.entry.name}" (${view.version})`,
        `Setup snap "${view.entry.name}" (${view.version}) security profiles`,
        `${view.entry.name} ${view.version} from ${view.entry.publisher ?? 'seed'} installed`,
      ].join('\n')).join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'error: snap not installed';
      return removed.map((view) => `${view.entry.name} removed`).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'No snaps are installed yet. Try \'snap install hello-world\'.';
      return table(
        ['Name', 'Version', 'Rev', 'Tracking', 'Publisher', 'Notes'],
        records.map((record, index) => [record.name, record.version, String(1000 + index), 'latest/stable', record.section === 'classic' ? 'canonical✓' : 'seed', record.section === 'classic' ? 'classic' : '-']),
      );
    },
    search({ query, matches }) {
      if (!matches.length) return `No matching snaps for "${query}"`;
      return table(
        ['Name', 'Version', 'Publisher', 'Notes', 'Summary'],
        matches.map((entry) => [entry.name, entry.version, entry.publisher ?? 'seed', entry.section === 'classic' ? 'classic' : '-', entry.description]),
      );
    },
    info({ name, entry, record }) {
      if (!entry) return `error: no snap found for "${name}"`;
      return [
        `name:      ${entry.name}`,
        `summary:   ${entry.description}`,
        `publisher: ${entry.publisher ?? 'seed'}`,
        'store-url: https://snapcraft.seed.local/' + entry.name,
        'license:   unset',
        `description: |`,
        `  ${entry.description}`,
        ...(entry.section === 'classic' ? ['snap-id:   seedclassic'] : []),
        ...(record ? [`installed:   ${record.version}  (1000) ${humanBytes(entry.sizeBytes ?? 67_108_864)} -`] : []),
        'channels:',
        `  latest/stable:    ${entry.version}`,
        `  latest/candidate: ${entry.version}`,
      ].join('\n');
    },
    refresh() { return 'All snaps up to date.'; },
    outdated({ entries }) {
      if (!entries.length) return 'All snaps up to date.';
      return table(['Name', 'Version', 'Rev', 'Publisher', 'Notes'], entries.map(({ record, latest }, index) => [record.name, latest, String(1001 + index), 'seed', `was ${record.version}`]));
    },
  },
});

export const flatpakManager = defineManager({
  id: 'flatpak',
  commands: ['flatpak'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', uninstall: 'remove', remove: 'remove', update: 'upgrade', list: 'list',
    search: 'search', info: 'info', run: 'run', 'remote-ls': 'search', remotes: 'list',
    'remote-add': 'refresh', repair: 'clean', help: 'help', history: 'list',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'update', 'list', 'search', 'info', 'run', 'remote-ls', 'remotes', 'repair'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `${view.entry.name}/x86_64/stable is already installed`).join('\n');
      const lines = ['Looking for matches…'];
      for (const view of installed) {
        lines.push(`Required runtime for ${view.entry.name}/x86_64/stable (runtime/org.freedesktop.Platform/x86_64/24.08) is already installed.`);
        lines.push('');
        lines.push(`        ID                            Branch          Op          Remote          Download`);
        lines.push(` 1. [✓] ${view.entry.name.padEnd(29)} stable          i           ${(view.entry.section ?? 'flathub').padEnd(15)} ${humanBytes(view.entry.sizeBytes ?? 134_217_728)}`);
        lines.push('');
      }
      lines.push(`Installation complete.`);
      return lines.join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'error: nothing matches';
      const lines = ['', '        ID                            Branch          Op'];
      removed.forEach((view, index) => lines.push(` ${index + 1}. [-] ${view.entry.name.padEnd(29)} stable          r`));
      lines.push('', 'Uninstall complete.');
      return lines.join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      return table(['Name', 'Application ID', 'Version', 'Branch', 'Installation'], records.map((record) => [record.description, record.name, record.version, 'stable', 'system']));
    },
    search({ query, matches }) {
      if (!matches.length) return `No matches found for "${query}"`;
      return table(['Name', 'Description', 'Application ID', 'Version', 'Branch', 'Remotes'], matches.map((entry) => [entry.name.split('.').at(-1) ?? entry.name, entry.description, entry.name, entry.version, 'stable', entry.section ?? 'flathub']));
    },
    info({ name, entry, record }) {
      if (!entry) return `error: ${name} not found`;
      return [
        '',
        `${entry.name.split('.').at(-1) ?? entry.name} - ${entry.description}`,
        '',
        `          ID: ${entry.name}`,
        `         Ref: app/${entry.name}/x86_64/stable`,
        `        Arch: x86_64`,
        `      Branch: stable`,
        `     Version: ${record?.version ?? entry.version}`,
        `     License: LGPL-2.1+`,
        `      Origin: ${entry.section ?? 'flathub'}`,
        `  Collection: org.flathub.Stable`,
        `Installation: ${record ? 'system' : '-'}`,
        `   Installed: ${humanBytes(entry.sizeBytes ?? 134_217_728)}`,
        `     Runtime: org.freedesktop.Platform/x86_64/24.08`,
        '',
      ].join('\n');
    },
    refresh() { return 'Looking for updates…\nNothing to do.'; },
    outdated({ entries }) {
      if (!entries.length) return 'Nothing to do.';
      return table(['Application ID', 'Installed', 'Available'], entries.map(({ record, latest }) => [record.name, record.version, latest]));
    },
  },
});
