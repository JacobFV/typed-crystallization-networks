import { defineManager, humanBytes, table } from './shared.js';

/** winget draws a dashed rule under its headers and pads to fixed columns. */
const rule = (headers: readonly string[], rows: readonly (readonly string[])[]): string => {
  const widths = headers.map((header, index) => Math.max(header.length, ...rows.map((row) => (row[index] ?? '').length)));
  const line = (cells: readonly string[]): string => cells.map((cell, index) => cell.padEnd(widths[index]!)).join(' ').trimEnd();
  return [line(headers), '-'.repeat(widths.reduce((sum, width) => sum + width + 1, -1))].concat(rows.map(line)).join('\n');
};

export const wingetManager = defineManager({
  id: 'winget',
  commands: ['winget'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', add: 'install', uninstall: 'remove', remove: 'remove', rm: 'remove',
    upgrade: 'upgrade', update: 'upgrade', list: 'list', ls: 'list', search: 'search', find: 'search',
    show: 'info', view: 'info', source: 'refresh', export: 'list', import: 'install',
    features: 'info', settings: 'info', help: 'help', pin: 'list', validate: 'info', hash: 'info',
  },
  defaultVerb: 'help',
  versionFlags: ['--version', '-v'],
  helpVerbs: ['install', 'uninstall', 'upgrade', 'list', 'search', 'show', 'source', 'export', 'import', 'pin', 'settings'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) {
        return reused.map((view) => [
          `Found ${view.entry.description} [${view.entry.name}] Version ${view.version}`,
          'No available upgrade found.',
          'No newer package versions are available from the configured sources.',
        ].join('\n')).join('\n');
      }
      const lines: string[] = [];
      for (const view of installed) {
        lines.push(`Found ${view.entry.description} [${view.entry.name}] Version ${view.version}`);
        lines.push('This application is licensed to you by its owner.');
        lines.push('Microsoft is not responsible for, nor does it grant any licenses to, third-party packages.');
        const bytes = view.entry.sizeBytes ?? 62_914_560;
        lines.push(`Downloading https://packages.seed.local/winget/${view.entry.name}/${view.version}/installer.exe`);
        lines.push(`  ██████████████████████████████  ${humanBytes(bytes)} / ${humanBytes(bytes)}`);
        lines.push('Successfully verified installer hash');
        lines.push('Starting package install...');
        for (const binary of view.entry.binaries ?? []) lines.push(`Command line alias added: ${binary}`);
        lines.push('Successfully installed');
      }
      return lines.join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'No installed package found matching input criteria.';
      return removed.flatMap((view) => [
        `Found ${view.entry.description} [${view.entry.name}]`,
        'Starting package uninstall...',
        '  ██████████████████████████████  100%',
        'Successfully uninstalled',
      ]).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'No installed package found matching input criteria.';
      return rule(
        ['Name', 'Id', 'Version', 'Available', 'Source'],
        records.map((record) => [record.description, record.name, record.version, '', record.section ?? 'winget']),
      );
    },
    search({ query, matches }) {
      if (!matches.length) return 'No package found matching input criteria.';
      return rule(
        ['Name', 'Id', 'Version', 'Match', 'Source'],
        matches.map((entry) => [entry.description, entry.name, entry.version, entry.name.toLowerCase().includes(query.toLowerCase()) ? `Moniker: ${query}` : `Tag: ${query}`, entry.section ?? 'winget']),
      );
    },
    info({ name, entry, record }) {
      if (!entry) return 'No package found matching input criteria.';
      return [
        `Found ${entry.description} [${entry.name}]`,
        `Version: ${record?.version ?? entry.version}`,
        `Publisher: ${entry.publisher ?? 'Unknown'}`,
        `Description: ${entry.description}`,
        `Homepage: https://packages.seed.local/winget/${entry.name}`,
        'License: Proprietary',
        'Installer:',
        '  Type: Exe',
        '  Locale: en-US',
        `  Download Url: https://packages.seed.local/winget/${entry.name}/${entry.version}/installer.exe`,
        '  SHA256: ' + '0'.repeat(64),
        ...(record ? [`  Installed Location: ${record.installPath}`] : []),
      ].join('\n');
    },
    refresh() {
      return [
        'Updating all sources...',
        'Updating source: winget...',
        'Done',
        'Updating source: msstore...',
        'Done',
      ].join('\n');
    },
    outdated({ entries }) {
      if (!entries.length) return 'No installed package found matching input criteria.';
      const rows = entries.map(({ record, latest }) => [record.description, record.name, record.version, latest, record.section ?? 'winget']);
      return `${rule(['Name', 'Id', 'Version', 'Available', 'Source'], rows)}\n${entries.length} upgrades available.`;
    },
  },
});
