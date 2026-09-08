import { applicationName } from '../layout.js';
import { defineManager, humanBytes, type PackageView } from './shared.js';

const BOTTLE_PLATFORM = 'arm64_sequoia';

/** Homebrew reports a keg as "<n> files, <size>". Derived from the entry's footprint. */
function kegSummary(view: PackageView): string {
  const bytes = view.entry.sizeBytes ?? 3_145_728;
  const files = Math.max(3, Math.round(bytes / 262_144) * 7 + 3);
  return `${files} files, ${humanBytes(bytes)}`;
}

export const brewManager = defineManager({
  id: 'brew',
  commands: ['brew'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', reinstall: 'install', remove: 'remove', uninstall: 'remove', rm: 'remove',
    autoremove: 'autoremove', update: 'refresh', upgrade: 'upgrade', list: 'list', ls: 'list',
    search: 'search', info: 'info', abv: 'info', outdated: 'outdated', cleanup: 'clean',
    services: 'services', help: 'help', deps: 'info', uses: 'info', doctor: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'reinstall', 'list', 'search', 'info', 'update', 'upgrade', 'outdated', 'autoremove', 'cleanup', 'services', 'deps'],
  renderers: {
    install({ installed, reused, cask, dryRun }) {
      if (!installed.length) {
        return reused.map((view) => [
          `Warning: ${view.entry.name} ${view.version} is already installed and up-to-date.`,
          `To reinstall ${view.version}, run:`,
          `  brew reinstall ${view.entry.name}`,
        ].join('\n')).join('\n');
      }
      const direct = installed.filter((view) => view.direct);
      const dependencies = installed.filter((view) => !view.direct);
      const lines: string[] = [];
      if (dryRun) {
        lines.push(`Would install ${installed.length} formulae:`);
        lines.push(installed.map((view) => view.entry.name).join(' '));
        return lines.join('\n');
      }
      if (dependencies.length && direct[0]) {
        lines.push(`==> Fetching dependencies for ${direct[0].entry.name}: ${dependencies.map((view) => view.entry.name).join(', ')}`);
      }
      for (const view of installed) {
        lines.push(`==> Fetching ${view.entry.cask ? 'cask ' : ''}${view.entry.name}`);
        lines.push(`==> Downloading https://ghcr.seed.local/v2/homebrew/${view.entry.cask ? 'cask' : 'core'}/${view.entry.name}/blobs/sha256-${view.version}`);
      }
      if (dependencies.length && direct[0]) {
        lines.push(`==> Installing dependencies for ${direct[0].entry.name}: ${dependencies.map((view) => view.entry.name).join(', ')}`);
      }
      for (const view of installed) {
        const label = view.direct || !direct[0] ? `==> Installing ${view.entry.name}` : `==> Installing ${direct[0].entry.name} dependency: ${view.entry.name}`;
        lines.push(label);
        if (view.entry.cask) {
          lines.push(`==> Installing Cask ${view.entry.name}`);
          lines.push(`==> Moving App '${applicationName(view.entry)}.app' to '/Applications/${applicationName(view.entry)}.app'`);
          for (const binary of view.entry.binaries ?? []) lines.push(`==> Linking Binary '${binary}' to '/opt/homebrew/bin/${binary}'`);
          lines.push(`🍺  ${view.entry.name} was successfully installed!`);
          continue;
        }
        lines.push(`==> Pouring ${view.entry.name}--${view.version}.${BOTTLE_PLATFORM}.bottle.tar.gz`);
        if (view.entry.service) {
          lines.push('==> Caveats');
          lines.push(`To start ${view.entry.name} now and restart at login:`);
          lines.push(`  brew services start ${view.entry.name}`);
          lines.push('==> Summary');
        }
        lines.push(`🍺  ${view.installPath}: ${kegSummary(view)}`);
      }
      lines.push(`==> Running \`brew cleanup ${direct.map((view) => view.entry.name).join(' ')}\`...`);
      if (cask && !installed.some((view) => view.entry.cask)) lines.push('Warning: --cask requested but the named formulae are not casks.');
      return lines.join('\n');
    },
    remove({ removed, autoremoved, orphans }) {
      if (!removed.length && !autoremoved.length) return 'Error: No such keg';
      const lines = removed.map((view) => `Uninstalling ${view.installPath}... (${kegSummary(view)})`);
      if (autoremoved.length) {
        lines.push(`==> Autoremoving ${autoremoved.length} unneeded ${autoremoved.length === 1 ? 'formula' : 'formulae'}:`);
        lines.push(autoremoved.map((view) => view.entry.name).join('\n'));
        for (const view of autoremoved) lines.push(`Uninstalling ${view.installPath}... (${kegSummary(view)})`);
      } else if (orphans.length) {
        lines.push(`==> ${orphans.length} unneeded ${orphans.length === 1 ? 'formula was' : 'formulae were'} left behind: ${orphans.join(' ')}`);
        lines.push('Run `brew autoremove` to remove them.');
      }
      return lines.join('\n');
    },
    list({ records }) {
      if (!records.length) return '';
      const formulae = records.filter((record) => !record.cask).sort((a, b) => a.name.localeCompare(b.name));
      const casks = records.filter((record) => record.cask).sort((a, b) => a.name.localeCompare(b.name));
      const lines: string[] = [];
      if (formulae.length) {
        lines.push('==> Formulae');
        for (const record of formulae) lines.push(`${record.name} ${record.version}`);
      }
      if (casks.length) {
        if (lines.length) lines.push('');
        lines.push('==> Casks');
        for (const record of casks) lines.push(`${record.name} ${record.version}`);
      }
      return lines.join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `No formulae or casks found for "${query}".`;
      const formulae = matches.filter((entry) => !entry.cask).map((entry) => entry.name);
      const casks = matches.filter((entry) => entry.cask).map((entry) => entry.name);
      const lines: string[] = [];
      if (formulae.length) { lines.push('==> Formulae'); lines.push(formulae.join('\n')); }
      if (casks.length) { if (lines.length) lines.push(''); lines.push('==> Casks'); lines.push(casks.join('\n')); }
      return lines.join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `Error: No available formula or cask with the name "${name}".`;
      const dependencies = Object.entries(entry.dependencies ?? {});
      const lines = [
        `==> ${entry.name}: stable ${entry.version}${entry.cask ? '' : ' (bottled)'}`,
        entry.description,
        entry.homepage ?? `https://formulae.brew.sh/formula/${entry.name}`,
      ];
      if (record) {
        lines.push('Installed');
        lines.push(`${record.installPath} (${record.files.length} files) *`);
        lines.push(`  Poured from bottle on ${record.installedAt}`);
      } else {
        lines.push('Not installed');
      }
      lines.push(`From: https://github.com/Homebrew/homebrew-${entry.cask ? 'cask' : 'core'}/blob/HEAD/${entry.cask ? 'Casks' : 'Formula'}/${entry.name[0]}/${entry.name}.rb`);
      if (entry.license) lines.push(`License: ${entry.license}`);
      if (dependencies.length) {
        lines.push('==> Dependencies');
        lines.push(`Required: ${dependencies.map(([dependency, range]) => `${dependency}${range === '*' ? '' : ` ${range}`}`).join(', ')}`);
      }
      if (entry.service) {
        lines.push('==> Caveats');
        lines.push(`To start ${entry.name} now and restart at login:`);
        lines.push(`  brew services start ${entry.name}`);
      }
      return lines.join('\n');
    },
    refresh() {
      return [
        '==> Updating Homebrew...',
        'Updated 2 taps (homebrew/core and homebrew/cask).',
        '==> New Formulae',
        'seed-cli',
        '==> Outdated Formulae',
        'You have outdated formulae installed.',
        'You can upgrade them with `brew upgrade`.',
      ].join('\n');
    },
    outdated({ entries }) {
      if (!entries.length) return '';
      return entries.map(({ record, latest }) => `${record.name} (${record.version}) < ${latest}`).join('\n');
    },
  },
});

export const masManager = defineManager({
  id: 'mas',
  commands: ['mas'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', purchase: 'install', lucky: 'install', uninstall: 'remove', remove: 'remove',
    list: 'list', search: 'search', info: 'info', outdated: 'outdated', upgrade: 'upgrade',
    account: 'info', version: 'info', help: 'help', open: 'help', home: 'info',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'list', 'search', 'info', 'outdated', 'upgrade', 'lucky', 'account'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `Warning: Already installed: "${view.entry.description}"`).join('\n');
      return installed.flatMap((view) => [
        `==> Downloading ${view.entry.description}`,
        `==> Installed ${view.entry.description}`,
      ]).join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return 'Error: Not installed';
      return removed.map((view) => `==> Uninstalled ${view.entry.description} (${view.installPath})`).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'No installed apps found';
      return records.map((record) => `${record.name.padEnd(12)}${record.description.padEnd(30)}(${record.version})`).join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `No results found for "${query}"`;
      return matches.map((entry) => `  ${entry.name.padEnd(12)}${entry.description.padEnd(30)}(${entry.version})`).join('\n');
    },
    info({ name, entry }) {
      if (!entry) return `Error: No results found for "${name}"`;
      return [
        `${entry.description} ${entry.version} [Free]`,
        `By: ${entry.publisher ?? 'Unknown'}`,
        'Released: 2026-01-15',
        `From: https://apps.apple.com/app/id${entry.name}`,
      ].join('\n');
    },
    refresh() { return 'mas: the Mac App Store catalog is always live'; },
    outdated({ entries }) {
      if (!entries.length) return '';
      return entries.map(({ record, latest }) => `${record.name} ${record.description} (${record.version} -> ${latest})`).join('\n');
    },
  },
});
