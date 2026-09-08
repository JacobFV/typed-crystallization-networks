import { archiveUrl, aptSize, defineManager, type PackageView, totalSize, wrapNames } from './shared.js';

/** Debian's own vocabulary for what a package release is called. */
const SUITE = 'noble';
const DB_FILES = 214_536;

function transcriptHeader(): string[] {
  return ['Reading package lists... Done', 'Building dependency tree... Done', 'Reading state information... Done'];
}

function unpackLines(views: readonly PackageView[]): string[] {
  const lines: string[] = [];
  for (const view of views) {
    if (view.previousVersion) {
      lines.push(`Preparing to unpack .../${view.entry.name}_${view.version}_amd64.deb ...`);
      lines.push(`Unpacking ${view.entry.name} (${view.version}) over (${view.previousVersion}) ...`);
      continue;
    }
    lines.push(`Selecting previously unselected package ${view.entry.name}.`);
    lines.push(`Preparing to unpack .../${view.entry.name}_${view.version}_amd64.deb ...`);
    lines.push(`Unpacking ${view.entry.name} (${view.version}) ...`);
  }
  for (const view of views) lines.push(`Setting up ${view.entry.name} (${view.version}) ...`);
  return lines;
}

export const aptManager = defineManager({
  id: 'apt',
  commands: ['apt', 'apt-get', 'apt-cache'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', reinstall: 'install', remove: 'remove', purge: 'remove', autoremove: 'autoremove',
    update: 'refresh', upgrade: 'upgrade', 'full-upgrade': 'upgrade', 'dist-upgrade': 'upgrade',
    list: 'list', search: 'search', show: 'info', policy: 'info', showpkg: 'info', depends: 'info',
    clean: 'clean', autoclean: 'clean', help: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'reinstall', 'remove', 'purge', 'autoremove', 'update', 'upgrade', 'full-upgrade', 'list', 'search', 'show', 'policy', 'depends', 'clean'],
  renderers: {
    install({ installed, reused, dryRun }) {
      const lines = transcriptHeader();
      const direct = installed.filter((view) => view.direct);
      const extra = installed.filter((view) => !view.direct);
      if (extra.length) {
        lines.push('The following additional packages will be installed:');
        lines.push(wrapNames(extra.map((view) => view.entry.name).sort()));
      }
      if (!installed.length) {
        for (const view of reused) lines.push(`${view.entry.name} is already the newest version (${view.version}).`);
        lines.push('0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.');
        return lines.join('\n');
      }
      const upgrades = installed.filter((view) => view.previousVersion);
      const fresh = installed.filter((view) => !view.previousVersion);
      if (fresh.length) {
        lines.push('The following NEW packages will be installed:');
        lines.push(wrapNames(fresh.map((view) => view.entry.name).sort()));
      }
      if (upgrades.length) {
        lines.push('The following packages will be upgraded:');
        lines.push(wrapNames(upgrades.map((view) => view.entry.name).sort()));
      }
      lines.push(`${upgrades.length} upgraded, ${fresh.length} newly installed, 0 to remove and 0 not upgraded.`);
      const download = totalSize(installed, 'downloadBytes');
      lines.push(`Need to get ${aptSize(download)} of archives.`);
      lines.push(`After this operation, ${aptSize(totalSize(installed, 'sizeBytes'))} of additional disk space will be used.`);
      if (dryRun) {
        lines.push('Conf ' + direct.map((view) => `${view.entry.name} (${view.version})`).join(', '));
        return lines.join('\n');
      }
      installed.forEach((view, index) => {
        lines.push(`Get:${index + 1} ${archiveUrl('apt', view.entry, view.version)} [${aptSize(view.entry.downloadBytes ?? 98_304)}]`);
      });
      lines.push(`Fetched ${aptSize(download)} in 1s (${aptSize(download)}/s)`);
      lines.push(`(Reading database ... ${DB_FILES} files and directories currently installed.)`);
      lines.push(...unpackLines(installed));
      lines.push('Processing triggers for man-db (2.12.1-3build1) ...');
      const services = installed.filter((view) => view.entry.service);
      for (const view of services) lines.push(`Created symlink /etc/systemd/system/multi-user.target.wants/${view.entry.service!.name}.service → /lib/systemd/system/${view.entry.service!.name}.service.`);
      return lines.join('\n');
    },
    remove({ removed, orphans, autoremoved, purge }) {
      const lines = transcriptHeader();
      if (!removed.length && !autoremoved.length) {
        lines.push('0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.');
        return lines.join('\n');
      }
      if (orphans.length) {
        lines.push('The following packages were automatically installed and are no longer required:');
        lines.push(wrapNames([...orphans].sort()));
        lines.push("Use 'apt autoremove' to remove them.");
      }
      const all = [...removed, ...autoremoved];
      lines.push(`The following packages will be ${purge ? 'REMOVED' : 'REMOVED'}:`);
      lines.push(wrapNames(all.map((view) => `${view.entry.name}${purge ? '*' : ''}`).sort()));
      lines.push(`0 upgraded, 0 newly installed, ${all.length} to remove and 0 not upgraded.`);
      lines.push(`After this operation, ${aptSize(totalSize(all, 'sizeBytes'))} disk space will be freed.`);
      lines.push(`(Reading database ... ${DB_FILES} files and directories currently installed.)`);
      for (const view of all) lines.push(`Removing ${view.entry.name} (${view.version}) ...`);
      if (purge) for (const view of all) lines.push(`Purging configuration files for ${view.entry.name} (${view.version}) ...`);
      lines.push('Processing triggers for man-db (2.12.1-3build1) ...');
      return lines.join('\n');
    },
    list({ records }) {
      const lines = ['Listing... Done'];
      for (const record of [...records].sort((a, b) => a.name.localeCompare(b.name))) {
        lines.push(`${record.name}/${SUITE},now ${record.version} amd64 [installed${record.dependencyType === 'transitive' ? ',automatic' : ''}]`);
      }
      return lines.join('\n');
    },
    search({ query, matches }) {
      const lines = ['Sorting... Done', 'Full Text Search... Done', ''];
      if (!matches.length) return `Sorting... Done\nFull Text Search... Done\n\nN: Unable to locate any package matching "${query}"`;
      for (const entry of matches) {
        lines.push(`${entry.name}/${SUITE} ${entry.version} amd64`);
        lines.push(`  ${entry.description}`);
        lines.push('');
      }
      return lines.join('\n').trimEnd();
    },
    info({ name, entry, record }) {
      if (!entry) return `N: Unable to locate package ${name}`;
      const depends = Object.entries(entry.dependencies ?? {}).map(([dependency, range]) => `${dependency}${range === '*' ? '' : ` (${range})`}`);
      return [
        `Package: ${entry.name}`,
        `Version: ${record?.version ?? entry.version}`,
        'Priority: optional',
        `Section: ${entry.section ?? 'misc'}`,
        'Origin: Ubuntu',
        'Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>',
        `Installed-Size: ${aptSize(entry.sizeBytes ?? 262_144)}`,
        ...(depends.length ? [`Depends: ${depends.join(', ')}`] : []),
        ...(Object.keys(entry.conflicts ?? {}).length ? [`Conflicts: ${Object.keys(entry.conflicts ?? {}).join(', ')}`] : []),
        `Download-Size: ${aptSize(entry.downloadBytes ?? 98_304)}`,
        `APT-Sources: https://packages.seed.local/ubuntu ${SUITE}/${entry.section ?? 'main'} amd64 Packages`,
        ...(record ? ['APT-Manual-Installed: ' + (record.dependencyType === 'direct' ? 'yes' : 'no')] : []),
        `Description: ${entry.description}`,
        '',
      ].join('\n');
    },
    refresh() {
      return [
        `Hit:1 https://packages.seed.local/ubuntu ${SUITE} InRelease`,
        `Get:2 https://packages.seed.local/ubuntu ${SUITE}-updates InRelease [126 kB]`,
        `Get:3 https://packages.seed.local/ubuntu ${SUITE}-security InRelease [126 kB]`,
        'Fetched 252 kB in 1s (252 kB/s)',
        'Reading package lists... Done',
        'Building dependency tree... Done',
        'Reading state information... Done',
        'All packages are up to date.',
      ].join('\n');
    },
    outdated({ entries }) {
      if (!entries.length) return 'Listing... Done';
      return ['Listing... Done', ...entries.map(({ record, latest }) => `${record.name}/${SUITE}-updates ${latest} amd64 [upgradable from: ${record.version}]`)].join('\n');
    },
  },
});

export const dpkgManager = defineManager({
  id: 'dpkg',
  commands: ['dpkg', 'dpkg-query'],
  family: 'native',
  projectScoped: false,
  verbs: {
    '-i': 'install', '--install': 'install', unpack: 'install', '--unpack': 'install', install: 'install',
    '-r': 'remove', '--remove': 'remove', remove: 'remove', '-P': 'remove', '--purge': 'remove', purge: 'remove',
    '-l': 'list', '--list': 'list', list: 'list', '-L': 'info', '--listfiles': 'info',
    '-s': 'info', '--status': 'info', status: 'info', '-S': 'search', '--search': 'search', search: 'search',
    '--configure': 'install', '--audit': 'list', '--help': 'help', help: 'help',
  },
  defaultVerb: 'help',
  helpVerbs: ['-i|--install', '-r|--remove', '-P|--purge', '-l|--list', '-L|--listfiles', '-s|--status', '-S|--search'],
  renderers: {
    install({ installed, reused }) {
      if (!installed.length) return reused.map((view) => `dpkg: warning: ${view.entry.name} is already installed (${view.version})`).join('\n');
      const lines: string[] = [];
      for (const view of installed) {
        lines.push(`Selecting previously unselected package ${view.entry.aliases?.[0] ?? view.entry.name}.`);
        lines.push(`(Reading database ... ${DB_FILES} files and directories currently installed.)`);
        lines.push(`Preparing to unpack ${view.entry.name} ...`);
        lines.push(`Unpacking ${view.entry.aliases?.[0] ?? view.entry.name} (${view.version}) ...`);
      }
      for (const view of installed) lines.push(`Setting up ${view.entry.aliases?.[0] ?? view.entry.name} (${view.version}) ...`);
      lines.push('Processing triggers for man-db (2.12.1-3build1) ...');
      return lines.join('\n');
    },
    remove({ removed, autoremoved }) {
      const all = [...removed, ...autoremoved];
      if (!all.length) return 'dpkg: warning: nothing to remove';
      const lines = [`(Reading database ... ${DB_FILES} files and directories currently installed.)`];
      for (const view of all) lines.push(`Removing ${view.entry.aliases?.[0] ?? view.entry.name} (${view.version}) ...`);
      return lines.join('\n');
    },
    list({ records }) {
      const rows = [...records].sort((a, b) => a.name.localeCompare(b.name)).map((record) => {
        const short = record.aliases[0] ?? record.name;
        return [short, record.version, 'amd64', record.description];
      });
      const widths = [
        Math.max(14, ...rows.map((row) => row[0]!.length)),
        Math.max(12, ...rows.map((row) => row[1]!.length)),
        12,
      ];
      const lines = [
        'Desired=Unknown/Install/Remove/Purge/Hold',
        '| Status=Not/Inst/Conf-files/Unpacked/halF-conf/Half-inst/trig-aWait/Trig-pend',
        '|/ Err?=(none)/Reinst-required (Status,Err: uppercase=bad)',
        `||/ ${'Name'.padEnd(widths[0]!)} ${'Version'.padEnd(widths[1]!)} ${'Architecture'.padEnd(widths[2]!)} Description`,
        `+++-${'='.repeat(widths[0]!)}-${'='.repeat(widths[1]!)}-${'='.repeat(widths[2]!)}-${'='.repeat(33)}`,
      ];
      for (const row of rows) lines.push(`ii  ${row[0]!.padEnd(widths[0]!)} ${row[1]!.padEnd(widths[1]!)} ${row[2]!.padEnd(widths[2]!)} ${row[3]}`);
      if (!rows.length) lines.push('dpkg-query: no packages found matching *');
      return lines.join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `dpkg-query: no path found matching pattern ${query}`;
      return matches.map((entry) => `${entry.aliases?.[0] ?? entry.name}: /usr/bin/${entry.binaries?.[0] ?? entry.name}`).join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `dpkg-query: package '${name}' is not installed and no information is available`;
      return [
        `Package: ${entry.aliases?.[0] ?? entry.name}`,
        `Status: install ok ${record ? 'installed' : 'not-installed'}`,
        'Priority: optional',
        `Section: ${entry.section ?? 'misc'}`,
        `Installed-Size: ${Math.round((entry.sizeBytes ?? 262_144) / 1024)}`,
        'Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>',
        'Architecture: amd64',
        `Version: ${record?.version ?? entry.version}`,
        `Description: ${entry.description}`,
        '',
      ].join('\n');
    },
    refresh() { return 'dpkg: no package index to refresh; use apt update'; },
    outdated({ entries }) {
      return entries.length
        ? entries.map(({ record, latest }) => `${record.name} ${record.version} -> ${latest}`).join('\n')
        : 'dpkg: all packages at their archive version';
    },
  },
});
