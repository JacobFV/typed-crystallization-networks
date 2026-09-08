import { defineManager, humanBytes, packageWord, table } from './shared.js';

const CHOCO_VERSION = 'Chocolatey v2.5.1';

export const chocoManager = defineManager({
  id: 'choco',
  commands: ['choco', 'chocolatey', 'cinst', 'cuninst'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', upgrade: 'upgrade', uninstall: 'remove', remove: 'remove',
    list: 'list', search: 'search', find: 'search', info: 'info', outdated: 'outdated',
    pin: 'list', source: 'refresh', sources: 'refresh', config: 'info', help: 'help', pack: 'build', push: 'build',
  },
  defaultVerb: 'help',
  versionFlags: ['--version'],
  helpVerbs: ['install', 'upgrade', 'uninstall', 'list', 'search', 'info', 'outdated', 'pin', 'source', 'config', 'pack'],
  renderers: {
    install({ installed, reused }) {
      const lines = [CHOCO_VERSION];
      if (!installed.length) {
        lines.push(...reused.map((view) => `${view.entry.name} v${view.version} already installed.`));
        lines.push(' Use --force to reinstall, specify a version to install, or try upgrade.');
        return lines.join('\n');
      }
      lines.push('Installing the following packages:');
      lines.push(installed.filter((view) => view.direct).map((view) => view.entry.name).join(';'));
      lines.push('By installing, you accept licenses for the packages.');
      lines.push('');
      for (const view of installed) {
        lines.push(`${view.entry.name} v${view.version} [Approved]`);
        lines.push(`${view.entry.name} package files install completed. Performing other installation steps.`);
        lines.push(` The install of ${view.entry.name} was successful.`);
        lines.push(`  Software installed to '${view.installPath.replace(/^\/C/, 'C:').replaceAll('/', '\\')}'`);
      }
      lines.push('');
      lines.push(`Chocolatey installed ${installed.length}/${installed.length} ${packageWord(installed.length)}.`);
      lines.push(" See the log for details (C:\\ProgramData\\chocolatey\\logs\\chocolatey.log).");
      return lines.join('\n');
    },
    remove({ removed }) {
      const lines = [CHOCO_VERSION];
      if (!removed.length) return `${CHOCO_VERSION}\nChocolatey uninstalled 0/0 packages.`;
      lines.push('Uninstalling the following packages:');
      lines.push(removed.map((view) => view.entry.name).join(';'));
      for (const view of removed) {
        lines.push('');
        lines.push(`${view.entry.name} v${view.version}`);
        lines.push(` ${view.entry.name} has been successfully uninstalled.`);
      }
      lines.push('');
      lines.push(`Chocolatey uninstalled ${removed.length}/${removed.length} ${packageWord(removed.length)}.`);
      return lines.join('\n');
    },
    list({ records }) {
      const lines = [CHOCO_VERSION, ...records.map((record) => `${record.name} ${record.version}`)];
      lines.push('');
      lines.push(`${records.length} ${packageWord(records.length)} installed.`);
      return lines.join('\n');
    },
    search({ query, matches }) {
      const lines = [CHOCO_VERSION];
      for (const entry of matches) {
        lines.push(`${entry.name} ${entry.version} [Approved]`);
        lines.push(` ${entry.description}`);
      }
      lines.push('');
      lines.push(matches.length ? `${matches.length} ${packageWord(matches.length)} found.` : `0 packages found matching '${query}'.`);
      return lines.join('\n');
    },
    info({ name, entry, record }) {
      if (!entry) return `${CHOCO_VERSION}\n0 packages found matching '${name}'.`;
      return [
        CHOCO_VERSION,
        `${entry.name} ${record?.version ?? entry.version} [Approved]`,
        ` Title: ${entry.description} | Published: 2026-06-01`,
        ' Package approved as a trusted package on Jun 01 2026.',
        ` Description: ${entry.description}`,
        ...(record ? [` Software Site: ${record.installPath}`] : []),
        '',
        '1 packages found.',
      ].join('\n');
    },
    refresh() {
      return [CHOCO_VERSION, 'chocolatey - https://community.chocolatey.seed.local/api/v2/ | Priority 0|Bypass Proxy - False|Self-Service - False|Visible to Admins Only - False'].join('\n');
    },
    outdated({ entries }) {
      const lines = [CHOCO_VERSION, 'Outdated Packages', ' Output is package name | current version | available version | pinned?', ''];
      for (const { record, latest } of entries) lines.push(`${record.name}|${record.version}|${latest}|false`);
      lines.push('');
      lines.push(`Chocolatey has determined ${entries.length} ${packageWord(entries.length)} are outdated.`);
      return lines.join('\n');
    },
  },
});

export const scoopManager = defineManager({
  id: 'scoop',
  commands: ['scoop'],
  family: 'native',
  projectScoped: false,
  verbs: {
    install: 'install', uninstall: 'remove', remove: 'remove', update: 'upgrade', list: 'list',
    search: 'search', info: 'info', status: 'outdated', bucket: 'refresh', cleanup: 'clean',
    cache: 'clean', hold: 'list', unhold: 'list', reset: 'install', which: 'info', help: 'help', prefix: 'info',
  },
  defaultVerb: 'help',
  helpVerbs: ['install', 'uninstall', 'update', 'list', 'search', 'info', 'status', 'bucket', 'cleanup', 'cache', 'hold', 'which', 'prefix'],
  renderers: {
    install({ installed, reused, context }) {
      if (!installed.length) {
        return reused.map((view) => [
          `WARN  '${view.entry.name}' (${view.version}) is already installed.`,
          `Use 'scoop update ${view.entry.name}' to install a new version.`,
        ].join('\n')).join('\n');
      }
      const lines: string[] = [];
      for (const view of installed) {
        const archive = `${view.entry.name}-${view.version}-x86_64-pc-windows-msvc.zip`;
        lines.push(`Installing '${view.entry.name}' (${view.version}) [64bit] from '${view.entry.section ?? 'main'}' bucket`);
        lines.push(`${archive} (${humanBytes(view.entry.sizeBytes ?? 3_145_728)}) [====================] 100%`);
        lines.push(`Checking hash of ${archive} ... ok.`);
        lines.push(`Extracting ${archive} ... done.`);
        lines.push(`Linking ~\\scoop\\apps\\${view.entry.name}\\current => ~\\scoop\\apps\\${view.entry.name}\\${view.version}`);
        for (const binary of view.entry.binaries ?? []) lines.push(`Creating shim for '${binary}'.`);
        lines.push(`'${view.entry.name}' (${view.version}) was installed successfully!`);
      }
      void context;
      return lines.join('\n');
    },
    remove({ removed }) {
      if (!removed.length) return "ERROR 'unknown' isn't installed.";
      return removed.flatMap((view) => [
        `Uninstalling '${view.entry.name}' (${view.version}).`,
        `Removing shim${(view.entry.binaries?.length ?? 0) === 1 ? '' : 's'} for ${(view.entry.binaries ?? []).map((binary) => `'${binary}'`).join(', ') || 'none'}.`,
        `Unlinking ~\\scoop\\apps\\${view.entry.name}\\current`,
        `'${view.entry.name}' was uninstalled.`,
      ]).join('\n');
    },
    list({ records }) {
      if (!records.length) return 'There aren\'t any apps installed.';
      return ['Installed apps:', '', table(
        ['Name', 'Version', 'Source', 'Updated', 'Info'],
        records.map((record) => [record.name, record.version, record.section ?? 'main', record.installedAt.replace('T', ' ').slice(0, 19), record.dependencyType === 'transitive' ? 'Dependency' : '']),
        '-',
      )].join('\n');
    },
    search({ query, matches }) {
      if (!matches.length) return `No matches found for '${query}'.`;
      const buckets = new Map<string, typeof matches>();
      for (const entry of matches) {
        const bucket = entry.section ?? 'main';
        buckets.set(bucket, [...(buckets.get(bucket) ?? []), entry]);
      }
      return [...buckets].map(([bucket, entries]) => [
        `'${bucket}' bucket:`,
        ...entries.map((entry) => `    ${entry.name} (${entry.version})${entry.binaries?.length ? ` --> includes '${entry.binaries[0]}'` : ''}`),
        '',
      ].join('\n')).join('\n').trimEnd();
    },
    info({ name, entry, record }) {
      if (!entry) return `Could not find manifest for '${name}'.`;
      return [
        `Name        : ${entry.name}`,
        `Description : ${entry.description}`,
        `Version     : ${record?.version ?? entry.version}`,
        `Bucket      : ${entry.section ?? 'main'}`,
        `Website     : https://scoop.seed.local/${entry.name}`,
        ...(record ? [`Installed   : ${record.installPath}`] : ['Installed   : No']),
        `Binaries    : ${(entry.binaries ?? []).join(' | ') || 'none'}`,
      ].join('\n');
    },
    refresh() { return "Updating Scoop...\nUpdating 'main' bucket...\nUpdating 'extras' bucket...\nScoop was updated successfully!"; },
    outdated({ entries }) {
      if (!entries.length) return 'Everything is ok!';
      return table(['Name', 'Installed Version', 'Latest Version', 'Missing Dependencies', 'Info'], entries.map(({ record, latest }) => [record.name, record.version, latest, '', 'Update available']), '-');
    },
  },
});
