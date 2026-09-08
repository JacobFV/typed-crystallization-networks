import type { PackageManagerKind } from '@tcn-computer/protocol';
import { findEntry } from './catalog.js';
import type { PackageSpecifier } from './types.js';
import type { RangeOptions } from './versions.js';

/**
 * Command-line package references, parsed per family.
 *
 * The families genuinely disagree: npm writes `@scope/pkg@^1.2.3`, pip writes
 * `pkg[extra]==1.2`, apt writes `pkg=1.2-3`, Composer writes `vendor/pkg:^1.2`,
 * and Homebrew puts the version *inside* the name (`python@3.13`). Folding all
 * of that into "split on @" is what made `npm install react@18.2.0` install a
 * package called `react@18.2.0`.
 */

const DIST_TAGS = new Set(['latest', 'next', 'beta', 'alpha', 'canary', 'rc', 'stable', 'dev', 'nightly', 'experimental']);

type Family = 'npm' | 'python' | 'debian' | 'brew' | 'cargo' | 'go' | 'gem' | 'composer' | 'nuget' | 'plain';

const families: Readonly<Record<PackageManagerKind, Family>> = {
  npm: 'npm', pnpm: 'npm', yarn: 'npm', bun: 'npm',
  pip: 'python', pipx: 'python', uv: 'python', poetry: 'python', conda: 'python',
  apt: 'debian', dpkg: 'debian',
  brew: 'brew', mas: 'plain', snap: 'plain', flatpak: 'plain',
  winget: 'plain', choco: 'plain', scoop: 'plain',
  cargo: 'cargo', go: 'go', gem: 'gem', composer: 'composer',
  dotnet: 'nuget', nuget: 'nuget', vcpkg: 'plain',
};

/** Managers whose bare `1.2.3` means `^1.2.3`. */
export const bareCaretManagers = new Set<PackageManagerKind>(['cargo', 'composer']);

/** Managers where `~1.2` allows the last named component to vary, not the patch. */
const compatibleTildeManagers = new Set<PackageManagerKind>(['composer', 'gem']);

/** Range grammar options for a manager, so every caller reads a range the same way. */
export function rangeOptionsFor(manager: PackageManagerKind): RangeOptions {
  return { bareCaret: bareCaretManagers.has(manager), tildeCompatible: compatibleTildeManagers.has(manager) };
}

function nonRegistrySource(raw: string): PackageSpecifier | undefined {
  if (/^(https?|git\+https?|git|ssh):\/\//.test(raw)) {
    const name = raw.replace(/\.git$/, '').split('/').at(-1) ?? raw;
    return { raw, name, range: '*', source: { kind: /^git/.test(raw) ? 'git' : 'url', locator: raw } };
  }
  if (/^(file:|link:|portal:)/.test(raw)) {
    const locator = raw.replace(/^(file:|link:|portal:)/, '');
    return { raw, name: locator.split('/').filter((part) => part && part !== '.' && part !== '..').at(-1) ?? locator, range: '*', source: { kind: 'file', locator } };
  }
  if (/^(github|gitlab|bitbucket):/.test(raw)) {
    const locator = raw.split(':')[1] ?? '';
    return { raw, name: locator.split('/').at(-1) ?? locator, range: '*', source: { kind: 'git', locator: raw } };
  }
  if (/^(\.{1,2}\/|\/)/.test(raw) && !raw.endsWith('.deb')) {
    return { raw, name: raw.split('/').filter(Boolean).at(-1) ?? raw, range: '*', source: { kind: 'path', locator: raw } };
  }
  return undefined;
}

function rangeOrTag(raw: string, value: string): Pick<PackageSpecifier, 'range' | 'tag'> {
  const text = value.trim();
  if (!text) return { range: '*' };
  if (DIST_TAGS.has(text.toLowerCase())) return { range: '*', tag: text.toLowerCase() };
  return { range: text };
}

/** `@scope/pkg@^1.2.3` → the `@` that starts the version is the one after index 0. */
function splitNpm(raw: string): { name: string; version: string } {
  const at = raw.indexOf('@', raw.startsWith('@') ? 1 : 0);
  if (at <= 0) return { name: raw, version: '' };
  return { name: raw.slice(0, at), version: raw.slice(at + 1) };
}

function parsePython(raw: string): PackageSpecifier {
  const extrasMatch = /^([A-Za-z0-9._-]+)\s*\[([^\]]*)\]\s*(.*)$/.exec(raw);
  const head = extrasMatch ? extrasMatch[1]! : raw;
  const extras = extrasMatch ? extrasMatch[2]!.split(',').map((item) => item.trim()).filter(Boolean) : undefined;
  const rest = extrasMatch ? extrasMatch[3]! : '';
  const operatorMatch = /^([A-Za-z0-9._-]+)\s*((?:[<>=!~]=|[<>]|===)\s*.*)$/.exec(extrasMatch ? `${head}${rest}` : raw);
  if (operatorMatch) {
    return { raw, name: operatorMatch[1]!, range: operatorMatch[2]!.replace(/\s+/g, ''), ...(extras ? { extras } : {}) };
  }
  // conda's `pkg=1.2` is an exact-ish pin, not a comparison.
  const equals = /^([A-Za-z0-9._-]+)=([^=].*)$/.exec(head + rest);
  if (equals) return { raw, name: equals[1]!, range: `==${equals[2]!}`, ...(extras ? { extras } : {}) };
  return { raw, name: head, range: '*', ...(extras ? { extras } : {}) };
}

function parseDebian(raw: string): PackageSpecifier {
  let text = raw;
  let architecture: string | undefined;
  const arch = /^([^=/\s]+):([a-z0-9]+)(.*)$/.exec(text);
  if (arch && ['amd64', 'arm64', 'i386', 'all', 'armhf'].includes(arch[2]!)) {
    architecture = arch[2];
    text = `${arch[1]!}${arch[3]!}`;
  }
  const release = /^([^=]+)\/([A-Za-z][\w.-]*)$/.exec(text);
  if (release) return { raw, name: release[1]!, range: '*', ...(architecture ? { architecture } : {}) };
  const pinned = /^([^=]+)=(.+)$/.exec(text);
  if (pinned) return { raw, name: pinned[1]!, range: `=${pinned[2]!}`, ...(architecture ? { architecture } : {}) };
  return { raw, name: text, range: '*', ...(architecture ? { architecture } : {}) };
}

export function parseSpecifier(manager: PackageManagerKind, raw: string): PackageSpecifier {
  const text = raw.trim();
  if (!text) return { raw, name: '', range: '*' };
  const family = families[manager];

  const external = family !== 'debian' || !text.endsWith('.deb') ? nonRegistrySource(text) : undefined;
  if (external) return external;

  // A catalog name that itself contains the separator wins over splitting it:
  // `python@3.13`, `node@22`, `@types/node`, `laravel/framework`.
  const literal = findEntry(manager, text);
  if (literal) return { raw, name: literal.name, range: '*' };

  switch (family) {
    case 'npm': {
      const { name, version } = splitNpm(text);
      if (!version) return { raw, name, range: '*' };
      return { raw, name, ...rangeOrTag(raw, version) };
    }
    case 'python':
      return parsePython(text);
    case 'debian':
      return parseDebian(text);
    case 'brew': {
      // Versioned formulae (`postgresql@17`) are distinct formulae, so only an
      // `@` followed by something version-shaped that is *not* a formula splits.
      const at = text.lastIndexOf('@');
      if (at > 0) {
        const head = text.slice(0, at);
        const tail = text.slice(at + 1);
        if (findEntry(manager, `${head}@${tail}`)) return { raw, name: `${head}@${tail}`, range: '*' };
        if (/^\d/.test(tail) && !findEntry(manager, text)) return { raw, name: head, ...rangeOrTag(raw, tail) };
      }
      return { raw, name: text, range: '*' };
    }
    case 'cargo': {
      const at = text.lastIndexOf('@');
      if (at > 0) return { raw, name: text.slice(0, at), ...rangeOrTag(raw, text.slice(at + 1)) };
      return { raw, name: text, range: '*' };
    }
    case 'go': {
      const at = text.lastIndexOf('@');
      if (at > 0) return { raw, name: text.slice(0, at), ...rangeOrTag(raw, text.slice(at + 1)) };
      return { raw, name: text, range: '*', tag: 'latest' };
    }
    case 'gem': {
      const colon = /^([\w.-]+):(.+)$/.exec(text);
      if (colon) return { raw, name: colon[1]!, ...rangeOrTag(raw, colon[2]!) };
      return { raw, name: text, range: '*' };
    }
    case 'composer': {
      const colon = /^([^:]+):(.+)$/.exec(text);
      if (colon) return { raw, name: colon[1]!, ...rangeOrTag(raw, colon[2]!) };
      return { raw, name: text, range: '*' };
    }
    case 'nuget': {
      const at = text.lastIndexOf('@');
      if (at > 0) return { raw, name: text.slice(0, at), ...rangeOrTag(raw, text.slice(at + 1)) };
      return { raw, name: text, range: '*' };
    }
    default: {
      const at = text.lastIndexOf('@');
      if (at > 0 && /^[\d*]/.test(text.slice(at + 1))) return { raw, name: text.slice(0, at), ...rangeOrTag(raw, text.slice(at + 1)) };
      return { raw, name: text, range: '*' };
    }
  }
}

/**
 * Applies an out-of-band version flag (`--version 1.2.3`, `-v 1.2.3`) that
 * winget, choco, scoop, gem, cargo and dotnet accept instead of an inline pin.
 */
export function applyVersionFlag(specifiers: PackageSpecifier[], version: string | undefined): PackageSpecifier[] {
  if (!version) return specifiers;
  return specifiers.map((specifier) => (specifier.range === '*' && !specifier.tag ? { ...specifier, range: version } : specifier));
}
