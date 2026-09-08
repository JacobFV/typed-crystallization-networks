/**
 * Version algebra shared by every package-manager family.
 *
 * One parser has to cope with semver (`1.2.3-beta.1`), Debian revisions
 * (`1:1.24.0-2ubuntu7`), Homebrew bottle revisions (`2.48.1_1`) and PEP 440
 * (`2.2.0rc1`), because the resolver compares versions across families the
 * moment a manifest pins a dependency. The normalized shape is therefore
 * `epoch : parts : prerelease : revision`, and every dialect is folded into it.
 */

export interface ParsedVersion {
  /** Debian-style epoch. Absent epochs sort as 0, so `1:1.0` beats `9.9`. */
  epoch: number;
  /** Dotted numeric release segment, left-aligned and zero-extended on compare. */
  parts: number[];
  /** Prerelease identifiers. A non-empty list always sorts *below* an empty one. */
  pre: Array<string | number>;
  /** Packaging revision (`-2ubuntu7`, `_1`). Sorts above a missing revision. */
  revision: string;
  raw: string;
}

export type ComparisonOperator = '>=' | '>' | '<=' | '<' | '=' | '!=';

export interface Comparator {
  op: ComparisonOperator;
  version: ParsedVersion;
}

/** Disjunction of conjunctions: `a && b || c` is `[[a, b], [c]]`. An empty inner list matches anything. */
export type VersionRange = Comparator[][];

export interface RangeOptions {
  /**
   * Cargo and Composer read a bare `1.2.3` as `^1.2.3`; npm and pip read it as
   * an exact pin. Everything else about the grammar is shared.
   */
  bareCaret?: boolean;
  /**
   * `~` is not one operator. npm reads `~1.2` as `>=1.2 <1.3`; Composer and
   * RubyGems read it as "allow the last named component to vary", so `~1.2`
   * means `>=1.2 <2.0` — the same rule PEP 440 spells `~=`.
   */
  tildeCompatible?: boolean;
}

const PRERELEASE_WORD = /^(?:alpha|beta|rc|pre|preview|dev|snapshot|nightly|canary|next|a|b|c)\d*$/i;

function splitIdentifiers(value: string): Array<string | number> {
  const out: Array<string | number> = [];
  for (const chunk of value.split(/[.+]/)) {
    if (!chunk) continue;
    for (const token of chunk.match(/\d+|\D+/g) ?? []) out.push(/^\d+$/.test(token) ? Number(token) : token);
  }
  return out;
}

/**
 * Splits the tail after the first `-`/`_`/`~` into a prerelease or a packaging
 * revision. A tail that opens with a digit (`-2ubuntu7`) is a distro revision;
 * one that opens with a known prerelease word is a prerelease.
 */
function classifyTail(tail: string): { pre: Array<string | number>; revision: string } {
  if (!tail) return { pre: [], revision: '' };
  const head = tail.split(/[.+-]/)[0] ?? '';
  if (/^\d/.test(head)) return { pre: [], revision: tail };
  if (PRERELEASE_WORD.test(head)) return { pre: splitIdentifiers(tail), revision: '' };
  return { pre: [], revision: tail };
}

export function parseVersion(input: string): ParsedVersion {
  const raw = String(input ?? '').trim();
  let text = raw.replace(/^[v=\s]+/, '');
  let epoch = 0;
  const epochMatch = /^(\d+):/.exec(text);
  if (epochMatch) {
    epoch = Number(epochMatch[1]);
    text = text.slice(epochMatch[0].length);
  }
  text = text.replace(/\+.*$/, '');
  // PEP 440 glues the prerelease to the release (`2.2.0rc1`); give it a dash so
  // the shared splitter sees the same shape as semver.
  text = text.replace(/^(\d[\d.]*?)(a|b|c|rc|alpha|beta|dev|post)(\d*)$/i, '$1-$2$3');
  const separator = text.search(/[-_~]/);
  const core = separator === -1 ? text : text.slice(0, separator);
  const tail = separator === -1 ? '' : text.slice(separator + 1);
  const parts = core.split('.').map((part) => {
    const digits = /^\d+/.exec(part);
    return digits ? Number(digits[0]) : 0;
  });
  const { pre, revision } = classifyTail(tail);
  return { epoch, parts: parts.length ? parts : [0], pre, revision, raw };
}

function compareIdentifierLists(a: Array<string | number>, b: Array<string | number>): number {
  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const left = a[index];
    const right = b[index];
    if (left === undefined) return -1;
    if (right === undefined) return 1;
    if (typeof left === 'number' && typeof right === 'number') {
      if (left !== right) return left < right ? -1 : 1;
      continue;
    }
    // Numeric identifiers always sort below alphanumeric ones (semver §11).
    if (typeof left === 'number') return -1;
    if (typeof right === 'number') return 1;
    if (left !== right) return left < right ? -1 : 1;
  }
  return 0;
}

export function compareVersions(a: string | ParsedVersion, b: string | ParsedVersion): number {
  const left = typeof a === 'string' ? parseVersion(a) : a;
  const right = typeof b === 'string' ? parseVersion(b) : b;
  if (left.epoch !== right.epoch) return left.epoch < right.epoch ? -1 : 1;
  const length = Math.max(left.parts.length, right.parts.length);
  for (let index = 0; index < length; index += 1) {
    const l = left.parts[index] ?? 0;
    const r = right.parts[index] ?? 0;
    if (l !== r) return l < r ? -1 : 1;
  }
  if (left.pre.length !== right.pre.length && (!left.pre.length || !right.pre.length)) return left.pre.length ? -1 : 1;
  const pre = compareIdentifierLists(left.pre, right.pre);
  if (pre !== 0) return pre;
  if (left.revision !== right.revision) {
    if (!left.revision) return -1;
    if (!right.revision) return 1;
    return compareIdentifierLists(splitIdentifiers(left.revision), splitIdentifiers(right.revision));
  }
  return 0;
}

/** Highest version of the list, or `undefined` when the list is empty. */
export function highestVersion(versions: readonly string[]): string | undefined {
  return [...versions].sort(compareVersions).at(-1);
}

function bumpMajor(version: ParsedVersion): string {
  return `${(version.parts[0] ?? 0) + 1}.0.0`;
}

function bumpMinor(version: ParsedVersion): string {
  return `${version.parts[0] ?? 0}.${(version.parts[1] ?? 0) + 1}.0`;
}

function bumpPatch(version: ParsedVersion): string {
  return `${version.parts[0] ?? 0}.${version.parts[1] ?? 0}.${(version.parts[2] ?? 0) + 1}`;
}

/** `^1.2.3` → `>=1.2.3 <2.0.0`, with npm's 0.x and 0.0.x narrowing. */
function caretComparators(text: string): Comparator[] {
  const version = parseVersion(text);
  const [major = 0, minor = 0] = version.parts;
  const upper = major > 0 ? bumpMajor(version)
    : minor > 0 || version.parts.length < 2 ? bumpMinor(version)
      : version.parts.length < 3 ? bumpMinor(version) : bumpPatch(version);
  return [{ op: '>=', version }, { op: '<', version: parseVersion(upper) }];
}

/** `~1.2.3` → `>=1.2.3 <1.3.0`; `~1` → `>=1.0.0 <2.0.0`. */
function tildeComparators(text: string): Comparator[] {
  const version = parseVersion(text);
  const upper = version.parts.length >= 2 ? bumpMinor(version) : bumpMajor(version);
  return [{ op: '>=', version }, { op: '<', version: parseVersion(upper) }];
}

/** PEP 440 `~=1.2.3` → `>=1.2.3, ==1.2.*`; `~=1.2` → `>=1.2, ==1.*`. */
function compatibleComparators(text: string): Comparator[] {
  const version = parseVersion(text);
  const upper = version.parts.length >= 3 ? bumpMinor(version) : bumpMajor(version);
  return [{ op: '>=', version }, { op: '<', version: parseVersion(upper) }];
}

/** `1.2.x` / `1.2.*` → `>=1.2.0 <1.3.0`; a bare `*` matches everything. */
function wildcardComparators(text: string): Comparator[] {
  const core = text.replace(/\.[*xX]$/, '').replace(/[*xX]/g, '');
  if (!core.replace(/\./g, '')) return [];
  const version = parseVersion(core);
  const upper = version.parts.length >= 2 ? bumpMinor(version) : bumpMajor(version);
  return [{ op: '>=', version: parseVersion(`${core}${version.parts.length >= 2 ? '' : ''}`) }, { op: '<', version: parseVersion(upper) }];
}

const ANY_RANGE_TOKENS = new Set(['', '*', 'x', 'latest', 'any', 'stable', '@latest']);

function comparatorsForToken(token: string, options: RangeOptions): Comparator[] {
  const text = token.trim();
  if (ANY_RANGE_TOKENS.has(text.toLowerCase())) return [];
  if (text.startsWith('^')) return caretComparators(text.slice(1));
  if (text.startsWith('~=')) return compatibleComparators(text.slice(2));
  if (text.startsWith('~>')) return compatibleComparators(text.slice(2)); // RubyGems pessimistic
  if (text.startsWith('~')) return (options.tildeCompatible ? compatibleComparators : tildeComparators)(text.slice(1));
  const operator = /^(>=|<=|!==|!=|===|==|=|>|<)\s*(.+)$/.exec(text);
  if (operator) {
    const op = operator[1]!;
    const value = operator[2]!;
    if (op === '===' || op === '==' || op === '=') {
      return /[*xX]/.test(value) ? wildcardComparators(value) : [{ op: '=', version: parseVersion(value) }];
    }
    if (op === '!=' || op === '!==') return [{ op: '!=', version: parseVersion(value) }];
    return [{ op: op as ComparisonOperator, version: parseVersion(value) }];
  }
  if (/[*xX]/.test(text)) return wildcardComparators(text);
  if (options.bareCaret) return caretComparators(text);
  // npm reads a two-segment bare version as a prefix match, a full one as exact.
  if (/^\d+(\.\d+)?$/.test(text)) return wildcardComparators(`${text}.x`);
  return [{ op: '=', version: parseVersion(text) }];
}

/**
 * Parses a range in the dialect every manager in this simulation shares:
 * `||` for alternatives, whitespace or `,` for conjunction, and `a - b` for an
 * inclusive hyphen range.
 */
export function parseRange(input: string, options: RangeOptions = {}): VersionRange {
  const text = String(input ?? '').trim();
  if (!text || ANY_RANGE_TOKENS.has(text.toLowerCase())) return [[]];
  return text.split('||').map((group) => {
    const clean = group.trim();
    const hyphen = /^(\S+)\s+-\s+(\S+)$/.exec(clean);
    if (hyphen) return [{ op: '>=' as const, version: parseVersion(hyphen[1]!) }, { op: '<=' as const, version: parseVersion(hyphen[2]!) }];
    // `>= 2.0.0 < 3.0.0` and `~> 1.0` glue their operator to the version first,
    // so the remaining whitespace is unambiguously a conjunction separator.
    const glued = clean.replace(/([<>=!~^]+)\s+/g, '$1');
    const tokens = glued.split(',').flatMap((part) => part.trim().split(/\s+/));
    return tokens.flatMap((token) => comparatorsForToken(token, options));
  });
}

function satisfiesComparator(version: ParsedVersion, comparator: Comparator): boolean {
  const order = compareVersions(version, comparator.version);
  switch (comparator.op) {
    case '>=': return order >= 0;
    case '>': return order > 0;
    case '<=': return order <= 0;
    case '<': return order < 0;
    case '!=': return order !== 0;
    default: return order === 0;
  }
}

export function satisfies(version: string, range: string | VersionRange, options: RangeOptions = {}): boolean {
  const parsed = typeof range === 'string' ? parseRange(range, options) : range;
  const value = parseVersion(version);
  return parsed.some((group) => group.every((comparator) => satisfiesComparator(value, comparator)));
}

/** Highest version satisfying every range at once, which is how the resolver intersects requirements. */
export function maxSatisfyingAll(versions: readonly string[], ranges: readonly string[], options: RangeOptions = {}): string | undefined {
  const parsed = ranges.map((range) => parseRange(range, options));
  const matching = versions.filter((version) => parsed.every((range) => satisfies(version, range)));
  return highestVersion(matching);
}

/** Renders a range the way the owning manager writes it into a manifest. */
export function defaultRangeFor(manager: string, version: string): string {
  if (['cargo'].includes(manager)) return version;
  if (['pip', 'uv', 'pipx', 'conda'].includes(manager)) return `>=${version}`;
  if (['gem'].includes(manager)) return `~> ${version}`;
  if (['apt', 'dpkg', 'snap', 'flatpak', 'brew', 'mas', 'winget', 'choco', 'scoop'].includes(manager)) return version;
  return `^${version}`;
}
