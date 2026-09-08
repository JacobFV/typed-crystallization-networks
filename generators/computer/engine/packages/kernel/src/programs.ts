import type { ComputerSpec, DirectoryEntry } from '@tcn-computer/protocol';
import type { InternetFabric } from './network.js';
import type { ProcessManager } from './processes.js';
import type { VirtualFileSystem } from './vfs.js';

/* ========================================================================== *
 * Seed executables
 *
 * A package manager installs a *real* program by writing a mode `0o755` file
 * into a VFS bin directory whose content is this JSON descriptor. The shell
 * finds it by walking `PATH`, parses it, and dispatches on `behavior` into the
 * registry below. A descriptor whose behavior has no implementation still
 * *runs* — it fails loudly with a non-zero status, which is honest, instead of
 * reporting `command not found` after a successful install.
 * ========================================================================== */

export interface SeedExecutableDescriptor {
  seedExecutable: 1;
  name: string;
  package: string;
  manager: string;
  version: string;
  provides: string[];
  behavior: string;
}

/** Parse an executable descriptor; `undefined` when the file is not one. */
export function parseSeedExecutable(content: string): SeedExecutableDescriptor | undefined {
  let value: unknown;
  try { value = JSON.parse(content); } catch { return undefined; }
  if (typeof value !== 'object' || value === null) return undefined;
  const record = value as Record<string, unknown>;
  if (record.seedExecutable !== 1 || typeof record.name !== 'string' || typeof record.behavior !== 'string') return undefined;
  return {
    ...(record as unknown as SeedExecutableDescriptor),
    seedExecutable: 1,
    name: record.name,
    package: typeof record.package === 'string' ? record.package : record.name,
    manager: typeof record.manager === 'string' ? record.manager : 'unknown',
    version: typeof record.version === 'string' ? record.version : '0.0.0',
    provides: Array.isArray(record.provides) ? record.provides.filter((item): item is string => typeof item === 'string') : [record.name],
    behavior: record.behavior,
  };
}

export interface ProgramResult { stdout: string; stderr: string; exitCode: number }

/** Everything a program is allowed to touch. Nothing here reaches the host. */
export interface ProgramContext {
  /** `argv[0]` is the name the user typed. */
  argv: string[];
  args: string[];
  stdin: string;
  cwd: string;
  env: Record<string, string>;
  descriptor: SeedExecutableDescriptor;
  executablePath: string;
  spec: ComputerSpec;
  /** Hostname a program should publish network services under. */
  serviceHost: string;
  /** PID the shell allocated for this program. */
  pid: number;
  vfs: VirtualFileSystem;
  processes: ProcessManager;
  network: InternetFabric;
  /** Resolve a user-supplied path against the shell's cwd, expanding `~`. */
  resolve(input: string): string;
  /** Simulated wall clock. Never the host clock. */
  now(): Date;
}

export type SeedProgram = (context: ProgramContext) => Promise<ProgramResult>;

/* -------------------------------------------------------------------------- *
 * Shared helpers (also consumed by the shell; this module never imports it)
 * -------------------------------------------------------------------------- */

export const ok = (stdout = '', exitCode = 0): ProgramResult => ({ stdout, stderr: '', exitCode });
export const fail = (stderr: string, exitCode = 1): ProgramResult => ({ stdout: '', stderr: stderr.endsWith('\n') ? stderr : `${stderr}\n`, exitCode });

/** Terminate a stream the way a POSIX utility does: exactly one trailing newline. */
export function terminated(value: string): string {
  if (!value) return '';
  return value.endsWith('\n') ? value : `${value}\n`;
}

/** Split a stream into lines, dropping the single trailing empty field. */
export function toLines(value: string): string[] {
  if (value === '') return [];
  const lines = value.split('\n');
  if (lines.at(-1) === '') lines.pop();
  return lines;
}

export interface FlagSpec {
  /** Flag names without dashes, e.g. `['i', 'ignore-case']`. */
  boolean?: readonly string[];
  /** Flags that consume a value, e.g. `['n', 'max-count']`. */
  value?: readonly string[];
  /** Treat the first operand as ending option parsing (`find`, `sudo`, `xargs`). */
  stopAtOperand?: boolean;
  /** Accept unknown flags as operands instead of erroring. */
  passthrough?: boolean;
}

export interface FlagResult {
  flags: Record<string, string | true>;
  operands: string[];
  error?: string;
}

/**
 * POSIX-ish option parser with clustered shorts (`-la`), attached values
 * (`-n5`), long options (`--color=never`) and `--`. Unknown options are an
 * error, so a flag can never be silently ignored.
 */
export function parseFlags(args: readonly string[], spec: FlagSpec = {}): FlagResult {
  const booleans = new Set(spec.boolean ?? []);
  const values = new Set(spec.value ?? []);
  const flags: Record<string, string | true> = {};
  const operands: string[] = [];
  let index = 0;
  for (; index < args.length; index += 1) {
    const arg = args[index]!;
    if (arg === '--') { index += 1; break; }
    if (arg === '-' || !arg.startsWith('-')) {
      operands.push(arg);
      if (spec.stopAtOperand) { index += 1; break; }
      continue;
    }
    if (arg.startsWith('--')) {
      const body = arg.slice(2);
      const equals = body.indexOf('=');
      const name = equals === -1 ? body : body.slice(0, equals);
      if (values.has(name)) {
        const value = equals === -1 ? args[++index] : body.slice(equals + 1);
        if (value === undefined) return { flags, operands, error: `option '--${name}' requires an argument` };
        flags[name] = value;
      } else if (booleans.has(name)) {
        flags[name] = true;
      } else if (spec.passthrough) {
        operands.push(arg);
      } else {
        return { flags, operands, error: `unrecognized option '--${name}'` };
      }
      continue;
    }
    const cluster = arg.slice(1);
    for (let position = 0; position < cluster.length; position += 1) {
      const name = cluster[position]!;
      if (values.has(name)) {
        const attached = cluster.slice(position + 1);
        const value = attached || args[++index];
        if (value === undefined) return { flags, operands, error: `option requires an argument -- '${name}'` };
        flags[name] = value;
        position = cluster.length;
      } else if (booleans.has(name)) {
        flags[name] = true;
      } else if (spec.passthrough) {
        operands.push(`-${cluster.slice(position)}`);
        position = cluster.length;
      } else {
        return { flags, operands, error: `invalid option -- '${name}'` };
      }
    }
  }
  operands.push(...args.slice(index));
  return { flags, operands };
}

/** True when any of the aliases was supplied. */
export function flagged(result: FlagResult, ...names: string[]): boolean {
  return names.some((name) => result.flags[name] !== undefined);
}

/** First supplied value among the aliases. */
export function flagValue(result: FlagResult, ...names: string[]): string | undefined {
  for (const name of names) {
    const value = result.flags[name];
    if (typeof value === 'string') return value;
  }
  return undefined;
}

/** Translate a shell glob into a regular expression. `**` crosses directories. */
export function globToRegExp(pattern: string, caseSensitive = true): RegExp {
  let source = '';
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index]!;
    if (char === '*') {
      if (pattern[index + 1] === '*') {
        index += 1;
        if (pattern[index + 1] === '/') { index += 1; source += '(?:[^/]*/)*'; } else source += '.*';
      } else source += '[^/]*';
    } else if (char === '?') source += '[^/]';
    else if (char === '[') {
      const end = pattern.indexOf(']', pattern[index + 1] === '!' || pattern[index + 1] === '^' ? index + 3 : index + 2);
      if (end === -1) source += '\\[';
      else {
        const body = pattern.slice(index + 1, end);
        source += `[${body.startsWith('!') ? `^${body.slice(1)}` : body}]`;
        index = end;
      }
    } else source += char.replace(/[.+^${}()|\\\/]/g, '\\$&');
  }
  return new RegExp(`^${source}$`, caseSensitive ? '' : 'i');
}

export const GLOB_CHARACTERS = /[*?[\]]/;

export interface WalkEntry { path: string; name: string; depth: number; kind: DirectoryEntry['inode']['kind']; inode: DirectoryEntry['inode'] }

/** Depth-first VFS walk that never follows symlinks and tolerates unreadable directories. */
export function walkTree(vfs: VirtualFileSystem, root: string, options: { maxDepth?: number; includeHidden?: boolean } = {}): WalkEntry[] {
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY;
  const results: WalkEntry[] = [];
  const visit = (directory: string, depth: number): void => {
    if (depth > maxDepth) return;
    let entries: DirectoryEntry[];
    try { entries = vfs.list(directory); } catch { return; }
    for (const entry of entries) {
      if (!options.includeHidden && entry.name.startsWith('.')) continue;
      results.push({ path: entry.path, name: entry.name, depth, kind: entry.inode.kind, inode: entry.inode });
      if (entry.inode.kind === 'directory') visit(entry.path, depth + 1);
    }
  };
  visit(root, 1);
  return results;
}

/* -------------------------------------------------------------------------- *
 * Program implementations
 * -------------------------------------------------------------------------- */

async function searchFiles(context: ProgramContext, pattern: RegExp, roots: string[], options: { hidden: boolean }): Promise<Array<{ path: string; line: number; text: string }>> {
  const hits: Array<{ path: string; line: number; text: string }> = [];
  for (const root of roots) {
    const stat = context.vfs.statSync(root);
    if (!stat) continue;
    const files = stat.kind === 'directory'
      ? walkTree(context.vfs, root, { includeHidden: options.hidden }).filter((entry) => entry.kind === 'file').map((entry) => entry.path)
      : [root];
    for (const file of files) {
      if (await context.vfs.isBinary(file).catch(() => true)) continue;
      const content = await context.vfs.readFile(file).catch(() => '');
      toLines(content).forEach((text, index) => { if (pattern.test(text)) hits.push({ path: file, line: index + 1, text }); });
    }
  }
  return hits;
}

const ripgrep: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, {
    boolean: ['i', 'ignore-case', 'n', 'line-number', 'N', 'no-line-number', 'l', 'files-with-matches', 'c', 'count', 'v', 'invert-match', 'w', 'word-regexp', 'F', 'fixed-strings', 'hidden', 'S', 'smart-case', 'no-heading', 'version', 'V'],
    value: ['e', 'regexp', 'g', 'glob', 't', 'type', 'm', 'max-count'],
  });
  if (parsed.error) return fail(`rg: ${parsed.error}`, 2);
  if (flagged(parsed, 'version', 'V')) return ok(`ripgrep ${context.descriptor.version}\n`);
  const explicit = flagValue(parsed, 'e', 'regexp');
  const pattern = explicit ?? parsed.operands.shift();
  if (pattern === undefined) return fail('rg: missing PATTERN', 2);
  const roots = (parsed.operands.length ? parsed.operands : ['.']).map((item) => context.resolve(item));
  const literal = flagged(parsed, 'F', 'fixed-strings');
  const body = literal ? pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : pattern;
  const wrapped = flagged(parsed, 'w', 'word-regexp') ? `\\b(?:${body})\\b` : body;
  let expression: RegExp;
  try { expression = new RegExp(wrapped, flagged(parsed, 'i', 'ignore-case') ? 'i' : ''); }
  catch (error) { return fail(`rg: invalid pattern: ${(error as Error).message}`, 2); }
  const invert = flagged(parsed, 'v', 'invert-match');
  const matcher = invert ? { test: (line: string) => !expression.test(line) } as RegExp : expression;
  const glob = flagValue(parsed, 'g', 'glob');
  let hits = await searchFiles(context, matcher, roots, { hidden: flagged(parsed, 'hidden') });
  if (glob) { const globExpression = globToRegExp(glob, context.vfs.caseSensitive); hits = hits.filter((hit) => globExpression.test(hit.path.split('/').at(-1)!)); }
  if (!hits.length) return { stdout: '', stderr: '', exitCode: 1 };
  const relative = (value: string) => (value.startsWith(`${context.cwd}/`) ? value.slice(context.cwd.length + 1) : value);
  if (flagged(parsed, 'l', 'files-with-matches')) return ok(terminated([...new Set(hits.map((hit) => relative(hit.path)))].join('\n')));
  if (flagged(parsed, 'c', 'count')) {
    const counts = new Map<string, number>();
    for (const hit of hits) counts.set(relative(hit.path), (counts.get(relative(hit.path)) ?? 0) + 1);
    return ok(terminated([...counts].map(([file, count]) => `${file}:${count}`).join('\n')));
  }
  const numbered = !flagged(parsed, 'N', 'no-line-number');
  const single = roots.length === 1 && context.vfs.statSync(roots[0]!)?.kind === 'file';
  return ok(terminated(hits.map((hit) => [single ? undefined : relative(hit.path), numbered ? String(hit.line) : undefined, hit.text].filter((part) => part !== undefined).join(':')).join('\n')));
};

const fdFind: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, {
    boolean: ['H', 'hidden', 'a', 'absolute-path', 'l', 'list-details', 'version', 'V'],
    value: ['e', 'extension', 't', 'type', 'd', 'max-depth', 'g', 'glob'],
  });
  if (parsed.error) return fail(`fd: ${parsed.error}`, 2);
  if (flagged(parsed, 'version', 'V')) return ok(`fd ${context.descriptor.version}\n`);
  const [rawPattern, rawRoot] = parsed.operands;
  const root = context.resolve(rawRoot ?? '.');
  const depth = flagValue(parsed, 'd', 'max-depth');
  const entries = walkTree(context.vfs, root, { includeHidden: flagged(parsed, 'H', 'hidden'), maxDepth: depth ? Number(depth) : undefined });
  const kind = flagValue(parsed, 't', 'type');
  const extension = flagValue(parsed, 'e', 'extension');
  const glob = flagValue(parsed, 'g', 'glob');
  let matcher: RegExp | undefined;
  if (glob) matcher = globToRegExp(glob, context.vfs.caseSensitive);
  else if (rawPattern) { try { matcher = new RegExp(rawPattern, context.vfs.caseSensitive ? '' : 'i'); } catch (error) { return fail(`fd: invalid pattern: ${(error as Error).message}`, 2); } }
  const wanted = kind === 'f' || kind === 'file' ? 'file' : kind === 'd' || kind === 'directory' ? 'directory' : kind === 'l' || kind === 'symlink' ? 'symlink' : undefined;
  if (kind && !wanted) return fail(`fd: unknown filter type '${kind}'`, 2);
  const matches = entries
    .filter((entry) => (wanted ? entry.kind === wanted : true))
    .filter((entry) => (extension ? entry.name.endsWith(`.${extension.replace(/^\./, '')}`) : true))
    .filter((entry) => (matcher ? matcher.test(entry.name) : true))
    .map((entry) => (flagged(parsed, 'a', 'absolute-path') ? entry.path : entry.path.startsWith(`${root}/`) ? entry.path.slice(root.length + 1) : entry.path));
  return matches.length ? ok(terminated(matches.join('\n'))) : { stdout: '', stderr: '', exitCode: 1 };
};

const bat: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, { boolean: ['p', 'plain', 'n', 'number', 'A', 'show-all', 'version', 'V'], value: ['l', 'language', 'style', 'theme', 'color'] });
  if (parsed.error) return fail(`bat: ${parsed.error}`, 2);
  if (flagged(parsed, 'version', 'V')) return ok(`bat ${context.descriptor.version}\n`);
  const plain = flagged(parsed, 'p', 'plain');
  const chunks: string[] = [];
  const targets = parsed.operands.length ? parsed.operands : ['-'];
  for (const operand of targets) {
    let content: string;
    if (operand === '-') content = context.stdin;
    else {
      try { content = await context.vfs.readFile(context.resolve(operand)); }
      catch { return fail(`bat: ${operand}: No such file or directory`); }
    }
    const lines = toLines(content);
    if (plain) { chunks.push(lines.join('\n')); continue; }
    const width = String(lines.length).length;
    chunks.push([
      `───────┬${'─'.repeat(60)}`,
      `       │ File: ${operand}`,
      `───────┼${'─'.repeat(60)}`,
      ...lines.map((line, index) => `${String(index + 1).padStart(Math.max(width, 5))}  │ ${line}`),
      `───────┴${'─'.repeat(60)}`,
    ].join('\n'));
  }
  return ok(terminated(chunks.join('\n')));
};

const tree: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, { boolean: ['a', 'all', 'd', 'directories', 'f', 'full-path', 'version'], value: ['L', 'level'] });
  if (parsed.error) return fail(`tree: ${parsed.error}`, 2);
  if (flagged(parsed, 'version')) return ok(`tree ${context.descriptor.version}\n`);
  const root = context.resolve(parsed.operands[0] ?? '.');
  if (context.vfs.statSync(root)?.kind !== 'directory') return fail(`tree: ${parsed.operands[0] ?? '.'}: No such directory`);
  const level = Number(flagValue(parsed, 'L', 'level') ?? Number.POSITIVE_INFINITY);
  const lines = [parsed.operands[0] ?? '.'];
  let directories = 0;
  let files = 0;
  const draw = (directory: string, prefix: string, depth: number): void => {
    if (depth > level) return;
    const entries = context.vfs.list(directory)
      .filter((entry) => (flagged(parsed, 'a', 'all') ? true : !entry.name.startsWith('.')))
      .filter((entry) => (flagged(parsed, 'd', 'directories') ? entry.inode.kind === 'directory' : true));
    entries.forEach((entry, index) => {
      const last = index === entries.length - 1;
      lines.push(`${prefix}${last ? '└── ' : '├── '}${flagged(parsed, 'f', 'full-path') ? entry.path : entry.name}`);
      if (entry.inode.kind === 'directory') { directories += 1; draw(entry.path, `${prefix}${last ? '    ' : '│   '}`, depth + 1); }
      else files += 1;
    });
  };
  draw(root, '', 1);
  lines.push('', `${directories} director${directories === 1 ? 'y' : 'ies'}, ${files} file${files === 1 ? '' : 's'}`);
  return ok(terminated(lines.join('\n')));
};

/* ------------------------------- jq --------------------------------------- */

type JsonValue = unknown;

/** Apply one `jq` path/function step to a value, returning the produced stream. */
function jqStep(values: JsonValue[], step: string): JsonValue[] {
  const trimmed = step.trim();
  if (trimmed === '' || trimmed === '.') return values;
  if (trimmed === 'length') return values.map((value) => (Array.isArray(value) ? value.length : typeof value === 'string' ? value.length : value && typeof value === 'object' ? Object.keys(value).length : 0));
  if (trimmed === 'keys' || trimmed === 'keys_unsorted') return values.map((value) => (value && typeof value === 'object' ? (Array.isArray(value) ? value.map((_, index) => index) : Object.keys(value as object).sort()) : null));
  if (trimmed === 'type') return values.map((value) => (value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value));
  if (trimmed === 'tostring') return values.map((value) => (typeof value === 'string' ? value : JSON.stringify(value)));
  if (trimmed === 'tonumber') return values.map((value) => Number(value));
  if (trimmed === 'add') return values.map((value) => (Array.isArray(value) ? value.reduce((sum: number, item) => sum + Number(item), 0) : null));
  if (trimmed === 'first') return values.map((value) => (Array.isArray(value) ? value[0] ?? null : null));
  if (trimmed === 'last') return values.map((value) => (Array.isArray(value) ? value.at(-1) ?? null : null));
  if (/^select\(/.test(trimmed)) {
    const body = trimmed.slice(7, trimmed.lastIndexOf(')'));
    const comparison = body.match(/^(.+?)\s*(==|!=)\s*(.+)$/);
    if (!comparison) throw new Error(`jq: unsupported select expression: ${body}`);
    const [, left, operator, right] = comparison;
    const expected: JsonValue = JSON.parse(right!.trim().replace(/^'(.*)'$/, '"$1"'));
    return values.filter((value) => {
      const actual = jqApply([value], left!.trim())[0];
      return operator === '==' ? JSON.stringify(actual) === JSON.stringify(expected) : JSON.stringify(actual) !== JSON.stringify(expected);
    });
  }
  // Path expression: .a.b[0][] and friends.
  if (!trimmed.startsWith('.')) throw new Error(`jq: unsupported filter: ${trimmed}`);
  let current = values;
  const tokens = trimmed.slice(1).match(/(?:[A-Za-z_][A-Za-z0-9_]*)|(?:\[[^\]]*\])/g) ?? [];
  for (const token of tokens) {
    if (token.startsWith('[')) {
      const inner = token.slice(1, -1).trim();
      if (inner === '') current = current.flatMap((value) => (Array.isArray(value) ? value : value && typeof value === 'object' ? Object.values(value as object) : []));
      else if (/^-?\d+$/.test(inner)) current = current.map((value) => (Array.isArray(value) ? value.at(Number(inner)) ?? null : null));
      else current = current.map((value) => (value && typeof value === 'object' ? (value as Record<string, JsonValue>)[JSON.parse(inner) as string] ?? null : null));
    } else current = current.map((value) => (value && typeof value === 'object' ? (value as Record<string, JsonValue>)[token] ?? null : null));
  }
  return current;
}

/** Split on `|` at depth zero, then fold the steps. */
function jqApply(values: JsonValue[], filter: string): JsonValue[] {
  const steps: string[] = [];
  let depth = 0;
  let buffer = '';
  let quote = '';
  for (const char of filter) {
    if (quote) { buffer += char; if (char === quote) quote = ''; continue; }
    if (char === '"' || char === "'") { quote = char; buffer += char; continue; }
    if (char === '(' || char === '[') depth += 1;
    if (char === ')' || char === ']') depth -= 1;
    if (char === '|' && depth === 0) { steps.push(buffer); buffer = ''; continue; }
    buffer += char;
  }
  steps.push(buffer);
  return steps.reduce((current, step) => jqStep(current, step), values);
}

const jq: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, { boolean: ['r', 'raw-output', 'c', 'compact-output', 'e', 'exit-status', 'n', 'null-input', 's', 'slurp', 'version'] });
  if (parsed.error) return fail(`jq: ${parsed.error}`, 2);
  if (flagged(parsed, 'version')) return ok(`jq-${context.descriptor.version}\n`);
  const filter = parsed.operands.shift() ?? '.';
  let documents: string[] = [];
  if (parsed.operands.length) {
    for (const operand of parsed.operands) {
      try { documents.push(await context.vfs.readFile(context.resolve(operand))); }
      catch { return fail(`jq: error: Could not open ${operand}`, 2); }
    }
  } else documents = [context.stdin];
  const inputs: JsonValue[] = [];
  for (const document of documents) {
    if (!document.trim()) continue;
    try { inputs.push(JSON.parse(document)); }
    catch (error) { return fail(`jq: error (at <stdin>:0): ${(error as Error).message}`, 4); }
  }
  let results: JsonValue[];
  try { results = jqApply(flagged(parsed, 's', 'slurp') ? [inputs] : inputs, filter); }
  catch (error) { return fail((error as Error).message, 3); }
  const raw = flagged(parsed, 'r', 'raw-output');
  const compact = flagged(parsed, 'c', 'compact-output');
  const rendered = results.map((value) => (raw && typeof value === 'string' ? value : JSON.stringify(value, null, compact ? 0 : 2) ?? 'null'));
  const empty = results.length === 0 || results.every((value) => value === null || value === false);
  return { stdout: terminated(rendered.join('\n')), stderr: '', exitCode: flagged(parsed, 'e', 'exit-status') && empty ? 1 : 0 };
};

/* ------------------------------ gh ---------------------------------------- */

const gh: SeedProgram = async (context) => {
  const [subcommand, ...rest] = context.args;
  const host = context.env.GH_HOST ?? 'git.seed.local';
  if (!subcommand || subcommand === '--help' || subcommand === 'help') {
    return ok(['Work seamlessly with the Seed git service.', '', 'USAGE', '  gh <command> <subcommand> [flags]', '', 'CORE COMMANDS', '  api:      make an authenticated request', '  auth:     inspect authentication state', '  repo:     view repositories', '  version:  print the version', ''].join('\n'));
  }
  if (subcommand === '--version' || subcommand === 'version') return ok(`gh version ${context.descriptor.version} (seed)\nhttps://${host}\n`);
  if (subcommand === 'auth') {
    if (rest[0] !== 'status') return fail(`gh: unknown auth subcommand: ${rest[0] ?? ''}`, 2);
    return ok([`${host}`, `  ✓ Logged in to ${host} as ${context.env.USER ?? 'agent'} (seed keychain)`, '  ✓ Git operations for ' + host + ' configured to use https protocol.', ''].join('\n'));
  }
  if (subcommand === 'api') {
    const route = rest.find((item) => !item.startsWith('-'));
    if (!route) return fail('gh: the required argument <endpoint> was not provided', 2);
    try {
      const response = await context.network.request(context.spec.id, `https://${host}/api/${route.replace(/^\//, '')}`);
      return { stdout: terminated(response.body), stderr: '', exitCode: response.status >= 400 ? 1 : 0 };
    } catch (error) { return fail(`gh: ${(error as Error).message}`); }
  }
  if (subcommand === 'repo') {
    const action = rest[0];
    if (action !== 'view' && action !== 'list') return fail(`gh: unknown repo subcommand: ${action ?? ''}`, 2);
    const repository = rest.find((item, index) => index > 0 && !item.startsWith('-')) ?? 'seed/example.git';
    try {
      const response = await context.network.request(context.spec.id, `https://${host}/api/repos/${repository.replace(/^\//, '')}`);
      if (response.status >= 400) return fail(`gh: repository ${repository} not found`);
      const state = JSON.parse(response.body) as { branches?: Record<string, string>; commits?: Array<{ hash: string; message: string }> };
      return ok(terminated([
        `${repository}`,
        `branches: ${Object.keys(state.branches ?? {}).join(', ') || '(none)'}`,
        `commits:  ${state.commits?.length ?? 0}`,
        ...(state.commits ?? []).slice(0, 10).map((commit) => `  ${commit.hash.slice(0, 7)} ${commit.message}`),
      ].join('\n')));
    } catch (error) { return fail(`gh: ${(error as Error).message}`); }
  }
  return fail(`gh: unknown command "${subcommand}" for "gh"`, 2);
};

/* ----------------------------- nginx -------------------------------------- */

const CONTENT_TYPES: Record<string, string> = {
  html: 'text/html; charset=utf-8', htm: 'text/html; charset=utf-8', css: 'text/css', js: 'text/javascript',
  json: 'application/json', txt: 'text/plain; charset=utf-8', md: 'text/markdown', svg: 'image/svg+xml',
};

const nginx: SeedProgram = async (context) => {
  const parsed = parseFlags(context.args, { boolean: ['v', 'V', 't', 'T', 'h'], value: ['s', 'p', 'c', 'g'] });
  if (parsed.error) return fail(`nginx: ${parsed.error}`, 2);
  if (flagged(parsed, 'v') || flagged(parsed, 'V')) return ok(`nginx version: nginx/${context.descriptor.version} (seed)\n`);
  const signal = flagValue(parsed, 's');
  // A daemon descriptor carries the port and document root its package shipped.
  const service = (context.descriptor as { service?: { port?: number; documentRoot?: string } }).service;
  const port = Number(context.env.NGINX_PORT ?? service?.port ?? 80);
  const serviceId = `nginx-${context.spec.id}-${port}`;
  if (signal) {
    if (!['stop', 'quit', 'reload', 'reopen'].includes(signal)) return fail(`nginx: invalid option: "-s ${signal}"`, 1);
    const running = context.processes.list().find((record) => record.executable === 'nginx' && record.pid !== context.pid);
    if (!running) return fail('nginx: [error] open() "/run/nginx.pid" failed (2: No such file or directory)');
    if (signal === 'stop' || signal === 'quit') {
      context.network.unregisterServicesForProcess(context.spec.id, running.pid);
      context.processes.kill(running.pid, signal === 'stop' ? 'SIGTERM' : 'SIGQUIT');
      return ok('');
    }
    return ok('');
  }
  const prefix = context.resolve(flagValue(parsed, 'p') ?? service?.documentRoot ?? '/usr/share/nginx/html');
  if (flagged(parsed, 't')) {
    return context.vfs.statSync(prefix)
      ? ok('nginx: configuration file /etc/nginx/nginx.conf test is successful\n')
      : fail(`nginx: [emerg] root directory "${prefix}" does not exist\nnginx: configuration file /etc/nginx/nginx.conf test failed`);
  }
  if (!context.vfs.statSync(prefix)) {
    // Nothing shipped a document root, so stand up the default landing page a
    // fresh install serves rather than answering every request with a 404.
    await context.vfs.mkdir(prefix);
    await context.vfs.writeFile(`${prefix}/index.html`, '<!doctype html><html><head><title>Welcome to nginx!</title></head><body><h1>Welcome to nginx!</h1><p>If you see this page, the nginx web server is successfully installed and working.</p></body></html>');
  }
  // The shell forks a process named after the command, so ignore our own entry.
  if (context.processes.list().some((record) => record.executable === 'nginx' && record.pid !== context.pid)) {
    return fail(`nginx: [emerg] bind() to 0.0.0.0:${port} failed (98: Address already in use)`);
  }
  const master = context.processes.spawn({ executable: 'nginx', argv: ['master process', 'nginx'], cwd: prefix, env: context.env, ppid: 1, listeningPorts: [port], memoryBytes: 12 * 1024 * 1024 });
  for (let worker = 0; worker < Math.min(2, context.spec.cpuCores); worker += 1) {
    context.processes.spawn({ executable: 'nginx', argv: ['worker process'], cwd: prefix, env: context.env, ppid: master.pid, memoryBytes: 8 * 1024 * 1024 });
  }
  context.network.registerService({
    id: serviceId, computerId: context.spec.id, host: context.serviceHost, port, protocol: 'http', pid: master.pid,
    handle: async (requestPath) => {
      const relative = requestPath.split('?')[0]!.replace(/^\//, '') || 'index.html';
      const target = `${prefix}/${relative}`;
      const resolved = context.vfs.statSync(target)?.kind === 'directory' ? `${target}/index.html` : target;
      try {
        const body = await context.vfs.readFile(resolved);
        return { status: 200, headers: { 'content-type': CONTENT_TYPES[resolved.split('.').at(-1) ?? ''] ?? 'application/octet-stream', server: `nginx/${context.descriptor.version}` }, body };
      } catch {
        return { status: 404, headers: { 'content-type': 'text/html; charset=utf-8', server: `nginx/${context.descriptor.version}` }, body: '<html><head><title>404 Not Found</title></head><body><center><h1>404 Not Found</h1></center><hr><center>nginx</center></body></html>' };
      }
    },
  });
  return ok('');
};

/* -------------------------------------------------------------------------- *
 * Registry
 * -------------------------------------------------------------------------- */

/** behavior string → implementation. Package managers write the behavior name. */
export const programRegistry: Record<string, SeedProgram> = {
  ripgrep, rg: ripgrep,
  fd: fdFind, 'fd-find': fdFind,
  bat,
  tree,
  jq,
  gh,
  nginx,
};

export function findProgram(behavior: string): SeedProgram | undefined {
  return programRegistry[behavior] ?? programRegistry[behavior.toLowerCase()];
}

export function listPrograms(): string[] { return Object.keys(programRegistry).sort(); }

/** Honest failure for an installed executable whose behavior nobody implements. */
export function unimplementedProgram(descriptor: SeedExecutableDescriptor, name: string): ProgramResult {
  return fail(
    `${name}: installed from ${descriptor.manager} package '${descriptor.package}' (${descriptor.version}), but behavior '${descriptor.behavior}' is not implemented in this simulation`,
    127,
  );
}
