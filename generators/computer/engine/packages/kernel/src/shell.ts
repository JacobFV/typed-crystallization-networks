import { createHash } from 'node:crypto';
import type { AppManifest, ComputerSpec, InstalledApp, OSRuntimeProfile } from '@tcn-computer/protocol';
import type { InternetFabric } from './network.js';
import type { ProcessManager, SeedSignal } from './processes.js';
import type { VirtualFileSystem } from './vfs.js';
import type { SoftwareEnvironment } from './software.js';
import {
  findProgram, flagValue, flagged, globToRegExp, GLOB_CHARACTERS, parseFlags, parseSeedExecutable, terminated, toLines,
  unimplementedProgram, walkTree, type FlagResult, type ProgramResult, type SeedExecutableDescriptor,
} from './programs.js';
import {
  lex, parse, ShellSyntaxError,
  type CommandNode, type Redirection, type Statement, type Word, type WordPart,
} from './shell-grammar.js';

export interface ShellResult { stdout: string; stderr: string; exitCode: number; cwd: string }

interface ShellDependencies {
  spec: ComputerSpec;
  profile: OSRuntimeProfile;
  vfs: VirtualFileSystem;
  processes: ProcessManager;
  network: InternetFabric;
  software: SoftwareEnvironment;
  listApps(): InstalledApp[];
  catalog(): AppManifest[];
  install(appId: string): Promise<InstalledApp>;
  onAction(action: string, data?: Record<string, unknown>): void;
}

/** What a command produces. Streams are POSIX-terminated (trailing newline). */
export interface CommandOutcome { stdout: string; stderr: string; exitCode: number }

export interface CommandContext {
  /** The name as typed, so error messages echo the user's spelling. */
  name: string;
  argv: string[];
  args: string[];
  stdin: string;
  /** PID of the process the shell forked for this command. */
  pid: number;
}

export interface CommandSpec {
  summary: string;
  usage?: string;
  run(context: CommandContext): Promise<CommandOutcome> | CommandOutcome;
  /** Shell builtins do not fork a process. */
  builtin?: boolean;
}

interface JobRecord { id: number; pid: number; pgid: number; command: string; status: 'running' | 'done'; exitCode: number }

const KERNELS: Record<ComputerSpec['os'], { sysname: string; release: string; version: string; machine: string; operating: string }> = {
  macos: { sysname: 'Darwin', release: '25.0.0', version: 'Darwin Kernel Version 25.0.0: Seed deterministic build; root:xnu-seed', machine: 'arm64', operating: 'Darwin' },
  ubuntu: { sysname: 'Linux', release: '6.11.0-seed', version: '#1 SMP Seed deterministic build', machine: 'x86_64', operating: 'GNU/Linux' },
  windows: { sysname: 'Windows_NT', release: '11.0.26100', version: 'Seed deterministic build', machine: 'AMD64', operating: 'Msys' },
};

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const SIGNAL_ALIASES: Record<string, SeedSignal> = {
  '1': 'SIGHUP', '2': 'SIGINT', '3': 'SIGQUIT', '9': 'SIGKILL', '15': 'SIGTERM', '18': 'SIGCONT', '19': 'SIGSTOP',
  HUP: 'SIGHUP', INT: 'SIGINT', QUIT: 'SIGQUIT', KILL: 'SIGKILL', TERM: 'SIGTERM', CONT: 'SIGCONT', STOP: 'SIGSTOP',
  USR1: 'SIGUSR1', USR2: 'SIGUSR2',
};

/** Mode bits rendered the way `ls -l` does. */
function modeString(kind: 'file' | 'directory' | 'symlink', mode: number): string {
  const type = kind === 'directory' ? 'd' : kind === 'symlink' ? 'l' : '-';
  const bits = [6, 3, 0].map((shift) => {
    const value = (mode >> shift) & 7;
    return `${value & 4 ? 'r' : '-'}${value & 2 ? 'w' : '-'}${value & 1 ? 'x' : '-'}`;
  });
  return type + bits.join('');
}

function humanSize(bytes: number): string {
  const units = ['B', 'K', 'M', 'G', 'T'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return unit === 0 ? `${value}${units[0]}` : `${value >= 10 ? Math.round(value) : value.toFixed(1)}${units[unit]}`;
}

/** Longest-common-subsequence table used by `diff`. */
function lcsMatrix(left: string[], right: string[]): number[][] {
  const table: number[][] = Array.from({ length: left.length + 1 }, () => new Array<number>(right.length + 1).fill(0));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      table[i]![j] = left[i] === right[j] ? table[i + 1]![j + 1]! + 1 : Math.max(table[i + 1]![j]!, table[i]![j + 1]!);
    }
  }
  return table;
}

/**
 * Interactive shell for one simulated computer.
 *
 * The dispatch table is chosen by `profile.shell.promptDialect`: a
 * case-sensitive POSIX table for zsh/bash and a case-insensitive PowerShell
 * table with real cmdlet names. A cmdlet never resolves on a POSIX shell and
 * `LS` never resolves on a case-sensitive one, because the tables are separate
 * and the POSIX lookup is exact.
 */
export class ShellSession {
  cwd: string;
  readonly history: string[] = [];
  private readonly dialect: 'posix' | 'powershell';
  private readonly commands = new Map<string, CommandSpec>();
  private readonly aliases = new Map<string, string>();
  private readonly vars = new Map<string, string>();
  private readonly exported = new Set<string>();
  private readonly jobs: JobRecord[] = [];
  private readonly pathListSeparator: string;
  private readonly executableSuffix: string;
  private readonly caseSensitive: boolean;
  private status = 0;
  private lastBackgroundPid = 0;
  private jobCounter = 0;
  private shellPid = 0;
  /** Simulated seconds since this shell's deterministic boot instant. */
  private ticks = 0;
  private readonly bootMillis: number;
  /** Assignments that apply only to the command currently being dispatched. */
  private overlay: Record<string, string> = {};

  constructor(private readonly deps: ShellDependencies) {
    const { profile, spec } = deps;
    this.dialect = profile.shell.promptDialect;
    this.caseSensitive = profile.filesystem.caseSensitive;
    this.executableSuffix = profile.conventions.executableSuffix;
    this.pathListSeparator = profile.filesystem.pathSeparator === '\\' ? ';' : ':';
    this.cwd = profile.filesystem.home;
    this.bootMillis = Date.UTC(2026, 0, 1, 9, 0, 0) + (Number.parseInt(createHash('sha256').update(spec.id).digest('hex').slice(0, 6), 16) % 86_400) * 1000;
    const windows = profile.filesystem.pathSeparator === '\\';
    const home = profile.filesystem.home;
    // Every directory a package manager installs executables into, so a freshly
    // installed binary is on PATH without the user re-exporting anything.
    const search = windows
      ? ['C:\\Windows\\System32', 'C:\\Windows', 'C:\\Program Files', `${home}\\AppData\\Local\\Microsoft\\WindowsApps`,
        `${home}\\AppData\\Local\\Microsoft\\WinGet\\Links`, 'C:\\ProgramData\\chocolatey\\bin', `${home}\\scoop\\shims`,
        `${home}\\AppData\\Roaming\\npm`, `${home}\\AppData\\Roaming\\Python\\Scripts`, `${home}\\.cargo\\bin`, `${home}\\go\\bin`, `${home}\\.dotnet\\tools`]
      : ['/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin', '/opt/homebrew/bin', '/home/linuxbrew/.linuxbrew/bin',
        '/snap/bin', '/var/lib/flatpak/exports/bin', `${home}/.local/bin`, `${home}/.local/share/pnpm`, `${home}/.bun/bin`,
        `${home}/.yarn/bin`, `${home}/.cargo/bin`, `${home}/go/bin`, `${home}/.gem/ruby/3.4.0/bin`, `${home}/.composer/vendor/bin`,
        `${home}/.dotnet/tools`, `${home}/miniconda3/bin`];
    for (const [key, value] of Object.entries({
      HOME: home, USER: 'agent', USERNAME: 'agent', LOGNAME: 'agent', HOSTNAME: spec.hostname,
      SHELL: profile.shell.executable, PWD: this.cwd, OLDPWD: this.cwd, TERM: 'xterm-256color', LANG: 'en_US.UTF-8',
      [profile.conventions.environmentPathKey]: search.join(this.pathListSeparator),
      TMPDIR: profile.filesystem.temporary, SEED_COMPUTER: spec.id, SEED_OS: spec.os,
    })) { this.vars.set(key, value); this.exported.add(key); }
    this.registerCommands();
  }

  /* ------------------------------------------------------------------ shell */

  prompt(): string {
    if (this.dialect === 'powershell') return `PS ${this.display(this.cwd)}> `;
    const user = this.variable('USER') ?? 'agent';
    return `${user}@${this.deps.spec.hostname}:${this.display(this.cwd)}${user === 'root' ? '#' : '$'} `;
  }

  /** Path as the platform shows it: `C:\Users\agent` on Windows, `~` elsewhere. */
  private display(value: string): string {
    if (this.deps.profile.filesystem.pathSeparator === '\\') return this.nativePath(value);
    const home = this.variable('HOME') ?? '/';
    return value === home ? '~' : value.startsWith(`${home}/`) ? `~${value.slice(home.length)}` : value;
  }

  /** Platform spelling with no `~` abbreviation. `pwd` always reports absolutely. */
  private nativePath(value: string): string {
    if (this.deps.profile.filesystem.pathSeparator === '\\') return value.replace(/^\/([A-Za-z])(?=\/|$)/, '$1:').replaceAll('/', '\\');
    return value;
  }

  private resolvePath(value: string): string {
    const home = this.variable('HOME') ?? '/';
    const expanded = value === '~' ? home : value.startsWith('~/') || value.startsWith('~\\') ? `${home}/${value.slice(2)}` : value;
    return this.deps.vfs.resolve(expanded, this.cwd);
  }

  /** Deterministic, topology-derived clock. The host wall clock never leaks in. */
  private now(): Date { return new Date(this.bootMillis + this.ticks * 1000); }

  private variable(name: string): string | undefined {
    if (name in this.overlay) return this.overlay[name];
    if (this.vars.has(name)) return this.vars.get(name);
    if (this.dialect === 'powershell') {
      const folded = name.toLowerCase().replace(/^env:/, '');
      for (const [key, value] of this.vars) if (key.toLowerCase() === folded) return value;
    }
    return undefined;
  }

  private setVariable(name: string, value: string, exportIt = false): void {
    const target = this.dialect === 'powershell' ? name.replace(/^env:/i, '') : name;
    this.vars.set(target, value);
    if (exportIt) this.exported.add(target);
  }

  /** Environment handed to forked processes: exported variables only. */
  private environment(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const key of this.exported) { const value = this.vars.get(key); if (value !== undefined) env[key] = value; }
    return { ...env, ...this.overlay };
  }

  private ensureShellProcess(): number {
    if (this.shellPid && this.deps.processes.get(this.shellPid)) return this.shellPid;
    const image = this.deps.profile.shell.executable.split(/[\\/]/).at(-1) ?? 'sh';
    const record = this.deps.processes.spawn({ executable: image, argv: ['-i'], cwd: this.cwd, env: this.environment(), ppid: 1, memoryBytes: 3 * 1024 * 1024 });
    this.deps.processes.setsid(record.pid);
    this.shellPid = record.pid;
    return record.pid;
  }

  /* --------------------------------------------------------------- entrypoint */

  async execute(line: string): Promise<ShellResult> {
    const trimmed = line.trim();
    if (!trimmed) return { stdout: '', stderr: '', exitCode: 0, cwd: this.cwd };
    this.history.push(trimmed);
    this.deps.onAction('shell.execute', { command: trimmed, cwd: this.cwd });
    this.ensureShellProcess();
    this.ticks += 1;
    let statements: Statement[];
    try {
      statements = parse(this.tokenize(line));
    } catch (error) {
      const message = error instanceof ShellSyntaxError ? error.message : String(error);
      this.status = 2;
      return { stdout: '', stderr: `${this.shellName()}: ${message}`, exitCode: 2, cwd: this.cwd };
    }
    const outcome = await this.runStatements(statements, '');
    this.status = outcome.exitCode;
    return {
      stdout: outcome.stdout.replace(/\n$/, ''),
      stderr: outcome.stderr.replace(/\n$/, ''),
      exitCode: outcome.exitCode,
      cwd: this.cwd,
    };
  }

  private shellName(): string { return this.deps.profile.shell.executable.split(/[\\/]/).at(-1) ?? 'sh'; }

  /** Lex in this shell's dialect: POSIX escapes with `\`, PowerShell with a backtick. */
  private tokenize(source: string): ReturnType<typeof lex> { return lex(source, { dialect: this.dialect }); }

  private async runStatements(statements: Statement[], stdin: string): Promise<CommandOutcome> {
    let stdout = '';
    let stderr = '';
    let exitCode = 0;
    for (const statement of statements) {
      if (statement.background) {
        const outcome = await this.runBackground(statement, stdin);
        stdout += outcome.stdout;
        stderr += outcome.stderr;
        exitCode = 0;
        continue;
      }
      for (const entry of statement.entries) {
        if (entry.operator === '&&' && exitCode !== 0) continue;
        if (entry.operator === '||' && exitCode === 0) continue;
        const outcome = await this.runPipeline(entry.pipeline, stdin);
        stdout += outcome.stdout;
        stderr += outcome.stderr;
        exitCode = outcome.exitCode;
        this.status = exitCode;
      }
    }
    return { stdout, stderr, exitCode };
  }

  private async runBackground(statement: Statement, stdin: string): Promise<CommandOutcome> {
    const description = statement.entries.map((entry) => entry.pipeline.commands.map((command) => this.describe(command)).join(' | ')).join(' ');
    const record = this.deps.processes.spawn({ executable: this.commandName(statement) ?? 'job', argv: [], cwd: this.cwd, env: this.environment(), ppid: this.ensureShellProcess(), memoryBytes: 2 * 1024 * 1024 });
    this.lastBackgroundPid = record.pid;
    const job: JobRecord = { id: ++this.jobCounter, pid: record.pid, pgid: record.pgid, command: description, status: 'running', exitCode: 0 };
    this.jobs.push(job);
    if (this.jobs.length > 50) this.jobs.shift();
    const outcome = await this.runStatements([{ ...statement, background: false }], stdin);
    job.status = 'done';
    job.exitCode = outcome.exitCode;
    this.deps.processes.exit(record.pid, outcome.exitCode);
    return { stdout: outcome.stdout, stderr: `${outcome.stderr}[${job.id}] ${record.pid}\n`, exitCode: 0 };
  }

  private commandName(statement: Statement): string | undefined {
    const first = statement.entries[0]?.pipeline.commands[0];
    if (first?.kind !== 'simple') return undefined;
    return first.words[0]?.parts.map((part) => (part.kind === 'literal' ? part.text : '')).join('') || undefined;
  }

  private describe(command: CommandNode): string {
    if (command.kind === 'subshell') return '( ... )';
    return command.words.map((word) => word.parts.map((part) => (part.kind === 'literal' ? part.text : `$${part.kind === 'variable' ? part.name : '(...)'}`)).join('')).join(' ');
  }

  private async runPipeline(pipeline: { commands: CommandNode[] }, stdin: string): Promise<CommandOutcome> {
    let piped = stdin;
    let stderr = '';
    let exitCode = 0;
    let pgid: number | undefined;
    for (let index = 0; index < pipeline.commands.length; index += 1) {
      const outcome = await this.runCommand(pipeline.commands[index]!, piped, { pgid, onPid: (pid, group) => { pgid ??= group ?? pid; } });
      stderr += outcome.stderr;
      exitCode = outcome.exitCode;
      piped = outcome.stdout;
    }
    return { stdout: piped, stderr, exitCode };
  }

  /* ------------------------------------------------------------- expansion */

  private async expandWord(word: Word, options: { split?: boolean; glob?: boolean } = {}): Promise<string[]> {
    const split = options.split !== false;
    const fields: Array<{ text: string; globbable: boolean }> = [{ text: '', globbable: false }];
    const push = (text: string, quote: 'none' | 'quoted', splittable: boolean): void => {
      if (splittable && quote === 'none' && /\s/.test(text)) {
        const chunks = text.split(/\s+/);
        chunks.forEach((chunk, index) => {
          if (index === 0) fields.at(-1)!.text += chunk;
          else if (chunk !== '' || index < chunks.length - 1) fields.push({ text: chunk, globbable: false });
        });
        return;
      }
      fields.at(-1)!.text += text;
    };
    for (const part of word.parts) {
      if (part.kind === 'literal') {
        fields.at(-1)!.text += part.text;
        if (part.quote === 'none' && GLOB_CHARACTERS.test(part.text)) fields.at(-1)!.globbable = true;
        continue;
      }
      const value = part.kind === 'variable'
        ? await this.variableValue(part)
        : part.kind === 'arithmetic'
          ? this.arithmetic(part.source)
          : (await this.substitute(part.source));
      push(value, part.quote === 'none' ? 'none' : 'quoted', split);
    }
    // Tilde expansion happens here, so argv carries the real path exactly as it
    // does in a real shell — commands never see a literal `~`.
    const first = word.parts[0];
    const home = this.variable('HOME') ?? '/';
    if (first?.kind === 'literal' && first.quote === 'none' && fields[0] && /^~($|[/\\])/.test(fields[0].text)) {
      fields[0].text = `${home}${fields[0].text.slice(1)}`;
    }
    const literal = word.parts.some((part) => part.kind === 'literal');
    const expanded: string[] = [];
    for (const field of fields) {
      if (field.text === '' && fields.length > 1) continue;
      if (options.glob !== false && field.globbable) {
        const matches = this.glob(field.text);
        if (matches.length) { expanded.push(...matches); continue; }
      }
      expanded.push(field.text);
    }
    // A word made only of expansions that produced nothing disappears, exactly
    // as `echo $UNSET` passes no argument at all.
    if (!literal && expanded.length === 1 && expanded[0] === '') return [];
    return expanded.filter((value, index) => value !== '' || index === 0 || fields.length === 1);
  }

  /**
   * POSIX `$(( … ))` integer arithmetic: the usual precedence ladder, unary
   * operators, comparisons, logical/bitwise operators, a ternary, and bare
   * variable names (an unset name is 0, as in a real shell). Evaluated with an
   * explicit parser rather than the host evaluator so a shell expression can
   * never reach the simulation process.
   */
  private arithmetic(source: string): string {
    const tokens = source.match(/\d+|[A-Za-z_][A-Za-z0-9_]*|\*\*|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%()<>!~^&|?:]/g) ?? [];
    let position = 0;
    const peek = () => tokens[position];
    const take = (value: string) => (tokens[position] === value ? (position++, true) : false);

    const primary = (): number => {
      const token = tokens[position];
      if (token === undefined) throw new Error(`bad arithmetic expression: ${source}`);
      if (take('(')) { const inner = ternary(); if (!take(')')) throw new Error(`missing ) in: ${source}`); return inner; }
      if (take('-')) return -primary();
      if (take('+')) return primary();
      if (take('!')) return primary() === 0 ? 1 : 0;
      if (take('~')) return ~primary();
      position++;
      if (/^\d+$/.test(token)) return Number(token);
      if (/^[A-Za-z_]/.test(token)) return Number(this.variable(token) ?? 0) || 0;
      throw new Error(`unexpected token '${token}' in: ${source}`);
    };
    const power = (): number => { const base = primary(); return take('**') ? base ** power() : base; };
    const binary = (next: () => number, operators: string[]) => (): number => {
      let value = next();
      for (;;) {
        const operator = peek();
        if (operator === undefined || !operators.includes(operator)) return value;
        position++;
        const right = next();
        if ((operator === '/' || operator === '%') && right === 0) throw new Error('division by 0');
        value = operator === '*' ? value * right : operator === '/' ? Math.trunc(value / right) : operator === '%' ? value % right
          : operator === '+' ? value + right : operator === '-' ? value - right
          : operator === '<<' ? value << right : operator === '>>' ? value >> right
          : operator === '<' ? Number(value < right) : operator === '<=' ? Number(value <= right)
          : operator === '>' ? Number(value > right) : operator === '>=' ? Number(value >= right)
          : operator === '==' ? Number(value === right) : operator === '!=' ? Number(value !== right)
          : operator === '&' ? value & right : operator === '^' ? value ^ right : operator === '|' ? value | right
          : operator === '&&' ? Number(Boolean(value) && Boolean(right)) : Number(Boolean(value) || Boolean(right));
      }
    };
    const ternary = (): number => {
      const condition = binary(binary(binary(binary(binary(binary(binary(binary(binary(power,
        ['*', '/', '%']), ['+', '-']), ['<<', '>>']), ['<', '<=', '>', '>=']), ['==', '!=']),
        ['&']), ['^']), ['|']), ['&&', '||'])();
      if (!take('?')) return condition;
      const whenTrue = ternary();
      if (!take(':')) throw new Error(`missing : in: ${source}`);
      const whenFalse = ternary();
      return condition !== 0 ? whenTrue : whenFalse;
    };

    const result = ternary();
    if (position !== tokens.length) throw new Error(`unexpected '${tokens[position]}' in: ${source}`);
    if (!Number.isFinite(result)) throw new Error(`bad arithmetic result in: ${source}`);
    return String(Math.trunc(result));
  }

  private async expandWords(words: Word[]): Promise<string[]> {
    const fields: string[] = [];
    for (const word of words) fields.push(...await this.expandWord(word));
    return fields;
  }

  private async expandSingle(word: Word): Promise<string> {
    const fields = await this.expandWord(word, { split: false });
    return fields[0] ?? '';
  }

  private async variableValue(part: Extract<WordPart, { kind: 'variable' }>): Promise<string> {
    let value: string | undefined;
    if (part.name === '?') value = String(this.status);
    else if (part.name === '$') value = String(this.shellPid);
    else if (part.name === '!') value = String(this.lastBackgroundPid);
    else if (part.name === '#') value = '0';
    else if (part.name === '0') value = this.shellName();
    else if (/^\d$/.test(part.name)) value = '';
    else value = this.variable(part.name);
    if (part.length) return String((value ?? '').length);
    const missing = part.operator?.startsWith(':') ? value === undefined || value === '' : value === undefined;
    if (part.operator) {
      const fallback = part.word ?? '';
      const operator = part.operator.replace(':', '');
      if (operator === '-' && missing) return this.expandInline(fallback);
      if (operator === '=' && missing) { const resolved = await this.expandInline(fallback); this.setVariable(part.name, resolved); return resolved; }
      if (operator === '+' && !missing) return this.expandInline(fallback);
      if (operator === '?' && missing) throw new ShellSyntaxError(`${part.name}: ${fallback || 'parameter null or not set'}`);
    }
    return value ?? '';
  }

  /** Expand a `${VAR:-...}` replacement word, which may itself contain expansions. */
  private async expandInline(source: string): Promise<string> {
    if (!/[$`]/.test(source)) return source;
    const tokens = this.tokenize(source);
    const word = tokens.find((token) => token.kind === 'word');
    return word?.kind === 'word' ? (await this.expandWord(word.word, { split: false, glob: false }))[0] ?? '' : source;
  }

  /** `$(...)` / backticks: run the body and strip trailing newlines, as POSIX requires. */
  private async substitute(source: string): Promise<string> {
    const outcome = await this.runStatements(parse(this.tokenize(source)), '');
    return outcome.stdout.replace(/\n+$/, '');
  }

  /** Filename generation for `*`, `?`, `[...]` and `**`. */
  private glob(pattern: string): string[] {
    const absolute = pattern.startsWith('/') || /^[A-Za-z]:[\\/]/.test(pattern) || pattern.startsWith('~');
    const base = absolute ? '' : this.cwd;
    const full = this.resolvePath(pattern);
    const segments = full.split('/').filter(Boolean);
    let current = ['/'];
    for (const segment of segments) {
      if (segment === '**') {
        // `**` matches this directory and every descendant directory.
        const reachable = new Set(current);
        for (const directory of current) for (const entry of walkTree(this.deps.vfs, directory, { includeHidden: false })) {
          if (entry.kind === 'directory') reachable.add(entry.path);
        }
        current = [...reachable];
        continue;
      }
      if (!GLOB_CHARACTERS.test(segment)) {
        current = current.map((directory) => (directory === '/' ? `/${segment}` : `${directory}/${segment}`)).filter((candidate) => this.deps.vfs.statSync(candidate));
        continue;
      }
      const expression = globToRegExp(segment, this.caseSensitive);
      const next: string[] = [];
      for (const directory of current) {
        let entries;
        try { entries = this.deps.vfs.list(directory); } catch { continue; }
        for (const entry of entries) {
          if (entry.name.startsWith('.') && !segment.startsWith('.')) continue;
          if (expression.test(entry.name)) next.push(entry.path);
        }
      }
      current = next;
    }
    const results = current.filter((candidate) => candidate !== '/' || pattern === '/');
    if (!results.length) return [];
    return results
      .map((candidate) => (base && candidate.startsWith(`${base}/`) ? candidate.slice(base.length + 1) : candidate))
      .sort((left, right) => left.localeCompare(right));
  }

  /* --------------------------------------------------------------- commands */

  private async runCommand(node: CommandNode, stdin: string, options: { pgid?: number; onPid?: (pid: number, pgid?: number) => void } = {}): Promise<CommandOutcome> {
    let input = stdin;
    try {
      for (const redirection of node.redirections) {
        if (redirection.operator === '<') input = await this.deps.vfs.readFile(this.resolvePath(await this.expandSingle(redirection.target!)));
        if (redirection.operator === '<<') input = redirection.heredoc!.expand ? await this.expandHeredoc(redirection.heredoc!.text) : redirection.heredoc!.text;
      }
    } catch (error) {
      return { stdout: '', stderr: terminated(`${this.shellName()}: ${(error as Error).message}`), exitCode: 1 };
    }

    let outcome: CommandOutcome;
    if (node.kind === 'subshell') {
      const savedCwd = this.cwd;
      const savedVars = new Map(this.vars);
      const savedExports = new Set(this.exported);
      outcome = await this.runStatements(node.statements, input);
      this.cwd = savedCwd;
      this.vars.clear();
      for (const [key, value] of savedVars) this.vars.set(key, value);
      this.exported.clear();
      for (const key of savedExports) this.exported.add(key);
    } else {
      outcome = await this.runSimple(node, input, options);
    }
    return this.applyOutputRedirections(node.redirections, outcome);
  }

  private async expandHeredoc(text: string): Promise<string> {
    const lines: string[] = [];
    for (const line of text.split('\n')) {
      if (!/[$`]/.test(line)) { lines.push(line); continue; }
      const tokens = this.tokenize(line.replaceAll(' ', '\u0001'));
      const parts: string[] = [];
      for (const token of tokens) if (token.kind === 'word') parts.push((await this.expandWord(token.word, { split: false, glob: false }))[0] ?? '');
      lines.push(parts.join('').replaceAll('\u0001', ' '));
    }
    return lines.join('\n');
  }

  private async applyOutputRedirections(redirections: Redirection[], outcome: CommandOutcome): Promise<CommandOutcome> {
    let result = outcome;
    for (const redirection of redirections) {
      if (redirection.operator !== '>' && redirection.operator !== '>>') continue;
      const target = this.resolvePath(await this.expandSingle(redirection.target!));
      const payload = redirection.fd === 2 ? result.stderr : redirection.fd === 'both' ? result.stdout + result.stderr : result.stdout;
      try {
        await this.writeStream(target, payload, redirection.operator === '>>');
      } catch (error) {
        return { stdout: '', stderr: terminated(`${this.shellName()}: ${target}: ${(error as Error).message}`), exitCode: 1 };
      }
      result = redirection.fd === 2 ? { ...result, stderr: '' } : redirection.fd === 'both' ? { ...result, stdout: '', stderr: '' } : { ...result, stdout: '' };
    }
    return result;
  }

  /**
   * Redirection storage convention: the payload lands byte-for-byte without the
   * stream's terminating newline, and `>>` re-inserts one separator so appended
   * records stay on their own lines.
   */
  private async writeStream(target: string, payload: string, append: boolean): Promise<void> {
    const body = payload.replace(/\n$/, '');
    if (!append) { await this.deps.vfs.writeFile(target, body); return; }
    let existing = '';
    try { existing = await this.deps.vfs.readFile(target); } catch { existing = ''; }
    const separator = existing === '' || existing.endsWith('\n') ? '' : '\n';
    await this.deps.vfs.writeFile(target, `${existing}${separator}${body}`);
  }

  private async runSimple(node: Extract<CommandNode, { kind: 'simple' }>, stdin: string, options: { pgid?: number; onPid?: (pid: number, pgid?: number) => void }): Promise<CommandOutcome> {
    const assignments: Record<string, string> = {};
    for (const word of node.assignments) {
      const text = await this.expandSingle(word);
      const split = text.indexOf('=');
      assignments[text.slice(0, split)] = text.slice(split + 1);
    }
    let argv: string[];
    try {
      argv = await this.expandWords(node.words);
    } catch (error) {
      return { stdout: '', stderr: terminated(`${this.shellName()}: ${(error as Error).message}`), exitCode: 1 };
    }
    if (!argv.length) {
      for (const [key, value] of Object.entries(assignments)) this.setVariable(key, value);
      return { stdout: '', stderr: '', exitCode: 0 };
    }
    const previousOverlay = this.overlay;
    this.overlay = { ...previousOverlay, ...assignments };
    try {
      return await this.invoke(argv, stdin, options);
    } finally {
      this.overlay = previousOverlay;
    }
  }

  /**
   * Resolve and run one command: alias, dialect table, package manager, then a
   * real `PATH` walk for installed executables. Anything unresolved is 127.
   */
  private async invoke(argv: string[], stdin: string, options: { pgid?: number; onPid?: (pid: number, pgid?: number) => void } = {}): Promise<CommandOutcome> {
    const [name = '', ...rawArgs] = argv;
    // PowerShell spells long parameters with one dash (`-Recurse`); normalize
    // them to the long form the option parser understands.
    const args = this.dialect === 'powershell'
      ? rawArgs.map((argument) => (/^-[A-Za-z][A-Za-z0-9-]+$/.test(argument) ? `-${argument}` : argument))
      : rawArgs;
    const alias = this.aliases.get(this.key(name));
    if (alias !== undefined && !this.aliasGuard.has(name)) {
      this.aliasGuard.add(name);
      try {
        const tokens = this.tokenize(alias);
        const words = tokens.filter((token) => token.kind === 'word').flatMap((token) => (token.kind === 'word' ? [token.word] : []));
        return await this.invoke([...await this.expandWords(words), ...args], stdin, options);
      } finally { this.aliasGuard.delete(name); }
    }
    const spec = this.lookup(name);
    if (spec?.builtin) return this.finish(await spec.run({ name, argv, args, stdin, pid: this.shellPid }));

    const pid = this.deps.processes.spawn({
      executable: name, argv: args, cwd: this.cwd, env: this.environment(),
      ppid: this.ensureShellProcess(), pgid: options.pgid, memoryBytes: 2 * 1024 * 1024,
    });
    options.onPid?.(pid.pid, pid.pgid);
    this.deps.processes.tick(4);
    let outcome: CommandOutcome;
    try {
      if (spec) outcome = await spec.run({ name, argv, args, stdin, pid: pid.pid });
      else {
        // `npx` and friends must reach the install-on-demand exec path, not the
        // bare PATH descriptor the nodejs package drops in.
        const external = this.deps.software.supportsExec(name) ? undefined : await this.resolveExecutable(name);
        if (external) outcome = await this.runExternal(external, argv, stdin, pid.pid);
        else if ((this.deps.software.supports(name) || this.deps.software.supportsExec(name)) && this.dialectAllowsManager(name)) {
          outcome = { stdout: terminated(await this.deps.software.packageCommand(name, args, this.cwd)), stderr: '', exitCode: 0 };
        } else outcome = { stdout: '', stderr: terminated(this.notFound(name)), exitCode: 127 };
      }
    } catch (error) {
      outcome = { stdout: '', stderr: terminated(`${name}: ${(error as Error).message}`), exitCode: 1 };
    }
    // The forked process exits with the status the command actually produced,
    // and the shell reaps it, so `ps` never accumulates phantom entries.
    this.deps.processes.exit(pid.pid, outcome.exitCode);
    this.deps.processes.wait(this.shellPid, { pid: pid.pid });
    return this.finish(outcome);
  }

  private readonly aliasGuard = new Set<string>();

  private finish(outcome: CommandOutcome): CommandOutcome {
    this.status = outcome.exitCode;
    return outcome;
  }

  private notFound(name: string): string {
    return this.dialect === 'powershell'
      ? `${name} : The term '${name}' is not recognized as a name of a cmdlet, function, script file, or executable program.`
      : `${this.shellName()}: command not found: ${name}`;
  }

  private key(name: string): string { return this.dialect === 'powershell' ? name.toLowerCase() : name; }

  private lookup(name: string): CommandSpec | undefined {
    const direct = this.commands.get(this.key(name));
    if (direct) return direct;
    if (this.executableSuffix && name.toLowerCase().endsWith(this.executableSuffix.toLowerCase())) {
      return this.commands.get(this.key(name.slice(0, -this.executableSuffix.length)));
    }
    return undefined;
  }

  /** Package managers are gated by dialect so `apt` cannot appear on PowerShell. */
  private dialectAllowsManager(name: string): boolean {
    const native = new Set<string>(this.deps.profile.packageManagers.native.map(String));
    const language = new Set<string>(this.deps.profile.packageManagers.language.map(String));
    const canonical = name.toLowerCase().replace(/^apt-get$/, 'apt').replace(/^pip3$/, 'pip').replace(/^chocolatey$/, 'choco');
    return native.has(canonical) || language.has(canonical);
  }

  /* ------------------------------------------------------- PATH resolution */

  private searchPath(): string[] {
    const raw = this.variable(this.deps.profile.conventions.environmentPathKey) ?? this.variable('PATH') ?? '';
    const directories = raw.split(this.pathListSeparator).filter(Boolean).map((entry) => this.resolvePath(entry));
    const local: string[] = [];
    let cursor = this.cwd;
    for (;;) {
      local.push(`${cursor === '/' ? '' : cursor}/node_modules/.bin`);
      if (cursor === '/') break;
      cursor = this.deps.vfs.resolve('..', cursor);
    }
    return [...local, ...directories];
  }

  /**
   * Find an executable: a mode `0o755` VFS file holding a seed executable
   * descriptor. Honors `conventions.executableSuffix` and `node_modules/.bin`.
   */
  private async resolveExecutable(name: string): Promise<{ path: string; descriptor: SeedExecutableDescriptor } | undefined> {
    const candidates: string[] = [];
    if (name.includes('/') || name.includes('\\')) candidates.push(this.resolvePath(name));
    else {
      for (const directory of this.searchPath()) {
        candidates.push(`${directory}/${name}`);
        if (this.executableSuffix && !name.endsWith(this.executableSuffix)) candidates.push(`${directory}/${name}${this.executableSuffix}`);
      }
    }
    for (const candidate of candidates) {
      const inode = this.deps.vfs.statSync(candidate);
      if (!inode || inode.kind !== 'file' || (inode.mode & 0o111) === 0) continue;
      let content: string;
      try { content = await this.deps.vfs.readFile(candidate); } catch { continue; }
      const descriptor = parseSeedExecutable(content);
      if (descriptor) return { path: candidate, descriptor };
    }
    return undefined;
  }

  private async runExternal(found: { path: string; descriptor: SeedExecutableDescriptor }, argv: string[], stdin: string, pid: number): Promise<ProgramResult> {
    const program = findProgram(found.descriptor.behavior);
    if (!program) return unimplementedProgram(found.descriptor, argv[0]!);
    return program({
      argv, args: argv.slice(1), stdin, cwd: this.cwd, env: this.environment(),
      descriptor: found.descriptor, executablePath: found.path, spec: this.deps.spec,
      serviceHost: `${this.deps.spec.hostname}.seed.local`, pid,
      vfs: this.deps.vfs, processes: this.deps.processes, network: this.deps.network,
      resolve: (input) => this.resolvePath(input),
      now: () => this.now(),
    });
  }

  /* ------------------------------------------------------ command utilities */

  private ok(stdout = ''): CommandOutcome { return { stdout: terminated(stdout), stderr: '', exitCode: 0 }; }
  private silent(): CommandOutcome { return { stdout: '', stderr: '', exitCode: 0 }; }
  private error(message: string, exitCode = 1): CommandOutcome { return { stdout: '', stderr: terminated(message), exitCode }; }
  private outcome(stdout: string, stderr: string, exitCode: number): CommandOutcome { return { stdout: terminated(stdout), stderr: terminated(stderr), exitCode }; }

  /** Reject an unparsable option instead of ignoring it. */
  private optionError(name: string, parsed: FlagResult): CommandOutcome | undefined {
    return parsed.error ? this.error(`${name}: ${parsed.error}`, 2) : undefined;
  }

  /** Read every operand (or stdin when there are none) as a named stream. */
  private async inputs(name: string, operands: string[], stdin: string): Promise<{ streams: Array<{ name: string; content: string }>; errors: string[] }> {
    if (!operands.length) return { streams: [{ name: '-', content: stdin }], errors: [] };
    const streams: Array<{ name: string; content: string }> = [];
    const errors: string[] = [];
    for (const operand of operands) {
      if (operand === '-') { streams.push({ name: '-', content: stdin }); continue; }
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) { errors.push(`${name}: ${operand}: No such file or directory`); continue; }
      if (inode.kind === 'directory') { errors.push(`${name}: ${operand}: Is a directory`); continue; }
      streams.push({ name: operand, content: await this.deps.vfs.readFile(target) });
    }
    return { streams, errors };
  }

  private owner(uid: number): string { return uid === 0 ? 'root' : this.variable('USER') ?? 'agent'; }
  private group(gid: number): string { return gid === 0 ? (this.deps.spec.os === 'macos' ? 'wheel' : 'root') : this.deps.spec.os === 'macos' ? 'staff' : this.variable('USER') ?? 'agent'; }

  private stamp(iso: string): string {
    const date = new Date(iso);
    return `${MONTHS[date.getUTCMonth()]} ${String(date.getUTCDate()).padStart(2)} ${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
  }

  private formatDate(date: Date, format?: string): string {
    const pad = (value: number, width = 2) => String(value).padStart(width, '0');
    if (!format) return `${DAYS[date.getUTCDay()]} ${MONTHS[date.getUTCMonth()]} ${String(date.getUTCDate()).padStart(2)} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())} UTC ${date.getUTCFullYear()}`;
    return format.replace(/%[A-Za-z%]/g, (token) => {
      switch (token) {
        case '%Y': return String(date.getUTCFullYear());
        case '%m': return pad(date.getUTCMonth() + 1);
        case '%d': return pad(date.getUTCDate());
        case '%H': return pad(date.getUTCHours());
        case '%M': return pad(date.getUTCMinutes());
        case '%S': return pad(date.getUTCSeconds());
        case '%F': return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
        case '%T': return `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
        case '%s': return String(Math.floor(date.getTime() / 1000));
        case '%a': return DAYS[date.getUTCDay()]!;
        case '%b': return MONTHS[date.getUTCMonth()]!;
        case '%Z': return 'UTC';
        case '%%': return '%';
        default: return token;
      }
    });
  }

  /* ------------------------------------------------------------- dispatch tables */

  private registerCommands(): void {
    const add = (names: string[], summary: string, run: CommandSpec['run'], options: { builtin?: boolean; usage?: string } = {}): void => {
      for (const name of names) this.commands.set(this.key(name), { summary, run, ...options });
    };
    if (this.dialect === 'powershell') this.registerPowerShell(add); else this.registerPosix(add);
  }

  /* ------------------------------------------------------------ shell builtins */

  private cmdCd = async (context: CommandContext): Promise<CommandOutcome> => {
    const target = context.args[0];
    const home = this.variable('HOME') ?? '/';
    const requested = target === undefined || target === '~' ? home : target === '-' ? this.variable('OLDPWD') ?? this.cwd : target;
    const next = this.resolvePath(requested);
    const inode = this.deps.vfs.statSync(next);
    if (!inode) return this.error(`cd: no such file or directory: ${target ?? requested}`);
    if (inode.kind !== 'directory') return this.error(`cd: not a directory: ${target}`);
    this.setVariable('OLDPWD', this.cwd, true);
    this.cwd = this.deps.vfs.realpath(next);
    this.setVariable('PWD', this.cwd, true);
    return target === '-' ? this.ok(this.display(this.cwd)) : this.silent();
  };

  private cmdPwd = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['L', 'P'] });
    return this.optionError(context.name, parsed) ?? this.ok(flagged(parsed, 'P') ? this.nativePath(this.deps.vfs.realpath(this.cwd)) : this.nativePath(this.cwd));
  };

  private cmdExport = (context: CommandContext): CommandOutcome => {
    if (!context.args.length) return this.ok([...this.exported].sort().map((key) => `export ${key}="${this.vars.get(key) ?? ''}"`).join('\n'));
    for (const argument of context.args) {
      const split = argument.indexOf('=');
      if (split === -1) { if (this.vars.has(argument)) this.exported.add(argument); else return this.error(`export: ${argument}: not set`); continue; }
      this.setVariable(argument.slice(0, split), argument.slice(split + 1), true);
    }
    return this.silent();
  };

  private cmdUnset = (context: CommandContext): CommandOutcome => {
    for (const name of context.args) { this.vars.delete(name); this.exported.delete(name); }
    return this.silent();
  };

  private cmdSet = (context: CommandContext): CommandOutcome => {
    if (!context.args.length) return this.ok([...this.vars].sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => `${key}=${value}`).join('\n'));
    return this.error(`set: options are not modelled in this simulation: ${context.args.join(' ')}`, 2);
  };

  private cmdEnv = (context: CommandContext): CommandOutcome => {
    const index = context.args.findIndex((argument) => !argument.includes('='));
    if (index >= 0) return this.error(`env: running a command through env is not implemented in this simulation`, 2);
    for (const argument of context.args) { const split = argument.indexOf('='); this.setVariable(argument.slice(0, split), argument.slice(split + 1), true); }
    return this.ok(Object.entries(this.environment()).sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => `${key}=${value}`).join('\n'));
  };

  private cmdAlias = (context: CommandContext): CommandOutcome => {
    if (!context.args.length) return this.ok([...this.aliases].sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => `${key}='${value}'`).join('\n'));
    for (const argument of context.args) {
      const split = argument.indexOf('=');
      if (split === -1) {
        const value = this.aliases.get(this.key(argument));
        if (value === undefined) return this.error(`alias: ${argument}: not found`);
        return this.ok(`${argument}='${value}'`);
      }
      this.aliases.set(this.key(argument.slice(0, split)), argument.slice(split + 1));
    }
    return this.silent();
  };

  private cmdUnalias = (context: CommandContext): CommandOutcome => {
    for (const name of context.args) if (!this.aliases.delete(this.key(name))) return this.error(`unalias: no such hash table element: ${name}`);
    return this.silent();
  };

  private cmdSource = async (context: CommandContext): Promise<CommandOutcome> => {
    const target = context.args[0];
    if (!target) return this.error(`${context.name}: filename argument required`, 2);
    let script: string;
    try { script = await this.deps.vfs.readFile(this.resolvePath(target)); }
    catch { return this.error(`${context.name}: no such file or directory: ${target}`); }
    return this.runStatements(parse(this.tokenize(script)), context.stdin);
  };

  private cmdExit = (context: CommandContext): CommandOutcome => {
    const code = Number(context.args[0] ?? this.status);
    this.deps.onAction('shell.exit', { exitCode: code });
    return { stdout: '', stderr: '', exitCode: Number.isFinite(code) ? code : 0 };
  };

  private cmdHistory = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['c'], value: ['n'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    if (flagged(parsed, 'c')) { this.history.length = 0; return this.silent(); }
    const limit = Number(parsed.operands[0] ?? flagValue(parsed, 'n') ?? this.history.length);
    const entries = this.history.slice(Math.max(0, this.history.length - limit));
    const offset = this.history.length - entries.length;
    return this.ok(entries.map((item, index) => `${String(offset + index + 1).padStart(4)}  ${item}`).join('\n'));
  };

  private cmdJobs = (): CommandOutcome => {
    if (!this.jobs.length) return this.silent();
    return this.ok(this.jobs.map((job) => `[${job.id}]  ${job.status === 'done' ? `Done${job.exitCode ? `(${job.exitCode})` : ''}` : 'Running'}                 ${job.command}`).join('\n'));
  };

  private cmdWait = (context: CommandContext): CommandOutcome => {
    const target = context.args[0];
    const pid = target ? Number(target.replace(/^%/, '')) : undefined;
    if (target && !Number.isFinite(pid)) return this.error(`wait: ${target}: no such job`, 127);
    const status = this.deps.processes.wait(this.shellPid, pid === undefined ? {} : { pid });
    if (!status) return target ? this.error(`wait: pid ${target} is not a child of this shell`, 127) : this.silent();
    const job = this.jobs.find((entry) => entry.pid === status.pid);
    if (job) job.status = 'done';
    return { stdout: '', stderr: '', exitCode: status.exitCode };
  };

  private cmdTrue = (): CommandOutcome => ({ stdout: '', stderr: '', exitCode: 0 });
  private cmdFalse = (): CommandOutcome => ({ stdout: '', stderr: '', exitCode: 1 });

  private cmdTest = (context: CommandContext): CommandOutcome => {
    const args = context.name === '[' ? context.args.slice(0, context.args.at(-1) === ']' ? -1 : undefined) : context.args;
    const truth = (value: boolean): CommandOutcome => ({ stdout: '', stderr: '', exitCode: value ? 0 : 1 });
    if (!args.length) return truth(false);
    if (args.length === 1) return truth(args[0] !== '');
    if (args.length === 2) {
      const [operator, operand] = args as [string, string];
      const inode = this.deps.vfs.statSync(this.resolvePath(operand));
      switch (operator) {
        case '-e': return truth(Boolean(inode));
        case '-f': return truth(inode?.kind === 'file');
        case '-d': return truth(inode?.kind === 'directory');
        case '-L': case '-h': return truth(this.deps.vfs.lstatSync(this.resolvePath(operand))?.kind === 'symlink');
        case '-s': return truth((inode?.size ?? 0) > 0);
        case '-r': return truth(Boolean(inode));
        case '-w': return truth(Boolean(inode) && ((inode!.mode >> 6) & 2) === 2);
        case '-x': return truth(Boolean(inode) && (inode!.mode & 0o111) !== 0);
        case '-z': return truth(operand === '');
        case '-n': return truth(operand !== '');
        case '!': return truth(operand === '');
        default: return this.error(`${context.name}: ${operator}: unary operator expected`, 2);
      }
    }
    const [left, operator, right] = args as [string, string, string];
    switch (operator) {
      case '=': case '==': return truth(left === right);
      case '!=': return truth(left !== right);
      case '-eq': return truth(Number(left) === Number(right));
      case '-ne': return truth(Number(left) !== Number(right));
      case '-lt': return truth(Number(left) < Number(right));
      case '-le': return truth(Number(left) <= Number(right));
      case '-gt': return truth(Number(left) > Number(right));
      case '-ge': return truth(Number(left) >= Number(right));
      default: return this.error(`${context.name}: ${operator}: binary operator expected`, 2);
    }
  };

  private cmdEval = async (context: CommandContext): Promise<CommandOutcome> => {
    if (!context.args.length) return this.silent();
    return this.runStatements(parse(this.tokenize(context.args.join(' '))), context.stdin);
  };

  private cmdType = (context: CommandContext): CommandOutcome => {
    const lines: string[] = [];
    let exitCode = 0;
    for (const name of context.args) {
      const alias = this.aliases.get(this.key(name));
      if (alias) { lines.push(`${name} is an alias for ${alias}`); continue; }
      const spec = this.lookup(name);
      if (spec) { lines.push(`${name} is a shell ${spec.builtin ? 'builtin' : 'command'}`); continue; }
      lines.push(`${name} not found`);
      exitCode = 1;
    }
    return this.outcome(lines.join('\n'), '', exitCode);
  };

  private cmdWhich = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['a', 's'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const lines: string[] = [];
    let exitCode = 0;
    for (const name of parsed.operands) {
      const spec = this.lookup(name);
      const external = await this.resolveExecutable(name);
      if (external) lines.push(external.path);
      else if (spec) lines.push(`${name}: shell built-in command`);
      else { lines.push(`${name} not found`); exitCode = 1; }
    }
    return flagged(parsed, 's') ? { stdout: '', stderr: '', exitCode } : this.outcome(lines.join('\n'), '', exitCode);
  };

  private cmdHelp = async (context: CommandContext): Promise<CommandOutcome> => {
    const topic = context.args.find((argument) => !argument.startsWith('-'));
    if (topic) return this.cmdMan({ ...context, name: 'man', args: [topic] });
    const names = [...new Set([...this.commands.keys()])].sort();
    const gitHelp = await this.deps.software.gitCommand(['help'], this.cwd).catch(() => 'git');
    const managers = this.deps.software.supportedManagers().join('  ');
    const columns: string[] = [];
    for (let index = 0; index < names.length; index += 6) columns.push(`  ${names.slice(index, index + 6).map((name) => name.padEnd(18)).join('')}`.trimEnd());
    return this.ok([
      `${this.shellName()} — ${this.deps.spec.os} (${this.dialect} dialect, ${this.caseSensitive ? 'case-sensitive' : 'case-insensitive'} filesystem)`,
      '',
      'commands',
      ...columns,
      '',
      'grammar    quoting  $VAR  ${VAR:-default}  $(...)  `...`  globs  ;  &&  ||  |  &  ( )  >  >>  <  2>  &>  <<',
      '',
      gitHelp,
      '',
      `packages   ${managers}`,
      `programs   installed executables resolve through ${this.deps.profile.conventions.environmentPathKey} (${this.searchPath().length} directories, including node_modules/.bin)`,
    ].join('\n'));
  };

  private cmdMan = (context: CommandContext): CommandOutcome => {
    const name = context.args.find((argument) => !argument.startsWith('-'));
    if (!name) return this.error('What manual page do you want?', 1);
    const spec = this.lookup(name);
    if (!spec) return this.error(`No manual entry for ${name}`, 16);
    return this.ok([
      `${name.toUpperCase()}(1)                     Seed Simulation Manual                    ${name.toUpperCase()}(1)`,
      '',
      'NAME',
      `     ${name} -- ${spec.summary}`,
      '',
      'SYNOPSIS',
      `     ${spec.usage ?? `${name} [options] [operands]`}`,
      '',
      'DESCRIPTION',
      `     ${spec.summary}. Implemented by the Seed kernel against the virtual`,
      '     filesystem, process table and network fabric; unknown options are',
      '     reported as errors rather than silently ignored.',
      '',
      `Seed                            ${this.formatDate(this.now(), '%F')}                            Seed`,
    ].join('\n'));
  };

  /* ---------------------------------------------------------------- filesystem */

  private cmdLs = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['a', 'A', 'l', '1', 'h', 'R', 't', 'S', 'r', 'd', 'F', 'p'], value: ['color'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const long = flagged(parsed, 'l');
    const operands = parsed.operands.length ? parsed.operands : ['.'];
    const blocks: string[] = [];
    const errors: string[] = [];
    let labelled = false;
    const render = (label: string, directory: string): void => {
      const entries = this.deps.vfs.list(directory).map((entry) => ({ name: entry.name, path: entry.path, inode: entry.inode }));
      if (flagged(parsed, 'a')) {
        const self = this.deps.vfs.statSync(directory)!;
        entries.unshift({ name: '.', path: directory, inode: self }, { name: '..', path: `${directory}/..`, inode: this.deps.vfs.statSync(`${directory}/..`) ?? self });
      }
      let visible = flagged(parsed, 'a') || flagged(parsed, 'A') ? entries : entries.filter((entry) => !entry.name.startsWith('.'));
      if (flagged(parsed, 't')) visible = [...visible].sort((left, right) => right.inode.modifiedAt.localeCompare(left.inode.modifiedAt));
      else if (flagged(parsed, 'S')) visible = [...visible].sort((left, right) => right.inode.size - left.inode.size);
      if (flagged(parsed, 'r')) visible = [...visible].reverse();
      blocks.push([
        labelled || flagged(parsed, 'R') ? `${label}:` : undefined,
        long ? `total ${visible.reduce((sum, entry) => sum + Math.ceil(entry.inode.size / 512), 0)}` : undefined,
        ...visible.map((entry) => this.formatEntry(entry.name, entry.path, entry.inode, { long, human: flagged(parsed, 'h'), classify: flagged(parsed, 'F') || flagged(parsed, 'p') })),
      ].filter((line) => line !== undefined).join('\n'));
      if (flagged(parsed, 'R')) {
        for (const entry of visible) {
          if (entry.inode.kind === 'directory' && !['.', '..'].includes(entry.name)) render(`${label}/${entry.name}`, entry.path);
        }
      }
    };
    // Non-directory operands are listed together first, exactly as ls does.
    const files: string[] = [];
    const directories: Array<{ label: string; path: string }> = [];
    for (const operand of operands) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) { errors.push(`${context.name}: cannot access '${operand}': No such file or directory`); continue; }
      if (inode.kind !== 'directory' || flagged(parsed, 'd')) files.push(this.formatEntry(operand, target, inode, { long, human: flagged(parsed, 'h'), classify: flagged(parsed, 'F') }));
      else directories.push({ label: operand, path: target });
    }
    if (files.length) blocks.push(files.join('\n'));
    labelled = directories.length + (files.length ? 1 : 0) > 1;
    for (const directory of directories) render(directory.label, directory.path);
    return this.outcome(blocks.join('\n\n'), errors.join('\n'), errors.length ? 2 : 0);
  };

  private formatEntry(name: string, path: string, inode: { kind: 'file' | 'directory' | 'symlink'; mode: number; size: number; modifiedAt: string; target?: string }, options: { long: boolean; human: boolean; classify: boolean }): string {
    const suffix = options.classify ? (inode.kind === 'directory' ? '/' : inode.kind === 'symlink' ? '@' : (inode.mode & 0o111) !== 0 ? '*' : '') : '';
    const label = inode.kind === 'symlink' && options.long ? `${name} -> ${inode.target ?? ''}` : `${name}${suffix}`;
    if (!options.long) return label;
    const owner = this.deps.vfs.owner(path);
    const size = options.human ? humanSize(inode.size).padStart(5) : String(inode.size).padStart(7);
    return `${modeString(inode.kind, inode.mode)}  1 ${this.owner(owner.uid).padEnd(8)} ${this.group(owner.gid).padEnd(8)} ${size} ${this.stamp(inode.modifiedAt)} ${label}`;
  }

  private cmdCat = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['n', 'b', 'E', 's', 'A', 'v'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    let body = streams.map((stream) => stream.content).join('');
    if (flagged(parsed, 's')) body = body.replace(/\n{3,}/g, '\n\n');
    if (flagged(parsed, 'E') || flagged(parsed, 'A')) body = toLines(body).map((line) => `${line}$`).join('\n');
    if (flagged(parsed, 'n') || flagged(parsed, 'b')) {
      let counter = 0;
      body = toLines(body).map((line) => (flagged(parsed, 'b') && line === '' ? '      \t' : `${String(++counter).padStart(6)}\t${line}`)).join('\n');
    }
    return { stdout: flagged(parsed, 'n') || flagged(parsed, 'b') || flagged(parsed, 'E') ? terminated(body) : body, stderr: terminated(errors.join('\n')), exitCode: errors.length ? 1 : 0 };
  };

  private cmdEcho = (context: CommandContext): CommandOutcome => {
    let args = [...context.args];
    let newline = true;
    let escapes = false;
    while (args[0] === '-n' || args[0] === '-e' || args[0] === '-E') {
      if (args[0] === '-n') newline = false;
      if (args[0] === '-e') escapes = true;
      if (args[0] === '-E') escapes = false;
      args = args.slice(1);
    }
    let text = args.join(' ');
    if (escapes) {
      text = text.replace(/\\(n|t|r|a|b|f|v|0|\\|x[0-9A-Fa-f]{2})/g, (_, code: string) => {
        if (code.startsWith('x')) return String.fromCharCode(Number.parseInt(code.slice(1), 16));
        return { n: '\n', t: '\t', r: '\r', a: '', b: '\b', f: '\f', v: '\v', 0: '\0', '\\': '\\' }[code] ?? code;
      });
    }
    return { stdout: newline ? `${text}\n` : text, stderr: '', exitCode: 0 };
  };

  private cmdPrintf = (context: CommandContext): CommandOutcome => {
    const [format, ...operands] = context.args;
    if (format === undefined) return this.error('printf: usage: printf format [arguments]', 2);
    let index = 0;
    const body = format
      .replace(/\\(n|t|r|\\)/g, (_, code: string) => ({ n: '\n', t: '\t', r: '\r', '\\': '\\' }[code] ?? code))
      .replace(/%(-?\d+)?(\.\d+)?([sdifq%])/g, (_, width: string | undefined, precision: string | undefined, kind: string) => {
        if (kind === '%') return '%';
        const raw = operands[index++] ?? '';
        let value = kind === 'd' || kind === 'i' ? String(Math.trunc(Number(raw) || 0)) : kind === 'f' ? (Number(raw) || 0).toFixed(precision ? Number(precision.slice(1)) : 6) : raw;
        if (kind === 'q') value = `'${value}'`;
        if (!width) return value;
        const size = Number(width);
        return size < 0 ? value.padEnd(-size) : value.padStart(size);
      });
    return { stdout: body, stderr: '', exitCode: 0 };
  };

  private cmdMkdir = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['p', 'v', 'Force'], value: ['m', 'mode', 'ItemType', 'Path'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const operands = [...parsed.operands, ...(flagValue(parsed, 'Path') ? [flagValue(parsed, 'Path')!] : [])];
    if (!operands.length) return this.error(`${context.name}: missing operand`, 2);
    const file = (flagValue(parsed, 'ItemType') ?? 'directory').toLowerCase() === 'file';
    const lines: string[] = [];
    const errors: string[] = [];
    for (const operand of operands) {
      const target = this.resolvePath(operand);
      if (this.deps.vfs.statSync(target)) {
        if (flagged(parsed, 'p') || flagged(parsed, 'Force')) continue;
        errors.push(`${context.name}: ${operand}: File exists`);
        continue;
      }
      if (!flagged(parsed, 'p') && !this.deps.vfs.statSync(this.deps.vfs.resolve('..', target))) {
        errors.push(`${context.name}: ${operand}: No such file or directory`);
        continue;
      }
      if (file) await this.deps.vfs.writeFile(target, '');
      else await this.deps.vfs.mkdir(target);
      const mode = flagValue(parsed, 'm', 'mode');
      if (mode) await this.deps.vfs.chmod(target, Number.parseInt(mode, 8));
      if (flagged(parsed, 'v')) lines.push(`${context.name}: created directory '${operand}'`);
      if (this.dialect === 'powershell') lines.push(`    Directory: ${this.display(this.deps.vfs.resolve('..', target))}\n\nMode                 LastWriteTime         Length Name\n----                 -------------         ------ ----\nd-----   ${this.stamp(this.now().toISOString())}                ${operand.split('/').at(-1)}`);
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdRmdir = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['p', 'v'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const errors: string[] = [];
    for (const operand of parsed.operands) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) { errors.push(`rmdir: ${operand}: No such file or directory`); continue; }
      if (inode.kind !== 'directory') { errors.push(`rmdir: ${operand}: Not a directory`); continue; }
      try { await this.deps.vfs.remove(target, { recursive: false }); }
      catch (error) { errors.push(`rmdir: ${operand}: ${/not empty/.test((error as Error).message) ? 'Directory not empty' : (error as Error).message}`); }
    }
    return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
  };

  /**
   * `touch` creates a file when it is absent and only refreshes the modification
   * time when it exists. It must never truncate: rewriting the same bytes is how
   * this VFS stamps an inode.
   */
  private cmdTouch = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['c', 'a', 'm', 'v'], value: ['d', 't', 'r'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    if (flagValue(parsed, 'd', 't', 'r')) return this.error('touch: explicit timestamps are not implemented in this simulation', 1);
    if (!parsed.operands.length) return this.error('touch: missing file operand', 2);
    const errors: string[] = [];
    for (const operand of parsed.operands) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) {
        if (flagged(parsed, 'c')) continue;
        try { await this.deps.vfs.writeFile(target, ''); }
        catch (error) { errors.push(`touch: ${operand}: ${(error as Error).message}`); }
        continue;
      }
      if (inode.kind !== 'file') continue;
      const bytes = await this.deps.vfs.readBytes(target);
      await this.deps.vfs.writeFile(target, bytes);
    }
    return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdRm = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['r', 'R', 'recursive', 'f', 'force', 'v', 'd', 'i', 'Recurse', 'Force'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    if (flagged(parsed, 'i')) return this.error('rm: interactive prompts are not implemented in this simulation', 1);
    const recursive = flagged(parsed, 'r', 'R', 'recursive', 'Recurse');
    const force = flagged(parsed, 'f', 'force', 'Force');
    if (!parsed.operands.length && !force) return this.error(`${context.name}: missing operand`, 2);
    const lines: string[] = [];
    const errors: string[] = [];
    for (const operand of parsed.operands) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.lstatSync(target);
      if (!inode) { if (!force) errors.push(`${context.name}: ${operand}: No such file or directory`); continue; }
      if (inode.kind === 'directory' && !recursive && !flagged(parsed, 'd')) {
        errors.push(`${context.name}: ${operand}: is a directory`);
        continue;
      }
      try {
        await this.deps.vfs.remove(target, { recursive });
        if (flagged(parsed, 'v')) lines.push(`removed '${operand}'`);
      } catch (error) {
        errors.push(`${context.name}: ${operand}: ${/not empty/.test((error as Error).message) ? 'Directory not empty' : (error as Error).message}`);
      }
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdMv = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['f', 'i', 'n', 'v', 'Force'], value: ['Path', 'Destination'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const operands = [...(flagValue(parsed, 'Path') ? [flagValue(parsed, 'Path')!] : []), ...parsed.operands, ...(flagValue(parsed, 'Destination') ? [flagValue(parsed, 'Destination')!] : [])];
    if (operands.length < 2) return this.error(`${context.name}: missing destination file operand`, 2);
    const destination = operands.pop()!;
    const target = this.resolvePath(destination);
    const intoDirectory = this.deps.vfs.statSync(target)?.kind === 'directory';
    if (operands.length > 1 && !intoDirectory) return this.error(`${context.name}: target '${destination}' is not a directory`, 2);
    const lines: string[] = [];
    const errors: string[] = [];
    for (const operand of operands) {
      try {
        await this.deps.vfs.rename(this.resolvePath(operand), target, { overwrite: !flagged(parsed, 'n') });
        if (flagged(parsed, 'v')) lines.push(`renamed '${operand}' -> '${destination}'`);
      } catch (error) { errors.push(`${context.name}: ${operand}: ${(error as Error).message}`); }
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdCp = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['r', 'R', 'recursive', 'Recurse', 'a', 'f', 'i', 'n', 'v', 'L', 'p', 'Force'], value: ['Path', 'Destination'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const operands = [...(flagValue(parsed, 'Path') ? [flagValue(parsed, 'Path')!] : []), ...parsed.operands, ...(flagValue(parsed, 'Destination') ? [flagValue(parsed, 'Destination')!] : [])];
    if (operands.length < 2) return this.error(`${context.name}: missing destination file operand`, 2);
    const destination = operands.pop()!;
    const recursive = flagged(parsed, 'r', 'R', 'recursive', 'Recurse', 'a');
    const lines: string[] = [];
    const errors: string[] = [];
    for (const operand of operands) {
      const source = this.resolvePath(operand);
      if (this.deps.vfs.statSync(source)?.kind === 'directory' && !recursive) {
        errors.push(`${context.name}: -r not specified; omitting directory '${operand}'`);
        continue;
      }
      try {
        await this.deps.vfs.copy(source, this.resolvePath(destination), { recursive, overwrite: !flagged(parsed, 'n'), dereference: flagged(parsed, 'L') });
        if (flagged(parsed, 'v')) lines.push(`'${operand}' -> '${destination}'`);
      } catch (error) { errors.push(`${context.name}: ${operand}: ${(error as Error).message}`); }
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdLn = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['s', 'f', 'v', 'n'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const [source, link] = parsed.operands;
    if (!source) return this.error('ln: missing file operand', 2);
    const linkPath = this.resolvePath(link ?? source.split('/').at(-1)!);
    if (flagged(parsed, 'f') && this.deps.vfs.lstatSync(linkPath)) await this.deps.vfs.remove(linkPath, { recursive: false });
    try {
      if (flagged(parsed, 's')) await this.deps.vfs.symlink(source, linkPath);
      else await this.deps.vfs.link(this.resolvePath(source), linkPath);
    } catch (error) { return this.error(`ln: ${(error as Error).message}`); }
    return flagged(parsed, 'v') ? this.ok(`'${link ?? source}' -> '${source}'`) : this.silent();
  };

  private cmdBasename = (context: CommandContext): CommandOutcome => {
    const [value, suffix] = context.args;
    if (!value) return this.error('basename: missing operand', 2);
    let name = value.replace(/\/+$/, '').split('/').at(-1) ?? value;
    if (suffix && name.endsWith(suffix) && name !== suffix) name = name.slice(0, -suffix.length);
    return this.ok(name);
  };

  private cmdDirname = (context: CommandContext): CommandOutcome => {
    const value = context.args[0];
    if (!value) return this.error('dirname: missing operand', 2);
    const trimmed = value.replace(/\/+$/, '');
    const index = trimmed.lastIndexOf('/');
    return this.ok(index === -1 ? '.' : index === 0 ? '/' : trimmed.slice(0, index));
  };

  private cmdRealpath = (context: CommandContext): CommandOutcome => {
    const lines: string[] = [];
    const errors: string[] = [];
    for (const operand of context.args.length ? context.args : ['.']) {
      try { lines.push(this.deps.vfs.realpath(this.resolvePath(operand))); }
      catch { errors.push(`realpath: ${operand}: No such file or directory`); }
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdReadlink = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['f', 'n'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const operand = parsed.operands[0];
    if (!operand) return this.error('readlink: missing operand', 2);
    try { return this.ok(flagged(parsed, 'f') ? this.deps.vfs.realpath(this.resolvePath(operand)) : this.deps.vfs.readlink(this.resolvePath(operand))); }
    catch { return { stdout: '', stderr: '', exitCode: 1 }; }
  };

  private cmdStat = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['L', 't'], value: ['c', 'format'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const blocks: string[] = [];
    const errors: string[] = [];
    for (const operand of parsed.operands) {
      const target = this.resolvePath(operand);
      const inode = flagged(parsed, 'L') ? this.deps.vfs.statSync(target) : this.deps.vfs.lstatSync(target);
      if (!inode) { errors.push(`stat: cannot statx '${operand}': No such file or directory`); continue; }
      const owner = this.deps.vfs.owner(target);
      const format = flagValue(parsed, 'c', 'format');
      if (format) {
        blocks.push(format.replace(/%[a-zA-Z]/g, (token) => ({
          '%n': operand, '%s': String(inode.size), '%f': inode.mode.toString(16), '%a': (inode.mode & 0o7777).toString(8),
          '%F': inode.kind === 'directory' ? 'directory' : inode.kind === 'symlink' ? 'symbolic link' : 'regular file',
          '%u': String(owner.uid), '%g': String(owner.gid), '%y': inode.modifiedAt, '%i': inode.id,
        }[token] ?? token)));
        continue;
      }
      blocks.push([
        `  File: ${operand}${inode.kind === 'symlink' ? ` -> ${inode.target ?? ''}` : ''}`,
        `  Size: ${String(inode.size).padEnd(10)} Blocks: ${Math.ceil(inode.size / 512)}          IO Block: 4096   ${inode.kind === 'directory' ? 'directory' : inode.kind === 'symlink' ? 'symbolic link' : 'regular file'}`,
        `Device: ${inode.diskId}   Inode: ${inode.id}   Links: 1`,
        `Access: (${(inode.mode & 0o7777).toString(8).padStart(4, '0')}/${modeString(inode.kind, inode.mode)})  Uid: (${String(owner.uid).padStart(5)}/${this.owner(owner.uid).padStart(8)})   Gid: (${String(owner.gid).padStart(5)}/${this.group(owner.gid).padStart(8)})`,
        `Modify: ${inode.modifiedAt}`,
        `Birth : ${inode.createdAt}`,
      ].join('\n'));
    }
    return this.outcome(blocks.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdFile = async (context: CommandContext): Promise<CommandOutcome> => {
    const lines: string[] = [];
    let exitCode = 0;
    for (const operand of context.args) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.lstatSync(target);
      if (!inode) { lines.push(`${operand}: cannot open (No such file or directory)`); exitCode = 1; continue; }
      if (inode.kind === 'directory') { lines.push(`${operand}: directory`); continue; }
      if (inode.kind === 'symlink') { lines.push(`${operand}: symbolic link to ${inode.target ?? ''}`); continue; }
      if (inode.size === 0) { lines.push(`${operand}: empty`); continue; }
      if (await this.deps.vfs.isBinary(target)) {
        const bytes = await this.deps.vfs.readBytes(target);
        const header = bytes.subarray(0, 5).toString('latin1');
        lines.push(`${operand}: ${header.startsWith('%PDF') ? 'PDF document' : header.startsWith('ELF') ? 'ELF executable' : 'data'}`);
        continue;
      }
      const content = await this.deps.vfs.readFile(target);
      const kind = content.startsWith('{') || content.startsWith('[') ? 'JSON text data'
        : content.startsWith('#!') ? `a ${content.slice(2, content.indexOf('\n')).trim()} script, ASCII text executable`
          : content.startsWith('<') ? 'HTML document text' : 'ASCII text';
      lines.push(`${operand}: ${kind}`);
    }
    return this.outcome(lines.join('\n'), '', exitCode);
  };

  private cmdChmod = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['R', 'v'], stopAtOperand: false });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const [mode, ...targets] = parsed.operands;
    if (!mode || !targets.length) return this.error('chmod: missing operand', 2);
    const compute = (current: number): number | undefined => {
      if (/^[0-7]{3,4}$/.test(mode)) return Number.parseInt(mode, 8);
      const match = mode.match(/^([ugoa]*)([+\-=])([rwxX]+)$/);
      if (!match) return undefined;
      const [, who = 'a', operator, bits] = match;
      const value = (bits!.includes('r') ? 4 : 0) | (bits!.includes('w') ? 2 : 0) | (bits!.includes('x') || bits!.includes('X') ? 1 : 0);
      const shifts = (who === 'a' || who === '' ? ['u', 'g', 'o'] : [...who]).map((entry) => ({ u: 6, g: 3, o: 0 }[entry] ?? 0));
      let next = current;
      for (const shift of shifts) {
        if (operator === '+') next |= value << shift;
        else if (operator === '-') next &= ~(value << shift);
        else next = (next & ~(7 << shift)) | (value << shift);
      }
      return next;
    };
    const errors: string[] = [];
    for (const operand of targets) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) { errors.push(`chmod: ${operand}: No such file or directory`); continue; }
      const paths = flagged(parsed, 'R') && inode.kind === 'directory' ? [target, ...walkTree(this.deps.vfs, target, { includeHidden: true }).map((entry) => entry.path)] : [target];
      for (const path of paths) {
        const current = this.deps.vfs.statSync(path)!;
        const next = compute(current.mode);
        if (next === undefined) return this.error(`chmod: invalid mode: '${mode}'`, 2);
        await this.deps.vfs.chmod(path, next);
      }
    }
    return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdChown = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['R', 'v'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const [spec, ...targets] = parsed.operands;
    if (!spec || !targets.length) return this.error('chown: missing operand', 2);
    const identity = (name: string): number => (name === 'root' || name === '0' ? 0 : /^\d+$/.test(name) ? Number(name) : this.deps.spec.os === 'macos' ? 501 : 1000);
    const [user, groupName] = spec.split(':');
    const uid = identity(user ?? 'root');
    const gid = groupName ? identity(groupName) : this.deps.spec.os === 'macos' ? (uid === 0 ? 0 : 20) : uid;
    const errors: string[] = [];
    for (const operand of targets) {
      const target = this.resolvePath(operand);
      if (!this.deps.vfs.statSync(target)) { errors.push(`chown: ${operand}: No such file or directory`); continue; }
      const paths = flagged(parsed, 'R') ? [target, ...walkTree(this.deps.vfs, target, { includeHidden: true }).map((entry) => entry.path)] : [target];
      for (const path of paths) await this.deps.vfs.chown(path, uid, gid);
    }
    return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdDf = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['h', 'k', 'i', 'T'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const usage = this.deps.vfs.usage();
    const human = flagged(parsed, 'h');
    const size = (bytes: number): string => (human ? humanSize(bytes) : String(Math.round(bytes / 1024)));
    const rows = this.deps.spec.disks.map((disk, index) => {
      const used = index === 0 ? usage.bytes : 0;
      const percent = disk.capacityBytes ? Math.min(100, Math.ceil((used / disk.capacityBytes) * 100)) : 0;
      return `${`/dev/${disk.id}`.padEnd(16)} ${size(disk.capacityBytes).padStart(9)} ${size(used).padStart(9)} ${size(disk.capacityBytes - used).padStart(9)} ${`${percent}%`.padStart(4)} ${disk.mount}`;
    });
    return this.ok([`${'Filesystem'.padEnd(16)} ${(human ? 'Size' : '1K-blocks').padStart(9)} ${'Used'.padStart(9)} ${'Avail'.padStart(9)} Use% Mounted on`, ...rows].join('\n'));
  };

  private cmdDu = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['s', 'h', 'a', 'c'], value: ['d', 'max-depth'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const rows: string[] = [];
    let total = 0;
    for (const operand of parsed.operands.length ? parsed.operands : ['.']) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) return this.error(`du: ${operand}: No such file or directory`);
      const entries = inode.kind === 'directory' ? walkTree(this.deps.vfs, target, { includeHidden: true }) : [];
      const sizeOf = (path: string): number => entries.filter((entry) => entry.path === path || entry.path.startsWith(`${path}/`)).reduce((sum, entry) => sum + entry.inode.size, 0);
      const bytes = inode.kind === 'directory' ? sizeOf(target) : inode.size;
      total += bytes;
      const format = (value: number): string => (flagged(parsed, 'h') ? humanSize(value) : String(Math.max(1, Math.ceil(value / 1024))));
      if (!flagged(parsed, 's')) {
        const depth = Number(flagValue(parsed, 'd', 'max-depth') ?? Number.POSITIVE_INFINITY);
        for (const entry of entries) {
          if (entry.depth > depth) continue;
          if (entry.kind !== 'directory' && !flagged(parsed, 'a')) continue;
          rows.push(`${format(entry.kind === 'directory' ? sizeOf(entry.path) : entry.inode.size)}\t${entry.path.startsWith(`${target}/`) ? `${operand}/${entry.path.slice(target.length + 1)}` : entry.path}`);
        }
      }
      rows.push(`${format(bytes)}\t${operand}`);
    }
    if (flagged(parsed, 'c')) rows.push(`${flagged(parsed, 'h') ? humanSize(total) : Math.max(1, Math.ceil(total / 1024))}\ttotal`);
    return this.ok(rows.join('\n'));
  };

  private cmdFind = async (context: CommandContext): Promise<CommandOutcome> => {
    const roots: string[] = [];
    let index = 0;
    while (index < context.args.length && !context.args[index]!.startsWith('-') && !['(', ')', '!'].includes(context.args[index]!)) roots.push(context.args[index++]!);
    if (!roots.length) roots.push('.');
    interface Predicate { (entry: { path: string; name: string; kind: string; depth: number }): boolean }
    const predicates: Predicate[] = [];
    let maxDepth = Number.POSITIVE_INFINITY;
    let minDepth = 0;
    let action: { kind: 'print' } | { kind: 'delete' } | { kind: 'exec'; argv: string[]; batch: boolean } = { kind: 'print' };
    while (index < context.args.length) {
      const token = context.args[index++]!;
      const value = context.args[index];
      switch (token) {
        case '-name': index += 1; { const expression = globToRegExp(value ?? '', this.caseSensitive); predicates.push((entry) => expression.test(entry.name)); break; }
        case '-iname': index += 1; { const expression = globToRegExp(value ?? '', false); predicates.push((entry) => expression.test(entry.name)); break; }
        case '-path': index += 1; { const expression = globToRegExp(value ?? '', this.caseSensitive); predicates.push((entry) => expression.test(entry.path)); break; }
        case '-type': index += 1; { const wanted = value === 'f' ? 'file' : value === 'd' ? 'directory' : value === 'l' ? 'symlink' : undefined; if (!wanted) return this.error(`find: unknown type '${value ?? ''}'`, 2); predicates.push((entry) => entry.kind === wanted); break; }
        case '-maxdepth': index += 1; maxDepth = Number(value ?? 0); break;
        case '-mindepth': index += 1; minDepth = Number(value ?? 0); break;
        case '-empty': predicates.push((entry) => (this.deps.vfs.statSync(entry.path)?.size ?? 0) === 0); break;
        case '-print': action = { kind: 'print' }; break;
        case '-delete': action = { kind: 'delete' }; break;
        case '-exec': {
          const argv: string[] = [];
          while (index < context.args.length && context.args[index] !== ';' && context.args[index] !== '+') argv.push(context.args[index++]!);
          const batch = context.args[index] === '+';
          index += 1;
          action = { kind: 'exec', argv, batch };
          break;
        }
        default: return this.error(`find: unknown predicate '${token}'`, 2);
      }
    }
    const results: string[] = [];
    const errors: string[] = [];
    for (const root of roots) {
      const target = this.resolvePath(root);
      if (!this.deps.vfs.statSync(target)) { errors.push(`find: ${root}: No such file or directory`); continue; }
      const candidates = [
        { path: target, name: target.split('/').at(-1) ?? target, kind: this.deps.vfs.statSync(target)!.kind as string, depth: 0 },
        ...walkTree(this.deps.vfs, target, { includeHidden: true, maxDepth: Number.isFinite(maxDepth) ? maxDepth : undefined }).map((entry) => ({ path: entry.path, name: entry.name, kind: entry.kind as string, depth: entry.depth })),
      ];
      for (const candidate of candidates) {
        if (candidate.depth > maxDepth || candidate.depth < minDepth) continue;
        if (!predicates.every((predicate) => predicate(candidate))) continue;
        results.push(candidate.path === target ? root : `${root.replace(/\/$/, '')}/${candidate.path.slice(target.length + 1)}`);
      }
    }
    if (action.kind === 'delete') {
      for (const path of [...results].reverse()) await this.deps.vfs.remove(this.resolvePath(path), { recursive: true });
      return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
    }
    if (action.kind === 'exec') {
      const outputs: string[] = [];
      const runs = action.batch ? [results] : results.map((entry) => [entry]);
      for (const batch of runs) {
        const argv = action.argv.flatMap((token) => (token === '{}' ? batch : [token]));
        const outcome = await this.invoke(argv, '');
        outputs.push(outcome.stdout);
        if (outcome.stderr) errors.push(outcome.stderr.replace(/\n$/, ''));
      }
      return { stdout: outputs.join(''), stderr: terminated(errors.join('\n')), exitCode: errors.length ? 1 : 0 };
    }
    return this.outcome(results.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  /* ------------------------------------------------------------------- text */

  /**
   * Real grep: opens the files it is given, matches actual regular expressions,
   * is case-sensitive by default, and exits 1 when nothing matched.
   */
  private cmdGrep = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, {
      boolean: ['i', 'ignore-case', 'v', 'invert-match', 'n', 'line-number', 'c', 'count', 'r', 'R', 'recursive', 'l', 'files-with-matches', 'L', 'q', 'quiet', 'w', 'word-regexp', 'x', 'line-regexp', 'E', 'extended-regexp', 'F', 'fixed-strings', 'h', 'H', 's', 'o'],
      value: ['e', 'regexp', 'm', 'max-count'],
    });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const explicit = flagValue(parsed, 'e', 'regexp');
    const fixed = flagged(parsed, 'F', 'fixed-strings') || context.name === 'fgrep';
    const pattern = explicit ?? parsed.operands.shift();
    if (pattern === undefined) return this.error(`usage: ${context.name} [-invcrlEF] pattern [file ...]`, 2);
    const escaped = fixed ? pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : pattern;
    const bounded = flagged(parsed, 'x', 'line-regexp') ? `^(?:${escaped})$` : flagged(parsed, 'w', 'word-regexp') ? `\\b(?:${escaped})\\b` : escaped;
    let expression: RegExp;
    try { expression = new RegExp(bounded, flagged(parsed, 'i', 'ignore-case') ? 'i' : ''); }
    catch (error) { return this.error(`${context.name}: ${(error as Error).message}`, 2); }
    const recursive = flagged(parsed, 'r', 'R', 'recursive');
    const targets: Array<{ name: string; content: string }> = [];
    const errors: string[] = [];
    if (!parsed.operands.length) targets.push({ name: '(standard input)', content: context.stdin });
    for (const operand of parsed.operands) {
      const target = this.resolvePath(operand);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) { if (!flagged(parsed, 's')) errors.push(`${context.name}: ${operand}: No such file or directory`); continue; }
      if (inode.kind === 'directory') {
        if (!recursive) { errors.push(`${context.name}: ${operand}: Is a directory`); continue; }
        for (const entry of walkTree(this.deps.vfs, target, { includeHidden: true })) {
          if (entry.kind !== 'file') continue;
          if (await this.deps.vfs.isBinary(entry.path).catch(() => true)) continue;
          targets.push({ name: `${operand.replace(/\/$/, '')}/${entry.path.slice(target.length + 1)}`, content: await this.deps.vfs.readFile(entry.path) });
        }
        continue;
      }
      targets.push({ name: operand, content: await this.deps.vfs.readFile(target) });
    }
    const invert = flagged(parsed, 'v', 'invert-match');
    const limit = Number(flagValue(parsed, 'm', 'max-count') ?? Number.POSITIVE_INFINITY);
    const showName = flagged(parsed, 'H') || (targets.length > 1 && !flagged(parsed, 'h'));
    const lines: string[] = [];
    let matched = false;
    for (const target of targets) {
      let count = 0;
      const hits: string[] = [];
      toLines(target.content).forEach((line, index) => {
        if (count >= limit) return;
        if (expression.test(line) === invert) return;
        count += 1;
        matched = true;
        const body = flagged(parsed, 'o') ? line.match(new RegExp(expression.source, expression.flags))?.[0] ?? line : line;
        hits.push([showName ? target.name : undefined, flagged(parsed, 'n', 'line-number') ? String(index + 1) : undefined, body].filter((part) => part !== undefined).join(':'));
      });
      if (flagged(parsed, 'c', 'count')) lines.push(showName ? `${target.name}:${count}` : String(count));
      else if (flagged(parsed, 'l', 'files-with-matches')) { if (count) lines.push(target.name); }
      else if (flagged(parsed, 'L')) { if (!count) lines.push(target.name); }
      else lines.push(...hits);
    }
    if (errors.length && !matched) return this.outcome('', errors.join('\n'), 2);
    if (flagged(parsed, 'q', 'quiet')) return { stdout: '', stderr: '', exitCode: matched ? 0 : 1 };
    return this.outcome(lines.join('\n'), errors.join('\n'), matched ? 0 : 1);
  };

  /** `head -3` / `tail -5`: POSIX numeric shorthand for `-n 3`. */
  private static numericLineShorthand(args: string[]): string[] {
    return args.flatMap((arg) => /^-\d+$/.test(arg) ? ['-n', arg.slice(1)] : [arg]);
  }

  private cmdHead = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(ShellSession.numericLineShorthand(context.args), { boolean: ['q', 'v'], value: ['n', 'lines', 'c', 'bytes'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const count = Number(flagValue(parsed, 'n', 'lines') ?? 10);
    const bytes = flagValue(parsed, 'c', 'bytes');
    const blocks = streams.map((stream) => {
      const body = bytes ? stream.content.slice(0, Number(bytes)) : toLines(stream.content).slice(0, count).join('\n');
      return streams.length > 1 || flagged(parsed, 'v') ? `==> ${stream.name} <==\n${body}` : body;
    });
    return this.outcome(blocks.join('\n\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdTail = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(ShellSession.numericLineShorthand(context.args), { boolean: ['q', 'v', 'f', 'F'], value: ['n', 'lines', 'c', 'bytes'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    if (flagged(parsed, 'f', 'F')) return this.error('tail: follow mode is not implemented in this simulation', 1);
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const raw = flagValue(parsed, 'n', 'lines') ?? '10';
    const fromStart = raw.startsWith('+');
    const count = Number(raw.replace(/^[+-]/, ''));
    const bytes = flagValue(parsed, 'c', 'bytes');
    const blocks = streams.map((stream) => {
      const lines = toLines(stream.content);
      const body = bytes ? stream.content.slice(-Number(bytes)) : (fromStart ? lines.slice(count - 1) : lines.slice(Math.max(0, lines.length - count))).join('\n');
      return streams.length > 1 || flagged(parsed, 'v') ? `==> ${stream.name} <==\n${body}` : body;
    });
    return this.outcome(blocks.join('\n\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdWc = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['l', 'lines', 'w', 'words', 'c', 'bytes', 'm', 'chars', 'L'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const wantLines = flagged(parsed, 'l', 'lines');
    const wantWords = flagged(parsed, 'w', 'words');
    const wantBytes = flagged(parsed, 'c', 'bytes', 'm', 'chars');
    const none = !wantLines && !wantWords && !wantBytes && !flagged(parsed, 'L');
    const totals = { lines: 0, words: 0, bytes: 0 };
    const rows = streams.map((stream) => {
      const lines = toLines(stream.content).length;
      const words = stream.content.split(/\s+/).filter(Boolean).length;
      const bytes = Buffer.byteLength(stream.content, 'utf8');
      totals.lines += lines; totals.words += words; totals.bytes += bytes;
      const columns: string[] = [];
      if (none || wantLines) columns.push(String(lines).padStart(none ? 8 : 0));
      if (none || wantWords) columns.push(String(words).padStart(none ? 8 : 0));
      if (none || wantBytes) columns.push(String(bytes).padStart(none ? 8 : 0));
      if (flagged(parsed, 'L')) columns.push(String(Math.max(0, ...toLines(stream.content).map((line) => line.length))));
      return `${columns.join(' ')}${stream.name === '-' ? '' : ` ${stream.name}`}`;
    });
    if (streams.length > 1) {
      const columns: string[] = [];
      if (none || wantLines) columns.push(String(totals.lines).padStart(none ? 8 : 0));
      if (none || wantWords) columns.push(String(totals.words).padStart(none ? 8 : 0));
      if (none || wantBytes) columns.push(String(totals.bytes).padStart(none ? 8 : 0));
      rows.push(`${columns.join(' ')} total`);
    }
    return this.outcome(rows.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdSort = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['r', 'n', 'u', 'f', 'b', 'V', 'Descending', 'Unique'], value: ['k', 't', 'o'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    let lines = streams.flatMap((stream) => toLines(stream.content));
    const separator = flagValue(parsed, 't');
    const column = flagValue(parsed, 'k');
    const keyOf = (line: string): string => {
      if (!column) return line;
      const index = Number(column.split(',')[0]) - 1;
      const fields = separator ? line.split(separator) : line.trim().split(/\s+/);
      return fields[index] ?? '';
    };
    lines.sort((left, right) => {
      const a = keyOf(flagged(parsed, 'f') ? left.toLowerCase() : left);
      const b = keyOf(flagged(parsed, 'f') ? right.toLowerCase() : right);
      return flagged(parsed, 'n') ? (Number.parseFloat(a) || 0) - (Number.parseFloat(b) || 0) : a.localeCompare(b);
    });
    if (flagged(parsed, 'r', 'Descending')) lines.reverse();
    if (flagged(parsed, 'u', 'Unique')) lines = lines.filter((line, index, all) => index === 0 || all[index - 1] !== line);
    const output = lines.join('\n');
    const target = flagValue(parsed, 'o');
    if (target) { await this.writeStream(this.resolvePath(target), terminated(output), false); return this.outcome('', errors.join('\n'), errors.length ? 1 : 0); }
    return this.outcome(output, errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdUniq = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['c', 'd', 'u', 'i'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const lines = streams.flatMap((stream) => toLines(stream.content));
    const groups: Array<{ line: string; count: number }> = [];
    for (const line of lines) {
      const previous = groups.at(-1);
      const same = previous && (flagged(parsed, 'i') ? previous.line.toLowerCase() === line.toLowerCase() : previous.line === line);
      if (same) previous!.count += 1; else groups.push({ line, count: 1 });
    }
    const selected = groups.filter((group) => (flagged(parsed, 'd') ? group.count > 1 : flagged(parsed, 'u') ? group.count === 1 : true));
    return this.outcome(selected.map((group) => (flagged(parsed, 'c') ? `${String(group.count).padStart(7)} ${group.line}` : group.line)).join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdCut = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['s'], value: ['d', 'f', 'c', 'b'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const fields = flagValue(parsed, 'f');
    const characters = flagValue(parsed, 'c', 'b');
    if (!fields && !characters) return this.error('cut: you must specify a list of bytes, characters, or fields', 2);
    const delimiter = flagValue(parsed, 'd') ?? '\t';
    const ranges = (fields ?? characters!).split(',').map((entry) => {
      const [from, to] = entry.split('-');
      return { from: Number(from || 1), to: to === '' ? Number.POSITIVE_INFINITY : Number(to ?? from) };
    });
    const pick = <T,>(values: T[]): T[] => values.filter((_, index) => ranges.some((range) => index + 1 >= range.from && index + 1 <= range.to));
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const output = streams.flatMap((stream) => toLines(stream.content).flatMap((line) => {
      if (characters) return [pick([...line]).join('')];
      if (!line.includes(delimiter)) return flagged(parsed, 's') ? [] : [line];
      return [pick(line.split(delimiter)).join(delimiter)];
    }));
    return this.outcome(output.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdTr = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['d', 's', 'c', 'delete', 'squeeze-repeats'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const expand = (set: string): string => set
      .replace(/\[:alpha:\]/g, 'A-Za-z').replace(/\[:digit:\]/g, '0-9').replace(/\[:alnum:\]/g, 'A-Za-z0-9')
      .replace(/\[:upper:\]/g, 'A-Z').replace(/\[:lower:\]/g, 'a-z').replace(/\[:space:\]/g, ' \t\n\r')
      .replace(/(.)-(.)/g, (_, from: string, to: string) => {
        const characters: string[] = [];
        for (let code = from.charCodeAt(0); code <= to.charCodeAt(0); code += 1) characters.push(String.fromCharCode(code));
        return characters.join('');
      });
    const [rawFrom, rawTo] = parsed.operands;
    if (rawFrom === undefined) return this.error('tr: missing operand', 2);
    const from = expand(rawFrom);
    const to = rawTo === undefined ? '' : expand(rawTo);
    let output = '';
    let previous = '';
    for (const character of context.stdin) {
      const index = from.indexOf(character);
      if (flagged(parsed, 'd', 'delete') && index >= 0) continue;
      const replaced = index >= 0 && to ? to[Math.min(index, to.length - 1)]! : character;
      if (flagged(parsed, 's', 'squeeze-repeats') && replaced === previous && (index >= 0 || to.includes(replaced))) continue;
      output += replaced;
      previous = replaced;
    }
    return { stdout: output, stderr: '', exitCode: 0 };
  };

  private cmdTee = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['a', 'i'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const errors: string[] = [];
    for (const operand of parsed.operands) {
      try { await this.writeStream(this.resolvePath(operand), context.stdin, flagged(parsed, 'a')); }
      catch (error) { errors.push(`tee: ${operand}: ${(error as Error).message}`); }
    }
    return { stdout: context.stdin, stderr: terminated(errors.join('\n')), exitCode: errors.length ? 1 : 0 };
  };

  private cmdSed = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['n', 'i', 'r', 'E'], value: ['e', 'f'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const script = flagValue(parsed, 'e') ?? parsed.operands.shift();
    if (!script) return this.error('sed: no script specified', 2);
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const programs = script.split(';').map((entry) => entry.trim()).filter(Boolean);
    const outputs: string[] = [];
    for (const stream of streams) {
      const result: string[] = [];
      for (const [index, line] of toLines(stream.content).entries()) {
        let current: string | undefined = line;
        let printed = false;
        for (const program of programs) {
          if (current === undefined) break;
          const substitution = program.match(/^(?:(\d+|\/(?:\\.|[^/])*\/)\s*)?s(.)((?:\\.|(?!\2).)*)\2((?:\\.|(?!\2).)*)\2([gip]*)$/);
          if (substitution) {
            const [, address, , pattern, replacement, flags] = substitution;
            if (address && !this.sedAddressMatches(address, current, index)) continue;
            const expression = new RegExp(pattern!, `${flags!.includes('g') ? 'g' : ''}${flags!.includes('i') ? 'i' : ''}`);
            current = current.replace(expression, replacement!.replace(/\\n/g, '\n'));
            if (flags!.includes('p')) { result.push(current); printed = true; }
            continue;
          }
          const deletion = program.match(/^(\d+|\/(?:\\.|[^/])*\/)?\s*d$/);
          if (deletion) { if (!deletion[1] || this.sedAddressMatches(deletion[1], current, index)) current = undefined; continue; }
          const print = program.match(/^(\d+|\/(?:\\.|[^/])*\/)?\s*p$/);
          if (print) { if (!print[1] || this.sedAddressMatches(print[1], current, index)) { result.push(current); printed = true; } continue; }
          if (program === 'q') break;
          return this.error(`sed: unsupported command: ${program}`, 2);
        }
        if (current !== undefined && !flagged(parsed, 'n') && !printed) result.push(current);
      }
      const body = result.join('\n');
      if (flagged(parsed, 'i') && stream.name !== '-') await this.writeStream(this.resolvePath(stream.name), terminated(body), false);
      else outputs.push(body);
    }
    return this.outcome(outputs.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private sedAddressMatches(address: string, line: string, index: number): boolean {
    if (/^\d+$/.test(address)) return index + 1 === Number(address);
    return new RegExp(address.slice(1, -1)).test(line);
  }

  /** A working subset of awk: BEGIN/END, regex and NR patterns, `print`. */
  private cmdAwk = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { value: ['F', 'v'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const program = parsed.operands.shift();
    if (!program) return this.error('awk: no program text', 2);
    const separator = flagValue(parsed, 'F');
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    const rules: Array<{ pattern?: string; body: string }> = [];
    const source = program.trim();
    const expression = /(BEGIN|END|\/(?:\\.|[^/])*\/|\$?[A-Za-z_][A-Za-z0-9_]*\s*[=!<>]=?\s*[^{]+|[^{]*)\{([^}]*)\}/g;
    let match = expression.exec(source);
    if (!match) rules.push({ body: source.replace(/^\{|\}$/g, '') });
    while (match) {
      rules.push({ pattern: match[1]?.trim() || undefined, body: match[2] ?? '' });
      match = expression.exec(source);
    }
    const output: string[] = [];
    const variables = new Map<string, string>();
    const assignment = flagValue(parsed, 'v');
    if (assignment) variables.set(assignment.split('=')[0]!, assignment.split('=').slice(1).join('='));
    let records = 0;
    const emit = (body: string, fields: string[], nr: number): void => {
      for (const statement of body.split(';').map((entry) => entry.trim()).filter(Boolean)) {
        if (!statement.startsWith('print')) continue;
        const rest = statement.slice(5).trim();
        if (!rest) { output.push(fields.join(' ') || ''); continue; }
        const pieces = rest.split(',').map((piece) => piece.trim()).map((piece) => {
          if (/^"(.*)"$/.test(piece)) return piece.slice(1, -1);
          if (piece === 'NR') return String(nr);
          if (piece === 'NF') return String(fields.length - 1);
          if (piece === '$0') return fields[0] ?? '';
          if (piece === '$NF') return fields.at(-1) ?? '';
          const field = piece.match(/^\$(\d+)$/);
          if (field) return fields[Number(field[1])] ?? '';
          return variables.get(piece) ?? piece;
        });
        output.push(pieces.join(' '));
      }
    };
    for (const rule of rules) if (rule.pattern === 'BEGIN') emit(rule.body, [''], 0);
    let lastFields = [''];
    for (const stream of streams) {
      for (const line of toLines(stream.content)) {
        records += 1;
        const columns = separator ? line.split(separator) : line.trim().split(/\s+/);
        const fields = [line, ...columns];
        lastFields = fields;
        for (const rule of rules) {
          if (rule.pattern === 'BEGIN' || rule.pattern === 'END') continue;
          if (rule.pattern) {
            if (/^\/.*\/$/.test(rule.pattern)) { if (!new RegExp(rule.pattern.slice(1, -1)).test(line)) continue; }
            else {
              const comparison = rule.pattern.match(/^(\S+)\s*(==|!=|<|>|<=|>=)\s*(.+)$/);
              if (!comparison) continue;
              const left = comparison[1] === 'NR' ? String(records) : comparison[1]!.startsWith('$') ? fields[Number(comparison[1]!.slice(1))] ?? '' : comparison[1]!;
              const right = comparison[3]!.replace(/^"(.*)"$/, '$1');
              const value = { '==': left === right, '!=': left !== right, '<': Number(left) < Number(right), '>': Number(left) > Number(right), '<=': Number(left) <= Number(right), '>=': Number(left) >= Number(right) }[comparison[2]!];
              if (!value) continue;
            }
          }
          emit(rule.body, fields, records);
        }
      }
    }
    for (const rule of rules) if (rule.pattern === 'END') emit(rule.body, lastFields, records);
    return this.outcome(output.join('\n'), errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdDiff = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['u', 'q', 'i', 'w', 'r', 'N'], value: ['U'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const [left, right] = parsed.operands;
    if (!left || !right) return this.error('diff: missing operand', 2);
    const read = async (operand: string): Promise<string | undefined> => {
      const target = this.resolvePath(operand);
      if (!this.deps.vfs.statSync(target)) return undefined;
      return this.deps.vfs.readFile(target);
    };
    const leftContent = await read(left);
    const rightContent = await read(right);
    if (leftContent === undefined) return this.error(`diff: ${left}: No such file or directory`, 2);
    if (rightContent === undefined) return this.error(`diff: ${right}: No such file or directory`, 2);
    const normalize = (value: string): string[] => toLines(value).map((line) => (flagged(parsed, 'w') ? line.replace(/\s+/g, '') : flagged(parsed, 'i') ? line.toLowerCase() : line));
    const a = normalize(leftContent);
    const b = normalize(rightContent);
    if (a.join('\n') === b.join('\n')) return this.silent();
    if (flagged(parsed, 'q')) return this.outcome(`Files ${left} and ${right} differ`, '', 1);
    const rawLeft = toLines(leftContent);
    const rawRight = toLines(rightContent);
    const table = lcsMatrix(a, b);
    const hunk: string[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] === b[j]) { hunk.push(` ${rawLeft[i]}`); i += 1; j += 1; }
      else if (table[i + 1]![j]! >= table[i]![j + 1]!) { hunk.push(`-${rawLeft[i]}`); i += 1; }
      else { hunk.push(`+${rawRight[j]}`); j += 1; }
    }
    while (i < a.length) { hunk.push(`-${rawLeft[i]}`); i += 1; }
    while (j < b.length) { hunk.push(`+${rawRight[j]}`); j += 1; }
    if (flagged(parsed, 'u') || flagValue(parsed, 'U')) {
      return this.outcome([
        `--- ${left}`,
        `+++ ${right}`,
        `@@ -1,${rawLeft.length} +1,${rawRight.length} @@`,
        ...hunk,
      ].join('\n'), '', 1);
    }
    const removed = hunk.filter((line) => line.startsWith('-')).map((line) => `< ${line.slice(1)}`);
    const added = hunk.filter((line) => line.startsWith('+')).map((line) => `> ${line.slice(1)}`);
    return this.outcome([`${removed.length ? `1,${rawLeft.length}` : '0a'}${removed.length && added.length ? 'c' : removed.length ? 'd' : 'a'}${added.length ? `1,${rawRight.length}` : ''}`, ...removed, ...(removed.length && added.length ? ['---'] : []), ...added].join('\n'), '', 1);
  };

  private cmdXargs = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['0', 't', 'r'], value: ['n', 'I', 'd'], stopAtOperand: true, passthrough: true });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const command = parsed.operands.length ? parsed.operands : ['echo'];
    const delimiter = flagged(parsed, '0') ? '\0' : flagValue(parsed, 'd');
    const items = (delimiter ? context.stdin.split(delimiter) : context.stdin.split(/\s+/)).filter(Boolean);
    if (!items.length && flagged(parsed, 'r')) return this.silent();
    const placeholder = flagValue(parsed, 'I');
    const size = placeholder ? 1 : Number(flagValue(parsed, 'n') ?? items.length);
    const batches: string[][] = [];
    for (let index = 0; index < Math.max(items.length, 1); index += Math.max(1, size)) batches.push(items.slice(index, index + Math.max(1, size)));
    let stdout = '';
    let stderr = '';
    let exitCode = 0;
    for (const batch of batches) {
      const argv = placeholder ? command.map((token) => token.replaceAll(placeholder, batch[0] ?? '')) : [...command, ...batch];
      const outcome = await this.invoke(argv, '');
      stdout += outcome.stdout;
      stderr += outcome.stderr;
      exitCode = outcome.exitCode;
    }
    return { stdout, stderr, exitCode };
  };

  private cmdSeq = (context: CommandContext): CommandOutcome => {
    const numbers = context.args.filter((argument) => !argument.startsWith('-')).map(Number);
    if (!numbers.length || numbers.some((value) => !Number.isFinite(value))) return this.error('seq: invalid operand', 2);
    const [first, second, third] = numbers;
    const start = numbers.length === 1 ? 1 : first!;
    const step = numbers.length === 3 ? second! : 1;
    const end = numbers.length === 1 ? first! : numbers.length === 3 ? third! : second!;
    const values: string[] = [];
    for (let value = start; step > 0 ? value <= end : value >= end; value += step) values.push(String(value));
    return this.ok(values.join('\n'));
  };

  /* --------------------------------------------------------------- archives */

  private async createArchive(files: string[], base: string): Promise<string> {
    const entries: Array<{ path: string; kind: string; mode: number; content?: string }> = [];
    for (const file of files) {
      const target = this.resolvePath(file);
      const inode = this.deps.vfs.statSync(target);
      if (!inode) throw new Error(`${file}: No such file or directory`);
      const record = async (path: string): Promise<void> => {
        const current = this.deps.vfs.statSync(path)!;
        const relative = path.startsWith(`${base}/`) ? path.slice(base.length + 1) : path.split('/').at(-1)!;
        entries.push({ path: relative, kind: current.kind, mode: current.mode, ...(current.kind === 'file' ? { content: (await this.deps.vfs.readBytes(path)).toString('base64') } : {}) });
      };
      await record(target);
      if (inode.kind === 'directory') for (const entry of walkTree(this.deps.vfs, target, { includeHidden: true })) await record(entry.path);
    }
    return JSON.stringify({ seedArchive: 1, createdAt: this.now().toISOString(), entries }, null, 2);
  }

  private async extractArchive(content: string, destination: string): Promise<string[]> {
    const parsed = JSON.parse(content) as { seedArchive?: number; entries?: Array<{ path: string; kind: string; mode: number; content?: string }> };
    if (parsed.seedArchive !== 1 || !parsed.entries) throw new Error('not a seed archive');
    const written: string[] = [];
    for (const entry of parsed.entries) {
      const target = `${destination}/${entry.path}`;
      if (entry.kind === 'directory') await this.deps.vfs.mkdir(target);
      else await this.deps.vfs.writeFile(target, Buffer.from(entry.content ?? '', 'base64'));
      await this.deps.vfs.chmod(target, entry.mode);
      written.push(entry.path);
    }
    return written;
  }

  private cmdTar = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['c', 'x', 't', 'v', 'z', 'j', 'a'], value: ['f', 'C'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const archive = flagValue(parsed, 'f');
    if (!archive) return this.error('tar: no archive file specified (-f)', 2);
    const directory = this.resolvePath(flagValue(parsed, 'C') ?? '.');
    try {
      if (flagged(parsed, 'c')) {
        const payload = await this.createArchive(parsed.operands, directory);
        await this.deps.vfs.writeFile(this.resolvePath(archive), payload);
        return this.outcome(flagged(parsed, 'v') ? parsed.operands.join('\n') : '', '', 0);
      }
      const content = await this.deps.vfs.readFile(this.resolvePath(archive));
      if (flagged(parsed, 't')) {
        const listed = (JSON.parse(content) as { entries: Array<{ path: string }> }).entries.map((entry) => entry.path);
        return this.ok(listed.join('\n'));
      }
      if (flagged(parsed, 'x')) {
        const written = await this.extractArchive(content, directory);
        return this.outcome(flagged(parsed, 'v') ? written.join('\n') : '', '', 0);
      }
      return this.error('tar: one of -c, -x or -t is required', 2);
    } catch (error) { return this.error(`tar: ${(error as Error).message}`, 2); }
  };

  private cmdZip = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['r', 'q', 'v'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const [archive, ...files] = parsed.operands;
    if (!archive || !files.length) return this.error('zip: usage: zip [-r] archive.zip file ...', 2);
    try {
      const payload = await this.createArchive(files, this.cwd);
      await this.deps.vfs.writeFile(this.resolvePath(archive), payload);
      return this.ok(files.map((file) => `  adding: ${file}`).join('\n'));
    } catch (error) { return this.error(`zip: ${(error as Error).message}`); }
  };

  private cmdUnzip = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['l', 'o', 'q'], value: ['d'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const archive = parsed.operands[0];
    if (!archive) return this.error('unzip: usage: unzip archive.zip [-d directory]', 2);
    try {
      const content = await this.deps.vfs.readFile(this.resolvePath(archive));
      if (flagged(parsed, 'l')) {
        const entries = (JSON.parse(content) as { entries: Array<{ path: string }> }).entries;
        return this.ok([`Archive:  ${archive}`, '  Length      Name', '---------  ----', ...entries.map((entry) => `        -  ${entry.path}`)].join('\n'));
      }
      const written = await this.extractArchive(content, this.resolvePath(flagValue(parsed, 'd') ?? '.'));
      return this.ok([`Archive:  ${archive}`, ...written.map((path) => `  inflating: ${path}`)].join('\n'));
    } catch (error) { return this.error(`unzip: ${(error as Error).message}`); }
  };

  /* ----------------------------------------------------------------- system */

  private cmdPs = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['e', 'f', 'a', 'x', 'u', 'l', 'A'], passthrough: true });
    const full = flagged(parsed, 'f', 'l', 'u');
    const records = this.deps.processes.list();
    const clock = (ms: number): string => `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, '0')}`;
    if (!full) {
      return this.ok([`${'PID'.padStart(6)} ${'TTY'.padEnd(8)} ${'TIME'.padStart(8)} CMD`,
        ...records.map((record) => `${String(record.pid).padStart(6)} ${(record.sid === record.pid ? 'console' : 'ttys000').padEnd(8)} ${clock(record.cpuTimeMs).padStart(8)} ${record.executable}`)].join('\n'));
    }
    return this.ok([`${'UID'.padEnd(8)} ${'PID'.padStart(6)} ${'PPID'.padStart(6)} ${'PGID'.padStart(6)} ${'STAT'.padEnd(6)} ${'RSS'.padStart(8)} ${'TIME'.padStart(8)} CMD`,
      ...records.map((record) => `${(record.env.USER ?? 'root').padEnd(8)} ${String(record.pid).padStart(6)} ${String(record.ppid).padStart(6)} ${String(record.pgid).padStart(6)} ${record.runState.slice(0, 6).padEnd(6)} ${`${Math.round(record.memoryBytes / 1024)}K`.padStart(8)} ${clock(record.cpuTimeMs).padStart(8)} ${[record.executable, ...record.argv].join(' ')}`)].join('\n'));
  };

  private cmdTop = (): CommandOutcome => {
    const stats = this.deps.processes.stats();
    const records = this.deps.processes.list().sort((left, right) => right.cpuTimeMs - left.cpuTimeMs).slice(0, 15);
    const totalCpu = Math.max(1, stats.cpuTimeMs);
    const uptimeMinutes = Math.floor(this.ticks / 60);
    return this.ok([
      `Processes: ${stats.total} total, ${stats.running} running, ${stats.sleeping} sleeping, ${stats.stopped} stopped, ${stats.zombies} zombie`,
      `Load Avg: ${(stats.running / this.deps.spec.cpuCores).toFixed(2)}, ${(stats.total / 32).toFixed(2)}, ${(stats.total / 64).toFixed(2)}  Uptime: ${uptimeMinutes}m  ${this.formatDate(this.now(), '%T')}`,
      `MemRegions: ${stats.total * 24} total, ${humanSize(this.deps.spec.memoryBytes)} installed, ${humanSize(records.reduce((sum, record) => sum + record.memoryBytes, 0))} used`,
      '',
      `${'PID'.padStart(6)} ${'COMMAND'.padEnd(24)} ${'%CPU'.padStart(6)} ${'TIME'.padStart(8)} ${'#TH'.padStart(4)} ${'MEM'.padStart(8)} STATE`,
      ...records.map((record) => `${String(record.pid).padStart(6)} ${record.executable.slice(0, 24).padEnd(24)} ${((record.cpuTimeMs / totalCpu) * 100).toFixed(1).padStart(6)} ${`${Math.floor(record.cpuTimeMs / 1000)}s`.padStart(8)} ${String(record.threads).padStart(4)} ${humanSize(record.memoryBytes).padStart(8)} ${record.runState}`),
    ].join('\n'));
  };

  private signalFrom(args: string[]): { signal: SeedSignal; rest: string[] } {
    const rest: string[] = [];
    let signal: SeedSignal = 'SIGTERM';
    for (const argument of args) {
      const match = argument.match(/^-(?:s\s*)?([A-Z0-9]+)$/i) ?? argument.match(/^--signal=(.+)$/i);
      if (match) {
        const name = match[1]!.toUpperCase().replace(/^SIG/, '');
        const resolved = SIGNAL_ALIASES[name] ?? SIGNAL_ALIASES[match[1]!];
        if (resolved) { signal = resolved; continue; }
      }
      rest.push(argument);
    }
    return { signal, rest };
  }

  private cmdKill = (context: CommandContext): CommandOutcome => {
    if (context.args[0] === '-l') return this.ok(Object.values(SIGNAL_ALIASES).filter((value, index, all) => all.indexOf(value) === index).join(' '));
    const { signal, rest } = this.signalFrom(context.args);
    if (!rest.length) return this.error(`usage: ${context.name} [-s signal] pid | %job ...`, 2);
    const errors: string[] = [];
    for (const target of rest) {
      const job = target.startsWith('%') ? this.jobs.find((entry) => entry.id === Number(target.slice(1))) : undefined;
      const pid = job ? job.pid : Number(target);
      if (!Number.isFinite(pid)) { errors.push(`${context.name}: ${target}: arguments must be process or job IDs`); continue; }
      if (!this.deps.processes.kill(pid, signal)) errors.push(`${context.name}: ${pid}: no such process or operation not permitted`);
      else this.deps.network.unregisterServicesForProcess(this.deps.spec.id, pid);
    }
    return this.outcome('', errors.join('\n'), errors.length ? 1 : 0);
  };

  private cmdKillByName = (context: CommandContext): CommandOutcome => {
    const { signal, rest } = this.signalFrom(context.args);
    const parsed = parseFlags(rest, { boolean: ['i', 'f', 'e', 'l'], passthrough: true });
    if (!parsed.operands.length) return this.error(`usage: ${context.name} [-signal] name`, 2);
    const listOnly = context.name === 'pgrep';
    const matched: number[] = [];
    for (const pattern of parsed.operands) {
      const expression = new RegExp(pattern, flagged(parsed, 'i') ? 'i' : '');
      for (const record of this.deps.processes.list()) {
        if (!expression.test(record.executable)) continue;
        matched.push(record.pid);
        if (listOnly) continue;
        this.deps.processes.kill(record.pid, signal);
        this.deps.network.unregisterServicesForProcess(this.deps.spec.id, record.pid);
      }
    }
    if (!matched.length) return this.outcome('', listOnly ? '' : `${context.name}: no process found`, 1);
    return this.outcome(listOnly ? matched.join('\n') : '', '', 0);
  };

  private cmdUname = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['a', 's', 'n', 'r', 'v', 'm', 'o', 'p', 'i'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const kernel = KERNELS[this.deps.spec.os];
    const host = this.deps.spec.hostname;
    if (flagged(parsed, 'a')) return this.ok(`${kernel.sysname} ${host} ${kernel.release} ${kernel.version} ${kernel.machine} ${kernel.operating}`);
    const parts: string[] = [];
    if (flagged(parsed, 's') || !Object.keys(parsed.flags).length) parts.push(kernel.sysname);
    if (flagged(parsed, 'n')) parts.push(host);
    if (flagged(parsed, 'r')) parts.push(kernel.release);
    if (flagged(parsed, 'v')) parts.push(kernel.version);
    if (flagged(parsed, 'm', 'p', 'i')) parts.push(kernel.machine);
    if (flagged(parsed, 'o')) parts.push(kernel.operating);
    return this.ok(parts.join(' '));
  };

  private cmdHostname = (): CommandOutcome => this.ok(this.deps.spec.hostname);

  private cmdWhoami = (): CommandOutcome => this.ok(this.dialect === 'powershell' ? `${this.deps.spec.hostname.toLowerCase()}\\${this.variable('USER') ?? 'agent'}` : this.variable('USER') ?? 'agent');

  private cmdId = (): CommandOutcome => {
    const user = this.variable('USER') ?? 'agent';
    const uid = user === 'root' ? 0 : this.deps.spec.os === 'macos' ? 501 : 1000;
    const gid = user === 'root' ? 0 : this.deps.spec.os === 'macos' ? 20 : 1000;
    return this.ok(`uid=${uid}(${user}) gid=${gid}(${this.group(gid)}) groups=${gid}(${this.group(gid)})`);
  };

  private cmdUptime = (): CommandOutcome => {
    const stats = this.deps.processes.stats();
    const minutes = Math.floor(this.ticks / 60);
    return this.ok(`${this.formatDate(this.now(), '%T')} up ${minutes} min, 1 user, load averages: ${(stats.running / this.deps.spec.cpuCores).toFixed(2)} ${(stats.total / 32).toFixed(2)} ${(stats.total / 64).toFixed(2)}`);
  };

  private cmdDate = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['u', 'R', 'Iseconds'], value: ['d', 'r', 'Format', 'UFormat'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const format = parsed.operands.find((operand) => operand.startsWith('+'))?.slice(1) ?? flagValue(parsed, 'UFormat')?.replace(/^\+/, '');
    const custom = flagValue(parsed, 'Format');
    if (custom) return this.ok(this.formatDate(this.now(), custom.replace(/yyyy/g, '%Y').replace(/MM/g, '%m').replace(/dd/g, '%d').replace(/HH/g, '%H').replace(/mm/g, '%M').replace(/ss/g, '%S')));
    if (flagged(parsed, 'R')) return this.ok(`${this.formatDate(this.now(), '%a')}, ${this.formatDate(this.now(), '%d %b %Y %T')} +0000`);
    return this.ok(this.formatDate(this.now(), format));
  };

  /** Deterministic: the simulated clock advances, nothing actually blocks. */
  private cmdSleep = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { value: ['Seconds', 'Milliseconds'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const raw = flagValue(parsed, 'Seconds') ?? parsed.operands[0] ?? (flagValue(parsed, 'Milliseconds') ? String(Number(flagValue(parsed, 'Milliseconds')) / 1000) : undefined);
    if (raw === undefined) return this.error(`usage: ${context.name} seconds`, 2);
    const seconds = Number(raw.replace(/s$/, ''));
    if (!Number.isFinite(seconds)) return this.error(`sleep: invalid time interval '${raw}'`, 2);
    this.ticks += Math.max(0, Math.round(seconds));
    this.deps.processes.tick(Math.max(0, seconds) * 1000);
    return this.silent();
  };

  private cmdSudo = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['s', 'i', 'k', 'n'], value: ['u', 'g'], stopAtOperand: true, passthrough: true });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    if (!parsed.operands.length) return this.error('usage: sudo [-u user] command [args]', 1);
    const user = flagValue(parsed, 'u') ?? 'root';
    this.deps.onAction('shell.sudo', { user, command: parsed.operands.join(' ') });
    const previous = this.overlay;
    this.overlay = { ...previous, USER: user, HOME: user === 'root' ? '/root' : previous.HOME ?? this.variable('HOME') ?? '/', SUDO_USER: this.variable('USER') ?? 'agent' };
    try { return await this.invoke(parsed.operands, context.stdin); }
    finally { this.overlay = previous; }
  };

  private cmdClear = (): CommandOutcome => ({ stdout: 'c', stderr: '', exitCode: 0 });

  private cmdPrintenv = (context: CommandContext): CommandOutcome => {
    const environment = this.environment();
    if (!context.args.length) return this.ok(Object.entries(environment).sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => `${key}=${value}`).join('\n'));
    const values = context.args.map((name) => environment[name]);
    return values.some((value) => value === undefined) ? { stdout: '', stderr: '', exitCode: 1 } : this.ok(values.join('\n'));
  };

  /* ---------------------------------------------------------------- network */

  private cmdIfconfig = (context: CommandContext): CommandOutcome => {
    const summary = this.deps.network.interfaceSummary(this.deps.spec.id);
    if (context.name !== 'ip') return this.ok(summary);
    const [subcommand = 'addr'] = context.args;
    if (['addr', 'a', 'address', 'link', 'l'].includes(subcommand)) return this.ok(summary);
    if (['route', 'r'].includes(subcommand)) {
      return this.ok(this.deps.network.listRoutes(this.deps.spec.id).map((route) => `${route.destination === '0.0.0.0/0' ? 'default' : route.destination}${route.via ? ` via ${route.via}` : ''} dev ${route.dev} proto ${route.source} metric ${route.metric}`).join('\n'));
    }
    if (['neigh', 'n', 'neighbour', 'neighbor'].includes(subcommand)) {
      return this.ok(this.deps.network.listNeighbors(this.deps.spec.id).map((entry) => `${entry.address} dev ${entry.dev} lladdr ${entry.mac} ${entry.state}`).join('\n'));
    }
    return this.error(`ip: unknown object "${subcommand}"`, 2);
  };

  private cmdArp = (): CommandOutcome => this.ok(this.deps.network.listNeighbors(this.deps.spec.id).map((entry) => `? (${entry.address}) at ${entry.mac} on ${entry.dev} ${entry.state.toLowerCase()} [${entry.resolvedBy}]`).join('\n'));

  private cmdPing = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['n'], value: ['c', 'Count', 'i', 'W', 't'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const host = parsed.operands.at(-1);
    if (!host) return this.error('usage: ping [-c count] host', 2);
    const count = Number(flagValue(parsed, 'c', 'Count') ?? 1);
    const output = this.deps.network.ping(this.deps.spec.id, host, count);
    return output.includes('cannot resolve') ? this.outcome(output, '', 2) : this.ok(output);
  };

  private cmdResolve = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['short', 'x'], value: ['type', 'Type', 'Name', 'Server'], passthrough: true });
    const host = flagValue(parsed, 'Name') ?? parsed.operands.find((operand) => !operand.startsWith('@') && !operand.startsWith('-'));
    if (!host) return this.error(`usage: ${context.name} name`, 2);
    const resolution = this.deps.network.resolveDetailed(host, { computerId: this.deps.spec.id });
    if (resolution.status !== 'ok' || !resolution.addresses.length) {
      return this.outcome('', `** server can't find ${host}: ${resolution.status.toUpperCase()}`, 1);
    }
    if (context.name === 'host') return this.ok(resolution.addresses.map((address) => `${host} has address ${address}`).join('\n'));
    if (context.name === 'dig') {
      return this.ok([
        `; <<>> Seed DiG 9.20 <<>> ${host}`,
        ';; global options: +cmd',
        ';; Got answer:',
        `;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: ${resolution.chain.length + resolution.addresses.length}`,
        '',
        ';; QUESTION SECTION:',
        `;${host}.\t\t\tIN\tA`,
        '',
        ';; ANSWER SECTION:',
        ...resolution.records.map((record) => `${record.name}.\t\t${record.ttl}\tIN\t${record.type}\t${record.value}`),
        '',
        `;; SERVER: ${this.deps.network.networkConfig().dns}#53(${this.deps.network.networkConfig().dns})`,
      ].join('\n'));
    }
    return this.ok([
      `Server:  dns.${this.deps.network.networkConfig().domain}`,
      `Address: ${this.deps.network.networkConfig().dns}`,
      '',
      ...(resolution.chain.length > 1 ? [`Aliases: ${resolution.chain.slice(1).join(', ')}`] : []),
      `Name:    ${host}`,
      ...resolution.addresses.map((address) => `Address: ${address}`),
    ].join('\n'));
  };

  private cmdCurl = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, {
      boolean: ['i', 'I', 's', 'S', 'L', 'k', 'v', 'f', 'UseBasicParsing', 'q'],
      value: ['X', 'd', 'H', 'o', 'O', 'A', 'Method', 'Body', 'Uri', 'OutFile', 'm'],
      passthrough: true,
    });
    const url = flagValue(parsed, 'Uri') ?? parsed.operands.find((operand) => /^(https?:\/\/|[\w.-]+(?::\d+)?(?:\/|$))/.test(operand));
    if (!url) return this.error(`${context.name}: no URL specified`, 2);
    const method = (flagValue(parsed, 'X', 'Method') ?? (flagValue(parsed, 'd', 'Body') ? 'POST' : 'GET')).toUpperCase();
    const header = flagValue(parsed, 'H');
    const headers = header ? Object.fromEntries([header.split(/:\s*/, 2) as [string, string]]) : undefined;
    try {
      const response = await this.deps.network.request(this.deps.spec.id, url, method, flagValue(parsed, 'd', 'Body'), {
        ...(headers ? { headers } : {}),
        ...(flagged(parsed, 'L') ? { maxRedirects: 5 } : {}),
      });
      const headerText = `HTTP/1.1 ${response.status}\n${Object.entries(response.headers).map(([key, value]) => `${key}: ${value}`).join('\n')}`;
      const body = flagged(parsed, 'I') ? headerText : flagged(parsed, 'i') ? `${headerText}\n\n${response.body}` : response.body;
      const output = flagValue(parsed, 'o', 'OutFile');
      if (output) { await this.writeStream(this.resolvePath(output), terminated(body), false); return this.silent(); }
      if (flagged(parsed, 'f') && response.status >= 400) return this.error(`${context.name}: The requested URL returned error: ${response.status}`, 22);
      return { stdout: body, stderr: '', exitCode: 0 };
    } catch (error) {
      return this.error(`${context.name}: ${(error as Error).message}`);
    }
  };

  private cmdWget = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['q', 'S', 'spider', 'c', 'nv'], value: ['O', 'P', 'T', 'output-document'], passthrough: true });
    const url = parsed.operands.find((operand) => /^(https?:\/\/|[\w.-]+(?::\d+)?\/)/.test(operand));
    if (!url) return this.error('wget: missing URL', 2);
    try {
      const response = await this.deps.network.request(this.deps.spec.id, url);
      const target = flagValue(parsed, 'O', 'output-document') ?? url.split('/').filter(Boolean).at(-1) ?? 'index.html';
      if (target === '-') return { stdout: response.body, stderr: '', exitCode: 0 };
      const directory = flagValue(parsed, 'P');
      const destination = this.resolvePath(directory ? `${directory}/${target}` : target);
      if (!flagged(parsed, 'spider')) await this.writeStream(destination, terminated(response.body), false);
      return this.outcome('', [
        `--${this.formatDate(this.now(), '%F %T')}--  ${url}`,
        `HTTP request sent, awaiting response... ${response.status} ${response.status === 200 ? 'OK' : ''}`.trimEnd(),
        `Length: ${Buffer.byteLength(response.body)} [${response.headers['content-type'] ?? 'application/octet-stream'}]`,
        `Saving to: '${target}'`,
        '',
        `'${target}' saved [${Buffer.byteLength(response.body)}]`,
      ].join('\n'), 0);
    } catch (error) { return this.error(`wget: ${(error as Error).message}`, 4); }
  };

  private cmdNetstat = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['a', 'n', 'l', 't', 'u', 'p', 'r', 'i'], passthrough: true });
    if (flagged(parsed, 'r')) {
      return this.ok(['Destination        Gateway            Flags  Netif', ...this.deps.network.listRoutes(this.deps.spec.id).map((route) => `${(route.destination === '0.0.0.0/0' ? 'default' : route.destination).padEnd(18)} ${(route.via ?? 'link#1').padEnd(18)} ${(route.via ? 'UG' : 'U').padEnd(6)} ${route.dev}`)].join('\n'));
    }
    const sockets = this.deps.network.listSockets(this.deps.spec.id).filter((socket) => (flagged(parsed, 'l') ? socket.state === 'LISTEN' : true));
    return this.ok([
      `${'Proto'.padEnd(6)} ${'Local Address'.padEnd(24)} ${'Foreign Address'.padEnd(24)} State`,
      ...sockets.map((socket) => `${socket.protocol.toUpperCase().padEnd(6)} ${`${socket.localAddress}:${socket.localPort}`.padEnd(24)} ${`${socket.remoteAddress ?? '*'}:${socket.remotePort ?? '*'}`.padEnd(24)} ${socket.state}`),
    ].join('\n'));
  };

  private cmdServe = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { value: ['p', 'host'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const port = Number(flagValue(parsed, 'p') ?? parsed.operands[0] ?? 8080);
    if (!Number.isFinite(port)) return this.error(`serve: invalid port: ${parsed.operands[0]}`, 2);
    const target = this.resolvePath(parsed.operands[1] ?? '.');
    if (!this.deps.vfs.statSync(target)) return this.error(`serve: ${parsed.operands[1] ?? '.'}: No such file or directory`);
    const hostname = flagValue(parsed, 'host') ?? parsed.operands[2] ?? `${this.deps.spec.hostname}.${this.deps.network.networkConfig().domain}`;
    const record = this.deps.processes.spawn({
      executable: 'seed-httpd', argv: [String(port), target], cwd: this.cwd, env: this.environment(),
      ppid: this.shellPid, listeningPorts: [port], memoryBytes: 6 * 1024 * 1024,
    });
    try {
      this.deps.network.registerService({
        id: `httpd-${record.pid}`, computerId: this.deps.spec.id, host: hostname, port, protocol: 'http', pid: record.pid,
        handle: async (requestPath) => {
          try {
            const resolved = this.deps.vfs.statSync(target)?.kind === 'directory' ? `${target}/${requestPath === '/' ? 'index.html' : requestPath.replace(/^\//, '')}` : target;
            return { status: 200, headers: { 'content-type': resolved.endsWith('.html') ? 'text/html' : 'text/plain', server: 'seed-httpd/1.0' }, body: await this.deps.vfs.readFile(resolved) };
          } catch { return { status: 404, headers: { 'content-type': 'text/plain', server: 'seed-httpd/1.0' }, body: '404 not found' }; }
        },
      });
    } catch (error) {
      this.deps.processes.kill(record.pid, 'SIGKILL');
      this.deps.processes.wait(this.shellPid, { pid: record.pid });
      return this.error(`serve: cannot bind ${hostname}:${port}: ${(error as Error).message}`);
    }
    return this.ok(`serving ${this.display(target)} at http://${hostname}:${port} (pid ${record.pid})`);
  };

  /* -------------------------------------------------------------- ecosystem */

  private cmdApps = (): CommandOutcome => this.ok(this.deps.listApps().map((item) => `${item.id.padEnd(16)} ${item.version.padEnd(10)} ${item.name}`).join('\n'));

  private cmdStore = async (context: CommandContext): Promise<CommandOutcome> => {
    const [verb, appId] = context.args;
    if (verb === 'install' && appId) {
      const installed = await this.deps.install(appId);
      return this.ok(`installed ${installed.name} ${installed.version} → ${this.display(installed.installPath)}`);
    }
    if (verb && verb !== 'list') return this.error(`store: unknown subcommand '${verb}'`, 2);
    return this.ok(this.deps.catalog().filter((item) => item.supportedOS.includes(this.deps.spec.os)).map((item) => `${item.id.padEnd(16)} ${item.name.padEnd(24)} ${item.publisher}`).join('\n'));
  };

  private cmdGateway = (): CommandOutcome => this.ok(this.deps.network.gateways.map((rule) => `${rule.enabled ? 'ALLOW' : 'DENY '} ${rule.name}: ${rule.protocols.join(',')} ${rule.hostnames.join(',')} ports=${rule.ports === '*' ? '*' : rule.ports.join(',')}`).join('\n'));

  private cmdGit = async (context: CommandContext): Promise<CommandOutcome> => {
    try { return this.ok(await this.deps.software.gitCommand(context.args, this.cwd)); }
    catch (error) { return this.error(`${(error as Error).message}`, 128); }
  };

  private cmdPackageManager = async (context: CommandContext): Promise<CommandOutcome> => {
    try { return this.ok(await this.deps.software.packageCommand(context.name, context.args, this.cwd)); }
    catch (error) { return this.error((error as Error).message); }
  };

  private cmdPager = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['N', 'R', 'S', 'F', 'e'], passthrough: true });
    const { streams, errors } = await this.inputs(context.name, parsed.operands, context.stdin);
    return { stdout: streams.map((stream) => stream.content).join(''), stderr: terminated(errors.join('\n')), exitCode: errors.length ? 1 : 0 };
  };

  /* ------------------------------------------------------------- powershell */

  private cmdGetChildItem = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['Force', 'Recurse', 'Directory', 'File', 'Name'], value: ['Path', 'Filter', 'Include', 'Exclude', 'Depth'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const target = this.resolvePath(flagValue(parsed, 'Path') ?? parsed.operands[0] ?? '.');
    const inode = this.deps.vfs.statSync(target);
    if (!inode) return this.error(`Get-ChildItem : Cannot find path '${flagValue(parsed, 'Path') ?? parsed.operands[0] ?? '.'}' because it does not exist.`);
    const entries = inode.kind === 'directory'
      ? (flagged(parsed, 'Recurse')
        ? walkTree(this.deps.vfs, target, { includeHidden: flagged(parsed, 'Force'), maxDepth: flagValue(parsed, 'Depth') ? Number(flagValue(parsed, 'Depth')) + 1 : undefined }).map((entry) => ({ name: entry.name, path: entry.path, inode: entry.inode }))
        : this.deps.vfs.list(target).filter((entry) => flagged(parsed, 'Force') || !entry.name.startsWith('.')))
      : [{ name: target.split('/').at(-1)!, path: target, inode }];
    const filter = flagValue(parsed, 'Filter', 'Include');
    const expression = filter ? globToRegExp(filter, false) : undefined;
    const visible = entries
      .filter((entry) => (expression ? expression.test(entry.name) : true))
      .filter((entry) => (flagged(parsed, 'Directory') ? entry.inode.kind === 'directory' : flagged(parsed, 'File') ? entry.inode.kind === 'file' : true));
    if (flagged(parsed, 'Name')) return this.ok(visible.map((entry) => entry.name).join('\n'));
    return this.ok([
      '',
      `    Directory: ${this.display(target)}`,
      '',
      'Mode                 LastWriteTime         Length Name',
      '----                 -------------         ------ ----',
      ...visible.map((entry) => {
        const mode = `${entry.inode.kind === 'directory' ? 'd' : '-'}${entry.name.startsWith('.') ? 'h' : '-'}---${(entry.inode.mode & 0o111) !== 0 ? 'a' : '-'}`;
        const written = new Date(entry.inode.modifiedAt);
        const timestamp = `${written.getUTCFullYear()}-${String(written.getUTCMonth() + 1).padStart(2, '0')}-${String(written.getUTCDate()).padStart(2, '0')} ${String(written.getUTCHours()).padStart(2, '0')}:${String(written.getUTCMinutes()).padStart(2, '0')}`;
        return `${mode.padEnd(20)} ${timestamp.padEnd(20)} ${String(entry.inode.kind === 'directory' ? '' : entry.inode.size).padStart(6)} ${entry.name}`;
      }),
    ].join('\n'));
  };

  private cmdGetContent = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['Raw', 'Force'], value: ['Path', 'TotalCount', 'Tail', 'Head', 'Encoding'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const operands = [...(flagValue(parsed, 'Path') ? [flagValue(parsed, 'Path')!] : []), ...parsed.operands];
    const { streams, errors } = await this.inputs(context.name, operands, context.stdin);
    let body = streams.map((stream) => stream.content).join('');
    const head = flagValue(parsed, 'TotalCount', 'Head');
    const tail = flagValue(parsed, 'Tail');
    if (head) body = terminated(toLines(body).slice(0, Number(head)).join('\n'));
    if (tail) body = terminated(toLines(body).slice(-Number(tail)).join('\n'));
    return { stdout: body, stderr: terminated(errors.map((message) => message.replace(context.name, 'Get-Content')).join('\n')), exitCode: errors.length ? 1 : 0 };
  };

  private cmdSetContent = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['Force', 'NoNewline', 'Append'], value: ['Path', 'Value', 'FilePath', 'Encoding'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const target = flagValue(parsed, 'Path', 'FilePath') ?? parsed.operands[0];
    if (!target) return this.error(`${context.name} : Cannot bind argument to parameter 'Path' because it is null.`, 2);
    const value = flagValue(parsed, 'Value') ?? (parsed.operands.length > 1 ? parsed.operands.slice(1).join('\n') : context.stdin);
    const append = flagged(parsed, 'Append') || context.name.toLowerCase() === 'add-content';
    await this.writeStream(this.resolvePath(target), terminated(value), append);
    return this.silent();
  };

  private cmdWriteOutput = (context: CommandContext): CommandOutcome => this.ok(context.args.join('\n'));

  private cmdSelectString = async (context: CommandContext): Promise<CommandOutcome> => {
    const parsed = parseFlags(context.args, { boolean: ['CaseSensitive', 'SimpleMatch', 'NotMatch', 'Quiet', 'List'], value: ['Pattern', 'Path'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const pattern = flagValue(parsed, 'Pattern') ?? parsed.operands.shift();
    if (!pattern) return this.error('Select-String : Cannot bind argument to parameter \'Pattern\' because it is null.', 2);
    const body = flagged(parsed, 'SimpleMatch') ? pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : pattern;
    const expression = new RegExp(body, flagged(parsed, 'CaseSensitive') ? '' : 'i');
    const operands = [...(flagValue(parsed, 'Path') ? [flagValue(parsed, 'Path')!] : []), ...parsed.operands];
    const { streams, errors } = await this.inputs(context.name, operands, context.stdin);
    const lines: string[] = [];
    for (const stream of streams) {
      toLines(stream.content).forEach((line, index) => {
        if (expression.test(line) === flagged(parsed, 'NotMatch')) return;
        lines.push(stream.name === '-' ? line : `${stream.name}:${index + 1}:${line}`);
      });
    }
    if (flagged(parsed, 'Quiet')) return { stdout: '', stderr: '', exitCode: lines.length ? 0 : 1 };
    return this.outcome(lines.join('\n'), errors.join('\n'), lines.length ? 0 : 1);
  };

  private cmdFindstr = async (context: CommandContext): Promise<CommandOutcome> => {
    const flags = context.args.filter((argument) => argument.startsWith('/'));
    const operands = context.args.filter((argument) => !argument.startsWith('/'));
    const literal = flags.some((flag) => /^\/C:/i.test(flag));
    const pattern = literal ? flags.find((flag) => /^\/C:/i.test(flag))!.slice(3).replace(/^"|"$/g, '') : operands.shift();
    if (pattern === undefined) return this.error('FINDSTR: Bad command line', 2);
    const insensitive = flags.some((flag) => /^\/i$/i.test(flag));
    const invert = flags.some((flag) => /^\/v$/i.test(flag));
    const numbered = flags.some((flag) => /^\/n$/i.test(flag));
    const escaped = literal ? pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : pattern;
    let expression: RegExp;
    try { expression = new RegExp(escaped, insensitive ? 'i' : ''); }
    catch { return this.error('FINDSTR: Bad regular expression', 2); }
    const { streams, errors } = await this.inputs('findstr', operands, context.stdin);
    const lines: string[] = [];
    for (const stream of streams) {
      toLines(stream.content).forEach((line, index) => {
        if (expression.test(line) === invert) return;
        lines.push(`${streams.length > 1 ? `${stream.name}:` : ''}${numbered ? `${index + 1}:` : ''}${line}`);
      });
    }
    return this.outcome(lines.join('\n'), errors.join('\n'), lines.length ? 0 : 1);
  };

  private cmdMeasureObject = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['Line', 'Word', 'Character', 'Sum', 'Average', 'Maximum', 'Minimum'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const lines = toLines(context.stdin);
    const words = context.stdin.split(/\s+/).filter(Boolean).length;
    const characters = context.stdin.length;
    const rows: string[] = ['', `Count    : ${lines.length}`];
    if (flagged(parsed, 'Word')) rows.push(`Words    : ${words}`);
    if (flagged(parsed, 'Character')) rows.push(`Characters : ${characters}`);
    return this.ok(rows.join('\n'));
  };

  private cmdSelectObject = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { boolean: ['Unique', 'Last'], value: ['First', 'Skip', 'Property'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    let lines = toLines(context.stdin);
    const skip = flagValue(parsed, 'Skip');
    if (skip) lines = lines.slice(Number(skip));
    const first = flagValue(parsed, 'First');
    if (first) lines = lines.slice(0, Number(first));
    if (flagged(parsed, 'Unique')) lines = [...new Set(lines)];
    return this.ok(lines.join('\n'));
  };

  private cmdWhereObject = (context: CommandContext): CommandOutcome => {
    const source = context.args.join(' ').replace(/^\{|\}$/g, '').trim();
    const match = source.match(/^\$_\s+-(match|notmatch|like|eq|ne|contains)\s+(.+)$/i);
    if (!match) return this.error("Where-Object : only '$_ -match/-notmatch/-like/-eq/-ne <value>' filters are implemented in this simulation", 2);
    const [, operator, rawValue] = match;
    const value = rawValue!.trim().replace(/^['"]|['"]$/g, '');
    const lines = toLines(context.stdin).filter((line) => {
      switch (operator!.toLowerCase()) {
        case 'match': return new RegExp(value, 'i').test(line);
        case 'notmatch': return !new RegExp(value, 'i').test(line);
        case 'like': return globToRegExp(value, false).test(line);
        case 'eq': return line === value;
        case 'ne': return line !== value;
        default: return line.includes(value);
      }
    });
    return this.ok(lines.join('\n'));
  };

  private cmdForEachObject = (): CommandOutcome => this.error('ForEach-Object : script-block object pipelines are not implemented in this simulation', 2);

  private cmdTestPath = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { value: ['Path', 'PathType'] });
    const invalid = this.optionError(context.name, parsed);
    if (invalid) return invalid;
    const target = flagValue(parsed, 'Path') ?? parsed.operands[0];
    if (!target) return this.error('Test-Path : Cannot bind argument to parameter \'Path\'.', 2);
    const inode = this.deps.vfs.statSync(this.resolvePath(target));
    const kind = flagValue(parsed, 'PathType')?.toLowerCase();
    const value = Boolean(inode) && (kind === 'container' ? inode!.kind === 'directory' : kind === 'leaf' ? inode!.kind === 'file' : true);
    return this.ok(value ? 'True' : 'False');
  };

  private cmdGetCommand = async (context: CommandContext): Promise<CommandOutcome> => {
    const names = context.args.filter((argument) => !argument.startsWith('-'));
    const rows: string[] = ['', 'CommandType     Name                                               Version    Source', '-----------     ----                                               -------    ------'];
    let exitCode = 0;
    for (const name of names.length ? names : [...this.commands.keys()].sort()) {
      const spec = this.lookup(name);
      const external = await this.resolveExecutable(name);
      if (external) rows.push(`Application     ${name.padEnd(50)} ${external.descriptor.version.padEnd(10)} ${external.path}`);
      else if (spec) rows.push(`Cmdlet          ${name.padEnd(50)} ${'1.0.0'.padEnd(10)} Seed.Shell`);
      else { rows.push(`Get-Command : The term '${name}' is not recognized.`); exitCode = 1; }
    }
    return this.outcome(rows.join('\n'), '', exitCode);
  };

  private cmdGetProcess = (context: CommandContext): CommandOutcome => {
    const filter = context.args.filter((argument) => !argument.startsWith('-') && !argument.startsWith('/'));
    const records = this.deps.processes.list().filter((record) => (filter.length ? filter.some((name) => record.executable.toLowerCase().includes(name.toLowerCase())) : true));
    return this.ok([
      '',
      ' NPM(K)    PM(M)      WS(M)     CPU(s)      Id  SI ProcessName',
      ' ------    -----      -----     ------      --  -- -----------',
      ...records.map((record) => `${String(Math.round(record.memoryBytes / 1024 / 32)).padStart(7)} ${(record.memoryBytes / 1024 / 1024).toFixed(2).padStart(8)} ${(record.memoryBytes / 1024 / 1024).toFixed(2).padStart(10)} ${(record.cpuTimeMs / 1000).toFixed(2).padStart(10)} ${String(record.pid).padStart(7)} ${String(record.sid).padStart(3)} ${record.executable}`),
    ].join('\n'));
  };

  private cmdTaskkill = (context: CommandContext): CommandOutcome => {
    const pidFlag = context.args.findIndex((argument) => /^\/pid$/i.test(argument));
    const imageFlag = context.args.findIndex((argument) => /^\/im$/i.test(argument));
    if (pidFlag >= 0) {
      const pid = Number(context.args[pidFlag + 1]);
      if (!this.deps.processes.kill(pid, context.args.some((argument) => /^\/f$/i.test(argument)) ? 'SIGKILL' : 'SIGTERM')) {
        return this.error(`ERROR: The process "${pid}" not found.`, 128);
      }
      this.deps.network.unregisterServicesForProcess(this.deps.spec.id, pid);
      return this.ok(`SUCCESS: Sent termination signal to the process with PID ${pid}.`);
    }
    if (imageFlag >= 0) return this.cmdKillByName({ ...context, name: 'taskkill', args: [context.args[imageFlag + 1] ?? ''] });
    return this.error('ERROR: Invalid syntax. Use /PID or /IM.', 1);
  };

  private cmdVer = (): CommandOutcome => this.ok(`\nSeed Microsoft Windows [Version ${KERNELS.windows.release}]`);

  private cmdSystemInfo = (): CommandOutcome => this.ok([
    `Host Name:                 ${this.deps.spec.hostname.toUpperCase()}`,
    `OS Name:                   Seed Microsoft Windows 11 Pro`,
    `OS Version:                ${KERNELS.windows.release}`,
    `System Type:               ${KERNELS.windows.machine}-based PC`,
    `Processor(s):              ${this.deps.spec.cpuCores} Core(s)`,
    `Total Physical Memory:     ${humanSize(this.deps.spec.memoryBytes)}`,
    `Network Card(s):           ${this.deps.spec.ipv4}`,
  ].join('\n'));

  private cmdGetVolume = (): CommandOutcome => {
    const usage = this.deps.vfs.usage();
    return this.ok([
      '',
      'DriveLetter FileSystem  FileSystemLabel      Size Remaining        Size',
      '----------- ----------  ---------------      -------------        ----',
      ...this.deps.spec.disks.map((disk, index) => `${disk.mount.replace(/[:\\/]/g, '').padEnd(11)} ${'NTFS'.padEnd(11)} ${disk.label.padEnd(20)} ${humanSize(disk.capacityBytes - (index === 0 ? usage.bytes : 0)).padStart(13)} ${humanSize(disk.capacityBytes).padStart(11)}`),
    ].join('\n'));
  };

  private cmdGetVariable = (context: CommandContext): CommandOutcome => {
    const names = context.args.filter((argument) => !argument.startsWith('-'));
    const entries = [...this.vars].filter(([key]) => (names.length ? names.some((name) => key.toLowerCase() === name.toLowerCase().replace(/^\$?env:/, '')) : true));
    return this.ok(['', 'Name                           Value', '----                           -----', ...entries.map(([key, value]) => `${key.padEnd(30)} ${value}`)].join('\n'));
  };

  private cmdTestNetConnection = (context: CommandContext): CommandOutcome => {
    const parsed = parseFlags(context.args, { value: ['ComputerName', 'Port'], passthrough: true });
    const host = flagValue(parsed, 'ComputerName') ?? parsed.operands[0];
    if (!host) return this.error('Test-NetConnection : a ComputerName is required', 2);
    const address = this.deps.network.resolve(host, this.deps.spec.id);
    if (!address) return this.error(`Test-NetConnection : Name resolution of ${host} failed`, 1);
    const port = flagValue(parsed, 'Port');
    const sockets = this.deps.network.listSockets().filter((socket) => socket.localAddress === address && socket.state === 'LISTEN');
    return this.ok([
      `ComputerName     : ${host}`,
      `RemoteAddress    : ${address}`,
      ...(port ? [`RemotePort       : ${port}`, `TcpTestSucceeded : ${sockets.some((socket) => socket.localPort === Number(port)) ? 'True' : 'False'}`] : []),
      `PingSucceeded    : ${this.deps.network.ping(this.deps.spec.id, host).includes('bytes from') ? 'True' : 'False'}`,
    ].join('\n'));
  };

  /* ------------------------------------------------------------- registration */

  private registerPosix(add: (names: string[], summary: string, run: CommandSpec['run'], options?: { builtin?: boolean; usage?: string }) => void): void {
    add(['cd'], 'change the working directory', this.cmdCd, { builtin: true, usage: 'cd [directory|-]' });
    add(['pwd'], 'print the working directory', this.cmdPwd, { builtin: true });
    add(['export'], 'mark shell variables for export to commands', this.cmdExport, { builtin: true, usage: 'export NAME=value' });
    add(['unset'], 'remove shell variables', this.cmdUnset, { builtin: true });
    add(['set'], 'list shell variables', this.cmdSet, { builtin: true });
    add(['env'], 'print the exported environment', this.cmdEnv, { builtin: true });
    add(['printenv'], 'print environment variables', this.cmdPrintenv);
    add(['alias'], 'define or list command aliases', this.cmdAlias, { builtin: true });
    add(['unalias'], 'remove command aliases', this.cmdUnalias, { builtin: true });
    add(['source', '.'], 'execute a script in the current shell', this.cmdSource, { builtin: true });
    add(['exit'], 'exit the shell with a status', this.cmdExit, { builtin: true });
    add(['history'], 'show the command history', this.cmdHistory, { builtin: true });
    add(['jobs'], 'list background jobs', this.cmdJobs, { builtin: true });
    add(['wait'], 'wait for a background job and reap it', this.cmdWait, { builtin: true });
    add(['true'], 'return success', this.cmdTrue, { builtin: true });
    add(['false'], 'return failure', this.cmdFalse, { builtin: true });
    add(['test', '['], 'evaluate a conditional expression', this.cmdTest, { builtin: true, usage: 'test expression | [ expression ]' });
    add(['eval'], 'evaluate arguments as a command line', this.cmdEval, { builtin: true });
    add(['type'], 'describe how a name would be resolved', this.cmdType, { builtin: true });
    add(['which'], 'locate a command on PATH', this.cmdWhich);
    add(['help'], 'list the commands this shell implements', this.cmdHelp, { builtin: true });
    add(['man'], 'display a manual page', this.cmdMan);

    add(['ls'], 'list directory contents', this.cmdLs, { usage: 'ls [-alh1RtSrdF] [file ...]' });
    add(['cat'], 'concatenate and print files', this.cmdCat);
    add(['echo'], 'write arguments to standard output', this.cmdEcho, { usage: 'echo [-neE] [string ...]' });
    add(['printf'], 'format and print data', this.cmdPrintf);
    add(['mkdir'], 'create directories', this.cmdMkdir, { usage: 'mkdir [-pv] [-m mode] directory ...' });
    add(['rmdir'], 'remove empty directories', this.cmdRmdir);
    add(['touch'], 'create a file or update its modification time', this.cmdTouch, { usage: 'touch [-c] file ...' });
    add(['rm'], 'remove files and directories', this.cmdRm, { usage: 'rm [-rRf] file ...' });
    add(['mv'], 'move or rename files', this.cmdMv);
    add(['cp'], 'copy files and directories', this.cmdCp, { usage: 'cp [-r] source ... destination' });
    add(['ln'], 'create hard and symbolic links', this.cmdLn);
    add(['find'], 'walk a file hierarchy', this.cmdFind, { usage: 'find path ... [-name glob] [-type f|d|l] [-maxdepth n] [-exec cmd {} ;]' });
    add(['stat'], 'display file status', this.cmdStat);
    add(['file'], 'identify file type', this.cmdFile);
    add(['chmod'], 'change file mode bits', this.cmdChmod);
    add(['chown'], 'change file owner and group', this.cmdChown);
    add(['df'], 'report filesystem usage', this.cmdDf);
    add(['du'], 'summarize disk usage', this.cmdDu);
    add(['basename'], 'strip directory from a path', this.cmdBasename);
    add(['dirname'], 'strip the last path component', this.cmdDirname);
    add(['realpath'], 'resolve a path to its canonical form', this.cmdRealpath);
    add(['readlink'], 'print a symbolic link target', this.cmdReadlink);
    add(['tree'], 'list a directory as a tree', async (context) => {
      const external = await this.resolveExecutable('tree');
      return external ? this.runExternal(external, context.argv, context.stdin, context.pid) : this.error('tree: not installed — install it with a package manager first', 127);
    });

    add(['grep', 'egrep', 'fgrep'], 'search files with regular expressions', this.cmdGrep, { usage: 'grep [-invcrlEFwxq] pattern [file ...]' });
    add(['head'], 'print the first lines of a file', this.cmdHead);
    add(['tail'], 'print the last lines of a file', this.cmdTail);
    add(['wc'], 'count lines, words and bytes', this.cmdWc, { usage: 'wc [-lwc] [file ...]' });
    add(['sort'], 'sort lines of text', this.cmdSort);
    add(['uniq'], 'report or filter repeated lines', this.cmdUniq);
    add(['cut'], 'select fields or characters from lines', this.cmdCut);
    add(['tr'], 'translate or delete characters', this.cmdTr);
    add(['tee'], 'copy standard input to files and stdout', this.cmdTee);
    add(['sed'], 'stream editor for substitution and deletion', this.cmdSed, { usage: "sed [-n] [-i] 's/pattern/replacement/flags' [file ...]" });
    add(['awk'], 'pattern-directed scanning and processing', this.cmdAwk, { usage: "awk [-F sep] 'pattern { print $1 }' [file ...]" });
    add(['diff'], 'compare files line by line', this.cmdDiff);
    add(['xargs'], 'build and execute command lines from input', this.cmdXargs);
    add(['seq'], 'print a sequence of numbers', this.cmdSeq);
    add(['less', 'more'], 'page through text (non-interactive here)', this.cmdPager);
    add(['tar'], 'create and extract archives', this.cmdTar, { usage: 'tar -c|-x|-t -f archive [-C directory] [file ...]' });
    add(['zip'], 'package files into an archive', this.cmdZip);
    add(['unzip'], 'extract files from an archive', this.cmdUnzip);

    add(['ps'], 'report process status', this.cmdPs);
    add(['top'], 'display process activity', this.cmdTop);
    add(['kill'], 'send a signal to a process', this.cmdKill, { usage: 'kill [-SIGNAL] pid | %job' });
    add(['killall', 'pkill', 'pgrep'], 'signal or list processes by name', this.cmdKillByName);
    add(['uname'], 'print system information', this.cmdUname);
    add(['arch'], 'print the machine architecture', () => this.ok(KERNELS[this.deps.spec.os].machine));
    add(['hostname'], 'print the hostname', this.cmdHostname);
    add(['whoami'], 'print the effective user', this.cmdWhoami);
    add(['id'], 'print user and group identity', this.cmdId);
    add(['uptime'], 'show how long the system has been running', this.cmdUptime);
    add(['date'], 'print the simulated date and time', this.cmdDate, { usage: 'date [+FORMAT]' });
    add(['sleep'], 'advance the simulated clock', this.cmdSleep);
    add(['sudo'], 'execute a command as another user', this.cmdSudo, { usage: 'sudo [-u user] command [args ...]' });
    add(['clear'], 'clear the terminal', this.cmdClear);
    add(['nproc'], 'print the number of processing units', () => this.ok(String(this.deps.spec.cpuCores)));

    add(['ping'], 'send ICMP echo requests', this.cmdPing);
    add(['ifconfig', 'ip'], 'show interfaces, routes and neighbours', this.cmdIfconfig, { usage: 'ip addr|route|neigh' });
    add(['arp'], 'show the neighbour cache', this.cmdArp);
    add(['nslookup', 'dig', 'host'], 'query the virtual DNS service', this.cmdResolve);
    add(['curl'], 'transfer a URL over the virtual fabric', this.cmdCurl, { usage: 'curl [-i|-I] [-X method] [-d body] [-o file] url' });
    add(['wget'], 'download a URL to a file', this.cmdWget);
    add(['netstat', 'ss'], 'show sockets and routes', this.cmdNetstat);
    add(['serve'], 'publish a path as an HTTP service', this.cmdServe, { usage: 'serve [port] [path] [hostname]' });

    add(['apps'], 'list installed applications', this.cmdApps);
    add(['store'], 'browse or install from the app registry', this.cmdStore);
    add(['gateway'], 'show egress gateway policy', this.cmdGateway);
    add(['git'], 'the Seed content-addressed git implementation', this.cmdGit);
    this.registerPackageManagers(add);
  }

  private registerPowerShell(add: (names: string[], summary: string, run: CommandSpec['run'], options?: { builtin?: boolean; usage?: string }) => void): void {
    add(['Set-Location', 'sl', 'cd', 'chdir'], 'set the working location', this.cmdCd, { builtin: true });
    add(['Get-Location', 'gl', 'pwd'], 'get the working location', this.cmdPwd, { builtin: true });
    add(['Get-ChildItem', 'gci', 'ls', 'dir'], 'get items in a container', this.cmdGetChildItem, { usage: 'Get-ChildItem [-Path] [-Recurse] [-Filter glob]' });
    add(['Get-Content', 'gc', 'cat', 'type'], 'get the content of an item', this.cmdGetContent);
    add(['Set-Content', 'sc'], 'write content to an item', this.cmdSetContent);
    add(['Add-Content', 'ac'], 'append content to an item', this.cmdSetContent);
    add(['Out-File'], 'send output to a file', this.cmdSetContent);
    add(['Write-Output', 'echo', 'write'], 'write objects to the pipeline', this.cmdWriteOutput);
    add(['Write-Host'], 'write directly to the host', this.cmdWriteOutput);
    add(['New-Item', 'ni', 'mkdir', 'md'], 'create a file or directory', this.cmdMkdir, { usage: 'New-Item -ItemType Directory|File -Path path' });
    add(['Remove-Item', 'ri', 'rm', 'del', 'erase', 'rd', 'rmdir'], 'delete items', this.cmdRm, { usage: 'Remove-Item [-Recurse] [-Force] path' });
    add(['Copy-Item', 'cpi', 'cp', 'copy'], 'copy items', this.cmdCp);
    add(['Move-Item', 'mi', 'mv', 'move'], 'move items', this.cmdMv);
    add(['Rename-Item', 'rni', 'ren'], 'rename an item', this.cmdMv);
    add(['Test-Path'], 'test whether a path exists', this.cmdTestPath);
    add(['Get-Item', 'gi'], 'get an item', this.cmdGetChildItem);
    add(['Select-String', 'sls'], 'find text in strings and files', this.cmdSelectString, { usage: 'Select-String [-Pattern] regex [-Path file]' });
    add(['findstr'], 'search for strings in files', this.cmdFindstr, { usage: 'findstr [/I] [/V] [/N] [/C:"literal"] pattern [file ...]' });
    add(['Measure-Object', 'measure'], 'count objects and characters', this.cmdMeasureObject);
    add(['Select-Object', 'select'], 'select the first, last or unique items', this.cmdSelectObject);
    add(['Where-Object', 'where'], 'filter pipeline items', this.cmdWhereObject);
    add(['where.exe'], 'locate a program on Path', this.cmdWhich); // native tool; `where` alone is Where-Object
    add(['ForEach-Object', 'foreach'], 'run a script block per item', this.cmdForEachObject);
    add(['Sort-Object', 'sort'], 'sort items', this.cmdSort);
    add(['Get-Unique', 'gu'], 'return unique items from a sorted list', this.cmdUniq);
    add(['Get-Command', 'gcm'], 'get the commands this session can run', this.cmdGetCommand);
    add(['Get-Help', 'help', 'man'], 'display help', this.cmdHelp);
    add(['Get-History', 'h', 'history'], 'get the session history', this.cmdHistory, { builtin: true });
    add(['Clear-Host', 'cls', 'clear'], 'clear the host display', this.cmdClear);
    add(['Get-Process', 'gps', 'ps', 'tasklist'], 'get running processes', this.cmdGetProcess);
    add(['Stop-Process', 'spps', 'kill'], 'stop a process', this.cmdKill);
    add(['taskkill'], 'terminate a process by PID or image name', this.cmdTaskkill);
    add(['Get-Date', 'date'], 'get the simulated date and time', this.cmdDate);
    add(['Start-Sleep', 'sleep'], 'advance the simulated clock', this.cmdSleep);
    add(['Get-Variable', 'gv'], 'get shell variables', this.cmdGetVariable);
    add(['exit'], 'exit the shell', this.cmdExit, { builtin: true });
    add(['hostname'], 'print the hostname', this.cmdHostname);
    add(['whoami'], 'print the effective user', this.cmdWhoami);
    add(['ver'], 'print the Windows version', this.cmdVer);
    add(['systeminfo'], 'print system information', this.cmdSystemInfo);
    add(['Get-Volume'], 'get storage volumes', this.cmdGetVolume);
    add(['Get-NetIPConfiguration', 'ipconfig'], 'show interfaces and routes', this.cmdIfconfig);
    add(['Get-NetNeighbor', 'arp'], 'show the neighbour cache', this.cmdArp);
    add(['Resolve-DnsName', 'nslookup'], 'resolve a DNS name', this.cmdResolve);
    add(['Test-NetConnection', 'tnc'], 'test connectivity to a host and port', this.cmdTestNetConnection);
    add(['ping'], 'send ICMP echo requests', this.cmdPing);
    add(['Invoke-WebRequest', 'iwr', 'curl', 'wget'], 'send an HTTP request', this.cmdCurl);
    add(['Invoke-RestMethod', 'irm'], 'send an HTTP request and return the body', this.cmdCurl);
    add(['Get-NetTCPConnection', 'netstat'], 'show TCP sockets', this.cmdNetstat);
    add(['serve'], 'publish a path as an HTTP service', this.cmdServe);
    add(['more'], 'page through text (non-interactive here)', this.cmdPager);
    add(['tar'], 'create and extract archives', this.cmdTar);
    add(['apps'], 'list installed applications', this.cmdApps);
    add(['store'], 'browse or install from the app registry', this.cmdStore);
    add(['gateway'], 'show egress gateway policy', this.cmdGateway);
    add(['git'], 'the Seed content-addressed git implementation', this.cmdGit);
    this.registerPackageManagers(add);
  }

  private registerPackageManagers(add: (names: string[], summary: string, run: CommandSpec['run'], options?: { builtin?: boolean; usage?: string }) => void): void {
    const aliases: Record<string, string> = { 'apt-get': 'apt', pip3: 'pip', chocolatey: 'choco', mamba: 'conda' };
    const supported = this.deps.software.supportedManagers().map(String);
    for (const manager of supported) add([manager], `${manager} package manager`, this.cmdPackageManager);
    for (const [alias, canonical] of Object.entries(aliases)) if (supported.includes(canonical)) add([alias], `${canonical} package manager`, this.cmdPackageManager);
  }
}
