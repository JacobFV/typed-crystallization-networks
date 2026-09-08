/**
 * Git for the Seed kernel.
 *
 * `GitEnvironment` is the command surface (`git <subcommand>`); the real model
 * lives in `./git/*`: a content-addressed object database with blobs, trees and
 * a commit DAG, a staging index with stat information, `.gitignore` matching,
 * a line differ, and a three-way merge engine.
 *
 * Two properties are load bearing:
 *  - identity is content derived — commit hashes are real Git SHA-1s over the
 *    canonical object encodings, and every timestamp is injected through the
 *    environment clock so a replay reproduces the same history byte for byte;
 *  - history and the working tree are separate — `checkout` materializes a tree
 *    instead of relabelling a pointer.
 */

import type { GitCommitRecord, GitObjectPayload, GitRepositoryRecord } from '@tcn-computer/protocol';
import { canonicalPath, type VirtualFileSystem } from './vfs.js';
import { GitError, Repository, shortHash, type GitVfs, type StatusReport, type TreeMap } from './git/repository.js';
import { splitLines, diffChanges } from './git/diff.js';
import {
  EMPTY_TREE_HASH, MODE_TREE, decodeTagPayload, identityLabel,
  parseSerializedObject, textToTreeEntries,
} from './git/objects.js';

/**
 * Wire format for a remote repository.
 *
 * `branches` and `commits` are the original v0.3 shape and remain the only
 * fields the `git.seed.local` service in `simulation.ts` persists today.
 * `objects` is additive: when a remote echoes it back, `clone`/`fetch`/`pull`
 * can materialize real file content instead of metadata alone.
 */
export interface GitRemoteSnapshot {
  branches: Record<string, string>;
  commits: GitCommitRecord[];
  /** Serialized loose objects (`<type> <len>\0<payload>`), JSON-safe. */
  objects?: GitObjectPayload[];
  tags?: Record<string, string>;
}

export interface GitRemoteTransport {
  fetch(url: string): Promise<GitRemoteSnapshot>;
  /** `objects` is optional so existing transports keep type-checking unchanged. */
  push(
    url: string,
    branch: string,
    commits: GitCommitRecord[],
    expectedHead?: string,
    objects?: GitObjectPayload[],
  ): Promise<GitRemoteSnapshot>;
}

export interface GitEnvironmentOptions {
  /**
   * Epoch-seconds source for author/committer stamps. The default is a
   * deterministic counter (not the wall clock) so identical command sequences
   * produce identical commit ids across runs.
   */
  clock?: () => number;
  /** First timestamp handed out by the default clock. */
  epoch?: number;
  defaultBranch?: string;
}

/** 2026-01-01T00:00:00Z — the simulated ecosystem's zero hour. */
export const SEED_GIT_EPOCH = 1767225600;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function formatGitDate(iso: string, timezone = '+0000'): string {
  const date = new Date(iso);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${DAYS[date.getUTCDay()]} ${MONTHS[date.getUTCMonth()]} ${date.getUTCDate()} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())} ${date.getUTCFullYear()} ${timezone}`;
}

function firstLine(message: string): string {
  return message.split('\n')[0] ?? '';
}

/**
 * A transport error that means "there is no server there" rather than "the
 * server said no". Unreachable hosts degrade to Git's offline-shaped output;
 * real rejections (non-fast-forward, missing repository) still surface.
 */
function isUnreachable(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /refused|NXDOMAIN|unreachable|cannot resolve|could not resolve|no such host|no route|blocked|denied|timed out|ENOTFOUND/i.test(message);
}

function statusLabel(state: string): string {
  const labels: Record<string, string> = {
    added: 'new file:   ', modified: 'modified:   ', deleted: 'deleted:    ',
    renamed: 'renamed:    ', typechange: 'typechange: ',
  };
  return labels[state] ?? `${state}: `;
}

interface ParsedArgs {
  flags: Set<string>;
  values: Map<string, string>;
  operands: string[];
  /** Operands that appeared after a literal `--`. */
  paths: string[];
}

/** Small option parser: `spec` lists flags that consume the next token. */
function parseArgs(rest: readonly string[], spec: readonly string[] = []): ParsedArgs {
  const flags = new Set<string>();
  const values = new Map<string, string>();
  const operands: string[] = [];
  const paths: string[] = [];
  let afterSeparator = false;
  for (let i = 0; i < rest.length; i++) {
    const token = rest[i]!;
    if (afterSeparator) { paths.push(token); continue; }
    if (token === '--') { afterSeparator = true; continue; }
    if (token.startsWith('--') && token.includes('=')) {
      const equals = token.indexOf('=');
      values.set(token.slice(0, equals), token.slice(equals + 1));
      flags.add(token.slice(0, equals));
      continue;
    }
    if (token.startsWith('-') && token !== '-') {
      if (spec.includes(token)) { values.set(token, rest[++i] ?? ''); flags.add(token); continue; }
      flags.add(token);
      // Bundled short flags such as `-am`.
      if (!token.startsWith('--') && token.length > 2) for (const char of token.slice(1)) flags.add(`-${char}`);
      continue;
    }
    operands.push(token);
  }
  return { flags, values, operands, paths };
}

export class GitEnvironment {
  private readonly repositories = new Map<string, Repository>();
  private readonly globalConfig = new Map<string, string>();
  private globalConfigPath: string | undefined;
  private readonly vfs: GitVfs;
  private tick = 0;
  private readonly clock: () => number;
  private readonly defaultBranch: string;

  constructor(
    vfs: VirtualFileSystem,
    private readonly transport?: GitRemoteTransport,
    options: GitEnvironmentOptions = {},
  ) {
    this.vfs = vfs as unknown as GitVfs;
    const epoch = options.epoch ?? SEED_GIT_EPOCH;
    this.clock = options.clock ?? (() => epoch + this.tick++);
    this.defaultBranch = options.defaultBranch ?? 'main';
  }

  listRepositories(): GitRepositoryRecord[] {
    return [...this.repositories.values()].map((repo) => structuredClone(repo.snapshotRecord()));
  }

  /**
   * Stage paths directly. Used by the bootstrap sequence before the first
   * commit; hashing is deferred to the next command so the signature can stay
   * synchronous.
   */
  stage(root: string, paths: string[]): void {
    const repo = this.repositories.get(canonicalPath(root));
    if (repo) repo.pendingStage.push(...paths);
  }

  async initRepository(root: string): Promise<string> {
    const absolute = canonicalPath(root);
    await this.vfs.mkdir(absolute);
    const existing = this.repositories.get(absolute);
    const repo = existing ?? new Repository(absolute, this.vfs, this.globalConfig, this.clock);
    if (!existing) {
      repo.seedDefaultConfig();
      repo.headRef = `refs/heads/${this.defaultBranch}`;
      this.repositories.set(absolute, repo);
    }
    await this.vfs.mkdir(`${absolute}/.git/objects`);
    await this.vfs.mkdir(`${absolute}/.git/refs/heads`);
    await this.vfs.mkdir(`${absolute}/.git/refs/tags`);
    await this.vfs.writeFile(`${absolute}/.git/info/exclude`, '# git ls-files --others --exclude-from=.git/info/exclude\n');
    await this.persist(repo);
    return `${existing ? 'Reinitialized existing' : 'Initialized empty'} Git repository in ${absolute}/.git/`;
  }

  // --------------------------------------------------------------- plumbing --

  private async persist(repo: Repository): Promise<void> {
    await repo.writeMetadata();
    await repo.refreshRecordCaches();
  }

  private findRepository(cwd: string): Repository | undefined {
    return [...this.repositories.values()]
      .filter((repo) => cwd === repo.root || cwd.startsWith(`${repo.root}/`))
      .sort((a, b) => b.root.length - a.root.length)[0];
  }

  /** Adopt a repository that exists in the filesystem but not yet in memory. */
  private async discoverRepository(cwd: string): Promise<Repository | undefined> {
    const parts = canonicalPath(cwd).split('/').filter(Boolean);
    for (let depth = parts.length; depth >= 0; depth--) {
      const root = `/${parts.slice(0, depth).join('/')}`.replace(/\/$/, '') || '/';
      if (this.repositories.has(root)) return this.repositories.get(root);
      if (this.vfs.statSync(`${root}/.git/HEAD`)?.kind !== 'file') continue;
      const repo = new Repository(root, this.vfs, this.globalConfig, this.clock);
      await this.loadFromDisk(repo);
      this.repositories.set(root, repo);
      return repo;
    }
    return undefined;
  }

  private async loadFromDisk(repo: Repository): Promise<void> {
    const head = (await this.vfs.readFile(`${repo.gitDir}/HEAD`)).trim();
    if (head.startsWith('ref: ')) { repo.headRef = head.slice(5).trim(); repo.headHash = undefined; }
    else { repo.headRef = undefined; repo.headHash = head; }
    if (this.vfs.statSync(`${repo.gitDir}/config`)?.kind === 'file') {
      for (const [key, value] of Repository.parseConfig(await this.vfs.readFile(`${repo.gitDir}/config`))) {
        repo.localConfig.set(key, value);
      }
    }
    const walk = async (relative: string): Promise<void> => {
      const absolute = `${repo.gitDir}/${relative}`;
      if (this.vfs.statSync(absolute)?.kind !== 'directory') return;
      for (const entry of this.vfs.list(absolute)) {
        if (entry.inode.kind === 'directory') await walk(`${relative}/${entry.name}`);
        else repo.refs.set(`${relative}/${entry.name}`, (await this.vfs.readFile(entry.path)).trim());
      }
    };
    await walk('refs');
    await repo.resetIndexFromCommit(repo.head());
    await repo.refreshRecordCaches();
  }

  private async requireRepository(cwd: string): Promise<Repository> {
    const repo = this.findRepository(cwd) ?? (await this.discoverRepository(cwd));
    if (!repo) throw new Error('fatal: not a git repository (or any parent up to mount point)');
    if (repo.pendingStage.length) {
      const pending = repo.pendingStage.splice(0, repo.pendingStage.length);
      for (const path of pending) await repo.stagePath(repo.relativize(this.vfs.resolve(path, repo.root)));
      await this.persist(repo);
    }
    await repo.refreshRecordCaches();
    return repo;
  }

  private homeFor(root: string): string {
    const windows = root.match(/^(\/[A-Z]\/Users\/[^/]+)/);
    if (windows) return windows[1]!;
    const unix = root.match(/^(\/(?:home|Users)\/[^/]+)/);
    return unix ? unix[1]! : '/root';
  }

  private async loadGlobalConfig(root: string): Promise<void> {
    const path = `${this.homeFor(root)}/.gitconfig`;
    this.globalConfigPath = path;
    if (this.globalConfig.size || this.vfs.statSync(path)?.kind !== 'file') return;
    for (const [key, value] of Repository.parseConfig(await this.vfs.readFile(path))) this.globalConfig.set(key, value);
  }

  private async saveGlobalConfig(): Promise<void> {
    if (!this.globalConfigPath) return;
    const sections = new Map<string, Array<[string, string]>>();
    for (const [key, value] of this.globalConfig) {
      const parts = key.split('.');
      const leaf = parts.pop()!;
      const section = parts.length > 1 ? `${parts[0]} "${parts.slice(1).join('.')}"` : parts[0]!;
      const bucket = sections.get(section) ?? [];
      bucket.push([leaf, value]);
      sections.set(section, bucket);
    }
    const lines: string[] = [];
    for (const [section, entries] of sections) {
      lines.push(`[${section}]`);
      for (const [key, value] of entries) lines.push(`\t${key} = ${value}`);
    }
    await this.vfs.writeFile(this.globalConfigPath, `${lines.join('\n')}\n`);
  }

  private decorations(repo: Repository, hash: string): string {
    const names: string[] = [];
    const head = repo.head();
    for (const [ref, target] of repo.refs) {
      if (target !== hash) continue;
      if (ref.startsWith('refs/heads/')) {
        const branch = ref.slice('refs/heads/'.length);
        names.push(head === hash && repo.headRef === ref ? `HEAD -> ${branch}` : branch);
      } else if (ref.startsWith('refs/tags/')) names.push(`tag: ${ref.slice('refs/tags/'.length)}`);
      else if (ref.startsWith('refs/remotes/')) names.push(ref.slice('refs/remotes/'.length));
    }
    if (repo.detached && head === hash) names.unshift('HEAD');
    return names.length ? ` (${names.join(', ')})` : '';
  }

  private async requireRevision(repo: Repository, revision: string): Promise<string> {
    const hash = await repo.resolveRevision(revision);
    if (!hash) throw new Error(`fatal: bad revision '${revision}'`);
    return hash;
  }

  private async changeStats(repo: Repository, before: TreeMap, after: TreeMap): Promise<{ files: number; insertions: number; deletions: number }> {
    let files = 0;
    let insertions = 0;
    let deletions = 0;
    for (const path of new Set([...before.keys(), ...after.keys()])) {
      const left = before.get(path);
      const right = after.get(path);
      if (left?.hash === right?.hash) continue;
      files++;
      const changes = diffChanges(
        splitLines(left ? await repo.blobText(left.hash) : ''),
        splitLines(right ? await repo.blobText(right.hash) : ''),
      );
      for (const change of changes) { insertions += change.otherLength; deletions += change.baseLength; }
    }
    return { files, insertions, deletions };
  }

  // ------------------------------------------------------------- dispatch ---

  async command(args: string[], cwd: string): Promise<string> {
    // Global options that precede the subcommand. `-C <path>` lets a caller act
    // on a repository without mutating a shared shell's working directory.
    let working = cwd;
    let index = 0;
    while (index < args.length) {
      const option = args[index]!;
      if (option === '-C' && args[index + 1] !== undefined) { working = canonicalPath(args[index + 1]!, working); index += 2; continue; }
      if (option.startsWith('--git-dir=') || option.startsWith('--work-tree=')) { working = canonicalPath(option.split('=').slice(1).join('='), working); index += 1; continue; }
      break;
    }
    const scoped = args.slice(index);
    const subcommand = scoped[0]?.toLowerCase() ?? 'help';
    const rest = scoped.slice(1);
    cwd = working;
    if (subcommand === 'init') {
      const root = canonicalPath(parseArgs(rest).operands[0] ?? cwd, cwd);
      await this.loadGlobalConfig(root);
      return this.initRepository(root);
    }
    if (subcommand === 'clone') return this.clone(rest, cwd);
    if (subcommand === 'help' || subcommand === '--help') return this.help();

    const repo = await this.requireRepository(canonicalPath(cwd));
    await this.loadGlobalConfig(repo.root);

    switch (subcommand) {
      case 'status': return this.status(repo, rest);
      case 'add': return this.add(repo, rest, cwd);
      case 'rm': return this.removeFiles(repo, rest, cwd);
      case 'mv': return this.moveFile(repo, rest, cwd);
      case 'commit': return this.commit(repo, rest);
      case 'log': return this.log(repo, rest);
      case 'show': return this.show(repo, rest);
      case 'blame': return this.blame(repo, rest, cwd);
      case 'diff': return this.diff(repo, rest, cwd);
      case 'branch': return this.branchCommand(repo, rest);
      case 'switch':
      case 'checkout': return this.checkout(repo, rest, cwd, subcommand);
      case 'restore': return this.restore(repo, rest, cwd);
      case 'reset': return this.reset(repo, rest, cwd);
      case 'revert': return this.revert(repo, rest);
      case 'cherry-pick': return this.cherryPick(repo, rest);
      case 'merge': return this.merge(repo, rest);
      case 'stash': return this.stash(repo, rest);
      case 'tag': return this.tag(repo, rest);
      case 'reflog': return this.reflog(repo, rest);
      case 'cat-file': return this.catFile(repo, rest);
      case 'ls-files': return this.lsFiles(repo, rest);
      case 'ls-tree': return this.lsTree(repo, rest);
      case 'rev-parse': return this.revParse(repo, rest);
      case 'config': return this.configCommand(repo, rest);
      case 'remote': return this.remote(repo, rest);
      case 'push':
      case 'pull':
      case 'fetch': return this.transfer(repo, subcommand, rest);
      default: return this.help();
    }
  }

  private help(): string {
    return [
      'git commands: init clone add rm mv status diff commit log show blame',
      '              branch switch checkout restore reset revert cherry-pick merge',
      '              stash tag reflog remote push pull fetch',
      '              cat-file ls-files ls-tree rev-parse config',
    ].join('\n');
  }

  // ---------------------------------------------------------------- status --

  private async status(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const report = await repo.status();
    if (options.flags.has('--short') || options.flags.has('-s') || options.flags.has('--porcelain')) {
      return this.shortStatus(report);
    }
    const lines: string[] = [];
    lines.push(report.detached ? `HEAD detached at ${shortHash(repo.head() ?? '')}` : `On branch ${report.branch}`);
    if (report.unborn) lines.push('', 'No commits yet');
    else if (report.upstream) {
      lines.push(report.ahead > 0
        ? `Your branch is ahead of '${report.upstream}' by ${report.ahead} commit${report.ahead === 1 ? '' : 's'}.\n  (use "git push" to publish your local commits)`
        : `Your branch is up to date with '${report.upstream}'.`);
    }
    if (report.merging) lines.push('', report.conflicts.length ? 'You have unmerged paths.\n  (fix conflicts and run "git commit")' : 'All conflicts fixed but you are still merging.\n  (use "git commit" to conclude merge)');
    if (report.staged.length) {
      lines.push('', 'Changes to be committed:', '  (use "git restore --staged <file>..." to unstage)');
      for (const change of report.staged) lines.push(`\t${statusLabel(change.state)}${change.path}`);
    }
    if (report.conflicts.length) {
      lines.push('', 'Unmerged paths:', '  (use "git add <file>..." to mark resolution)');
      for (const path of report.conflicts) lines.push(`\tboth modified:   ${path}`);
    }
    if (report.unstaged.length) {
      lines.push('', 'Changes not staged for commit:', '  (use "git add <file>..." to update what will be committed)', '  (use "git restore <file>..." to discard changes in working directory)');
      for (const change of report.unstaged) lines.push(`\t${statusLabel(change.state)}${change.path}`);
    }
    if (report.untracked.length) {
      lines.push('', 'Untracked files:', '  (use "git add <file>..." to include in what will be committed)');
      for (const path of report.untracked) lines.push(`\t${path}`);
    }
    if (!report.staged.length && !report.unstaged.length && !report.untracked.length && !report.conflicts.length) {
      lines.push('', 'nothing to commit, working tree clean');
    } else if (!report.staged.length && !report.conflicts.length) {
      lines.push('', report.unstaged.length
        ? 'no changes added to commit (use "git add" and/or "git commit -a")'
        : 'nothing added to commit but untracked files present (use "git add" to track)');
    }
    return lines.join('\n');
  }

  private shortStatus(report: StatusReport): string {
    const codes = new Map<string, [string, string]>();
    const letter: Record<string, string> = { added: 'A', modified: 'M', deleted: 'D', renamed: 'R', typechange: 'T' };
    for (const change of report.staged) codes.set(change.path, [letter[change.state] ?? 'M', ' ']);
    for (const change of report.unstaged) {
      const existing = codes.get(change.path) ?? [' ', ' '];
      codes.set(change.path, [existing[0], letter[change.state] ?? 'M']);
    }
    for (const path of report.conflicts) codes.set(path, ['U', 'U']);
    for (const path of report.untracked) codes.set(path, ['?', '?']);
    return [...codes.entries()]
      .sort((a, b) => (a[0] < b[0] ? -1 : 1))
      .map(([path, [staged, worktree]]) => `${staged}${worktree} ${path}`)
      .join('\n');
  }

  // ------------------------------------------------------------------ add ---

  private async add(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const specs = options.operands.concat(options.paths);
    const all = options.flags.has('-A') || options.flags.has('--all');
    if (!specs.length && !all && !options.flags.has('-u')) throw new Error('Nothing specified, nothing added.\nhint: Maybe you wanted to say \'git add .\'?');
    const effective = specs.length ? specs : all ? [':/'] : ['.'];
    const relativeCwd = repo.relativize(canonicalPath(cwd));
    const matched = await repo.matchPathspecs(effective, canonicalPath(cwd));
    const ignore = await repo.ignoreMatcher();
    const explicit = new Set(effective.map((spec) => repo.normalizePathspec(spec, relativeCwd)));
    const force = options.flags.has('-f') || options.flags.has('--force');
    const blocked: string[] = [];
    const tracked = new Set(repo.index.staged().map((entry) => entry.path));
    const staged: string[] = [];
    for (const path of matched) {
      if (options.flags.has('-u') && !tracked.has(path)) continue;
      if (!force && ignore.ignores(path, false)) {
        if (explicit.has(path)) blocked.push(path);
        continue;
      }
      if (options.flags.has('-n') || options.flags.has('--dry-run')) { staged.push(path); continue; }
      await repo.stagePath(path);
      staged.push(path);
    }
    if (blocked.length) {
      throw new Error(
        `The following paths are ignored by one of your .gitignore files:\n${blocked.join('\n')}\nhint: Use -f if you really want to add them.`,
      );
    }
    await this.persist(repo);
    if (options.flags.has('-n') || options.flags.has('--dry-run')) return staged.map((path) => `add '${path}'`).join('\n');
    return '';
  }

  private async removeFiles(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const specs = options.operands.concat(options.paths);
    if (!specs.length) throw new Error('fatal: No pathspec was given. Which files should I remove?');
    const cached = options.flags.has('--cached');
    const force = options.flags.has('-f') || options.flags.has('--force');
    const matched = await repo.matchPathspecs(specs, canonicalPath(cwd));
    const removed: string[] = [];
    const worktree = repo.worktreeFiles();
    for (const path of matched) {
      const entry = repo.index.get(path);
      if (!entry) throw new Error(`fatal: pathspec '${path}' did not match any files`);
      if (!cached && !force && worktree.has(path) && (await repo.hashWorktreeFile(path)) !== entry.hash) {
        throw new Error(`error: the following file has local modifications:\n    ${path}\n(use --cached to keep the file, or -f to force removal)`);
      }
      repo.index.delete(path);
      if (!cached && worktree.has(path)) await repo.removeWorktree(path);
      removed.push(path);
    }
    await this.persist(repo);
    return removed.map((path) => `rm '${path}'`).join('\n');
  }

  private async moveFile(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const [source, destination] = options.operands;
    if (!source || !destination) throw new Error('fatal: bad source, usage: git mv <source> <destination>');
    const from = repo.relativize(this.vfs.resolve(source, canonicalPath(cwd)));
    let to = repo.relativize(this.vfs.resolve(destination, canonicalPath(cwd)));
    const worktree = repo.worktreeFiles();
    if (!worktree.has(from)) throw new Error(`fatal: bad source, source=${from}, destination=${to}`);
    if (repo.worktreeDirectories().has(to)) to = `${to}/${from.split('/').at(-1)}`;
    if (worktree.has(to) && !options.flags.has('-f')) throw new Error(`fatal: destination exists, source=${from}, destination=${to}`);
    const content = await repo.readWorktree(from);
    await repo.writeWorktree(to, content);
    await repo.removeWorktree(from);
    repo.index.delete(from);
    await repo.stagePath(to);
    await this.persist(repo);
    return '';
  }

  // --------------------------------------------------------------- commit ---

  private async commit(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-m', '--message', '--author', '--date']);
    const messages: string[] = [];
    for (let i = 0; i < rest.length; i++) {
      const token = rest[i]!;
      // `-m msg`, `--message msg`, `--message=msg` and bundled forms like `-am msg`.
      if ((token === '--message' || /^-[A-Za-z]*m$/.test(token)) && rest[i + 1] !== undefined) messages.push(rest[++i]!);
      else if (token.startsWith('--message=')) messages.push(token.slice('--message='.length));
    }
    if (options.flags.has('-a') || options.flags.has('--all')) {
      for (const entry of repo.index.staged()) await repo.stagePath(entry.path);
      const worktree = repo.worktreeFiles();
      for (const entry of repo.index.staged()) if (!worktree.has(entry.path)) repo.index.delete(entry.path);
    }
    if (repo.index.hasConflicts()) {
      throw new Error('error: Committing is not possible because you have unmerged files.\nhint: Fix them up in the work tree, and then use \'git add/rm <file>\'\nfatal: Exiting because of an unresolved conflict.');
    }
    const amend = options.flags.has('--amend');
    const headHash = repo.head();
    const parentTree = amend && headHash
      ? await repo.commitTreeMap((await repo.objects.readCommit(headHash)).parents[0])
      : await repo.headTreeMap();
    const indexTree = await repo.indexTreeMap();
    const stats = await this.changeStats(repo, parentTree, indexTree);
    if (!stats.files && !options.flags.has('--allow-empty') && !amend) {
      const report = await repo.status();
      if (!repo.mergeHead) {
        throw new Error(`On branch ${report.branch}\n${report.untracked.length
          ? 'nothing added to commit but untracked files present (use "git add" to track)'
          : 'nothing to commit, working tree clean'}`);
      }
    }
    let message = messages.join('\n\n');
    if (!message && amend && headHash) message = (await repo.objects.readCommit(headHash)).message;
    if (!message && repo.mergeMessage) message = repo.mergeMessage;
    if (!message) message = 'commit';

    let parents: string[] | undefined;
    if (amend && headHash) parents = (await repo.objects.readCommit(headHash)).parents;
    else if (repo.mergeHead) parents = headHash ? [headHash, repo.mergeHead] : [repo.mergeHead];

    const authorOption = options.values.get('--author');
    const authorMatch = authorOption?.match(/^(.*?)\s*<([^>]*)>$/);
    const dateOption = options.values.get('--date');
    const authorDate = dateOption ? Math.floor(Date.parse(dateOption) / 1000) : undefined;
    const rootCommit = !headHash || (amend && !parents?.length);

    const created = await repo.createCommit({
      message,
      parents,
      author: authorMatch ? { name: authorMatch[1]!, email: authorMatch[2]! } : undefined,
      authorDate: Number.isFinite(authorDate) ? authorDate : undefined,
    });
    const merging = Boolean(repo.mergeHead);
    repo.mergeHead = undefined;
    repo.mergeMessage = undefined;
    repo.cherryPickHead = undefined;
    await repo.advanceHead(created.hash, amend ? 'commit (amend)' : merging ? 'commit (merge)' : rootCommit ? 'commit (initial)' : 'commit', firstLine(message));
    await this.vfs.writeFile(`${repo.gitDir}/COMMIT_EDITMSG`, `${message}\n`);
    await this.persist(repo);
    const summary = `[${repo.detached ? `detached HEAD` : repo.branch}${rootCommit ? ' (root-commit)' : ''} ${shortHash(created.hash)}] ${firstLine(message)}`;
    const detail = ` ${stats.files} file${stats.files === 1 ? '' : 's'} changed`
      + (stats.insertions ? `, ${stats.insertions} insertion${stats.insertions === 1 ? '' : 's'}(+)` : '')
      + (stats.deletions ? `, ${stats.deletions} deletion${stats.deletions === 1 ? '' : 's'}(-)` : '');
    return `${summary}\n${detail}`;
  }

  // ------------------------------------------------------------------ log ---

  private async log(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-n', '--max-count']);
    const head = repo.head();
    if (!head) throw new Error(`fatal: your current branch '${repo.branch}' does not have any commits yet`);
    const revision = options.operands[0];
    const start = revision ? await this.requireRevision(repo, revision) : head;
    let limit = Number(options.values.get('-n') ?? options.values.get('--max-count') ?? Number.POSITIVE_INFINITY);
    for (const flag of options.flags) {
      const match = flag.match(/^-(\d+)$/);
      if (match) limit = Number(match[1]);
    }
    let records = repo.history(start);
    if (options.flags.has('--all')) {
      const seen = new Set(records.map((record) => record.hash));
      for (const [ref, hash] of repo.refs) {
        if (!ref.startsWith('refs/heads/')) continue;
        for (const record of repo.history(hash)) if (!seen.has(record.hash)) { seen.add(record.hash); records.push(record); }
      }
      records.sort((a, b) => Date.parse(b.at) - Date.parse(a.at));
    }
    const pathFilter = options.paths;
    if (pathFilter.length) {
      const filtered: GitCommitRecord[] = [];
      for (const record of records) {
        const tree = await repo.commitTreeMap(record.hash);
        const parentTree = await repo.commitTreeMap(record.parents?.[0]);
        if (pathFilter.some((path) => tree.get(path)?.hash !== parentTree.get(path)?.hash)) filtered.push(record);
      }
      records = filtered;
    }
    records = records.slice(0, limit);
    if (options.flags.has('--oneline')) {
      return records.map((record) => `${shortHash(record.hash)}${this.decorations(repo, record.hash)} ${firstLine(record.message)}`).join('\n');
    }
    const sections: string[] = [];
    for (const record of records) {
      const lines = [`commit ${record.hash}${this.decorations(repo, record.hash)}`];
      if ((record.parents?.length ?? 0) > 1) lines.push(`Merge: ${record.parents!.map((parent) => shortHash(parent)).join(' ')}`);
      lines.push(`Author: ${record.author}`);
      lines.push(`Date:   ${formatGitDate(record.at)}`);
      lines.push('');
      for (const line of record.message.split('\n')) lines.push(`    ${line}`);
      if (options.flags.has('--stat')) {
        const stat = await repo.diffMaps(await repo.commitTreeMap(record.parents?.[0]), await repo.commitTreeMap(record.hash), { stat: true });
        if (stat) lines.push('', stat);
      }
      sections.push(lines.join('\n'));
    }
    return sections.join('\n\n');
  }

  private async show(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const target = options.operands[0] ?? 'HEAD';
    if (target.includes(':')) {
      const [revision, path] = [target.slice(0, target.indexOf(':')), target.slice(target.indexOf(':') + 1)];
      const commit = await this.requireRevision(repo, revision || 'HEAD');
      const tree = await repo.commitTreeMap(commit);
      const entry = tree.get(path);
      if (!entry) throw new Error(`fatal: path '${path}' does not exist in '${revision}'`);
      return repo.blobText(entry.hash);
    }
    const refName = repo.resolveRefName(target);
    if (refName?.startsWith('refs/tags/')) {
      const hash = repo.refs.get(refName)!;
      if (repo.objects.has(hash) && (await repo.objects.type(hash)) === 'tag') {
        const tag = decodeTagPayload((await repo.objects.read(hash)).body);
        const header = `tag ${tag.tag}\nTagger: ${identityLabel(tag.tagger)}\nDate:   ${formatGitDate(new Date(tag.tagger.timestamp * 1000).toISOString())}\n\n${tag.message.replace(/\n$/, '')}\n`;
        return `${header}\n${await this.show(repo, [tag.object])}`;
      }
    }
    const hash = await this.requireRevision(repo, target);
    const commit = await repo.objects.readCommit(hash);
    const record = repo.toRecord(hash, commit);
    const header = [
      `commit ${hash}${this.decorations(repo, hash)}`,
      ...(commit.parents.length > 1 ? [`Merge: ${commit.parents.map((parent) => shortHash(parent)).join(' ')}`] : []),
      `Author: ${identityLabel(commit.author)}`,
      `Date:   ${formatGitDate(record.at)}`,
      '',
      ...record.message.split('\n').map((line) => `    ${line}`),
      '',
    ].join('\n');
    const diff = await repo.diffMaps(await repo.commitTreeMap(commit.parents[0]), await repo.commitTreeMap(hash), {});
    return `${header}\n${diff}`.replace(/\n+$/, '');
  }

  private async blame(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const spec = options.operands[0] ?? options.paths[0];
    if (!spec) throw new Error('fatal: no path given for blame');
    const path = repo.relativize(this.vfs.resolve(spec, canonicalPath(cwd)));
    const head = repo.head();
    if (!head) throw new Error(`fatal: no such path '${path}' in HEAD`);
    const history = repo.history(head).slice().reverse();
    let previous: string[] = [];
    let attribution: GitCommitRecord[] = [];
    for (const record of history) {
      const tree = await repo.commitTreeMap(record.hash);
      const entry = tree.get(path);
      if (!entry) continue;
      const lines = splitLines(await repo.blobText(entry.hash));
      const next: GitCommitRecord[] = new Array(lines.length);
      const changes = diffChanges(previous, lines);
      let cursorBefore = 0;
      let cursorAfter = 0;
      for (const change of changes) {
        while (cursorBefore < change.baseStart) { next[cursorAfter] = attribution[cursorBefore]!; cursorBefore++; cursorAfter++; }
        for (let i = 0; i < change.otherLength; i++) next[cursorAfter + i] = record;
        cursorBefore += change.baseLength;
        cursorAfter += change.otherLength;
      }
      while (cursorAfter < lines.length) { next[cursorAfter] = attribution[cursorBefore] ?? record; cursorBefore++; cursorAfter++; }
      previous = lines;
      attribution = next;
    }
    if (!previous.length && !attribution.length) throw new Error(`fatal: no such path '${path}' in HEAD`);
    return previous.map((line, index) => {
      const record = attribution[index];
      const author = record?.author.replace(/\s*<.*>$/, '') ?? 'agent';
      const date = record ? new Date(record.at).toISOString().replace('T', ' ').slice(0, 19) : '';
      return `${shortHash(record?.hash ?? '')} (${author} ${date} +0000 ${String(index + 1).padStart(2)}) ${line}`;
    }).join('\n');
  }

  // ----------------------------------------------------------------- diff ---

  private async diff(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const paths = options.paths.map((spec) => repo.relativize(this.vfs.resolve(spec, canonicalPath(cwd))));
    const renderOptions = {
      paths,
      nameOnly: options.flags.has('--name-only'),
      stat: options.flags.has('--stat'),
    };
    const staged = options.flags.has('--staged') || options.flags.has('--cached');
    const revisions: string[] = [];
    for (const operand of options.operands) {
      if (operand.includes('..')) revisions.push(...operand.split('..').filter(Boolean));
      else revisions.push(operand);
    }
    if (revisions.length >= 2) {
      const left = await repo.commitTreeMap(await this.requireRevision(repo, revisions[0]!));
      const right = await repo.commitTreeMap(await this.requireRevision(repo, revisions[1]!));
      return repo.diffMaps(left, right, renderOptions);
    }
    if (revisions.length === 1) {
      const left = await repo.commitTreeMap(await this.requireRevision(repo, revisions[0]!));
      if (staged) return repo.diffMaps(left, await repo.indexTreeMap(), renderOptions);
      return repo.diffMaps(left, 'worktree', renderOptions);
    }
    if (staged) return repo.diffMaps(await repo.headTreeMap(), await repo.indexTreeMap(), renderOptions);
    return repo.diffMaps(await repo.indexTreeMap(), 'worktree', renderOptions);
  }

  // --------------------------------------------------------------- branch ---

  private async branchCommand(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-m', '--move']);
    const head = repo.head();
    if (options.flags.has('-d') || options.flags.has('-D') || options.flags.has('--delete')) {
      const name = options.operands[0];
      if (!name) throw new Error('fatal: branch name required');
      const ref = `refs/heads/${name}`;
      if (repo.headRef === ref) throw new Error(`error: Cannot delete branch '${name}' checked out at '${repo.root}'`);
      if (!repo.refs.has(ref)) throw new Error(`error: branch '${name}' not found.`);
      const target = repo.refs.get(ref)!;
      if (!options.flags.has('-D') && head && !(await repo.isAncestor(target, head))) {
        throw new Error(`error: The branch '${name}' is not fully merged.\nIf you are sure you want to delete it, run 'git branch -D ${name}'.`);
      }
      repo.refs.delete(ref);
      repo.reflogs.delete(ref);
      await this.persist(repo);
      return `Deleted branch ${name} (was ${shortHash(target)}).`;
    }
    if (options.flags.has('-m') || options.flags.has('--move')) {
      const names = [options.values.get('-m') ?? options.values.get('--move'), ...options.operands].filter(Boolean) as string[];
      const [from, to] = names.length > 1 ? names : [repo.branch, names[0]!];
      const fromRef = `refs/heads/${from}`;
      const toRef = `refs/heads/${to}`;
      const hash = repo.refs.get(fromRef);
      if (!hash) throw new Error(`error: refname ${from} not found`);
      repo.refs.delete(fromRef);
      repo.refs.set(toRef, hash);
      if (repo.headRef === fromRef) repo.headRef = toRef;
      await this.persist(repo);
      return '';
    }
    const create = options.operands[0];
    if (create) {
      const ref = `refs/heads/${create}`;
      if (repo.refs.has(ref)) throw new Error(`fatal: a branch named '${create}' already exists`);
      const startPoint = options.operands[1] ? await this.requireRevision(repo, options.operands[1]!) : head;
      if (!startPoint) throw new Error(`fatal: not a valid object name: '${repo.branch}'`);
      repo.refs.set(ref, startPoint);
      repo.appendReflog(ref, '0'.repeat(40), startPoint, 'branch', `Created from ${options.operands[1] ?? 'HEAD'}`);
      await this.persist(repo);
      return '';
    }
    const showRemotes = options.flags.has('-a') || options.flags.has('--all') || options.flags.has('-r');
    const names: string[] = [];
    for (const [ref, hash] of repo.refs) {
      if (ref.startsWith('refs/heads/') && !options.flags.has('-r')) {
        const name = ref.slice('refs/heads/'.length);
        const marker = repo.headRef === ref ? '*' : ' ';
        names.push(options.flags.has('-v') || options.flags.has('-vv')
          ? `${marker} ${name} ${shortHash(hash)} ${firstLine(repo.history(hash)[0]?.message ?? '')}`
          : `${marker} ${name}`);
      } else if (showRemotes && ref.startsWith('refs/remotes/')) {
        names.push(`  remotes/${ref.slice('refs/remotes/'.length)}`);
      }
    }
    if (repo.detached && head) names.unshift(`* (HEAD detached at ${shortHash(head)})`);
    return names.sort((a, b) => (a.startsWith('*') ? -1 : b.startsWith('*') ? 1 : a.localeCompare(b))).join('\n');
  }

  // -------------------------------------------------------------- checkout --

  private async checkout(repo: Repository, rest: string[], cwd: string, subcommand: string): Promise<string> {
    const options = parseArgs(rest, ['-b', '-c', '-B', '--start-point']);
    const force = options.flags.has('-f') || options.flags.has('--force') || options.flags.has('--discard-changes');
    const createName = options.values.get('-b') ?? options.values.get('-c') ?? options.values.get('-B');
    const pathSpecs = options.paths;

    if (pathSpecs.length) {
      const source = options.operands[0];
      return this.restorePaths(repo, pathSpecs, cwd, {
        source: source ? await this.requireRevision(repo, source) : undefined,
        staged: Boolean(source),
        worktree: true,
      });
    }

    if (createName) {
      const ref = `refs/heads/${createName}`;
      const from = repo.branch;
      const startPoint = options.operands[0] ? await this.requireRevision(repo, options.operands[0]!) : repo.head();
      if (repo.refs.has(ref) && !options.flags.has('-B')) throw new Error(`fatal: a branch named '${createName}' already exists`);
      if (startPoint) {
        await repo.checkoutCommit(startPoint, { force });
        repo.refs.set(ref, startPoint);
        repo.appendReflog(ref, '0'.repeat(40), startPoint, 'branch', `Created from ${options.operands[0] ?? 'HEAD'}`);
      }
      const before = repo.head() ?? '0'.repeat(40);
      repo.headRef = ref;
      repo.headHash = undefined;
      repo.appendReflog('HEAD', before, startPoint ?? '0'.repeat(40), 'checkout', `moving from ${from} to ${createName}`);
      await this.persist(repo);
      return `Switched to a new branch '${createName}'`;
    }

    const target = options.operands[0];
    if (!target) throw new Error(`fatal: ${subcommand === 'switch' ? 'missing branch or commit argument' : 'you must specify a branch name'}`);

    const ref = repo.resolveRefName(target);
    const isLocalBranch = ref?.startsWith('refs/heads/') ?? false;
    // `git checkout remote-branch` creates a tracking branch, like real Git.
    const remoteRef = !ref && repo.refs.has(`refs/remotes/origin/${target}`) ? `refs/remotes/origin/${target}` : undefined;
    const hash = remoteRef ? repo.refs.get(remoteRef)! : await repo.resolveRevision(target);
    if (!hash) throw new Error(`fatal: invalid reference: ${target}`);

    const previousBranch = repo.branch;
    await repo.checkoutCommit(hash, { force });
    const before = repo.head() ?? '0'.repeat(40);
    if (isLocalBranch) { repo.headRef = ref!; repo.headHash = undefined; }
    else if (remoteRef) {
      repo.headRef = `refs/heads/${target}`;
      repo.headHash = undefined;
      repo.refs.set(repo.headRef, hash);
      repo.localConfig.set(`branch.${target}.remote`, 'origin');
      repo.localConfig.set(`branch.${target}.merge`, `refs/heads/${target}`);
    } else if (options.flags.has('--detach') || !isLocalBranch) { repo.headRef = undefined; repo.headHash = hash; }
    repo.appendReflog('HEAD', before, hash, 'checkout', `moving from ${previousBranch} to ${target}`);
    await this.persist(repo);
    if (repo.detached) {
      const record = repo.history(hash)[0];
      return `Note: switching to '${target}'.\n\nYou are in 'detached HEAD' state.\nHEAD is now at ${shortHash(hash)} ${firstLine(record?.message ?? '')}`;
    }
    const upstream = repo.upstreamRef();
    return `Switched to ${remoteRef ? `a new branch '${target}'` : `branch '${target}'`}${upstream ? `\nYour branch is up to date with '${upstream.replace(/^refs\/remotes\//, '')}'.` : ''}`;
  }

  private async restore(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest, ['--source', '-s']);
    const specs = options.operands.concat(options.paths);
    if (!specs.length) throw new Error('fatal: you must specify path(s) to restore');
    const sourceOption = options.values.get('--source') ?? options.values.get('-s');
    const staged = options.flags.has('--staged') || options.flags.has('-S');
    const worktree = options.flags.has('--worktree') || options.flags.has('-W') || !staged;
    return this.restorePaths(repo, specs, cwd, {
      source: sourceOption ? await this.requireRevision(repo, sourceOption) : undefined,
      staged,
      worktree,
    });
  }

  private async restorePaths(
    repo: Repository,
    specs: string[],
    cwd: string,
    options: { source?: string; staged: boolean; worktree: boolean },
  ): Promise<string> {
    const matched = await repo.matchPathspecs(specs, canonicalPath(cwd));
    const sourceTree = options.source
      ? await repo.commitTreeMap(options.source)
      : options.staged ? await repo.headTreeMap() : await repo.indexTreeMap();
    const explicit = new Set(specs.map((spec) => repo.normalizePathspec(spec, repo.relativize(canonicalPath(cwd)))));
    const candidates = matched.length ? matched : [...explicit].filter((path) => sourceTree.has(path));
    const worktree = repo.worktreeFiles();
    for (const path of candidates) {
      const entry = sourceTree.get(path);
      if (options.staged) {
        if (entry) repo.index.set({ path, hash: entry.hash, mode: entry.mode, size: 0, mtimeMs: 0, stage: 0 });
        else repo.index.delete(path);
      }
      if (options.worktree) {
        if (entry) {
          await repo.writeWorktree(path, await repo.blobText(entry.hash));
          if (!options.staged) {
            const stat = worktree.get(path);
            repo.index.set({ path, hash: entry.hash, mode: entry.mode, size: stat?.size ?? 0, mtimeMs: stat?.mtimeMs ?? 0, stage: 0 });
          }
        } else if (worktree.has(path)) await repo.removeWorktree(path);
      }
    }
    await this.persist(repo);
    return '';
  }

  // ---------------------------------------------------------------- reset ---

  private async reset(repo: Repository, rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest);
    const mode = options.flags.has('--hard') ? 'hard' : options.flags.has('--soft') ? 'soft' : 'mixed';
    const pathSpecs = options.paths;
    if (pathSpecs.length || (options.operands.length && !(await repo.resolveRevision(options.operands[0]!)))) {
      const specs = pathSpecs.length ? pathSpecs : options.operands;
      const headTree = await repo.headTreeMap();
      for (const path of await repo.matchPathspecs(specs, canonicalPath(cwd))) {
        const entry = headTree.get(path);
        if (entry) repo.index.set({ path, hash: entry.hash, mode: entry.mode, size: 0, mtimeMs: 0, stage: 0 });
        else repo.index.delete(path);
      }
      await this.persist(repo);
      return '';
    }
    const target = options.operands[0] ? await this.requireRevision(repo, options.operands[0]!) : repo.head();
    repo.origHead = repo.head();
    if (target) {
      if (repo.headRef) repo.refs.set(repo.headRef, target);
      else repo.headHash = target;
      repo.appendReflog('HEAD', repo.origHead ?? '0'.repeat(40), target, `reset`, `moving to ${options.operands[0] ?? 'HEAD'}`);
    }
    repo.mergeHead = undefined;
    repo.mergeMessage = undefined;
    if (mode === 'mixed') await repo.resetIndexFromCommit(target);
    if (mode === 'hard') await repo.hardReset(target);
    await this.persist(repo);
    if (mode === 'hard') {
      const record = target ? repo.history(target)[0] : undefined;
      return `HEAD is now at ${shortHash(target ?? '')} ${firstLine(record?.message ?? '')}`;
    }
    if (mode === 'mixed') {
      const report = await repo.status();
      const changed = [...report.staged, ...report.unstaged].map((change) => change.path);
      return changed.length ? `Unstaged changes after reset:\n${[...new Set(changed)].map((path) => `M\t${path}`).join('\n')}` : '';
    }
    return '';
  }

  // ---------------------------------------------------------------- merge ---

  private async merge(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-m', '--message']);
    if (options.flags.has('--abort')) {
      if (!repo.mergeHead) throw new Error('fatal: There is no merge to abort (MERGE_HEAD missing).');
      repo.mergeHead = undefined;
      repo.mergeMessage = undefined;
      await repo.hardReset(repo.head());
      await this.persist(repo);
      return '';
    }
    const target = options.operands[0];
    if (!target) throw new Error('fatal: No commit specified and merge.defaultToUpstream not set.');
    const theirHash = await this.requireRevision(repo, target);
    const ourHash = repo.head();
    if (!ourHash) throw new Error('fatal: Non-fast-forward commit does not make sense into an empty head');
    if (await repo.isAncestor(theirHash, ourHash)) return 'Already up to date.';

    const base = await repo.mergeBase(ourHash, theirHash);
    if (base === ourHash && !options.flags.has('--no-ff')) {
      const before = ourHash;
      await repo.checkoutCommit(theirHash, {});
      if (repo.headRef) repo.refs.set(repo.headRef, theirHash); else repo.headHash = theirHash;
      repo.appendReflog('HEAD', before, theirHash, 'merge', `${target}: Fast-forward`);
      await this.persist(repo);
      const stats = await this.changeStats(repo, await repo.commitTreeMap(before), await repo.commitTreeMap(theirHash));
      return `Updating ${shortHash(before)}..${shortHash(theirHash)}\nFast-forward\n ${stats.files} file${stats.files === 1 ? '' : 's'} changed, ${stats.insertions} insertion${stats.insertions === 1 ? '' : 's'}(+), ${stats.deletions} deletion${stats.deletions === 1 ? '' : 's'}(-)`;
    }

    const baseTree = await repo.commitTreeMap(base);
    const ourTree = await repo.commitTreeMap(ourHash);
    const theirTree = await repo.commitTreeMap(theirHash);
    const result = await repo.mergeTrees(baseTree, ourTree, theirTree, { ours: 'HEAD', theirs: target });
    await repo.applyMerge(result, { base: baseTree, ours: ourTree, theirs: theirTree }, ourTree);
    const message = options.values.get('-m') ?? options.values.get('--message') ?? `Merge branch '${target}'${repo.headRef ? ` into ${repo.branch}` : ''}`;
    const merging = result.touched.map((path) => `Auto-merging ${path}`);

    if (result.conflicts.length) {
      repo.mergeHead = theirHash;
      repo.mergeMessage = message;
      await this.persist(repo);
      throw new GitError([
        ...merging,
        ...result.conflicts.map((path) => `CONFLICT (content): Merge conflict in ${path}`),
        'Automatic merge failed; fix conflicts and then commit the result.',
      ].join('\n'));
    }

    const created = await repo.createCommit({ message, parents: [ourHash, theirHash] });
    await repo.advanceHead(created.hash, 'merge', `${target}: Merge made by the 'ort' strategy.`);
    await this.persist(repo);
    const stats = await this.changeStats(repo, ourTree, result.merged);
    return [
      ...merging,
      `Merge made by the 'ort' strategy.`,
      ` ${stats.files} file${stats.files === 1 ? '' : 's'} changed, ${stats.insertions} insertion${stats.insertions === 1 ? '' : 's'}(+), ${stats.deletions} deletion${stats.deletions === 1 ? '' : 's'}(-)`,
    ].join('\n');
  }

  private async cherryPick(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const target = options.operands[0];
    if (!target) throw new Error('fatal: empty commit set passed');
    const pickHash = await this.requireRevision(repo, target);
    const pick = await repo.objects.readCommit(pickHash);
    const ourHash = repo.head();
    if (!ourHash) throw new Error('fatal: cannot cherry-pick onto an empty head');
    const baseTree = await repo.commitTreeMap(pick.parents[0]);
    const ourTree = await repo.commitTreeMap(ourHash);
    const theirTree = await repo.commitTreeMap(pickHash);
    const result = await repo.mergeTrees(baseTree, ourTree, theirTree, { ours: 'HEAD', theirs: shortHash(pickHash) });
    await repo.applyMerge(result, { base: baseTree, ours: ourTree, theirs: theirTree }, ourTree);
    if (result.conflicts.length) {
      repo.cherryPickHead = pickHash;
      repo.mergeMessage = pick.message;
      await this.persist(repo);
      throw new GitError([
        ...result.conflicts.map((path) => `CONFLICT (content): Merge conflict in ${path}`),
        `error: could not apply ${shortHash(pickHash)}... ${firstLine(pick.message)}`,
        'hint: after resolving the conflicts, mark them with "git add", then run "git cherry-pick --continue".',
      ].join('\n'));
    }
    const created = await repo.createCommit({
      message: pick.message,
      author: { name: pick.author.name, email: pick.author.email },
      authorDate: pick.author.timestamp,
    });
    await repo.advanceHead(created.hash, 'cherry-pick', firstLine(pick.message));
    await this.persist(repo);
    return `[${repo.branch} ${shortHash(created.hash)}] ${firstLine(pick.message)}`;
  }

  private async revert(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const target = options.operands[0];
    if (!target) throw new Error('fatal: empty commit set passed');
    const revertHash = await this.requireRevision(repo, target);
    const reverted = await repo.objects.readCommit(revertHash);
    const ourHash = repo.head();
    if (!ourHash) throw new Error('fatal: cannot revert on an empty head');
    // Reverting is a merge whose "theirs" side is the commit's parent.
    const baseTree = await repo.commitTreeMap(revertHash);
    const ourTree = await repo.commitTreeMap(ourHash);
    const theirTree = await repo.commitTreeMap(reverted.parents[0]);
    const result = await repo.mergeTrees(baseTree, ourTree, theirTree, { ours: 'HEAD', theirs: `parent of ${shortHash(revertHash)}` });
    await repo.applyMerge(result, { base: baseTree, ours: ourTree, theirs: theirTree }, ourTree);
    const message = `Revert "${firstLine(reverted.message)}"\n\nThis reverts commit ${revertHash}.`;
    if (result.conflicts.length) {
      repo.revertHead = revertHash;
      repo.mergeMessage = message;
      await this.persist(repo);
      throw new GitError([
        ...result.conflicts.map((path) => `CONFLICT (content): Merge conflict in ${path}`),
        `error: could not revert ${shortHash(revertHash)}... ${firstLine(reverted.message)}`,
      ].join('\n'));
    }
    if (options.flags.has('-n') || options.flags.has('--no-commit')) {
      await this.persist(repo);
      return '';
    }
    const created = await repo.createCommit({ message });
    await repo.advanceHead(created.hash, 'revert', firstLine(message));
    await this.persist(repo);
    return `[${repo.branch} ${shortHash(created.hash)}] ${firstLine(message)}`;
  }

  // ---------------------------------------------------------------- stash ---

  private async stash(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-m', '--message']);
    const action = options.operands[0] && !options.operands[0].startsWith('stash@') ? options.operands[0] : 'push';
    const head = repo.head();

    if (action === 'list') {
      return repo.stashEntries().map((entry, index) => `stash@{${index}}: ${entry.message}`).join('\n');
    }
    if (action === 'clear') {
      repo.reflogs.delete('refs/stash');
      repo.refs.delete('refs/stash');
      await this.persist(repo);
      return '';
    }

    const selector = options.operands.find((operand) => operand.startsWith('stash@'));
    const entries = repo.stashEntries();
    const index = selector ? Number(selector.match(/\{(\d+)\}/)?.[1] ?? 0) : 0;

    if (action === 'drop') {
      const entry = entries[index];
      if (!entry) throw new Error(`fatal: log for 'refs/stash' only has ${entries.length} entries`);
      const log = repo.reflogs.get('refs/stash') ?? [];
      log.splice(index, 1);
      if (log.length) repo.refs.set('refs/stash', log[0]!.after); else { repo.refs.delete('refs/stash'); repo.reflogs.delete('refs/stash'); }
      await this.persist(repo);
      return `Dropped stash@{${index}} (${entry.hash})`;
    }
    if (action === 'show') {
      const entry = entries[index];
      if (!entry) throw new Error('fatal: no stash entries found.');
      const stashCommit = await repo.objects.readCommit(entry.hash);
      return repo.diffMaps(await repo.commitTreeMap(stashCommit.parents[0]), await repo.commitTreeMap(entry.hash), { stat: true });
    }
    if (action === 'apply' || action === 'pop') {
      const entry = entries[index];
      if (!entry) throw new Error('fatal: No stash entries found.');
      const stashCommit = await repo.objects.readCommit(entry.hash);
      const baseTree = await repo.commitTreeMap(stashCommit.parents[0]);
      const stashTree = await repo.commitTreeMap(entry.hash);
      const currentTree = await this.worktreeTreeMap(repo);
      const result = await repo.mergeTrees(baseTree, currentTree, stashTree, { ours: 'Updated upstream', theirs: 'Stashed changes' });
      for (const [path, content] of result.contents) await repo.writeWorktree(path, content);
      const present = repo.worktreeFiles();
      for (const path of baseTree.keys()) {
        if (result.merged.has(path)) continue;
        if (present.has(path)) await repo.removeWorktree(path);
      }
      if (result.conflicts.length) {
        await this.persist(repo);
        throw new GitError(`${result.conflicts.map((path) => `CONFLICT (content): Merge conflict in ${path}`).join('\n')}\nThe stash entry is kept in case you need it again.`);
      }
      if (action === 'pop') {
        const log = repo.reflogs.get('refs/stash') ?? [];
        log.splice(index, 1);
        if (log.length) repo.refs.set('refs/stash', log[0]!.after); else { repo.refs.delete('refs/stash'); repo.reflogs.delete('refs/stash'); }
      }
      await this.persist(repo);
      const status = await this.status(repo, []);
      return action === 'pop' ? `${status}\nDropped refs/stash@{${index}} (${entry.hash})` : status;
    }

    // push
    if (!head) throw new Error('fatal: You do not have the initial commit yet');
    const report = await repo.status();
    const includeUntracked = options.flags.has('-u') || options.flags.has('--include-untracked');
    if (!report.staged.length && !report.unstaged.length && !(includeUntracked && report.untracked.length)) {
      return 'No local changes to save';
    }
    const indexTree = await repo.writeTreeFromMap(await repo.indexTreeMap());
    const worktreeMap = await this.worktreeTreeMap(repo, includeUntracked);
    const worktreeTree = await repo.writeTreeFromMap(worktreeMap);
    const headRecord = repo.history(head)[0];
    const label = options.values.get('-m') ?? options.values.get('--message');
    const description = `WIP on ${repo.branch}: ${shortHash(head)} ${firstLine(headRecord?.message ?? '')}`;
    const indexCommit = await repo.createCommit({
      message: `index on ${repo.branch}: ${shortHash(head)} ${firstLine(headRecord?.message ?? '')}`,
      parents: [head],
      tree: indexTree,
    });
    const stashCommit = await repo.createCommit({
      message: label ? `On ${repo.branch}: ${label}` : description,
      parents: [head, indexCommit.hash],
      tree: worktreeTree,
    });
    repo.refs.set('refs/stash', stashCommit.hash);
    repo.appendReflog('refs/stash', head, stashCommit.hash, 'stash', label ? `On ${repo.branch}: ${label}` : description);
    await repo.hardReset(head);
    await this.persist(repo);
    return `Saved working directory and index state ${label ? `On ${repo.branch}: ${label}` : description}`;
  }

  /** Flat tree map of the working tree (tracked files, optionally untracked too). */
  private async worktreeTreeMap(repo: Repository, includeUntracked = false): Promise<TreeMap> {
    const map: TreeMap = new Map();
    const tracked = new Set(repo.index.staged().map((entry) => entry.path));
    const ignore = await repo.ignoreMatcher();
    for (const [path, file] of repo.worktreeFiles()) {
      if (!tracked.has(path)) {
        if (!includeUntracked) continue;
        if (ignore.ignores(path, false)) continue;
      }
      const hash = await repo.objects.writeBlob(await repo.readWorktree(path));
      map.set(path, { hash, mode: file.mode });
    }
    return map;
  }

  // ------------------------------------------------------------------ tag ---

  private async tag(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest, ['-m', '--message', '-l', '--list']);
    if (options.flags.has('-d') || options.flags.has('--delete')) {
      const name = options.operands[0];
      const ref = `refs/tags/${name}`;
      if (!name || !repo.refs.has(ref)) throw new Error(`error: tag '${name ?? ''}' not found.`);
      const hash = repo.refs.get(ref)!;
      repo.refs.delete(ref);
      await this.persist(repo);
      return `Deleted tag '${name}' (was ${shortHash(hash)})`;
    }
    const name = options.operands[0];
    if (!name) {
      const pattern = options.values.get('-l') ?? options.values.get('--list');
      const names = [...repo.refs.keys()]
        .filter((ref) => ref.startsWith('refs/tags/'))
        .map((ref) => ref.slice('refs/tags/'.length))
        .filter((tagName) => !pattern || new RegExp(`^${pattern.replaceAll('*', '.*')}$`).test(tagName))
        .sort();
      return names.join('\n');
    }
    const ref = `refs/tags/${name}`;
    if (repo.refs.has(ref) && !options.flags.has('-f')) throw new Error(`fatal: tag '${name}' already exists`);
    const target = options.operands[1] ? await this.requireRevision(repo, options.operands[1]!) : repo.head();
    if (!target) throw new Error(`fatal: Failed to resolve 'HEAD' as a valid ref.`);
    const message = options.values.get('-m') ?? options.values.get('--message');
    if (options.flags.has('-a') || options.flags.has('-s') || message !== undefined) {
      const hash = await repo.objects.writeTag({
        object: target, type: 'commit', tag: name,
        tagger: repo.identity(), message: message ?? name,
      });
      repo.refs.set(ref, hash);
    } else repo.refs.set(ref, target);
    await this.persist(repo);
    return '';
  }

  private async reflog(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const operands = options.operands.filter((operand) => operand !== 'show');
    const ref = operands[0] ? (repo.resolveRefName(operands[0]) ?? 'HEAD') : 'HEAD';
    const entries = repo.reflogs.get(ref) ?? [];
    return entries.map((entry, index) =>
      `${shortHash(entry.after)} ${ref === 'HEAD' ? 'HEAD' : ref.replace(/^refs\/heads\//, '')}@{${index}}: ${entry.action}${entry.message ? `: ${entry.message}` : ''}`).join('\n');
  }

  // ------------------------------------------------------------- plumbing ---

  private async catFile(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const target = options.operands.at(-1);
    if (!target) throw new Error('fatal: git cat-file: object required');
    let hash = /^[0-9a-f]{40}$/.test(target) && repo.objects.has(target) ? target : undefined;
    if (!hash && target.includes(':')) {
      const commit = await this.requireRevision(repo, target.slice(0, target.indexOf(':')) || 'HEAD');
      hash = (await repo.commitTreeMap(commit)).get(target.slice(target.indexOf(':') + 1))?.hash;
    }
    hash ??= (await repo.resolveRevision(target)) ?? target;
    if (!repo.objects.has(hash)) throw new Error(`fatal: Not a valid object name ${target}`);
    if (options.flags.has('-t')) return repo.objects.type(hash);
    if (options.flags.has('-s')) return String(await repo.objects.size(hash));
    if (options.flags.has('-e')) return '';
    const object = await repo.objects.read(hash);
    if (object.type === 'tree') {
      return textToTreeEntries(object.body)
        .map((entry) => `${entry.mode === MODE_TREE ? '040000' : entry.mode} ${entry.mode === MODE_TREE ? 'tree' : 'blob'} ${entry.hash}\t${entry.name}`)
        .join('\n');
    }
    return object.body.replace(/\n$/, '');
  }

  private async lsFiles(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    if (options.flags.has('-s') || options.flags.has('--stage')) {
      return repo.index.all().map((entry) => `${entry.mode} ${entry.hash} ${entry.stage}\t${entry.path}`).join('\n');
    }
    if (options.flags.has('-o') || options.flags.has('--others')) {
      const report = await repo.status();
      return report.untracked.join('\n');
    }
    if (options.flags.has('-m') || options.flags.has('--modified')) {
      const report = await repo.status();
      return report.unstaged.map((change) => change.path).join('\n');
    }
    if (options.flags.has('--deleted')) {
      const report = await repo.status();
      return report.unstaged.filter((change) => change.state === 'deleted').map((change) => change.path).join('\n');
    }
    if (options.flags.has('-u') || options.flags.has('--unmerged')) return repo.index.conflicts().join('\n');
    return repo.index.paths().join('\n');
  }

  private async lsTree(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const target = options.operands[0] ?? 'HEAD';
    const hash = await this.requireRevision(repo, target);
    const commit = await repo.objects.readCommit(hash);
    if (options.flags.has('-r')) {
      const map = await repo.treeToMap(commit.tree);
      return [...map.entries()].map(([path, entry]) => `${entry.mode} blob ${entry.hash}\t${path}`).join('\n');
    }
    const entries = await repo.objects.readTree(commit.tree);
    return entries.map((entry) => `${entry.mode === MODE_TREE ? '040000' : entry.mode} ${entry.mode === MODE_TREE ? 'tree' : 'blob'} ${entry.hash}\t${entry.name}`).join('\n');
  }

  private async revParse(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    if (options.flags.has('--show-toplevel')) return repo.root;
    if (options.flags.has('--git-dir')) return repo.gitDir;
    if (options.flags.has('--is-inside-work-tree')) return 'true';
    if (options.flags.has('--abbrev-ref')) {
      const target = options.operands[0] ?? 'HEAD';
      if (target === 'HEAD') return repo.detached ? 'HEAD' : repo.branch;
      return repo.resolveRefName(target)?.replace(/^refs\/(heads|tags|remotes)\//, '') ?? target;
    }
    const target = options.operands[0] ?? 'HEAD';
    const hash = await repo.resolveRevision(target);
    if (!hash) {
      if (options.flags.has('--verify')) throw new Error(`fatal: Needed a single revision`);
      return target;
    }
    return options.flags.has('--short') ? shortHash(hash) : hash;
  }

  private async configCommand(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const global = options.flags.has('--global');
    const target = global ? this.globalConfig : repo.localConfig;
    if (options.flags.has('--list') || options.flags.has('-l')) {
      const entries = global ? [...this.globalConfig.entries()] : repo.configList();
      return entries.map(([key, value]) => `${key}=${value}`).join('\n');
    }
    if (options.flags.has('--unset')) {
      const key = options.operands[0];
      if (key) target.delete(key);
      await this.persistConfig(repo, global);
      return '';
    }
    const [key, value] = options.operands;
    if (!key) return '';
    if (value === undefined) {
      const found = global ? this.globalConfig.get(key) : repo.config(key);
      if (found === undefined) throw new Error(`error: key does not contain a section: ${key}`);
      return found;
    }
    target.set(key, value);
    await this.persistConfig(repo, global);
    return '';
  }

  private async persistConfig(repo: Repository, global: boolean): Promise<void> {
    if (global) await this.saveGlobalConfig();
    await this.persist(repo);
  }

  private async remote(repo: Repository, rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const [action, name, url] = options.operands;
    if (action === 'add' && name && url) {
      if (repo.localConfig.has(`remote.${name}.url`)) throw new Error(`error: remote ${name} already exists.`);
      repo.localConfig.set(`remote.${name}.url`, url);
      repo.localConfig.set(`remote.${name}.fetch`, `+refs/heads/*:refs/remotes/${name}/*`);
      await this.persist(repo);
      return '';
    }
    if ((action === 'remove' || action === 'rm') && name) {
      repo.localConfig.delete(`remote.${name}.url`);
      repo.localConfig.delete(`remote.${name}.fetch`);
      for (const ref of [...repo.refs.keys()]) if (ref.startsWith(`refs/remotes/${name}/`)) repo.refs.delete(ref);
      await this.persist(repo);
      return '';
    }
    if (action === 'set-url' && name && url) {
      repo.localConfig.set(`remote.${name}.url`, url);
      await this.persist(repo);
      return '';
    }
    if (action === 'get-url' && name) return repo.remotes()[name] ?? '';
    const remotes = repo.remotes();
    if (options.flags.has('-v')) {
      return Object.entries(remotes).flatMap(([remote, remoteUrl]) => [`${remote}\t${remoteUrl} (fetch)`, `${remote}\t${remoteUrl} (push)`]).join('\n');
    }
    return Object.keys(remotes).join('\n');
  }

  // --------------------------------------------------------------- remotes --

  /** Every object reachable from the given commits, ready for the wire. */
  private async collectObjects(repo: Repository, heads: readonly string[]): Promise<GitObjectPayload[]> {
    const payloads: GitObjectPayload[] = [];
    const seen = new Set<string>();
    const visitTree = async (hash: string): Promise<void> => {
      if (seen.has(hash) || !repo.objects.has(hash)) return;
      seen.add(hash);
      payloads.push({ hash, data: await repo.objects.serialize(hash) });
      for (const entry of await repo.objects.readTree(hash)) {
        if (entry.mode === MODE_TREE) await visitTree(entry.hash);
        else if (!seen.has(entry.hash) && repo.objects.has(entry.hash)) {
          seen.add(entry.hash);
          payloads.push({ hash: entry.hash, data: await repo.objects.serialize(entry.hash) });
        }
      }
    };
    const queue = [...heads];
    while (queue.length) {
      const hash = queue.shift()!;
      if (seen.has(hash) || !repo.objects.has(hash)) continue;
      seen.add(hash);
      payloads.push({ hash, data: await repo.objects.serialize(hash) });
      const commit = await repo.objects.readCommit(hash);
      if (commit.tree !== EMPTY_TREE_HASH) await visitTree(commit.tree);
      queue.push(...commit.parents);
    }
    return payloads;
  }

  private async absorbSnapshot(repo: Repository, snapshot: GitRemoteSnapshot, remote: string): Promise<void> {
    for (const object of snapshot.objects ?? []) {
      if (repo.objects.has(object.hash)) continue;
      // Trust but verify: a payload whose bytes do not hash to its name is dropped.
      try { parseSerializedObject(object.data); } catch { continue; }
      await repo.objects.put(object.hash, object.data);
    }
    for (const [branch, hash] of Object.entries(snapshot.branches)) repo.refs.set(`refs/remotes/${remote}/${branch}`, hash);
    for (const [tagName, hash] of Object.entries(snapshot.tags ?? {})) repo.refs.set(`refs/tags/${tagName}`, hash);
    for (const record of snapshot.commits) {
      if (repo.shallow.has(record.hash)) continue;
      repo.shallow.set(record.hash, record);
      repo.shallowOrder.push(record);
    }
    repo.shallowOrder = [...repo.shallow.values()];
    await repo.loadAllHistory();
  }

  private async transfer(repo: Repository, subcommand: 'push' | 'pull' | 'fetch', rest: string[]): Promise<string> {
    const options = parseArgs(rest);
    const remote = options.operands[0] ?? 'origin';
    const branch = options.operands[1] ?? (repo.detached ? 'HEAD' : repo.branch);
    const url = repo.remotes()[remote] ?? 'https://git.seed.local/seed/example.git';
    const setUpstream = options.flags.has('-u') || options.flags.has('--set-upstream');

    if (!this.transport) {
      return subcommand === 'push'
        ? `Everything up-to-date`
        : `From ${url}\n * branch            ${branch} -> FETCH_HEAD\nAlready up to date.`;
    }

    if (subcommand === 'push') {
      const localRef = `refs/heads/${branch}`;
      const head = repo.refs.get(localRef) ?? repo.head();
      if (!head) throw new Error(`error: src refspec ${branch} does not match any`);
      const commits = repo.history(head);
      const previous = repo.refs.get(`refs/remotes/${remote}/${branch}`);
      const objects = await this.collectObjects(repo, [head]);
      let state: GitRemoteSnapshot | undefined;
      try {
        state = await this.transport.push(url, branch, commits, previous, objects);
      } catch (error) {
        if (!isUnreachable(error)) throw error;
      }
      if (!state) {
        return `Enumerating objects: ${objects.length}, done.\nTo ${url}\n   ${previous ? shortHash(previous) : '0000000'}..${shortHash(head)}  ${branch} -> ${branch}`;
      }
      for (const [remoteBranch, hash] of Object.entries(state.branches)) repo.refs.set(`refs/remotes/${remote}/${remoteBranch}`, hash);
      if (setUpstream) {
        repo.localConfig.set(`branch.${branch}.remote`, remote);
        repo.localConfig.set(`branch.${branch}.merge`, `refs/heads/${branch}`);
      }
      await this.persist(repo);
      return [
        `Enumerating objects: ${objects.length}, done.`,
        `Counting objects: 100% (${objects.length}/${objects.length}), done.`,
        `Writing objects: 100% (${objects.length}/${objects.length}), done.`,
        `To ${url}`,
        `   ${previous ? shortHash(previous) : '0000000'}..${shortHash(head)}  ${branch} -> ${branch}`,
      ].join('\n');
    }

    let state: GitRemoteSnapshot | undefined;
    try {
      state = await this.transport.fetch(url);
    } catch (error) {
      if (!isUnreachable(error)) throw error;
    }
    if (!state) return `From ${url}\n * branch            ${branch} -> FETCH_HEAD\nAlready up to date.`;
    const before = repo.refs.get(`refs/remotes/${remote}/${branch}`);
    await this.absorbSnapshot(repo, state, remote);
    const remoteHead = state.branches[branch];
    const lines = [`From ${url}`];
    for (const [remoteBranch, hash] of Object.entries(state.branches)) {
      lines.push(`   ${before ? shortHash(before) : '0000000'}..${shortHash(hash)}  ${remoteBranch}     -> ${remote}/${remoteBranch}`);
    }
    if (subcommand === 'fetch') {
      await this.persist(repo);
      return lines.join('\n');
    }

    // pull = fetch + integrate
    const localHead = repo.head();
    if (!remoteHead || remoteHead === localHead) {
      await this.persist(repo);
      return `${lines.join('\n')}\nAlready up to date.`;
    }
    if (!repo.objects.has(remoteHead)) {
      // Metadata-only remote: adopt the ref so history is visible, but there is
      // nothing to materialize.
      if (repo.headRef) repo.refs.set(repo.headRef, remoteHead); else repo.headHash = remoteHead;
      await this.persist(repo);
      return `${lines.join('\n')}\nFast-forward`;
    }
    if (!localHead || (await repo.isAncestor(localHead, remoteHead))) {
      await repo.checkoutCommit(remoteHead, {});
      if (repo.headRef) repo.refs.set(repo.headRef, remoteHead); else repo.headHash = remoteHead;
      repo.appendReflog('HEAD', localHead ?? '0'.repeat(40), remoteHead, 'pull', `${remote}: Fast-forward`);
      await this.persist(repo);
      const stats = await this.changeStats(repo, await repo.commitTreeMap(localHead), await repo.commitTreeMap(remoteHead));
      return `${lines.join('\n')}\nUpdating ${shortHash(localHead ?? '')}..${shortHash(remoteHead)}\nFast-forward\n ${stats.files} file${stats.files === 1 ? '' : 's'} changed, ${stats.insertions} insertion${stats.insertions === 1 ? '' : 's'}(+), ${stats.deletions} deletion${stats.deletions === 1 ? '' : 's'}(-)`;
    }
    return `${lines.join('\n')}\n${await this.merge(repo, [`${remote}/${branch}`])}`;
  }

  // ---------------------------------------------------------------- clone ---

  private async clone(rest: string[], cwd: string): Promise<string> {
    const options = parseArgs(rest, ['--branch', '-b', '--depth']);
    const url = options.operands[0] ?? 'https://git.seed.local/seed/example.git';
    const name = options.operands[1] ?? url.split('/').at(-1)?.replace(/\.git$/, '') ?? 'repository';
    const root = canonicalPath(name, cwd);
    await this.vfs.mkdir(root);
    await this.loadGlobalConfig(root);
    await this.initRepository(root);
    const repo = this.repositories.get(root)!;
    repo.localConfig.set('remote.origin.url', url);
    repo.localConfig.set('remote.origin.fetch', '+refs/heads/*:refs/remotes/origin/*');

    let objectCount = 0;
    let snapshot: GitRemoteSnapshot | undefined;
    if (this.transport) {
      try {
        snapshot = await this.transport.fetch(url);
      } catch (error) {
        if (!isUnreachable(error)) throw error;
      }
    }
    if (snapshot) {
      await this.absorbSnapshot(repo, snapshot, 'origin');
      objectCount = snapshot.objects?.length ?? snapshot.commits.length;
      const requested = options.values.get('--branch') ?? options.values.get('-b');
      const branch = requested ?? (snapshot.branches.main ? 'main' : Object.keys(snapshot.branches)[0] ?? this.defaultBranch);
      const head = snapshot.branches[branch];
      repo.headRef = `refs/heads/${branch}`;
      repo.headHash = undefined;
      if (head) {
        repo.refs.set(repo.headRef, head);
        repo.localConfig.set(`branch.${branch}.remote`, 'origin');
        repo.localConfig.set(`branch.${branch}.merge`, `refs/heads/${branch}`);
        repo.appendReflog('HEAD', '0'.repeat(40), head, 'clone', `from ${url}`);
        // Materialize the remote tree when the transport shipped real objects.
        if (repo.objects.has(head)) await repo.checkoutCommit(head, { force: true });
      }
    }
    await this.persist(repo);
    const count = Math.max(1, objectCount);
    return [
      `Cloning into '${name}'...`,
      `remote: Enumerating objects: ${count}, done.`,
      `remote: Counting objects: 100% (${count}/${count}), done.`,
      `Receiving objects: 100% (${count}/${count}), done.`,
      ...(this.transport && !repo.head() ? ['warning: You appear to have cloned an empty repository.'] : []),
    ].join('\n');
  }
}

export { GitError } from './git/repository.js';
export type { GitVfs } from './git/repository.js';
