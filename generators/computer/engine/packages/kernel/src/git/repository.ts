/**
 * A single Git repository: refs, objects, index, worktree and the operations
 * that move state between them.
 *
 * Everything here is content addressed. The working tree is never the source of
 * truth for history, and history is never the source of truth for the working
 * tree — `checkout` materializes, `status` compares, `commit` snapshots.
 */

import type { GitCommitRecord, GitRepositoryRecord } from '@tcn-computer/protocol';
import { GitIndex, encodeIndexFile } from './index-file.js';
import { IgnoreMatcher, parseIgnoreFile, type IgnoreRule } from './ignore.js';
import { diffChanges, mergeText, splitLines, unifiedDiff } from './diff.js';
import {
  EMPTY_TREE_HASH, GitObjectStore, MODE_EXEC, MODE_FILE, MODE_TREE,
  decodeTagPayload, hashBlob, identityIso, identityLabel,
  type GitCommitObject, type GitIdentity, type GitTreeEntry,
} from './objects.js';

/** The subset of the virtual filesystem Git is allowed to touch. */
export interface GitVfs {
  readFile(path: string): Promise<string>;
  writeFile(path: string, content: string | Uint8Array): Promise<unknown>;
  mkdir(path: string): Promise<unknown>;
  remove(path: string): Promise<void>;
  list(path: string): Array<{ name: string; path: string; inode: { kind: string } }>;
  statSync(path: string): { kind: string; mode: number; size: number; modifiedAt: string } | undefined;
  resolve(path: string, cwd?: string): string;
  hostLayout(): { paths: Record<string, string>; inodes: Record<string, { kind: string; mode: number; size: number; modifiedAt: string }> };
}

export interface TreeFile { hash: string; mode: string; }
export type TreeMap = Map<string, TreeFile>;

export interface WorktreeFile { mode: string; size: number; mtimeMs: number; }

export interface ReflogEntry {
  before: string;
  after: string;
  identity: GitIdentity;
  action: string;
  message: string;
}

export interface StatusChange { path: string; state: 'added' | 'modified' | 'deleted' | 'renamed' | 'typechange'; }

export interface StatusReport {
  branch: string;
  detached: boolean;
  unborn: boolean;
  staged: StatusChange[];
  unstaged: StatusChange[];
  untracked: string[];
  conflicts: string[];
  merging: boolean;
  ahead: number;
  upstream?: string;
}

export interface CommitOptions {
  message: string;
  /** Explicit root tree; defaults to a tree written from the current index. */
  tree?: string;
  /** Epoch seconds; injected so identity is reproducible. */
  authorDate?: number;
  committerDate?: number;
  author?: { name: string; email: string };
  parents?: string[];
  timezone?: string;
}

export interface CheckoutReport {
  updated: string[];
  removed: string[];
}

const DEFAULT_TZ = '+0000';

export function shortHash(hash: string): string { return hash.slice(0, 7); }

function joinPath(base: string, relative: string): string {
  return relative ? `${base}/${relative}` : base;
}

function sortPaths(paths: Iterable<string>): string[] {
  return [...paths].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

export class GitError extends Error {}

export class Repository {
  readonly gitDir: string;
  readonly objects: GitObjectStore;
  index = new GitIndex();
  /** `refs/heads/main` → commit hash. */
  readonly refs = new Map<string, string>();
  headRef: string | undefined = 'refs/heads/main';
  headHash: string | undefined;
  readonly localConfig = new Map<string, string>();
  readonly reflogs = new Map<string, ReflogEntry[]>();
  /** Commits known only as remote metadata (no object payload received). */
  readonly shallow = new Map<string, GitCommitRecord>();
  shallowOrder: GitCommitRecord[] = [];
  mergeHead: string | undefined;
  mergeMessage: string | undefined;
  cherryPickHead: string | undefined;
  revertHead: string | undefined;
  origHead: string | undefined;
  /** Paths handed to `stage()` before the async flush can hash them. */
  pendingStage: string[] = [];

  constructor(
    readonly root: string,
    private readonly vfs: GitVfs,
    private readonly globalConfig: Map<string, string>,
    private readonly clock: () => number,
  ) {
    this.gitDir = `${root}/.git`;
    this.objects = new GitObjectStore(vfs, this.gitDir);
  }

  // ---------------------------------------------------------------- config --

  config(key: string): string | undefined {
    return this.localConfig.get(key) ?? this.globalConfig.get(key);
  }

  configList(): Array<[string, string]> {
    const merged = new Map<string, string>();
    for (const [key, value] of this.globalConfig) merged.set(key, value);
    for (const [key, value] of this.localConfig) merged.set(key, value);
    return [...merged.entries()];
  }

  identity(timestamp = this.clock()): GitIdentity {
    return {
      name: this.config('user.name') ?? 'agent',
      email: this.config('user.email') ?? 'agent@seed.local',
      timestamp,
      timezone: this.config('user.timezone') ?? DEFAULT_TZ,
    };
  }

  get branch(): string {
    return this.headRef ? this.headRef.replace(/^refs\/heads\//, '') : `(HEAD detached at ${shortHash(this.headHash ?? '')})`;
  }

  get detached(): boolean { return !this.headRef; }

  head(): string | undefined {
    return this.headRef ? this.refs.get(this.headRef) : this.headHash;
  }

  remotes(): Record<string, string> {
    const remotes: Record<string, string> = {};
    for (const [key, value] of this.configList()) {
      const match = key.match(/^remote\.(.+)\.url$/);
      if (match) remotes[match[1]!] = value;
    }
    return remotes;
  }

  // ------------------------------------------------------------ worktree io --

  private worktreePath(relative: string): string { return joinPath(this.root, relative); }

  /** Every tracked-eligible file in the working tree, keyed by repo-relative path. */
  worktreeFiles(): Map<string, WorktreeFile> {
    const layout = this.vfs.hostLayout();
    const files = new Map<string, WorktreeFile>();
    const prefix = `${this.root}/`;
    for (const [absolute, inodeId] of Object.entries(layout.paths)) {
      if (!absolute.startsWith(prefix)) continue;
      const relative = absolute.slice(prefix.length);
      if (relative === '.git' || relative.startsWith('.git/')) continue;
      const inode = layout.inodes[inodeId];
      if (!inode || inode.kind !== 'file') continue;
      files.set(relative, {
        mode: (inode.mode & 0o111) === 0 ? MODE_FILE : MODE_EXEC,
        size: inode.size,
        mtimeMs: Date.parse(inode.modifiedAt) || 0,
      });
    }
    return files;
  }

  worktreeDirectories(): Set<string> {
    const layout = this.vfs.hostLayout();
    const dirs = new Set<string>();
    const prefix = `${this.root}/`;
    for (const [absolute, inodeId] of Object.entries(layout.paths)) {
      if (!absolute.startsWith(prefix)) continue;
      const relative = absolute.slice(prefix.length);
      if (relative === '.git' || relative.startsWith('.git/')) continue;
      if (layout.inodes[inodeId]?.kind === 'directory') dirs.add(relative);
    }
    return dirs;
  }

  async readWorktree(relative: string): Promise<string> {
    return this.vfs.readFile(this.worktreePath(relative));
  }

  async writeWorktree(relative: string, content: string): Promise<void> {
    await this.vfs.writeFile(this.worktreePath(relative), content);
  }

  async removeWorktree(relative: string): Promise<void> {
    await this.vfs.remove(this.worktreePath(relative));
    await this.pruneEmptyDirectories(relative);
  }

  private async pruneEmptyDirectories(relative: string): Promise<void> {
    const parts = relative.split('/');
    parts.pop();
    while (parts.length) {
      const candidate = joinPath(this.root, parts.join('/'));
      if (this.vfs.statSync(candidate)?.kind !== 'directory') return;
      if (this.vfs.list(candidate).length) return;
      await this.vfs.remove(candidate);
      parts.pop();
    }
  }

  // -------------------------------------------------------------- ignoring --

  async ignoreMatcher(): Promise<IgnoreMatcher> {
    const rules: IgnoreRule[] = [];
    const exclude = `${this.gitDir}/info/exclude`;
    if (this.vfs.statSync(exclude)?.kind === 'file') {
      rules.push(...parseIgnoreFile(await this.vfs.readFile(exclude), '', exclude));
    }
    const layout = this.vfs.hostLayout();
    const prefix = `${this.root}/`;
    const ignoreFiles: string[] = [];
    for (const absolute of Object.keys(layout.paths)) {
      if (!absolute.startsWith(prefix)) continue;
      const relative = absolute.slice(prefix.length);
      if (relative.startsWith('.git/')) continue;
      if (relative === '.gitignore' || relative.endsWith('/.gitignore')) ignoreFiles.push(relative);
    }
    // Shallower ignore files first; deeper ones override.
    ignoreFiles.sort((a, b) => a.split('/').length - b.split('/').length || (a < b ? -1 : 1));
    for (const file of ignoreFiles) {
      const base = file.includes('/') ? file.slice(0, file.lastIndexOf('/')) : '';
      rules.push(...parseIgnoreFile(await this.readWorktree(file), base, file));
    }
    return new IgnoreMatcher(rules);
  }

  // ---------------------------------------------------------------- objects --

  async hashWorktreeFile(relative: string): Promise<string> {
    return hashBlob(await this.readWorktree(relative));
  }

  /** Flatten a tree object into repo-relative paths. */
  async treeToMap(treeHash: string | undefined, prefix = ''): Promise<TreeMap> {
    const map: TreeMap = new Map();
    if (!treeHash) return map;
    if (!this.objects.has(treeHash) && treeHash !== EMPTY_TREE_HASH) return map;
    for (const entry of await this.objects.readTree(treeHash)) {
      const path = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.mode === MODE_TREE) {
        for (const [child, value] of await this.treeToMap(entry.hash, path)) map.set(child, value);
      } else map.set(path, { hash: entry.hash, mode: entry.mode });
    }
    return map;
  }

  async commitTreeMap(commitHash: string | undefined): Promise<TreeMap> {
    if (!commitHash || !this.objects.has(commitHash)) return new Map();
    const commit = await this.objects.readCommit(commitHash);
    return this.treeToMap(commit.tree);
  }

  async headTreeMap(): Promise<TreeMap> {
    return this.commitTreeMap(this.head());
  }

  /** Write tree objects for a flat path→blob map and return the root tree id. */
  async writeTreeFromMap(map: TreeMap): Promise<string> {
    interface Node { files: Map<string, TreeFile>; dirs: Map<string, Node>; }
    const rootNode: Node = { files: new Map(), dirs: new Map() };
    for (const path of sortPaths(map.keys())) {
      const value = map.get(path)!;
      const parts = path.split('/');
      let node = rootNode;
      for (const part of parts.slice(0, -1)) {
        let next = node.dirs.get(part);
        if (!next) { next = { files: new Map(), dirs: new Map() }; node.dirs.set(part, next); }
        node = next;
      }
      node.files.set(parts.at(-1)!, value);
    }
    const write = async (node: Node): Promise<string> => {
      const entries: GitTreeEntry[] = [];
      for (const [name, file] of node.files) entries.push({ mode: file.mode, name, hash: file.hash });
      for (const [name, child] of node.dirs) {
        const hash = await write(child);
        // Git prunes directories that contain nothing.
        if (hash !== EMPTY_TREE_HASH) entries.push({ mode: MODE_TREE, name, hash });
      }
      if (!entries.length) return EMPTY_TREE_HASH;
      return this.objects.writeTree(entries);
    };
    return write(rootNode);
  }

  async indexTreeMap(): Promise<TreeMap> {
    const map: TreeMap = new Map();
    for (const entry of this.index.staged()) map.set(entry.path, { hash: entry.hash, mode: entry.mode });
    return map;
  }

  // ------------------------------------------------------------- revisions --

  resolveRefName(name: string): string | undefined {
    const candidates = [name, `refs/${name}`, `refs/heads/${name}`, `refs/tags/${name}`, `refs/remotes/${name}`, `refs/remotes/${name}/HEAD`];
    for (const candidate of candidates) if (this.refs.has(candidate)) return candidate;
    return undefined;
  }

  /** Resolve `HEAD`, `main~2`, `abc1234`, `v1.0^`, `stash@{1}` … to a commit id. */
  async resolveRevision(revision: string): Promise<string | undefined> {
    let expression = revision.trim();
    if (!expression) return undefined;
    // `<rev>^{tree}` / `^{commit}` peeling, applied after the ancestry walk.
    let peel: string | undefined;
    const peelMatch = expression.match(/\^\{(\w*)\}$/);
    if (peelMatch) {
      peel = peelMatch[1] || 'commit';
      expression = expression.slice(0, -peelMatch[0].length);
    }
    let suffix = '';
    const suffixMatch = expression.match(/((?:[~^]\d*)+)$/);
    if (suffixMatch) {
      suffix = suffixMatch[1]!;
      expression = expression.slice(0, -suffix.length);
    }
    let hash: string | undefined;
    const stashMatch = expression.match(/^stash@\{(\d+)\}$/);
    if (expression === 'HEAD' || expression === '@') hash = this.head();
    else if (expression === 'ORIG_HEAD') hash = this.origHead;
    else if (expression === 'MERGE_HEAD') hash = this.mergeHead;
    else if (stashMatch) hash = this.stashEntries()[Number(stashMatch[1])]?.hash;
    else {
      const refName = this.resolveRefName(expression);
      if (refName) hash = this.refs.get(refName);
      else if (/^[0-9a-f]{4,40}$/.test(expression)) hash = this.expandHash(expression);
      else if (this.shallow.has(expression)) hash = expression;
    }
    if (hash && this.objects.has(hash) && (await this.objects.type(hash)) === 'tag') {
      hash = decodeTagPayload((await this.objects.read(hash)).body).object;
    }
    if (!hash) return undefined;
    for (const step of suffix.match(/[~^]\d*/g) ?? []) {
      const count = step.length > 1 ? Number(step.slice(1)) : 1;
      if (step.startsWith('~')) {
        for (let i = 0; i < count; i++) {
          if (!hash || !this.objects.has(hash)) return undefined;
          hash = (await this.objects.readCommit(hash)).parents[0];
        }
      } else {
        if (!hash || !this.objects.has(hash)) return undefined;
        const parents = (await this.objects.readCommit(hash)).parents;
        hash = count === 0 ? hash : parents[count - 1];
      }
    }
    if (peel === 'tree' && hash && this.objects.has(hash)) return (await this.objects.readCommit(hash)).tree;
    return hash;
  }

  private expandHash(prefix: string): string | undefined {
    if (prefix.length === 40) return prefix;
    for (const record of this.history(this.head())) {
      if (record.hash.startsWith(prefix)) return record.hash;
    }
    for (const hash of this.refs.values()) if (hash.startsWith(prefix)) return hash;
    return undefined;
  }

  /** Commit records reachable from `hash`, newest first. */
  history(hash: string | undefined, limit = Number.POSITIVE_INFINITY): GitCommitRecord[] {
    if (!hash) return [];
    if (!this.objects.has(hash)) {
      // Fetched from a remote that only shipped metadata: fall back to records.
      const index = this.shallowOrder.findIndex((record) => record.hash === hash);
      return index === -1 ? (this.shallow.get(hash) ? [this.shallow.get(hash)!] : []) : this.shallowOrder.slice(index);
    }
    const seen = new Set<string>();
    const records: GitCommitRecord[] = [];
    const queue = [hash];
    while (queue.length) {
      const current = queue.shift()!;
      if (seen.has(current)) continue;
      seen.add(current);
      if (!this.objects.has(current)) {
        const record = this.shallow.get(current);
        if (record) records.push(record);
        continue;
      }
      const commit = this.readCommitSync(current);
      if (!commit) continue;
      records.push(this.toRecord(current, commit));
      queue.push(...commit.parents);
    }
    records.sort((a, b) => Date.parse(b.at) - Date.parse(a.at) || (a.hash < b.hash ? 1 : -1));
    const headIndex = records.findIndex((record) => record.hash === hash);
    if (headIndex > 0) records.unshift(...records.splice(headIndex, 1));
    return records.slice(0, limit);
  }

  private commitCache = new Map<string, GitCommitObject>();

  /** Synchronous read backed by the object cache; safe once objects are loaded. */
  private readCommitSync(hash: string): GitCommitObject | undefined {
    const cached = this.commitCache.get(hash);
    if (cached) return cached;
    return undefined;
  }

  /** Warm the commit cache so `history()` can stay synchronous. */
  async loadHistory(hash: string | undefined): Promise<void> {
    if (!hash) return;
    const queue = [hash];
    const seen = new Set<string>();
    while (queue.length) {
      const current = queue.shift()!;
      if (seen.has(current) || !this.objects.has(current)) continue;
      seen.add(current);
      if ((await this.objects.type(current)) !== 'commit') continue;
      const commit = await this.objects.readCommit(current);
      this.commitCache.set(current, commit);
      queue.push(...commit.parents);
    }
  }

  async loadAllHistory(): Promise<void> {
    await this.loadHistory(this.head());
    for (const hash of this.refs.values()) await this.loadHistory(hash);
  }

  toRecord(hash: string, commit: GitCommitObject): GitCommitRecord {
    return {
      hash,
      message: commit.message.replace(/\n+$/, ''),
      author: identityLabel(commit.author),
      at: identityIso(commit.author),
      treeDigest: commit.tree,
      parents: [...commit.parents],
      committer: identityLabel(commit.committer),
      committedAt: identityIso(commit.committer),
    };
  }

  /** Best common ancestor of two commits (first-parent-aware BFS). */
  async mergeBase(a: string, b: string): Promise<string | undefined> {
    const ancestors = async (start: string): Promise<Map<string, number>> => {
      const depths = new Map<string, number>();
      const queue: Array<[string, number]> = [[start, 0]];
      while (queue.length) {
        const [hash, depth] = queue.shift()!;
        if (depths.has(hash) || !this.objects.has(hash)) continue;
        depths.set(hash, depth);
        for (const parent of (await this.objects.readCommit(hash)).parents) queue.push([parent, depth + 1]);
      }
      return depths;
    };
    const left = await ancestors(a);
    const right = await ancestors(b);
    let best: string | undefined;
    let bestDepth = Number.POSITIVE_INFINITY;
    for (const [hash, depth] of left) {
      if (!right.has(hash)) continue;
      const total = depth + right.get(hash)!;
      if (total < bestDepth) { best = hash; bestDepth = total; }
    }
    return best;
  }

  async isAncestor(candidate: string, descendant: string): Promise<boolean> {
    const queue = [descendant];
    const seen = new Set<string>();
    while (queue.length) {
      const hash = queue.shift()!;
      if (hash === candidate) return true;
      if (seen.has(hash) || !this.objects.has(hash)) continue;
      seen.add(hash);
      queue.push(...(await this.objects.readCommit(hash)).parents);
    }
    return false;
  }

  // ----------------------------------------------------------------- status --

  async status(): Promise<StatusReport> {
    const headTree = await this.headTreeMap();
    const worktree = this.worktreeFiles();
    const ignore = await this.ignoreMatcher();
    const staged: StatusChange[] = [];
    const unstaged: StatusChange[] = [];
    const untracked: string[] = [];
    const indexEntries = new Map(this.index.staged().map((entry) => [entry.path, entry] as const));

    for (const path of sortPaths(new Set([...indexEntries.keys(), ...headTree.keys()]))) {
      const entry = indexEntries.get(path);
      const head = headTree.get(path);
      if (entry && !head) staged.push({ path, state: 'added' });
      else if (!entry && head) staged.push({ path, state: 'deleted' });
      else if (entry && head && entry.hash !== head.hash) staged.push({ path, state: 'modified' });
      else if (entry && head && entry.mode !== head.mode) staged.push({ path, state: 'typechange' });
    }

    for (const path of sortPaths(indexEntries.keys())) {
      const entry = indexEntries.get(path)!;
      const file = worktree.get(path);
      if (!file) { unstaged.push({ path, state: 'deleted' }); continue; }
      const hash = await this.hashWorktreeFile(path);
      if (hash !== entry.hash) unstaged.push({ path, state: 'modified' });
      else if (file.mode !== entry.mode) unstaged.push({ path, state: 'typechange' });
    }

    for (const path of sortPaths(worktree.keys())) {
      if (indexEntries.has(path)) continue;
      if (this.index.get(path, 2) || this.index.get(path, 1) || this.index.get(path, 3)) continue;
      if (ignore.ignores(path, false)) continue;
      untracked.push(path);
    }

    let ahead = 0;
    const upstream = this.upstreamRef();
    if (upstream) {
      const remoteHash = this.refs.get(upstream);
      const local = this.head();
      if (local) {
        const remoteHistory = new Set(this.history(remoteHash).map((record) => record.hash));
        ahead = this.history(local).filter((record) => !remoteHistory.has(record.hash)).length;
      }
    }

    return {
      branch: this.branch,
      detached: this.detached,
      unborn: !this.head(),
      staged, unstaged, untracked,
      conflicts: this.index.conflicts(),
      merging: Boolean(this.mergeHead),
      ahead,
      upstream: upstream?.replace(/^refs\/remotes\//, ''),
    };
  }

  upstreamRef(): string | undefined {
    if (!this.headRef) return undefined;
    const branch = this.branch;
    const remote = this.config(`branch.${branch}.remote`);
    const merge = this.config(`branch.${branch}.merge`);
    if (remote && merge) {
      const candidate = `refs/remotes/${remote}/${merge.replace(/^refs\/heads\//, '')}`;
      if (this.refs.has(candidate)) return candidate;
    }
    const fallback = `refs/remotes/origin/${branch}`;
    return this.refs.has(fallback) ? fallback : undefined;
  }

  // -------------------------------------------------------------- staging ---

  /** Expand pathspecs into concrete repo-relative file paths. */
  async matchPathspecs(specs: readonly string[], cwd: string, options: { includeDeleted?: boolean } = {}): Promise<string[]> {
    const worktree = this.worktreeFiles();
    const tracked = new Set(this.index.staged().map((entry) => entry.path));
    const candidates = new Set<string>([...worktree.keys()]);
    if (options.includeDeleted !== false) for (const path of tracked) candidates.add(path);
    const relativeCwd = this.relativize(cwd);
    const matched = new Set<string>();
    for (const spec of specs) {
      const normalized = this.normalizePathspec(spec, relativeCwd);
      if (normalized === '') { for (const path of candidates) matched.add(path); continue; }
      for (const path of candidates) {
        if (path === normalized || path.startsWith(`${normalized}/`)) matched.add(path);
      }
    }
    return sortPaths(matched);
  }

  normalizePathspec(spec: string, relativeCwd: string): string {
    if (spec === '.' || spec === './') return relativeCwd;
    if (spec === '*' || spec === ':/' || spec === ':/.') return '';
    if (spec.startsWith('/')) return this.relativize(this.vfs.resolve(spec));
    const resolved = this.vfs.resolve(spec, joinPath(this.root, relativeCwd));
    return this.relativize(resolved);
  }

  relativize(absolute: string): string {
    const normalized = this.vfs.resolve(absolute);
    if (normalized === this.root) return '';
    return normalized.startsWith(`${this.root}/`) ? normalized.slice(this.root.length + 1) : normalized.replace(/^\//, '');
  }

  async stagePath(path: string): Promise<'added' | 'removed' | 'unchanged'> {
    const worktree = this.worktreeFiles().get(path);
    if (!worktree) {
      if (!this.index.get(path)) return 'unchanged';
      this.index.delete(path);
      return 'removed';
    }
    const content = await this.readWorktree(path);
    const hash = await this.objects.writeBlob(content);
    const existing = this.index.get(path);
    this.index.delete(path); // clears any conflict stages
    this.index.set({ path, hash, mode: worktree.mode, size: worktree.size, mtimeMs: worktree.mtimeMs, stage: 0 });
    return existing?.hash === hash ? 'unchanged' : 'added';
  }

  /** Rebuild the index from a tree (`read-tree`). */
  async readTreeIntoIndex(treeHash: string | undefined): Promise<void> {
    const map = await this.treeToMap(treeHash);
    const worktree = this.worktreeFiles();
    this.index.clear();
    for (const [path, file] of map) {
      const stat = worktree.get(path);
      this.index.set({
        path, hash: file.hash, mode: file.mode,
        size: stat?.size ?? 0, mtimeMs: stat?.mtimeMs ?? 0, stage: 0,
      });
    }
  }

  async resetIndexFromCommit(commitHash: string | undefined): Promise<void> {
    if (!commitHash) { this.index.clear(); return; }
    const commit = await this.objects.readCommit(commitHash);
    await this.readTreeIntoIndex(commit.tree);
  }

  // ------------------------------------------------------------- committing --

  async createCommit(options: CommitOptions): Promise<{ hash: string; commit: GitCommitObject; treeMap: TreeMap }> {
    const treeMap = options.tree ? await this.treeToMap(options.tree) : await this.indexTreeMap();
    const tree = options.tree ?? (await this.writeTreeFromMap(treeMap));
    const parents = options.parents ?? (this.head() ? [this.head()!] : []);
    const authorTime = options.authorDate ?? this.clock();
    const committerTime = options.committerDate ?? authorTime;
    const base = this.identity(authorTime);
    const author: GitIdentity = options.author
      ? { ...options.author, timestamp: authorTime, timezone: options.timezone ?? DEFAULT_TZ }
      : { ...base, timestamp: authorTime, timezone: options.timezone ?? DEFAULT_TZ };
    const committer: GitIdentity = { ...this.identity(committerTime), timezone: options.timezone ?? DEFAULT_TZ };
    const commit: GitCommitObject = {
      tree,
      parents,
      author,
      committer,
      message: options.message.endsWith('\n') ? options.message : `${options.message}\n`,
    };
    const hash = await this.objects.writeCommit(commit);
    this.commitCache.set(hash, commit);
    return { hash, commit, treeMap };
  }

  async advanceHead(hash: string, action: string, message: string): Promise<void> {
    const before = this.head() ?? '0'.repeat(40);
    if (this.headRef) this.refs.set(this.headRef, hash);
    else this.headHash = hash;
    this.appendReflog('HEAD', before, hash, action, message);
    if (this.headRef) this.appendReflog(this.headRef, before, hash, action, message);
  }

  appendReflog(ref: string, before: string, after: string, action: string, message: string): void {
    const entries = this.reflogs.get(ref) ?? [];
    entries.unshift({ before, after, identity: this.identity(), action, message });
    this.reflogs.set(ref, entries);
  }

  // -------------------------------------------------------------- checkout ---

  /**
   * Materialize a commit into the working tree. Files present in the target are
   * written, files absent from it are deleted, and uncommitted work blocks the
   * switch the same way it does in real Git.
   */
  async checkoutCommit(target: string | undefined, options: { force?: boolean } = {}): Promise<CheckoutReport> {
    const targetTree = await this.commitTreeMap(target);
    const worktree = this.worktreeFiles();
    const indexEntries = new Map(this.index.staged().map((entry) => [entry.path, entry] as const));
    const dirty = new Map<string, string | undefined>();
    for (const path of new Set([...indexEntries.keys(), ...worktree.keys()])) {
      const file = worktree.get(path);
      dirty.set(path, file ? await this.hashWorktreeFile(path) : undefined);
    }

    if (!options.force) {
      const blocked: string[] = [];
      const blockedUntracked: string[] = [];
      for (const [path, entry] of indexEntries) {
        const current = dirty.get(path);
        const changedLocally = current !== entry.hash;
        const targetEntry = targetTree.get(path);
        if (!changedLocally) continue;
        if (!targetEntry || targetEntry.hash !== entry.hash) blocked.push(path);
      }
      for (const [path, entry] of targetTree) {
        if (indexEntries.has(path)) continue;
        const current = dirty.get(path);
        if (current !== undefined && current !== entry.hash) blockedUntracked.push(path);
      }
      if (blocked.length) {
        throw new GitError(
          `error: Your local changes to the following files would be overwritten by checkout:\n${sortPaths(blocked).map((path) => `\t${path}`).join('\n')}\nPlease commit your changes or stash them before you switch branches.\nAborting`,
        );
      }
      if (blockedUntracked.length) {
        throw new GitError(
          `error: The following untracked working tree files would be overwritten by checkout:\n${sortPaths(blockedUntracked).map((path) => `\t${path}`).join('\n')}\nPlease move or remove them before you switch branches.\nAborting`,
        );
      }
    }

    const removed: string[] = [];
    const updated: string[] = [];
    for (const [path, entry] of indexEntries) {
      if (targetTree.has(path)) continue;
      if (!worktree.has(path)) continue;
      if (!options.force && dirty.get(path) !== entry.hash) continue;
      await this.removeWorktree(path);
      removed.push(path);
    }
    for (const [path, entry] of targetTree) {
      const current = dirty.get(path);
      if (current === entry.hash) continue;
      const indexEntry = indexEntries.get(path);
      // Preserve local edits when the file is identical on both sides.
      if (!options.force && indexEntry && current !== undefined && current !== indexEntry.hash && indexEntry.hash === entry.hash) continue;
      await this.writeWorktree(path, await this.objects.readBlob(entry.hash));
      updated.push(path);
    }

    await this.readTreeIntoIndex(target ? (await this.objects.readCommit(target)).tree : undefined);
    return { updated: sortPaths(updated), removed: sortPaths(removed) };
  }

  /** Reset worktree + index to a commit, discarding everything local. */
  async hardReset(target: string | undefined): Promise<CheckoutReport> {
    return this.checkoutCommit(target, { force: true });
  }

  // ------------------------------------------------------------------ diff ---

  async blobText(hash: string | undefined): Promise<string> {
    if (!hash) return '';
    return this.objects.readBlob(hash);
  }

  /** Unified diff between two flat path maps; `right === undefined` reads the worktree. */
  async diffMaps(
    left: TreeMap,
    right: TreeMap | 'worktree',
    options: { paths?: readonly string[]; nameOnly?: boolean; stat?: boolean; context?: number } = {},
  ): Promise<string> {
    const worktree = right === 'worktree' ? this.worktreeFiles() : undefined;
    const rightPaths = right === 'worktree' ? new Set(worktree!.keys()) : new Set(right.keys());
    // Worktree diffs only consider tracked files.
    const candidates = right === 'worktree'
      ? sortPaths(new Set([...left.keys(), ...[...rightPaths].filter((path) => left.has(path))]))
      : sortPaths(new Set([...left.keys(), ...rightPaths]));
    const filter = options.paths?.length ? options.paths : undefined;
    const sections: string[] = [];
    const stats: Array<{ path: string; added: number; removed: number }> = [];

    for (const path of candidates) {
      if (filter && !filter.some((spec) => path === spec || path.startsWith(`${spec}/`))) continue;
      const before = left.get(path);
      const afterEntry = right === 'worktree' ? worktree!.get(path) : right.get(path);
      if (!before && !afterEntry) continue;
      const beforeText = before ? await this.blobText(before.hash) : '';
      const afterText = afterEntry
        ? (right === 'worktree' ? await this.readWorktree(path) : await this.blobText((afterEntry as TreeFile).hash))
        : '';
      if (before && afterEntry) {
        const afterHash = right === 'worktree' ? hashBlob(afterText) : (afterEntry as TreeFile).hash;
        if (before.hash === afterHash) continue;
      }
      const beforeMode = before?.mode ?? MODE_FILE;
      const afterMode = afterEntry ? (right === 'worktree' ? (afterEntry as WorktreeFile).mode : (afterEntry as TreeFile).mode) : MODE_FILE;
      if (options.nameOnly) { sections.push(path); continue; }
      if (options.stat) {
        const changes = diffChanges(splitLines(beforeText), splitLines(afterText));
        stats.push({
          path,
          added: changes.reduce((sum, change) => sum + change.otherLength, 0),
          removed: changes.reduce((sum, change) => sum + change.baseLength, 0),
        });
        continue;
      }
      const header = [`diff --git a/${path} b/${path}`];
      if (!before) header.push(`new file mode ${afterMode}`);
      else if (!afterEntry) header.push(`deleted file mode ${beforeMode}`);
      else if (beforeMode !== afterMode) header.push(`old mode ${beforeMode}`, `new mode ${afterMode}`);
      const beforeHash = before ? before.hash : '0'.repeat(40);
      const afterHash = afterEntry
        ? (right === 'worktree' ? hashBlob(afterText) : (afterEntry as TreeFile).hash)
        : '0'.repeat(40);
      header.push(`index ${beforeHash.slice(0, 7)}..${afterHash.slice(0, 7)}${before && afterEntry ? ` ${afterMode}` : ''}`);
      const body = unifiedDiff(beforeText, afterText, {
        fromPath: path,
        toPath: path,
        fromLabel: before ? `a/${path}` : '/dev/null',
        toLabel: afterEntry ? `b/${path}` : '/dev/null',
        context: options.context,
      });
      if (!body && before && afterEntry) continue;
      sections.push(`${header.join('\n')}\n${body}`.replace(/\n$/, ''));
    }

    if (options.stat) {
      if (!stats.length) return '';
      const width = Math.max(...stats.map((entry) => entry.path.length));
      const lines = stats.map((entry) => ` ${entry.path.padEnd(width)} | ${entry.added + entry.removed} ${'+'.repeat(Math.min(entry.added, 40))}${'-'.repeat(Math.min(entry.removed, 40))}`);
      const files = stats.length;
      const insertions = stats.reduce((sum, entry) => sum + entry.added, 0);
      const deletions = stats.reduce((sum, entry) => sum + entry.removed, 0);
      lines.push(` ${files} file${files === 1 ? '' : 's'} changed, ${insertions} insertion${insertions === 1 ? '' : 's'}(+), ${deletions} deletion${deletions === 1 ? '' : 's'}(-)`);
      return lines.join('\n');
    }
    return sections.join('\n');
  }

  // ---------------------------------------------------------------- merging --

  /**
   * Three-way merge of flat trees. Returns the merged path map plus the file
   * contents that must land in the worktree, and any conflicted paths.
   */
  async mergeTrees(
    base: TreeMap,
    ours: TreeMap,
    theirs: TreeMap,
    labels: { ours: string; theirs: string },
  ): Promise<{ merged: TreeMap; contents: Map<string, string>; conflicts: string[]; touched: string[] }> {
    const merged: TreeMap = new Map();
    const contents = new Map<string, string>();
    const conflicts: string[] = [];
    const touched: string[] = [];
    for (const path of sortPaths(new Set([...base.keys(), ...ours.keys(), ...theirs.keys()]))) {
      const baseEntry = base.get(path);
      const ourEntry = ours.get(path);
      const theirEntry = theirs.get(path);
      const sameOurs = baseEntry?.hash === ourEntry?.hash;
      const sameTheirs = baseEntry?.hash === theirEntry?.hash;
      if (ourEntry?.hash === theirEntry?.hash) {
        if (ourEntry) merged.set(path, ourEntry);
        continue;
      }
      if (sameOurs) { // only they changed it
        if (theirEntry) { merged.set(path, theirEntry); contents.set(path, await this.blobText(theirEntry.hash)); }
        touched.push(path);
        continue;
      }
      if (sameTheirs) { // only we changed it
        if (ourEntry) merged.set(path, ourEntry);
        continue;
      }
      if (!ourEntry || !theirEntry) {
        // modify/delete conflict
        conflicts.push(path);
        const surviving = ourEntry ?? theirEntry!;
        merged.set(path, surviving);
        contents.set(path, await this.blobText(surviving.hash));
        touched.push(path);
        continue;
      }
      const result = mergeText(
        await this.blobText(ourEntry.hash),
        await this.blobText(baseEntry?.hash),
        await this.blobText(theirEntry.hash),
        { ours: labels.ours, theirs: labels.theirs },
      );
      const hash = await this.objects.writeBlob(result.content);
      merged.set(path, { hash, mode: ourEntry.mode });
      contents.set(path, result.content);
      touched.push(path);
      if (result.conflicted) conflicts.push(path);
    }
    return { merged, contents, conflicts, touched };
  }

  /** Apply a merge result: write files, update the index, record conflict stages. */
  async applyMerge(
    result: { merged: TreeMap; contents: Map<string, string>; conflicts: string[] },
    trees: { base: TreeMap; ours: TreeMap; theirs: TreeMap },
    previous: TreeMap,
  ): Promise<void> {
    for (const [path, content] of result.contents) await this.writeWorktree(path, content);
    const existing = this.worktreeFiles();
    for (const path of previous.keys()) {
      if (result.merged.has(path)) continue;
      if (existing.has(path)) await this.removeWorktree(path);
    }
    const worktree = this.worktreeFiles();
    this.index.clear();
    for (const [path, entry] of result.merged) {
      const stat = worktree.get(path);
      this.index.set({ path, hash: entry.hash, mode: entry.mode, size: stat?.size ?? 0, mtimeMs: stat?.mtimeMs ?? 0, stage: 0 });
    }
    for (const path of result.conflicts) {
      this.index.delete(path, 0);
      const stages: Array<[number, TreeFile | undefined]> = [
        [1, trees.base.get(path)], [2, trees.ours.get(path)], [3, trees.theirs.get(path)],
      ];
      for (const [stage, entry] of stages) {
        if (!entry) continue;
        this.index.set({ path, hash: entry.hash, mode: entry.mode, size: 0, mtimeMs: 0, stage });
      }
    }
  }

  // ---------------------------------------------------------------- stashes --

  stashEntries(): Array<{ hash: string; message: string }> {
    return (this.reflogs.get('refs/stash') ?? []).map((entry) => ({ hash: entry.after, message: entry.message }));
  }

  // ------------------------------------------------------------ persistence --

  /** Skip no-op writes: the VFS persists its whole path table on every write. */
  private readonly written = new Map<string, string>();

  private async writeIfChanged(path: string, content: string | Buffer): Promise<void> {
    const signature = typeof content === 'string' ? content : `bin:${content.length}:${content.subarray(content.length - 20).toString('hex')}`;
    if (this.written.get(path) === signature && this.vfs.statSync(path)) return;
    await this.vfs.writeFile(path, content);
    this.written.set(path, signature);
  }

  async writeMetadata(): Promise<void> {
    await this.vfs.mkdir(`${this.gitDir}/refs/heads`);
    await this.vfs.mkdir(`${this.gitDir}/refs/tags`);
    await this.vfs.mkdir(`${this.gitDir}/objects/info`);
    await this.vfs.mkdir(`${this.gitDir}/info`);
    await this.writeIfChanged(`${this.gitDir}/HEAD`, this.headRef ? `ref: ${this.headRef}\n` : `${this.headHash ?? ''}\n`);
    await this.writeIfChanged(`${this.gitDir}/description`, 'Unnamed repository; edit this file to name the repository.\n');
    await this.writeIfChanged(`${this.gitDir}/config`, this.serializeConfig());
    for (const [ref, hash] of this.refs) {
      await this.writeIfChanged(`${this.gitDir}/${ref}`, `${hash}\n`);
    }
    for (const [ref, entries] of this.reflogs) {
      const lines = [...entries].reverse().map((entry) =>
        `${entry.before} ${entry.after} ${identityLabel(entry.identity)} ${entry.identity.timestamp} ${entry.identity.timezone}\t${entry.action}${entry.message ? `: ${entry.message}` : ''}`);
      await this.writeIfChanged(`${this.gitDir}/logs/${ref}`, `${lines.join('\n')}\n`);
    }
    await this.writeIfChanged(`${this.gitDir}/index`, encodeIndexFile(this.index.all()));
    if (this.mergeHead) await this.vfs.writeFile(`${this.gitDir}/MERGE_HEAD`, `${this.mergeHead}\n`);
    else if (this.vfs.statSync(`${this.gitDir}/MERGE_HEAD`)) await this.vfs.remove(`${this.gitDir}/MERGE_HEAD`);
    if (this.mergeMessage) await this.vfs.writeFile(`${this.gitDir}/MERGE_MSG`, `${this.mergeMessage}\n`);
    if (this.cherryPickHead) await this.vfs.writeFile(`${this.gitDir}/CHERRY_PICK_HEAD`, `${this.cherryPickHead}\n`);
    else if (this.vfs.statSync(`${this.gitDir}/CHERRY_PICK_HEAD`)) await this.vfs.remove(`${this.gitDir}/CHERRY_PICK_HEAD`);
    if (this.origHead) await this.vfs.writeFile(`${this.gitDir}/ORIG_HEAD`, `${this.origHead}\n`);
    // Refs deleted in memory must disappear from disk too.
    for (const directory of ['refs/heads', 'refs/tags']) {
      const absolute = `${this.gitDir}/${directory}`;
      if (this.vfs.statSync(absolute)?.kind !== 'directory') continue;
      for (const entry of this.vfs.list(absolute)) {
        if (entry.inode.kind !== 'file') continue;
        if (!this.refs.has(`${directory}/${entry.name}`)) await this.vfs.remove(entry.path);
      }
    }
  }

  serializeConfig(): string {
    const sections = new Map<string, Array<[string, string]>>();
    for (const [key, value] of this.localConfig) {
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
    return `${lines.join('\n')}\n`;
  }

  static parseConfig(text: string): Map<string, string> {
    const config = new Map<string, string>();
    let section = '';
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (!line || line.startsWith('#') || line.startsWith(';')) continue;
      const header = line.match(/^\[([^\]\s]+)(?:\s+"([^"]*)")?\]$/);
      if (header) {
        section = header[2] ? `${header[1]}.${header[2]}` : header[1]!;
        continue;
      }
      const equals = line.indexOf('=');
      if (equals === -1) continue;
      config.set(`${section}.${line.slice(0, equals).trim()}`, line.slice(equals + 1).trim());
    }
    return config;
  }

  seedDefaultConfig(): void {
    this.localConfig.set('core.repositoryformatversion', '0');
    this.localConfig.set('core.filemode', 'false');
    this.localConfig.set('core.bare', 'false');
    this.localConfig.set('core.logallrefupdates', 'true');
  }

  /** Protocol projection consumed by snapshots and the simulator UI. */
  snapshotRecord(): GitRepositoryRecord {
    const branches: Record<string, string | undefined> = {};
    for (const [ref, hash] of this.refs) {
      if (ref.startsWith('refs/heads/')) branches[ref.slice('refs/heads/'.length)] = hash;
    }
    const remoteRefs: Record<string, string> = {};
    const tags: Record<string, string> = {};
    for (const [ref, hash] of this.refs) {
      if (ref.startsWith('refs/remotes/')) remoteRefs[ref.slice('refs/remotes/'.length)] = hash;
      if (ref.startsWith('refs/tags/')) tags[ref.slice('refs/tags/'.length)] = hash;
    }
    const head = this.head();
    return {
      root: this.root,
      branch: this.branch,
      head,
      branches,
      remotes: this.remotes(),
      remoteRefs,
      staged: this.stagedPathsForRecord(),
      commits: this.history(head),
      tags,
      detached: this.detached,
    };
  }

  private stagedPathsForRecord(): string[] {
    const head = this.headTreeCache;
    if (!head) return this.index.paths();
    const changed: string[] = [];
    for (const entry of this.index.staged()) {
      const existing = head.get(entry.path);
      if (!existing || existing.hash !== entry.hash) changed.push(entry.path);
    }
    for (const path of head.keys()) if (!this.index.get(path)) changed.push(path);
    return sortPaths(changed);
  }

  /** Refreshed by the environment after every mutating command. */
  headTreeCache: TreeMap | undefined;

  async refreshRecordCaches(): Promise<void> {
    await this.loadAllHistory();
    this.headTreeCache = await this.headTreeMap();
  }
}
