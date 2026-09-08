import { createHash } from 'node:crypto';
import { randomUUID } from './determinism.js';
import { appendFile, mkdir as hostMkdir, readFile as hostRead, rename as hostRename, rm as hostRemove, stat as hostStat, writeFile as hostWrite } from 'node:fs/promises';
import path from 'node:path';
import type { ComputerSpec, DirectoryEntry, InodeRecord, VfsFileTable, VfsIdentity, VfsInodeRecord } from '@tcn-computer/protocol';

/** Derived layout consumed by git/software; `paths` expands every hard link. */
export type FileTable = VfsFileTable;

interface ChildEntry { name: string; id: string }

/** Persisted point-in-time image of the inode graph. Directory children ride along with their directory. */
interface TableSnapshot {
  version: 2;
  sequence: number;
  root: string;
  inodes: Record<string, VfsInodeRecord & { children?: Record<string, string> }>;
}

/** Legacy flat table (`version: 1`) kept readable so existing state roots still boot. */
interface LegacyTable { version: 1; paths: Record<string, string>; inodes: Record<string, InodeRecord> }

type JournalEntry =
  | { s: number; t: 'inode'; n: VfsInodeRecord }
  | { s: number; t: 'link'; d: string; m: string; i: string }
  | { s: number; t: 'unlink'; d: string; m: string }
  | { s: number; t: 'drop'; i: string };

/** The same union before a sequence number is stamped on it. */
type JournalDraft = JournalEntry extends infer Entry ? (Entry extends JournalEntry ? Omit<Entry, 's'> : never) : never;

interface Located {
  /** Canonical path with the stored (case-preserved) names of every resolved component. */
  path: string;
  name: string;
  parentId?: string;
  parentPath: string;
  id?: string;
  inode?: VfsInodeRecord;
}

export interface VfsOptions {
  /** macOS/Windows profiles declare `false`: lookups fold case, listings keep it. */
  caseSensitive?: boolean;
  /** Off by default so existing callers never start failing. */
  enforcePermissions?: boolean;
  /** Identity used when a call does not pass one. Defaults to root (permissive). */
  identity?: VfsIdentity;
  /** Journal entries tolerated before the table is compacted back into a snapshot. */
  journalInterval?: number;
}

export interface VfsCallOptions { as?: VfsIdentity }

/** Identity-bound subset returned by {@link VirtualFileSystem.as}. */
export interface VfsUserView {
  readBytes(input: string): Promise<Buffer>;
  readFile(input: string): Promise<string>;
  writeFile(input: string, content: string | Uint8Array, diskId?: string): Promise<InodeRecord>;
  mkdir(input: string, diskId?: string): Promise<InodeRecord>;
  list(input: string): DirectoryEntry[];
  remove(input: string, options?: RemoveOptions): Promise<void>;
}
export interface RemoveOptions extends VfsCallOptions { recursive?: boolean }
export interface RenameOptions extends VfsCallOptions { overwrite?: boolean }
export interface CopyOptions extends VfsCallOptions { recursive?: boolean; overwrite?: boolean; dereference?: boolean }

const READ = 4;
const WRITE = 2;
const EXECUTE = 1;
const MAX_SYMLINK_HOPS = 40;
const ROOT_IDENTITY: VfsIdentity = { uid: 0, gid: 0 };

const now = () => new Date().toISOString();

export function canonicalPath(input: string, cwd = '/'): string {
  let value = input.trim().replaceAll('\\', '/');
  if (/^[a-zA-Z]:($|\/)/.test(value)) value = `/${value[0]!.toUpperCase()}${value.slice(2)}`;
  if (!value.startsWith('/')) value = `${cwd.replace(/\/$/, '')}/${value}`;
  const parts: string[] = [];
  for (const part of value.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') parts.pop(); else parts.push(part);
  }
  return `/${parts.join('/')}` || '/';
}

const joinPath = (dir: string, name: string) => (dir === '/' ? `/${name}` : `${dir}/${name}`);

/**
 * Inode filesystem over a host state directory.
 *
 * - Directories own a `name -> inodeId` child map, so lookup and listing cost the
 *   number of components/entries rather than the size of the whole filesystem.
 * - File content lives in one blob per inode (`<rootDir>/<diskId>/<inodeId>`), written
 *   through a temp file and renamed, and is always handled as bytes.
 * - Metadata persists as a snapshot plus an append-only journal: mutations queue a
 *   coalesced batch append instead of re-serializing the table, `flush()` forces the
 *   batch out, and the journal compacts back into a snapshot every `journalInterval`
 *   entries. Every write runs on one chain with unique temp names, so overlapping
 *   persists cannot race. A crash can lose an unflushed tail but never tears the
 *   table: `initialize()` loads the snapshot and replays only journal entries newer
 *   than it.
 */
export class VirtualFileSystem {
  readonly computerId: string;
  readonly rootDir: string;
  readonly caseSensitive: boolean;
  private readonly enforce: boolean;
  private readonly identity: VfsIdentity;
  private readonly journalInterval: number;
  private inodes = new Map<string, VfsInodeRecord>();
  private children = new Map<string, Map<string, ChildEntry>>();
  private rootId = '';
  private sequence = 0;
  private pending: JournalEntry[] = [];
  private journaled = 0;
  private tail: Promise<void> = Promise.resolve();
  private failure?: Error;
  private ops: Promise<unknown> = Promise.resolve();
  private tableFile: string;
  private journalFile: string;

  constructor(stateRoot: string, runId: string, private readonly spec: ComputerSpec, options: VfsOptions = {}) {
    this.computerId = spec.id;
    this.rootDir = path.join(stateRoot, runId, spec.id);
    this.tableFile = path.join(this.rootDir, 'file-table.json');
    this.journalFile = path.join(this.rootDir, 'file-table.journal');
    this.caseSensitive = options.caseSensitive ?? true;
    this.enforce = options.enforcePermissions ?? false;
    this.identity = options.identity ?? ROOT_IDENTITY;
    this.journalInterval = options.journalInterval ?? 512;
  }

  async initialize(): Promise<void> {
    await hostMkdir(this.rootDir, { recursive: true });
    for (const disk of this.spec.disks) await hostMkdir(path.join(this.rootDir, disk.id), { recursive: true });
    if (await this.restore()) return;
    this.inodes = new Map();
    this.children = new Map();
    this.sequence = 0;
    this.pending = [];
    const root = this.createInode('directory', this.defaultDisk, 0o755, ROOT_IDENTITY);
    this.rootId = root.id;
    await this.compact();
  }

  private get defaultDisk(): string { return this.spec.disks[0]?.id ?? 'disk0'; }

  // ---------------------------------------------------------------- persistence

  private async restore(): Promise<boolean> {
    let snapshot: TableSnapshot;
    try {
      const parsed = JSON.parse(await hostRead(this.tableFile, 'utf8')) as TableSnapshot | LegacyTable;
      snapshot = parsed.version === 1 ? migrateLegacy(parsed) : parsed;
      if (!snapshot.root || !snapshot.inodes[snapshot.root]) return false;
    } catch { return false; }
    this.inodes = new Map();
    this.children = new Map();
    this.rootId = snapshot.root;
    this.sequence = snapshot.sequence ?? 0;
    for (const [id, record] of Object.entries(snapshot.inodes)) {
      const { children, ...inode } = record;
      this.inodes.set(id, { ...inode, id });
      if (inode.kind === 'directory') this.children.set(id, new Map(Object.entries(children ?? {}).map(([name, target]) => [this.key(name), { name, id: target }])));
    }
    try {
      const journal = await hostRead(this.journalFile, 'utf8');
      for (const line of journal.split('\n')) {
        if (!line.trim()) continue;
        let entry: JournalEntry;
        try { entry = JSON.parse(line) as JournalEntry; } catch { continue; } // torn tail line
        if (entry.s <= this.sequence) continue;
        this.replay(entry);
        this.sequence = entry.s;
        this.journaled += 1;
      }
    } catch { /* no journal yet */ }
    return true;
  }

  private replay(entry: JournalEntry): void {
    if (entry.t === 'inode') {
      this.inodes.set(entry.n.id, entry.n);
      if (entry.n.kind === 'directory' && !this.children.has(entry.n.id)) this.children.set(entry.n.id, new Map());
    } else if (entry.t === 'link') {
      this.children.get(entry.d)?.set(this.key(entry.m), { name: entry.m, id: entry.i });
    } else if (entry.t === 'unlink') {
      this.children.get(entry.d)?.delete(this.key(entry.m));
    } else {
      this.inodes.delete(entry.i);
      this.children.delete(entry.i);
    }
  }

  private record(entry: JournalDraft): void {
    this.sequence += 1;
    this.pending.push({ ...entry, s: this.sequence } as JournalEntry);
    this.kick();
  }

  /** Queue a drain. Every persist runs on one chain, so temp files never collide. */
  private kick(): void {
    this.tail = this.tail.then(() => this.drain()).catch((error: unknown) => { this.failure ??= error as Error; });
  }

  private async drain(): Promise<void> {
    if (!this.pending.length) return;
    if (this.journaled + this.pending.length >= this.journalInterval) return this.compact();
    const batch = this.pending;
    this.pending = [];
    await appendFile(this.journalFile, `${batch.map((entry) => JSON.stringify(entry)).join('\n')}\n`);
    this.journaled += batch.length;
  }

  /** Atomic full write through a unique temp name, then the journal prefix it supersedes is dropped. */
  private async compact(): Promise<void> {
    const payload = JSON.stringify(this.snapshot());
    this.pending = [];
    const tmp = `${this.tableFile}.${randomUUID()}.tmp`;
    await hostWrite(tmp, payload);
    await hostRename(tmp, this.tableFile);
    await hostWrite(this.journalFile, '');
    this.journaled = 0;
  }

  private snapshot(): TableSnapshot {
    const inodes: TableSnapshot['inodes'] = {};
    for (const [id, inode] of this.inodes) {
      const entries = this.children.get(id);
      inodes[id] = entries ? { ...inode, children: Object.fromEntries([...entries.values()].map((entry) => [entry.name, entry.id])) } : { ...inode };
    }
    return { version: 2, sequence: this.sequence, root: this.rootId, inodes };
  }

  /** Durably write everything buffered so far, surfacing any deferred persistence failure. */
  async flush(): Promise<void> {
    this.kick();
    await this.tail;
    if (this.failure) { const failure = this.failure; this.failure = undefined; throw failure; }
  }

  /** Serializes every mutation, so overlapping callers can never interleave inside one operation. */
  private lock<T>(job: () => Promise<T>): Promise<T> {
    const run = this.ops.then(job, job);
    this.ops = run.then(() => undefined, () => undefined);
    return run;
  }

  // ------------------------------------------------------------------ traversal

  private key(name: string): string { return this.caseSensitive ? name : name.toLowerCase(); }

  private inodeOf(id: string): VfsInodeRecord {
    const inode = this.inodes.get(id);
    if (!inode) throw new Error(`corrupt inode reference: ${id}`);
    return inode;
  }

  private entryOf(dirId: string, name: string): ChildEntry | undefined { return this.children.get(dirId)?.get(this.key(name)); }

  private who(options: VfsCallOptions | undefined): VfsIdentity { return options?.as ?? this.identity; }

  private require(inode: VfsInodeRecord, identity: VfsIdentity, need: number, target: string): void {
    if (!this.enforce || identity.uid === 0) return;
    const shift = inode.uid === identity.uid ? 6 : inode.gid === identity.gid ? 3 : 0;
    if (((inode.mode >> shift) & need) !== need) throw new Error(`permission denied: ${target}`);
  }

  private requireOwner(inode: VfsInodeRecord, identity: VfsIdentity, target: string): void {
    if (!this.enforce || identity.uid === 0 || inode.uid === identity.uid) return;
    throw new Error(`operation not permitted: ${target}`);
  }

  /**
   * Resolves a path one directory entry at a time — O(components), never a table scan.
   * Symlinks are followed for intermediate components always and for the final one when
   * `follow`, restarting from the root each hop until `MAX_SYMLINK_HOPS` trips ELOOP.
   */
  private locate(input: string, options: { follow?: boolean; identity?: VfsIdentity; cwd?: string } = {}): Located {
    const identity = options.identity ?? ROOT_IDENTITY;
    const follow = options.follow !== false;
    const target = canonicalPath(input, options.cwd);
    const pending = target.split('/').filter(Boolean).reverse();
    const root = (): Located => ({ path: '/', name: '', parentPath: '/', id: this.rootId, inode: this.inodeOf(this.rootId) });
    let parentId = this.rootId;
    let parentPath = '/';
    let hops = 0;
    while (pending.length) {
      const name = pending.pop()!;
      const dir = this.inodes.get(parentId);
      if (!dir) throw new Error(`no such file or directory: ${parentPath}`);
      if (dir.kind !== 'directory') throw new Error(`not a directory: ${parentPath}`);
      this.require(dir, identity, EXECUTE, parentPath);
      const entry = this.entryOf(parentId, name);
      const resolved = joinPath(parentPath, entry?.name ?? name);
      if (!entry) {
        if (pending.length) throw new Error(`no such file or directory: ${resolved}`);
        return { path: resolved, name, parentId, parentPath };
      }
      const inode = this.inodeOf(entry.id);
      if (inode.kind === 'symlink' && (follow || pending.length)) {
        if (++hops > MAX_SYMLINK_HOPS) throw new Error(`too many levels of symbolic links: ${target}`);
        const linked = canonicalPath(inode.target ?? '/', parentPath);
        const parts = linked.split('/').filter(Boolean);
        for (let index = parts.length - 1; index >= 0; index -= 1) pending.push(parts[index]!);
        parentId = this.rootId;
        parentPath = '/';
        continue;
      }
      if (!pending.length) return { path: resolved, name: entry.name, parentId, parentPath, id: entry.id, inode };
      parentId = entry.id;
      parentPath = resolved;
    }
    return root();
  }

  private tryLocate(input: string, options: { follow?: boolean; identity?: VfsIdentity } = {}): Located | undefined {
    try { return this.locate(input, options); } catch { return undefined; }
  }

  // ------------------------------------------------------------------- mutation

  private createInode(kind: InodeRecord['kind'], diskId: string, mode: number, identity: VfsIdentity, target?: string): VfsInodeRecord {
    const stamp = now();
    const inode: VfsInodeRecord = { id: randomUUID(), diskId, kind, mode, size: target ? Buffer.byteLength(target) : 0, createdAt: stamp, modifiedAt: stamp, uid: identity.uid, gid: identity.gid, links: 1, ...(target ? { target } : {}) };
    if (kind === 'directory') this.children.set(inode.id, new Map());
    this.save(inode);
    return inode;
  }

  private save(inode: VfsInodeRecord): void {
    this.inodes.set(inode.id, inode);
    this.record({ t: 'inode', n: { ...inode } });
  }

  private attach(dirId: string, name: string, id: string): void {
    this.children.get(dirId)!.set(this.key(name), { name, id });
    this.record({ t: 'link', d: dirId, m: name, i: id });
  }

  private detach(dirId: string, name: string): void {
    this.children.get(dirId)?.delete(this.key(name));
    this.record({ t: 'unlink', d: dirId, m: name });
  }

  private contentPath(inode: InodeRecord): string { return path.join(this.rootDir, inode.diskId, inode.id); }

  private async writeBlob(inode: InodeRecord, bytes: Buffer): Promise<void> {
    const blob = this.contentPath(inode);
    await hostMkdir(path.dirname(blob), { recursive: true });
    const tmp = `${blob}.${randomUUID()}.tmp`;
    await hostWrite(tmp, bytes);
    await hostRename(tmp, blob);
  }

  /** Drops one name; the inode and its blob only die when the last hard link goes. */
  private async release(inode: VfsInodeRecord): Promise<void> {
    if (inode.links > 1) { this.save({ ...inode, links: inode.links - 1 }); return; }
    if (inode.kind === 'file') await hostRemove(this.contentPath(inode), { force: true });
    this.inodes.delete(inode.id);
    this.children.delete(inode.id);
    this.record({ t: 'drop', i: inode.id });
  }

  private async removeTree(inode: VfsInodeRecord): Promise<void> {
    if (inode.kind === 'directory') {
      for (const entry of [...(this.children.get(inode.id)?.values() ?? [])]) {
        this.detach(inode.id, entry.name);
        await this.removeTree(this.inodeOf(entry.id));
      }
    }
    await this.release(inode);
  }

  // --------------------------------------------------------------- public write

  async mkdir(input: string, diskId = this.defaultDisk, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(() => this.mkdirAt(input, diskId, this.who(options)));
  }

  private async mkdirAt(input: string, diskId: string, identity: VfsIdentity): Promise<VfsInodeRecord> {
    const target = canonicalPath(input);
    if (target === '/') return this.inodeOf(this.rootId);
    const parentPath = canonicalPath(`${target}/..`);
    if (!this.tryLocate(parentPath)?.inode) await this.mkdirAt(parentPath, diskId, identity);
    const found = this.locate(target, { identity });
    if (found.inode) {
      if (found.inode.kind !== 'directory') throw new Error(`not a directory: ${found.path}`);
      return found.inode;
    }
    const parent = this.inodeOf(found.parentId!);
    this.require(parent, identity, WRITE | EXECUTE, found.parentPath);
    const inode = this.createInode('directory', diskId, 0o755, identity);
    this.attach(parent.id, found.name, inode.id);
    return inode;
  }

  async writeFile(input: string, content: string | Uint8Array, diskId = this.defaultDisk, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(() => this.writeFileAt(input, content, diskId, this.who(options)));
  }

  private async writeFileAt(input: string, content: string | Uint8Array, diskId: string, identity: VfsIdentity): Promise<VfsInodeRecord> {
    const target = canonicalPath(input);
    await this.mkdirAt(canonicalPath(`${target}/..`), diskId, identity);
    const found = this.locate(target, { identity });
    if (found.inode?.kind === 'directory') throw new Error(`is a directory: ${found.path}`);
    const bytes = typeof content === 'string' ? Buffer.from(content, 'utf8') : Buffer.from(content.buffer, content.byteOffset, content.byteLength);
    let inode = found.inode;
    if (inode) {
      this.require(inode, identity, WRITE, found.path);
    } else {
      const parent = this.inodeOf(found.parentId!);
      this.require(parent, identity, WRITE | EXECUTE, found.parentPath);
      inode = this.createInode('file', diskId, 0o644, identity);
      this.attach(parent.id, found.name, inode.id);
    }
    await this.writeBlob(inode, bytes);
    const updated = { ...inode, size: bytes.byteLength, modifiedAt: now() };
    this.save(updated);
    return updated;
  }

  // ---------------------------------------------------------------- public read

  /** Byte-exact primitive: what went in comes back out, magic numbers and all. */
  async readBytes(input: string, options: VfsCallOptions = {}): Promise<Buffer> {
    const identity = this.who(options);
    const found = this.locate(input, { identity });
    if (!found.inode) throw new Error(`no such file: ${found.path}`);
    if (found.inode.kind !== 'file') throw new Error(`not a file: ${found.path}`);
    this.require(found.inode, identity, READ, found.path);
    return hostRead(this.contentPath(found.inode));
  }

  /** Compatibility view of {@link readBytes}: a lossy utf8 decode for text callers. */
  async readFile(input: string, options: VfsCallOptions = {}): Promise<string> {
    return (await this.readBytes(input, options)).toString('utf8');
  }

  /** True when the bytes are not valid utf8 or contain a NUL, the usual `file`/`grep` heuristic. */
  async isBinary(input: string, options: VfsCallOptions = {}): Promise<boolean> {
    const bytes = await this.readBytes(input, options);
    return bytes.includes(0) || !Buffer.from(bytes.toString('utf8'), 'utf8').equals(bytes);
  }

  statSync(input: string): InodeRecord | undefined {
    const found = this.tryLocate(input);
    if (found?.inode) return found.inode;
    return this.tryLocate(input, { follow: false })?.inode; // dangling symlink: report the link itself
  }

  /** `stat` without following a final symlink. */
  lstatSync(input: string): InodeRecord | undefined { return this.tryLocate(input, { follow: false })?.inode; }

  exists(input: string): boolean { return this.statSync(input) !== undefined; }

  /** Canonical path with symlinks resolved and stored casing restored. */
  realpath(input: string, options: VfsCallOptions = {}): string {
    const found = this.locate(input, { identity: this.who(options) });
    if (!found.inode) throw new Error(`no such file or directory: ${found.path}`);
    return found.path;
  }

  readlink(input: string): string {
    const found = this.locate(input, { follow: false });
    if (found.inode?.kind !== 'symlink') throw new Error(`not a symlink: ${found.path}`);
    return found.inode.target ?? '';
  }

  list(input: string, options: VfsCallOptions = {}): DirectoryEntry[] {
    const identity = this.who(options);
    const found = this.locate(input, { identity });
    if (!found.inode || found.inode.kind !== 'directory') throw new Error(`not a directory: ${canonicalPath(input)}`);
    this.require(found.inode, identity, READ, found.path);
    const entries: DirectoryEntry[] = [];
    for (const entry of this.children.get(found.id!)?.values() ?? []) {
      const inode = this.inodes.get(entry.id);
      if (inode) entries.push({ name: entry.name, path: joinPath(found.path, entry.name), inode: { ...inode } });
    }
    return entries.sort((a, b) => Number(b.inode.kind === 'directory') - Number(a.inode.kind === 'directory') || a.name.localeCompare(b.name));
  }

  // ------------------------------------------------------------------ namespace

  /** Recursive by default for compatibility; `recursive: false` refuses to empty a directory. */
  async remove(input: string, options: RemoveOptions = {}): Promise<void> {
    return this.lock(async () => {
      const identity = this.who(options);
      let found: Located;
      try {
        found = this.locate(input, { follow: false, identity });
      } catch (error) {
        if (!/^no such file|^not a directory/.test((error as Error).message)) throw error; // never swallow EACCES
        return;
      }
      if (!found.inode || !found.parentId) return;
      const parent = this.inodeOf(found.parentId);
      this.require(parent, identity, WRITE | EXECUTE, found.parentPath);
      if (options.recursive === false && found.inode.kind === 'directory' && (this.children.get(found.inode.id)?.size ?? 0) > 0) {
        throw new Error(`directory not empty: ${found.path}`);
      }
      this.detach(found.parentId, found.name);
      await this.removeTree(found.inode);
    });
  }

  /**
   * POSIX `mv`: moves across directories, replaces an existing file unless
   * `overwrite: false`, and moves *into* `to` when `to` is an existing directory.
   */
  async rename(from: string, to: string, options: RenameOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const source = this.locate(from, { follow: false, identity });
      if (!source.inode || !source.parentId) throw new Error(`no such file or directory: ${source.path}`);
      const target = this.destination(to, source.name, identity);
      if (target.path === source.path) return source.inode;
      if (source.inode.kind === 'directory' && target.path.startsWith(`${source.path}/`)) throw new Error(`cannot move ${source.path} into itself`);
      this.require(this.inodeOf(source.parentId), identity, WRITE | EXECUTE, source.parentPath);
      const parent = this.inodeOf(target.parentId!);
      this.require(parent, identity, WRITE | EXECUTE, target.parentPath);
      if (target.inode) {
        if (options.overwrite === false) throw new Error(`file exists: ${target.path}`);
        if (target.inode.kind === 'directory') throw new Error(`is a directory: ${target.path}`);
        this.detach(target.parentId!, target.name);
        await this.release(target.inode);
      }
      this.detach(source.parentId, source.name);
      this.attach(target.parentId!, target.name, source.inode.id);
      const moved = { ...this.inodeOf(source.inode.id), modifiedAt: now() };
      this.save(moved);
      return moved;
    });
  }

  /** Recursive by default; copies symlinks as symlinks unless `dereference`. */
  async copy(from: string, to: string, options: CopyOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const source = this.locate(from, { follow: options.dereference === true, identity });
      if (!source.inode) throw new Error(`no such file or directory: ${source.path}`);
      const target = this.destination(to, source.name, identity);
      if (target.path === source.path || (source.inode.kind === 'directory' && target.path.startsWith(`${source.path}/`))) throw new Error(`cannot copy ${source.path} into itself`);
      if (source.inode.kind === 'directory' && options.recursive === false) throw new Error(`is a directory: ${source.path}`);
      return this.copyInto(source.inode, source.path, target, identity, options);
    });
  }

  private async copyInto(source: VfsInodeRecord, sourcePath: string, target: Located, identity: VfsIdentity, options: CopyOptions): Promise<VfsInodeRecord> {
    if (target.inode) {
      if (options.overwrite === false) throw new Error(`file exists: ${target.path}`);
      if (target.inode.kind === 'directory' && source.kind === 'directory') {
        for (const entry of [...(this.children.get(source.id)?.values() ?? [])]) {
          await this.copyInto(this.inodeOf(entry.id), joinPath(sourcePath, entry.name), this.locate(joinPath(target.path, entry.name), { follow: false, identity }), identity, options);
        }
        return target.inode;
      }
      this.detach(target.parentId!, target.name);
      await this.removeTree(target.inode);
    }
    const parent = this.inodeOf(target.parentId!);
    this.require(parent, identity, WRITE | EXECUTE, target.parentPath);
    this.require(source, identity, READ, sourcePath);
    const copy = this.createInode(source.kind, source.diskId, source.mode, identity, source.target);
    this.attach(parent.id, target.name, copy.id);
    if (source.kind === 'file') {
      await this.writeBlob(copy, await hostRead(this.contentPath(source)));
      this.save({ ...copy, size: source.size });
    }
    if (source.kind === 'directory') {
      for (const entry of [...(this.children.get(source.id)?.values() ?? [])]) {
        const child = this.locate(joinPath(target.path, entry.name), { follow: false, identity });
        await this.copyInto(this.inodeOf(entry.id), joinPath(sourcePath, entry.name), child, identity, options);
      }
    }
    return this.inodeOf(copy.id);
  }

  /** Shared destination rule for rename/copy: an existing directory means "put it inside". */
  private destination(to: string, sourceName: string, identity: VfsIdentity): Located {
    const found = this.locate(to, { follow: false, identity });
    if (found.inode?.kind === 'directory') return this.locate(joinPath(found.path, sourceName), { follow: false, identity });
    if (found.inode?.kind === 'symlink') {
      const behind = this.tryLocate(found.path, { identity });
      if (behind?.inode?.kind === 'directory') return this.locate(joinPath(behind.path, sourceName), { follow: false, identity });
    }
    return found;
  }

  async symlink(target: string, linkPath: string, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const found = this.locate(linkPath, { follow: false, identity });
      if (found.inode) throw new Error(`file exists: ${found.path}`);
      const parent = this.inodeOf(found.parentId!);
      this.require(parent, identity, WRITE | EXECUTE, found.parentPath);
      const inode = this.createInode('symlink', parent.diskId, 0o777, identity, target);
      this.attach(parent.id, found.name, inode.id);
      return inode;
    });
  }

  /** Hard link: a second name for the same inode, with a real link count. */
  async link(from: string, to: string, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const source = this.locate(from, { identity });
      if (!source.inode) throw new Error(`no such file or directory: ${source.path}`);
      if (source.inode.kind === 'directory') throw new Error(`hard link not allowed for directory: ${source.path}`);
      const target = this.destination(to, source.name, identity);
      if (target.inode) throw new Error(`file exists: ${target.path}`);
      const parent = this.inodeOf(target.parentId!);
      this.require(parent, identity, WRITE | EXECUTE, target.parentPath);
      const linked = { ...this.inodeOf(source.inode.id), links: source.inode.links + 1 };
      this.save(linked);
      this.attach(parent.id, target.name, linked.id);
      return linked;
    });
  }

  // --------------------------------------------------------------- permissions

  owner(input: string): { uid: number; gid: number; mode: number } {
    const found = this.locate(input, { follow: false });
    if (!found.inode) throw new Error(`no such file or directory: ${found.path}`);
    return { uid: found.inode.uid, gid: found.inode.gid, mode: found.inode.mode };
  }

  async chmod(input: string, mode: number, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const found = this.locate(input, { identity });
      if (!found.inode) throw new Error(`no such file or directory: ${found.path}`);
      this.requireOwner(found.inode, identity, found.path);
      const updated = { ...found.inode, mode: mode & 0o7777, modifiedAt: now() };
      this.save(updated);
      return updated;
    });
  }

  async chown(input: string, uid: number, gid: number, options: VfsCallOptions = {}): Promise<InodeRecord> {
    return this.lock(async () => {
      const identity = this.who(options);
      const found = this.locate(input, { identity });
      if (!found.inode) throw new Error(`no such file or directory: ${found.path}`);
      if (this.enforce && identity.uid !== 0) throw new Error(`operation not permitted: ${found.path}`);
      const updated = { ...found.inode, uid, gid, modifiedAt: now() };
      this.save(updated);
      return updated;
    });
  }

  /** Bound view that carries an identity into every call, for callers that act as a user. */
  as(identity: VfsIdentity): VfsUserView {
    return {
      readBytes: (input) => this.readBytes(input, { as: identity }),
      readFile: (input) => this.readFile(input, { as: identity }),
      writeFile: (input, content, diskId) => this.writeFile(input, content, diskId ?? this.defaultDisk, { as: identity }),
      mkdir: (input, diskId) => this.mkdir(input, diskId ?? this.defaultDisk, { as: identity }),
      list: (input) => this.list(input, { as: identity }),
      remove: (input, options) => this.remove(input, { ...options, as: identity }),
    };
  }

  // ---------------------------------------------------------------- inspection

  resolve(input: string, cwd = '/'): string { return canonicalPath(input, cwd); }

  usage(): { files: number; directories: number; bytes: number; digest: string } {
    this.kick();
    const values = [...this.inodes.values()];
    const layout = this.hostLayout();
    const summary = Object.entries(layout.paths).map(([candidate, id]) => {
      const inode = layout.inodes[id]!;
      return `${candidate}:${inode.kind}:${inode.size}:${inode.mode}`;
    }).sort();
    return {
      files: values.filter((value) => value.kind === 'file').length,
      directories: values.filter((value) => value.kind === 'directory').length,
      bytes: values.reduce((sum, value) => sum + value.size, 0),
      digest: createHash('sha256').update(JSON.stringify(summary)).digest('hex').slice(0, 16),
    };
  }

  /** Flat `{ version, paths, inodes }` view derived from the inode graph for git/software. */
  hostLayout(): FileTable {
    const paths: Record<string, string> = {};
    const inodes: Record<string, InodeRecord> = {};
    const walk = (id: string, prefix: string): void => {
      const inode = this.inodes.get(id);
      if (!inode) return;
      paths[prefix] = id;
      inodes[id] = { ...inode };
      for (const entry of this.children.get(id)?.values() ?? []) walk(entry.id, joinPath(prefix, entry.name));
    };
    walk(this.rootId, '/');
    return { version: 1, paths, inodes };
  }

  async verifyContent(): Promise<string[]> {
    await this.flush();
    const problems: string[] = [];
    for (const inode of this.inodes.values()) {
      if (inode.kind !== 'file') continue;
      try {
        const info = await hostStat(this.contentPath(inode));
        if (info.size !== inode.size) problems.push(`${inode.id}: expected ${inode.size}, got ${info.size}`);
      } catch { problems.push(`${inode.id}: missing content blob`); }
    }
    return problems;
  }
}

function migrateLegacy(table: LegacyTable): TableSnapshot {
  const inodes: TableSnapshot['inodes'] = {};
  for (const [id, inode] of Object.entries(table.inodes)) inodes[id] = { ...inode, id, uid: 0, gid: 0, links: 1, ...(inode.kind === 'directory' ? { children: {} } : {}) };
  const root = table.paths['/'] ?? '';
  for (const [candidate, id] of Object.entries(table.paths).sort(([a], [b]) => a.length - b.length)) {
    if (candidate === '/') continue;
    const parent = inodes[table.paths[canonicalPath(`${candidate}/..`)] ?? ''];
    if (parent?.children) parent.children[candidate.slice(candidate.lastIndexOf('/') + 1)] = id;
  }
  return { version: 2, sequence: 0, root, inodes };
}
