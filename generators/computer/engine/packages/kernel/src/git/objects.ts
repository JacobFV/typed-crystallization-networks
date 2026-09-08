/**
 * Content-addressed Git object model.
 *
 * Hashing uses Git's real canonical encodings (`<type> <length>\0<payload>` with
 * SHA-1), so a Seed blob/commit hash equals the hash produced by `git
 * hash-object` for the same bytes. Only the *on-disk* encoding deviates: loose
 * objects are stored uncompressed, and tree entries write the object id as 40
 * hex characters instead of 20 raw bytes, because the virtual filesystem can
 * only hand text back to a reader (`readFile` decodes UTF-8). Everything that
 * participates in identity is still the canonical binary form.
 */

import { createHash } from 'node:crypto';

export type GitObjectType = 'blob' | 'tree' | 'commit' | 'tag';

export const MODE_FILE = '100644';
export const MODE_EXEC = '100755';
export const MODE_TREE = '40000';

export interface GitTreeEntry {
  /** Octal mode string exactly as Git writes it (`100644`, `40000`, ...). */
  mode: string;
  name: string;
  hash: string;
}

export interface GitIdentity {
  name: string;
  email: string;
  /** Seconds since the Unix epoch. Never read from the wall clock in here. */
  timestamp: number;
  /** `+0000` style offset. */
  timezone: string;
}

export interface GitCommitObject {
  tree: string;
  parents: string[];
  author: GitIdentity;
  committer: GitIdentity;
  message: string;
}

export interface GitAnnotatedTag {
  object: string;
  type: GitObjectType;
  tag: string;
  tagger: GitIdentity;
  message: string;
}

export const EMPTY_TREE_HASH = '4b825dc642cb6eb9a060e54bf8d69288fbee4904';

export function formatIdentity(identity: GitIdentity): string {
  return `${identity.name} <${identity.email}> ${identity.timestamp} ${identity.timezone}`;
}

export function parseIdentity(value: string): GitIdentity {
  const match = value.match(/^(.*?) <([^>]*)> (\d+) ([+-]\d{4})$/);
  if (!match) return { name: value.trim(), email: '', timestamp: 0, timezone: '+0000' };
  return { name: match[1]!, email: match[2]!, timestamp: Number(match[3]), timezone: match[4]! };
}

export function identityLabel(identity: GitIdentity): string {
  return `${identity.name} <${identity.email}>`;
}

/** ISO-8601 rendering of an identity stamp, used for `GitCommitRecord.at`. */
export function identityIso(identity: GitIdentity): string {
  return new Date(identity.timestamp * 1000).toISOString();
}

/** Git sorts tree entries by name, with directory names compared as `name/`. */
export function sortTreeEntries(entries: readonly GitTreeEntry[]): GitTreeEntry[] {
  return [...entries].sort((a, b) => {
    const left = a.mode === MODE_TREE ? `${a.name}/` : a.name;
    const right = b.mode === MODE_TREE ? `${b.name}/` : b.name;
    return left < right ? -1 : left > right ? 1 : 0;
  });
}

/** Canonical (real Git) payload bytes for a tree. */
export function encodeTreePayload(entries: readonly GitTreeEntry[]): Buffer {
  const parts: Buffer[] = [];
  for (const entry of sortTreeEntries(entries)) {
    parts.push(Buffer.from(`${entry.mode} ${entry.name}\0`, 'utf8'));
    parts.push(Buffer.from(entry.hash, 'hex'));
  }
  return Buffer.concat(parts);
}

export function encodeCommitPayload(commit: GitCommitObject): Buffer {
  const lines = [`tree ${commit.tree}`];
  for (const parent of commit.parents) lines.push(`parent ${parent}`);
  lines.push(`author ${formatIdentity(commit.author)}`);
  lines.push(`committer ${formatIdentity(commit.committer)}`);
  const message = commit.message.endsWith('\n') ? commit.message : `${commit.message}\n`;
  return Buffer.from(`${lines.join('\n')}\n\n${message}`, 'utf8');
}

export function decodeCommitPayload(text: string): GitCommitObject {
  const separator = text.indexOf('\n\n');
  const header = separator === -1 ? text : text.slice(0, separator);
  const message = separator === -1 ? '' : text.slice(separator + 2);
  const commit: GitCommitObject = {
    tree: '', parents: [],
    author: { name: 'agent', email: 'agent@seed.local', timestamp: 0, timezone: '+0000' },
    committer: { name: 'agent', email: 'agent@seed.local', timestamp: 0, timezone: '+0000' },
    message,
  };
  for (const line of header.split('\n')) {
    const space = line.indexOf(' ');
    if (space === -1) continue;
    const key = line.slice(0, space);
    const value = line.slice(space + 1);
    if (key === 'tree') commit.tree = value;
    else if (key === 'parent') commit.parents.push(value);
    else if (key === 'author') commit.author = parseIdentity(value);
    else if (key === 'committer') commit.committer = parseIdentity(value);
  }
  return commit;
}

export function encodeTagPayload(tag: GitAnnotatedTag): Buffer {
  const message = tag.message.endsWith('\n') ? tag.message : `${tag.message}\n`;
  return Buffer.from(
    `object ${tag.object}\ntype ${tag.type}\ntag ${tag.tag}\ntagger ${formatIdentity(tag.tagger)}\n\n${message}`,
    'utf8',
  );
}

export function decodeTagPayload(text: string): GitAnnotatedTag {
  const separator = text.indexOf('\n\n');
  const header = separator === -1 ? text : text.slice(0, separator);
  const message = separator === -1 ? '' : text.slice(separator + 2);
  const tag: GitAnnotatedTag = {
    object: '', type: 'commit', tag: '',
    tagger: { name: 'agent', email: 'agent@seed.local', timestamp: 0, timezone: '+0000' },
    message,
  };
  for (const line of header.split('\n')) {
    const space = line.indexOf(' ');
    if (space === -1) continue;
    const key = line.slice(0, space);
    const value = line.slice(space + 1);
    if (key === 'object') tag.object = value;
    else if (key === 'type') tag.type = value as GitObjectType;
    else if (key === 'tag') tag.tag = value;
    else if (key === 'tagger') tag.tagger = parseIdentity(value);
  }
  return tag;
}

/** Real Git object id: sha1 over `<type> <payload length>\0<payload>`. */
export function hashObject(type: GitObjectType, payload: Buffer): string {
  return createHash('sha1')
    .update(Buffer.concat([Buffer.from(`${type} ${payload.length}\0`, 'utf8'), payload]))
    .digest('hex');
}

export function hashBlob(content: string): string {
  return hashObject('blob', Buffer.from(content, 'utf8'));
}

export function hashTree(entries: readonly GitTreeEntry[]): string {
  return hashObject('tree', encodeTreePayload(entries));
}

export function hashCommit(commit: GitCommitObject): string {
  return hashObject('commit', encodeCommitPayload(commit));
}

/** Text transliteration of a tree payload: 40 hex chars replace the raw id. */
function encodeTreeText(entries: readonly GitTreeEntry[]): string {
  return sortTreeEntries(entries).map((entry) => `${entry.mode} ${entry.name}\0${entry.hash}`).join('');
}

function decodeTreeText(text: string): GitTreeEntry[] {
  const entries: GitTreeEntry[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    const nul = text.indexOf('\0', cursor);
    if (nul === -1) break;
    const head = text.slice(cursor, nul);
    const space = head.indexOf(' ');
    const mode = head.slice(0, space);
    const name = head.slice(space + 1);
    const hash = text.slice(nul + 1, nul + 41);
    entries.push({ mode, name, hash });
    cursor = nul + 41;
  }
  return entries;
}

/**
 * Serialized loose-object text. For blobs, commits and tags this is byte for
 * byte the uncompressed canonical Git object; trees hex-expand their ids.
 */
export function serializeObject(type: GitObjectType, body: string): string {
  return `${type} ${Buffer.byteLength(body, 'utf8')}\0${body}`;
}

export function parseSerializedObject(raw: string): { type: GitObjectType; body: string } {
  const nul = raw.indexOf('\0');
  if (nul === -1) throw new Error('fatal: malformed object file');
  const header = raw.slice(0, nul);
  const space = header.indexOf(' ');
  const type = header.slice(0, space) as GitObjectType;
  return { type, body: raw.slice(nul + 1) };
}

export function textToTreeEntries(body: string): GitTreeEntry[] {
  return decodeTreeText(body);
}

/** Minimal filesystem surface the object store needs from the VFS. */
export interface ObjectStorage {
  readFile(path: string): Promise<string>;
  writeFile(path: string, content: string | Uint8Array): Promise<unknown>;
  mkdir(path: string): Promise<unknown>;
  statSync(path: string): { kind: string } | undefined;
}

export type LooseObject = { type: GitObjectType; body: string };

/**
 * Loose object database rooted at `<gitdir>/objects`, mirroring Git's
 * `xx/yyyy...` fan-out. An in-memory cache keeps replay cheap; the VFS is the
 * durable copy.
 */
export class GitObjectStore {
  private readonly cache = new Map<string, LooseObject>();

  constructor(private readonly storage: ObjectStorage, private gitDir: string) {}

  setGitDir(gitDir: string): void { this.gitDir = gitDir; }

  private pathFor(hash: string): string {
    return `${this.gitDir}/objects/${hash.slice(0, 2)}/${hash.slice(2)}`;
  }

  has(hash: string): boolean {
    if (!/^[0-9a-f]{40}$/.test(hash)) return false;
    return this.cache.has(hash) || this.storage.statSync(this.pathFor(hash))?.kind === 'file';
  }

  /** Insert an already-serialized object (used by the remote transport). */
  async put(hash: string, serialized: string): Promise<void> {
    const { type, body } = parseSerializedObject(serialized);
    this.cache.set(hash, { type, body });
    await this.storage.mkdir(`${this.gitDir}/objects/${hash.slice(0, 2)}`);
    await this.storage.writeFile(this.pathFor(hash), serialized);
  }

  private async write(hash: string, object: LooseObject): Promise<string> {
    if (!this.cache.has(hash)) this.cache.set(hash, object);
    if (this.storage.statSync(this.pathFor(hash))?.kind !== 'file') {
      await this.storage.mkdir(`${this.gitDir}/objects/${hash.slice(0, 2)}`);
      await this.storage.writeFile(this.pathFor(hash), serializeObject(object.type, object.body));
    }
    return hash;
  }

  async writeBlob(content: string): Promise<string> {
    return this.write(hashBlob(content), { type: 'blob', body: content });
  }

  async writeTree(entries: readonly GitTreeEntry[]): Promise<string> {
    const sorted = sortTreeEntries(entries);
    return this.write(hashTree(sorted), { type: 'tree', body: encodeTreeText(sorted) });
  }

  async writeCommit(commit: GitCommitObject): Promise<string> {
    const body = encodeCommitPayload(commit).toString('utf8');
    return this.write(hashCommit(commit), { type: 'commit', body });
  }

  async writeTag(tag: GitAnnotatedTag): Promise<string> {
    const payload = encodeTagPayload(tag);
    return this.write(hashObject('tag', payload), { type: 'tag', body: payload.toString('utf8') });
  }

  async read(hash: string): Promise<LooseObject> {
    const cached = this.cache.get(hash);
    if (cached) return cached;
    const raw = await this.storage.readFile(this.pathFor(hash));
    const object = parseSerializedObject(raw);
    this.cache.set(hash, object);
    return object;
  }

  async readBlob(hash: string): Promise<string> {
    const object = await this.read(hash);
    if (object.type !== 'blob') throw new Error(`fatal: not a blob object: ${hash}`);
    return object.body;
  }

  async readTree(hash: string): Promise<GitTreeEntry[]> {
    if (hash === EMPTY_TREE_HASH && !this.has(hash)) return [];
    const object = await this.read(hash);
    if (object.type !== 'tree') throw new Error(`fatal: not a tree object: ${hash}`);
    return decodeTreeText(object.body);
  }

  async readCommit(hash: string): Promise<GitCommitObject> {
    const object = await this.read(hash);
    if (object.type === 'tag') return this.readCommit(decodeTagPayload(object.body).object);
    if (object.type !== 'commit') throw new Error(`fatal: not a commit object: ${hash}`);
    return decodeCommitPayload(object.body);
  }

  async type(hash: string): Promise<GitObjectType> {
    return (await this.read(hash)).type;
  }

  /** Serialized form, for shipping objects across the virtual HTTPS fabric. */
  async serialize(hash: string): Promise<string> {
    const object = await this.read(hash);
    return serializeObject(object.type, object.body);
  }

  /** Byte size of the canonical (binary) object payload, as `cat-file -s` reports. */
  async size(hash: string): Promise<number> {
    const object = await this.read(hash);
    if (object.type === 'tree') return encodeTreePayload(decodeTreeText(object.body)).length;
    return Buffer.byteLength(object.body, 'utf8');
  }
}
