/**
 * The staging area.
 *
 * Entries carry the blob id plus enough stat information for `status` to tell
 * "staged" from "modified in the working tree", and a merge stage so conflicted
 * paths can hold base/ours/theirs simultaneously.
 *
 * Persistence writes a genuine binary index (`DIRC` version 2, SHA-1 trailer)
 * at the standard `.git/index` path. It is write-only from the simulator's
 * point of view: the virtual filesystem decodes file reads as UTF-8, so the
 * binary form cannot be read back — in-memory state is authoritative and a
 * cold start rebuilds the index from the HEAD tree.
 */

import { createHash } from 'node:crypto';
import { MODE_FILE } from './objects.js';

export interface GitIndexEntry {
  path: string;
  hash: string;
  mode: string;
  size: number;
  /** Milliseconds; mirrors the VFS `modifiedAt` stamp of the worktree file. */
  mtimeMs: number;
  /** 0 = merged, 1 = base, 2 = ours, 3 = theirs. */
  stage: number;
}

export class GitIndex {
  private entries = new Map<string, GitIndexEntry>();

  private static key(path: string, stage: number): string { return `${path}\0${stage}`; }

  clear(): void { this.entries.clear(); }

  get size(): number { return this.entries.size; }

  set(entry: GitIndexEntry): void {
    this.entries.set(GitIndex.key(entry.path, entry.stage), { ...entry });
  }

  get(path: string, stage = 0): GitIndexEntry | undefined {
    return this.entries.get(GitIndex.key(path, stage));
  }

  delete(path: string, stage?: number): void {
    if (stage === undefined) {
      for (const candidate of [0, 1, 2, 3]) this.entries.delete(GitIndex.key(path, candidate));
    } else this.entries.delete(GitIndex.key(path, stage));
  }

  /** Stage-0 (merged) entries, sorted by path. */
  staged(): GitIndexEntry[] {
    return [...this.entries.values()].filter((entry) => entry.stage === 0).sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  }

  all(): GitIndexEntry[] {
    return [...this.entries.values()].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0) || a.stage - b.stage);
  }

  /** Paths with unresolved merge stages. */
  conflicts(): string[] {
    const paths = new Set<string>();
    for (const entry of this.entries.values()) if (entry.stage > 0) paths.add(entry.path);
    return [...paths].sort();
  }

  hasConflicts(): boolean {
    for (const entry of this.entries.values()) if (entry.stage > 0) return true;
    return false;
  }

  paths(): string[] { return this.staged().map((entry) => entry.path); }

  replaceAll(entries: readonly GitIndexEntry[]): void {
    this.clear();
    for (const entry of entries) this.set(entry);
  }

}

function modeBits(mode: string): number {
  return Number.parseInt(mode, 8);
}

/** Real `DIRC` v2 index bytes for `.git/index`. */
export function encodeIndexFile(entries: readonly GitIndexEntry[]): Buffer {
  const sorted = [...entries].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0) || a.stage - b.stage);
  const header = Buffer.alloc(12);
  header.write('DIRC', 0, 'ascii');
  header.writeUInt32BE(2, 4);
  header.writeUInt32BE(sorted.length, 8);
  const chunks: Buffer[] = [header];
  for (const entry of sorted) {
    const pathBytes = Buffer.from(entry.path, 'utf8');
    const fixed = Buffer.alloc(62);
    const seconds = Math.floor(entry.mtimeMs / 1000);
    const nanos = Math.floor((entry.mtimeMs % 1000) * 1e6);
    fixed.writeUInt32BE(seconds, 0); // ctime seconds
    fixed.writeUInt32BE(nanos, 4);
    fixed.writeUInt32BE(seconds, 8); // mtime seconds
    fixed.writeUInt32BE(nanos, 12);
    fixed.writeUInt32BE(0, 16); // dev
    fixed.writeUInt32BE(0, 20); // ino
    fixed.writeUInt32BE(modeBits(entry.mode || MODE_FILE), 24);
    fixed.writeUInt32BE(0, 28); // uid
    fixed.writeUInt32BE(0, 32); // gid
    fixed.writeUInt32BE(entry.size, 36);
    Buffer.from(entry.hash, 'hex').copy(fixed, 40);
    fixed.writeUInt16BE(((entry.stage & 0x3) << 12) | Math.min(pathBytes.length, 0xfff), 60);
    const unpadded = 62 + pathBytes.length;
    const padded = Math.ceil((unpadded + 1) / 8) * 8;
    const record = Buffer.alloc(padded);
    fixed.copy(record, 0);
    pathBytes.copy(record, 62);
    chunks.push(record);
  }
  const body = Buffer.concat(chunks);
  const checksum = createHash('sha1').update(body).digest();
  return Buffer.concat([body, checksum]);
}
