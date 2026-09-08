import { randomUUID } from './determinism.js';
import { mkdtemp, readdir, readFile, stat, writeFile as writeHostFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import type { ComputerSpec } from '@tcn-computer/protocol';
import { canonicalPath, VirtualFileSystem, type VfsOptions } from './vfs.js';

const spec: ComputerSpec = {
  id: 'vfs-fixture', hostname: 'vfs-fixture', os: 'ubuntu', shell: 'bash', ipv4: '10.42.0.99',
  memoryBytes: 1024, cpuCores: 1, disks: [{ id: 'disk0', label: 'root', mount: '/', capacityBytes: 1_000_000 }], displays: [],
};

let stateRoot = '';

async function open(options: VfsOptions = {}, runId = 'run'): Promise<VirtualFileSystem> {
  const vfs = new VirtualFileSystem(stateRoot, runId, spec, options);
  await vfs.initialize();
  return vfs;
}

beforeEach(async () => { stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-vfs-unit-')); });

describe('binary fidelity', () => {
  it('round-trips arbitrary bytes without utf8 corruption', async () => {
    const vfs = await open();
    const png = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0xd8, 0x00, 0x01]);
    const inode = await vfs.writeFile('/assets/logo.png', png);
    const bytes = await vfs.readBytes('/assets/logo.png');
    expect([...bytes]).toEqual([...png]);
    expect(bytes.byteLength).toBe(png.byteLength);
    expect(inode.size).toBe(png.byteLength);
    expect(await vfs.isBinary('/assets/logo.png')).toBe(true);
    expect(await vfs.verifyContent()).toEqual([]);
  });

  it('survives a byte-level rewrite cycle through readBytes', async () => {
    const vfs = await open();
    const payload = Buffer.from([0x00, 0xff, 0xfe, 0x41, 0x80, 0x7f]);
    await vfs.writeFile('/blob.bin', payload);
    for (let round = 0; round < 3; round += 1) await vfs.writeFile('/blob.bin', await vfs.readBytes('/blob.bin'));
    expect(await vfs.readBytes('/blob.bin')).toEqual(payload);
  });

  it('keeps readFile a faithful utf8 view for text', async () => {
    const vfs = await open();
    const text = 'héllo — seed ✅\n';
    await vfs.writeFile('/notes.txt', text);
    expect(await vfs.readFile('/notes.txt')).toBe(text);
    expect((await vfs.readBytes('/notes.txt')).byteLength).toBe(Buffer.byteLength(text));
    expect(await vfs.isBinary('/notes.txt')).toBe(false);
  });

  it('writes a byte view without dragging in the rest of its backing buffer', async () => {
    const vfs = await open();
    const backing = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    await vfs.writeFile('/slice.bin', backing.subarray(2, 5));
    expect([...(await vfs.readBytes('/slice.bin'))]).toEqual([3, 4, 5]);
  });
});

describe('directory entries', () => {
  it('lists only real children and keeps the sort contract', async () => {
    const vfs = await open();
    await vfs.writeFile('/home/agent/a.txt', 'a');
    await vfs.writeFile('/home/agent/nested/deep/b.txt', 'b');
    await vfs.mkdir('/home/agent/zdir');
    await vfs.writeFile('/home/agentical/decoy.txt', 'decoy'); // prefix-matching trap for the old scanner
    const entries = vfs.list('/home/agent');
    expect(entries.map((entry) => entry.name)).toEqual(['nested', 'zdir', 'a.txt']);
    expect(entries.map((entry) => entry.path)).toEqual(['/home/agent/nested', '/home/agent/zdir', '/home/agent/a.txt']);
    expect(() => vfs.list('/home/agent/a.txt')).toThrow(/not a directory/);
  });

  it('derives a hostLayout table compatible with the flat path map', async () => {
    const vfs = await open();
    const inode = await vfs.writeFile('/home/agent/Desktop/capture.txt', 'trajectory-data');
    const layout = vfs.hostLayout();
    expect(layout.version).toBe(1);
    expect(layout.paths['/home/agent/Desktop/capture.txt']).toBe(inode.id);
    expect(layout.paths['/']).toBeDefined();
    expect(layout.paths['/home']).toBeDefined();
    expect(layout.inodes[inode.id]?.kind).toBe('file');
    expect(Object.keys(layout.paths).filter((candidate) => candidate.startsWith('/home/agent/'))).toEqual(['/home/agent/Desktop', '/home/agent/Desktop/capture.txt']);
  });

  it('scales listings to the directory, not the filesystem', async () => {
    const vfs = await open();
    await Promise.all(Array.from({ length: 400 }, (_, index) => vfs.writeFile(`/bulk/file-${index}.txt`, String(index))));
    await vfs.writeFile('/small/only.txt', 'x');
    expect(vfs.list('/bulk')).toHaveLength(400);
    expect(vfs.list('/small').map((entry) => entry.name)).toEqual(['only.txt']);
    expect(vfs.usage().files).toBe(401);
  });
});

describe('persistence', () => {
  it('restores an inode graph from the journal without rewriting the table per write', async () => {
    const vfs = await open({ journalInterval: 10_000 });
    await vfs.flush();
    const baseline = await readFile(path.join(vfs.rootDir, 'file-table.json'), 'utf8');
    for (let index = 0; index < 25; index += 1) await vfs.writeFile(`/logs/entry-${index}.txt`, `line ${index}`);
    await vfs.flush();
    expect(await readFile(path.join(vfs.rootDir, 'file-table.json'), 'utf8')).toBe(baseline);
    expect((await stat(path.join(vfs.rootDir, 'file-table.journal'))).size).toBeGreaterThan(0);

    const reopened = await open({ journalInterval: 10_000 });
    expect(reopened.list('/logs')).toHaveLength(25);
    expect(await reopened.readFile('/logs/entry-24.txt')).toBe('line 24');
    expect(await reopened.verifyContent()).toEqual([]);
  });

  it('compacts the journal into a snapshot and still restores', async () => {
    const vfs = await open({ journalInterval: 4 });
    for (let index = 0; index < 30; index += 1) await vfs.writeFile(`/c/${index}.txt`, String(index));
    await vfs.flush();
    expect((await stat(path.join(vfs.rootDir, 'file-table.journal'))).size).toBeLessThan(4096);
    const reopened = await open({ journalInterval: 4 });
    expect(reopened.list('/c')).toHaveLength(30);
    expect(await reopened.readFile('/c/29.txt')).toBe('29');
  });

  it('replays a truncated journal tail instead of failing to boot', async () => {
    const vfs = await open({ journalInterval: 10_000 });
    await vfs.writeFile('/a.txt', 'one');
    await vfs.writeFile('/b.txt', 'two');
    await vfs.flush();
    const journalPath = path.join(vfs.rootDir, 'file-table.journal');
    const journal = await readFile(journalPath, 'utf8');
    await writeHostFile(journalPath, `${journal}{"s":999,"t":"inod`);
    const reopened = await open({ journalInterval: 10_000 });
    expect(await reopened.readFile('/a.txt')).toBe('one');
    expect(await reopened.readFile('/b.txt')).toBe('two');
  });

  it('keeps the table intact under many overlapping writes', async () => {
    const vfs = await open();
    const pending = Array.from({ length: 200 }, (_, index) => vfs.writeFile(`/race/${index % 8}/file-${index}.txt`, `payload-${index}`));
    pending.push(vfs.mkdir('/race/9/deep') as unknown as Promise<never>);
    for (let index = 0; index < 40; index += 1) pending.push(vfs.writeFile('/race/contended.txt', `revision-${index}`) as unknown as Promise<never>);
    await Promise.all(pending);
    await vfs.flush();

    const table = JSON.parse(await readFile(path.join(vfs.rootDir, 'file-table.json'), 'utf8')) as { root: string };
    expect(table.root).toBeTruthy();
    expect((await readdir(vfs.rootDir)).some((entry) => entry.endsWith('.tmp'))).toBe(false);
    expect(await vfs.verifyContent()).toEqual([]);

    const reopened = await open();
    expect(reopened.list('/race').filter((entry) => entry.inode.kind === 'directory')).toHaveLength(9);
    for (let index = 0; index < 200; index += 1) expect(await reopened.readFile(`/race/${index % 8}/file-${index}.txt`)).toBe(`payload-${index}`);
    expect((await reopened.readFile('/race/contended.txt')).startsWith('revision-')).toBe(true);
    expect(reopened.statSync('/race/contended.txt')?.size).toBe((await reopened.readBytes('/race/contended.txt')).byteLength);
  });

  it('never produces two writers on one temp table file', async () => {
    const vfs = await open({ journalInterval: 2 });
    await Promise.all(Array.from({ length: 60 }, (_, index) => vfs.writeFile(`/burst/${index}.txt`, String(index))));
    await vfs.flush();
    expect((await readdir(vfs.rootDir)).filter((entry) => entry.includes('.tmp'))).toEqual([]);
    const reopened = await open({ journalInterval: 2 });
    expect(reopened.list('/burst')).toHaveLength(60);
  });
});

describe('rename and copy', () => {
  it('renames across directories and overwrites files by default', async () => {
    const vfs = await open();
    await vfs.writeFile('/src/one.txt', 'one');
    await vfs.writeFile('/dst/two.txt', 'two');
    const original = vfs.statSync('/src/one.txt')!.id;
    await vfs.rename('/src/one.txt', '/dst/two.txt');
    expect(vfs.statSync('/src/one.txt')).toBeUndefined();
    expect(await vfs.readFile('/dst/two.txt')).toBe('one');
    expect(vfs.statSync('/dst/two.txt')?.id).toBe(original);
    expect(await vfs.verifyContent()).toEqual([]);
  });

  it('moves into an existing directory and refuses unsafe moves', async () => {
    const vfs = await open();
    await vfs.writeFile('/a/keep.txt', 'keep');
    await vfs.mkdir('/b');
    await vfs.rename('/a/keep.txt', '/b');
    expect(await vfs.readFile('/b/keep.txt')).toBe('keep');
    await vfs.rename('/b', '/a');
    expect(await vfs.readFile('/a/b/keep.txt')).toBe('keep');
    await expect(vfs.rename('/a', '/a/b/loop')).rejects.toThrow(/into itself/);
    await expect(vfs.rename('/missing.txt', '/x.txt')).rejects.toThrow(/no such file/);
    await vfs.writeFile('/guard.txt', 'guard');
    await expect(vfs.rename('/a/b/keep.txt', '/guard.txt', { overwrite: false })).rejects.toThrow(/file exists/);
    await vfs.rename('/guard.txt', '/a'); // an existing directory means "move inside it"
    expect(await vfs.readFile('/a/guard.txt')).toBe('guard');
    await vfs.writeFile('/loose/b', 'collides with the directory /a/b');
    await expect(vfs.rename('/loose/b', '/a')).rejects.toThrow(/is a directory/); // /a/b already exists as a directory
  });

  it('copies files and directories with independent inodes', async () => {
    const vfs = await open();
    await vfs.writeFile('/tree/one.txt', 'one');
    await vfs.writeFile('/tree/nested/two.txt', 'two');
    await vfs.symlink('one.txt', '/tree/alias.txt');
    await vfs.copy('/tree', '/backup');
    expect(await vfs.readFile('/backup/one.txt')).toBe('one');
    expect(await vfs.readFile('/backup/nested/two.txt')).toBe('two');
    expect(vfs.lstatSync('/backup/alias.txt')?.kind).toBe('symlink');
    expect(vfs.statSync('/backup/one.txt')?.id).not.toBe(vfs.statSync('/tree/one.txt')?.id);
    await vfs.writeFile('/tree/one.txt', 'mutated');
    expect(await vfs.readFile('/backup/one.txt')).toBe('one');
    expect(await vfs.verifyContent()).toEqual([]);
  });

  it('honours copy overwrite, recursion, and into-directory rules', async () => {
    const vfs = await open();
    await vfs.writeFile('/f.txt', 'f');
    await vfs.mkdir('/into');
    await vfs.copy('/f.txt', '/into');
    expect(await vfs.readFile('/into/f.txt')).toBe('f');
    await expect(vfs.copy('/f.txt', '/into/f.txt', { overwrite: false })).rejects.toThrow(/file exists/);
    await vfs.mkdir('/dir/child');
    await expect(vfs.copy('/dir', '/copy-of-dir', { recursive: false })).rejects.toThrow(/is a directory/);
    await expect(vfs.copy('/dir', '/dir/child')).rejects.toThrow(/into itself/);
    await expect(vfs.copy('/f.txt', '/f.txt')).rejects.toThrow(/into itself/);
    await vfs.copy('/f.txt', '/g.txt');
    expect(await vfs.readFile('/g.txt')).toBe('f');
  });

  it('merges a directory copy over an existing tree', async () => {
    const vfs = await open();
    await vfs.writeFile('/left/keep.txt', 'keep');
    await vfs.writeFile('/left/shared.txt', 'new');
    await vfs.writeFile('/right/left/shared.txt', 'old');
    await vfs.writeFile('/right/left/own.txt', 'own');
    await vfs.copy('/left', '/right'); // lands on the existing /right/left and merges into it
    expect(await vfs.readFile('/right/left/shared.txt')).toBe('new');
    expect(await vfs.readFile('/right/left/keep.txt')).toBe('keep');
    expect(await vfs.readFile('/right/left/own.txt')).toBe('own');
    expect(await vfs.verifyContent()).toEqual([]);
  });
});

describe('remove semantics', () => {
  it('stays recursive by default and refuses non-empty directories when asked', async () => {
    const vfs = await open();
    await vfs.writeFile('/tree/nested/file.txt', 'x');
    await expect(vfs.remove('/tree', { recursive: false })).rejects.toThrow(/directory not empty/);
    expect(vfs.statSync('/tree/nested/file.txt')?.kind).toBe('file');
    await vfs.remove('/tree/nested/file.txt', { recursive: false });
    await vfs.remove('/tree/nested', { recursive: false });
    expect(vfs.statSync('/tree/nested')).toBeUndefined();
    await vfs.writeFile('/tree/again/file.txt', 'x');
    await vfs.remove('/tree');
    expect(vfs.statSync('/tree')).toBeUndefined();
    expect(vfs.usage().files).toBe(0);
    await expect(vfs.remove('/nowhere/at/all')).resolves.toBeUndefined();
  });

  it('deletes the backing blob only when the tree goes', async () => {
    const vfs = await open();
    const inode = await vfs.writeFile('/doomed/file.txt', 'bye');
    const blob = path.join(vfs.rootDir, inode.diskId, inode.id);
    expect((await stat(blob)).size).toBe(3);
    await vfs.remove('/doomed');
    await expect(stat(blob)).rejects.toThrow();
    expect(vfs.hostLayout().inodes[inode.id]).toBeUndefined();
  });
});

describe('symlinks', () => {
  it('resolves links to files and directories', async () => {
    const vfs = await open();
    await vfs.writeFile('/real/data.txt', 'payload');
    await vfs.symlink('/real', '/link-dir');
    await vfs.symlink('/real/data.txt', '/link-file');
    await vfs.symlink('data.txt', '/real/relative');
    expect(await vfs.readFile('/link-file')).toBe('payload');
    expect(await vfs.readFile('/link-dir/data.txt')).toBe('payload');
    expect(await vfs.readFile('/real/relative')).toBe('payload');
    expect(vfs.readlink('/link-dir')).toBe('/real');
    expect(vfs.lstatSync('/link-file')?.kind).toBe('symlink');
    expect(vfs.statSync('/link-file')?.kind).toBe('file');
    expect(vfs.realpath('/link-dir/data.txt')).toBe('/real/data.txt');
    await vfs.writeFile('/link-dir/written-through.txt', 'through');
    expect(await vfs.readFile('/real/written-through.txt')).toBe('through');
    expect(vfs.list('/link-dir').map((entry) => entry.name)).toContain('data.txt');
  });

  it('detects loops instead of hanging', async () => {
    const vfs = await open();
    await vfs.symlink('/loop-b', '/loop-a');
    await vfs.symlink('/loop-a', '/loop-b');
    await expect(vfs.readFile('/loop-a')).rejects.toThrow(/too many levels of symbolic links/);
    await vfs.symlink('/self', '/self-holder');
    await vfs.symlink('/self-holder', '/self');
    await expect(vfs.readBytes('/self/child')).rejects.toThrow(/too many levels of symbolic links/);
    expect(vfs.statSync('/loop-a')?.kind).toBe('symlink'); // lstat fallback keeps stat total
  });

  it('reports dangling links without inventing a file', async () => {
    const vfs = await open();
    await vfs.symlink('/gone.txt', '/dangling');
    expect(vfs.lstatSync('/dangling')?.kind).toBe('symlink');
    await expect(vfs.readFile('/dangling')).rejects.toThrow(/no such file/);
    await vfs.writeFile('/dangling', 'created through the link');
    expect(await vfs.readFile('/gone.txt')).toBe('created through the link');
    await expect(vfs.symlink('/x', '/dangling')).rejects.toThrow(/file exists/);
  });

  it('persists links across a reopen', async () => {
    const vfs = await open();
    await vfs.writeFile('/target.txt', 'kept');
    await vfs.symlink('/target.txt', '/alias');
    await vfs.flush();
    const reopened = await open();
    expect(await reopened.readFile('/alias')).toBe('kept');
    expect(reopened.readlink('/alias')).toBe('/target.txt');
  });
});

describe('hard links', () => {
  it('shares one inode across names and counts references', async () => {
    const vfs = await open();
    await vfs.writeFile('/original.txt', 'shared');
    const linked = await vfs.link('/original.txt', '/hard.txt');
    expect(linked.id).toBe(vfs.statSync('/original.txt')!.id);
    expect(vfs.statSync('/hard.txt')?.id).toBe(linked.id);
    expect(vfs.owner('/hard.txt').mode).toBe(0o644);
    await vfs.writeFile('/original.txt', 'updated through one name');
    expect(await vfs.readFile('/hard.txt')).toBe('updated through one name');
    const blob = path.join(vfs.rootDir, linked.diskId, linked.id);
    await vfs.remove('/original.txt');
    expect(await vfs.readFile('/hard.txt')).toBe('updated through one name');
    expect((await stat(blob)).size).toBeGreaterThan(0);
    await vfs.remove('/hard.txt');
    await expect(stat(blob)).rejects.toThrow();
    await vfs.mkdir('/d');
    await expect(vfs.link('/d', '/d-link')).rejects.toThrow(/hard link not allowed/);
  });
});

describe('case sensitivity', () => {
  it('keeps distinct names when case-sensitive', async () => {
    const vfs = await open();
    await vfs.writeFile('/Case.txt', 'upper');
    await vfs.writeFile('/case.txt', 'lower');
    expect(await vfs.readFile('/Case.txt')).toBe('upper');
    expect(await vfs.readFile('/case.txt')).toBe('lower');
    expect(vfs.list('/')).toHaveLength(2);
  });

  it('folds case while preserving it when case-insensitive', async () => {
    const vfs = await open({ caseSensitive: false });
    await vfs.writeFile('/Documents/Report.TXT', 'first');
    expect(await vfs.readFile('/documents/report.txt')).toBe('first');
    expect(vfs.statSync('/DOCUMENTS/REPORT.txt')?.kind).toBe('file');
    expect(vfs.list('/').map((entry) => entry.name)).toEqual(['Documents']);
    expect(vfs.list('/documents').map((entry) => entry.name)).toEqual(['Report.TXT']);
    expect(vfs.realpath('/documents/report.txt')).toBe('/Documents/Report.TXT');
    await vfs.writeFile('/documents/report.txt', 'second');
    expect(vfs.list('/Documents')).toHaveLength(1);
    expect(await vfs.readFile('/Documents/Report.TXT')).toBe('second');
    await vfs.flush();
    const reopened = await open({ caseSensitive: false });
    expect(await reopened.readFile('/DOCUMENTS/report.TXT')).toBe('second');
  });
});

describe('permissions and ownership', () => {
  it('records ownership and stays permissive by default', async () => {
    const vfs = await open();
    await vfs.writeFile('/root-owned.txt', 'x');
    expect(vfs.owner('/root-owned.txt')).toEqual({ uid: 0, gid: 0, mode: 0o644 });
    await vfs.chmod('/root-owned.txt', 0o600);
    await vfs.chown('/root-owned.txt', 1000, 1000);
    expect(vfs.owner('/root-owned.txt')).toEqual({ uid: 1000, gid: 1000, mode: 0o600 });
    // enforcement off: another user still reads it
    expect(await vfs.readFile('/root-owned.txt', { as: { uid: 4242, gid: 4242 } })).toBe('x');
  });

  it('enforces mode for non-root callers when asked', async () => {
    const vfs = await open({ enforcePermissions: true });
    const agent = { uid: 1000, gid: 1000 };
    const other = { uid: 1001, gid: 1001 };
    await vfs.mkdir('/home/agent');
    await vfs.chown('/home/agent', agent.uid, agent.gid);
    await vfs.chmod('/home/agent', 0o750);

    await vfs.writeFile('/home/agent/secret.txt', 'classified', 'disk0', { as: agent });
    expect(vfs.owner('/home/agent/secret.txt').uid).toBe(1000);
    await vfs.chmod('/home/agent/secret.txt', 0o600, { as: agent });
    expect(await vfs.readFile('/home/agent/secret.txt', { as: agent })).toBe('classified');

    await expect(vfs.readFile('/home/agent/secret.txt', { as: other })).rejects.toThrow(/permission denied/);
    expect(() => vfs.list('/home/agent', { as: other })).toThrow(/permission denied/);
    await expect(vfs.writeFile('/home/agent/intruder.txt', 'no', 'disk0', { as: other })).rejects.toThrow(/permission denied/);
    await expect(vfs.remove('/home/agent/secret.txt', { as: other })).rejects.toThrow(/permission denied/);
    await expect(vfs.chmod('/home/agent/secret.txt', 0o777, { as: other })).rejects.toThrow(/permission denied/);

    // reachable but not owned: the ownership check, not the traversal check, is what bites
    await vfs.writeFile('/public/notice.txt', 'public');
    await expect(vfs.chmod('/public/notice.txt', 0o777, { as: other })).rejects.toThrow(/not permitted/);
    await expect(vfs.chown('/public/notice.txt', 1001, 1001, { as: other })).rejects.toThrow(/not permitted/);

    // root and the default identity keep working
    expect(await vfs.readFile('/home/agent/secret.txt')).toBe('classified');
    // group bits apply to a peer in the owning group once the mode allows it
    const peer = { uid: 1002, gid: 1000 };
    await expect(vfs.readFile('/home/agent/secret.txt', { as: peer })).rejects.toThrow(/permission denied/);
    await vfs.chmod('/home/agent/secret.txt', 0o640, { as: agent });
    expect(await vfs.readFile('/home/agent/secret.txt', { as: peer })).toBe('classified');
    expect(vfs.list('/home/agent', { as: peer }).map((entry) => entry.name)).toEqual(['secret.txt']);

    // a directory without the execute bit blocks traversal entirely
    await vfs.writeFile('/vault/data.txt', 'x');
    await vfs.chmod('/vault', 0o700);
    await vfs.chown('/vault', 0, 0);
    await expect(vfs.readFile('/vault/data.txt', { as: agent })).rejects.toThrow(/permission denied/);
    expect(vfs.statSync('/vault/data.txt')?.kind).toBe('file');
  });

  it('binds an identity through the as() view', async () => {
    const vfs = await open({ enforcePermissions: true });
    await vfs.mkdir('/shared');
    await vfs.chown('/shared', 1000, 1000);
    const agent = vfs.as({ uid: 1000, gid: 1000 });
    await agent.writeFile('/shared/file.txt', 'from agent');
    expect(vfs.owner('/shared/file.txt').uid).toBe(1000);
    expect(agent.list('/shared').map((entry) => entry.name)).toEqual(['file.txt']);
    await expect(agent.writeFile('/forbidden.txt', 'no')).rejects.toThrow(/permission denied/);
    await vfs.chmod('/shared/file.txt', 0o600);
    await vfs.chown('/shared/file.txt', 0, 0);
    await expect(agent.readFile('/shared/file.txt')).rejects.toThrow(/permission denied/);
    await agent.remove('/shared/file.txt');
    expect(vfs.statSync('/shared/file.txt')).toBeUndefined();
  });

  it('enforces the constructor default identity when one is supplied', async () => {
    const root = { uid: 0, gid: 0 };
    const vfs = await open({ enforcePermissions: true, identity: { uid: 1000, gid: 1000 } });
    await vfs.mkdir('/mine', 'disk0', { as: root });
    await vfs.chown('/mine', 1000, 1000, { as: root });
    await vfs.writeFile('/mine/notes.txt', 'mine');
    expect(vfs.owner('/mine/notes.txt').uid).toBe(1000);
    await vfs.chmod('/mine/notes.txt', 0o000);
    await expect(vfs.readFile('/mine/notes.txt')).rejects.toThrow(/permission denied/);
    expect(await vfs.readFile('/mine/notes.txt', { as: root })).toBe('mine');
    await expect(vfs.writeFile('/top-level.txt', 'no')).rejects.toThrow(/permission denied/);
  });
});

describe('path handling', () => {
  it('canonicalises windows drives, traversal, and relative input', () => {
    expect(canonicalPath('C:\\Users\\agent\\file.txt')).toBe('/C/Users/agent/file.txt');
    expect(canonicalPath('../sibling', '/home/agent/docs')).toBe('/home/agent/sibling');
    expect(canonicalPath('./a//b/../c')).toBe('/a/c');
    expect(canonicalPath('/')).toBe('/');
  });

  it('resolves relative paths against a cwd and reports usage', async () => {
    const vfs = await open();
    expect(vfs.resolve('notes.txt', '/home/agent')).toBe('/home/agent/notes.txt');
    await vfs.writeFile('/home/agent/notes.txt', 'seed');
    const usage = vfs.usage();
    expect(usage.files).toBe(1);
    expect(usage.directories).toBe(3);
    expect(usage.bytes).toBe(4);
    expect(usage.digest).toHaveLength(16);
    expect(vfs.exists('/home/agent/notes.txt')).toBe(true);
    expect(vfs.exists('/home/agent/absent.txt')).toBe(false);
    await expect(vfs.readFile('/home/agent')).rejects.toThrow(/not a file/);
    await expect(vfs.writeFile('/home/agent', 'x')).rejects.toThrow(/is a directory/);
    await expect(vfs.mkdir('/home/agent/notes.txt')).rejects.toThrow(/not a directory/);
    expect(vfs.usage().digest).toBe(usage.digest);
  });

  it('isolates two filesystems that share a state root', async () => {
    const first = await open({}, `run-${randomUUID()}`);
    const second = await open({}, `run-${randomUUID()}`);
    await first.writeFile('/only-here.txt', 'a');
    await second.writeFile('/other.txt', 'b');
    expect(second.statSync('/only-here.txt')).toBeUndefined();
    expect(first.statSync('/other.txt')).toBeUndefined();
  });
});
