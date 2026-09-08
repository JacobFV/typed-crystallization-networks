import { createHash } from 'node:crypto';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import type { ComputerSpec, GitCommitRecord, GitObjectPayload } from '@tcn-computer/protocol';
import { VirtualFileSystem } from './vfs.js';
import { GitEnvironment, type GitRemoteSnapshot, type GitRemoteTransport } from './git.js';
import { encodeIndexFile } from './git/index-file.js';
import { IgnoreMatcher, parseIgnoreFile } from './git/ignore.js';
import { mergeText, unifiedDiff } from './git/diff.js';
import { hashBlob, EMPTY_TREE_HASH } from './git/objects.js';

/**
 * Ground truth captured from the real `git` binary, so the object model is
 * checked against Git itself rather than against itself:
 *
 *   git init -b main . && printf 'hello\n' > README.md && git add README.md
 *   GIT_AUTHOR_NAME=agent GIT_AUTHOR_EMAIL=agent@seed.local \
 *   GIT_COMMITTER_NAME=agent GIT_COMMITTER_EMAIL=agent@seed.local \
 *   GIT_AUTHOR_DATE="1767225600 +0000" GIT_COMMITTER_DATE="1767225600 +0000" \
 *   git commit -m "initial commit"
 */
const REAL_GIT = {
  blobHello: 'ce013625030ba8dba906f756967f9e9ca394464a',
  treeReadme: '853694aae8816094a0d875fee7ea26278dbf5d0f',
  commit: '8501ec4dfbe0d5a80b2d2998c7d8e807798d9ce1',
};

const spec: ComputerSpec = {
  id: 'git-unit', hostname: 'git-unit', os: 'ubuntu', shell: 'bash', ipv4: '10.42.0.30',
  memoryBytes: 8 * 1024 ** 3, cpuCores: 4,
  disks: [{ id: 'disk0', label: 'root', mount: '/', capacityBytes: 64 * 1024 ** 3 }],
  displays: [],
};

/** Quote-aware split so tests can be written as shell-ish one-liners. */
function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let quote = '';
  for (let i = 0; i < input.length; i++) {
    const char = input[i]!;
    if (quote) { if (char === quote) quote = ''; else current += char; }
    else if (char === '"' || char === "'") quote = char;
    else if (/\s/.test(char)) { if (current) { tokens.push(current); current = ''; } }
    else current += char;
  }
  if (current) tokens.push(current);
  return tokens;
}

interface Harness {
  vfs: VirtualFileSystem;
  git: GitEnvironment;
  root: string;
  git_(line: string, cwd?: string): Promise<string>;
  write(relative: string, content: string): Promise<unknown>;
  read(relative: string): Promise<string>;
  exists(relative: string): boolean;
}

let counter = 0;

async function harness(transport?: GitRemoteTransport, runId = `run-${counter++}`): Promise<Harness> {
  const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-git-unit-'));
  const vfs = new VirtualFileSystem(stateRoot, runId, spec);
  await vfs.initialize();
  const git = new GitEnvironment(vfs, transport);
  const root = '/home/agent/project';
  await vfs.mkdir(root);
  return {
    vfs, git, root,
    git_: (line, cwd = root) => git.command(tokenize(line).slice(1), cwd),
    write: (relative, content) => vfs.writeFile(`${root}/${relative}`, content),
    read: (relative) => vfs.readFile(`${root}/${relative}`),
    exists: (relative) => Boolean(vfs.statSync(`${root}/${relative}`)),
  };
}

/** init + one commit of `README.md` containing `hello\n`. */
async function seeded(transport?: GitRemoteTransport): Promise<Harness> {
  const env = await harness(transport);
  await env.git_('git init');
  await env.write('README.md', 'hello\n');
  await env.git_('git add README.md');
  await env.git_('git commit -m "initial commit"');
  return env;
}

describe('git object model', () => {
  it('produces the same object ids as the real git binary', async () => {
    const env = await seeded();
    const head = await env.git_('git rev-parse HEAD');
    expect(head).toBe(REAL_GIT.commit);
    expect(await env.git_('git rev-parse HEAD^{tree}')).toBe(REAL_GIT.treeReadme);
    expect(await env.git_('git cat-file -p HEAD')).toContain(`tree ${REAL_GIT.treeReadme}`);
    expect(hashBlob('hello\n')).toBe(REAL_GIT.blobHello);
    expect(await env.git_(`git cat-file -t ${REAL_GIT.blobHello}`)).toBe('blob');
    expect(await env.git_(`git cat-file -p ${REAL_GIT.blobHello}`)).toBe('hello');
    expect(await env.git_(`git cat-file -s ${REAL_GIT.blobHello}`)).toBe('6');
  });

  it('hashes file CONTENT, not path names: same names with different bytes differ', async () => {
    const first = await seeded();
    const second = await harness();
    await second.git_('git init');
    await second.write('README.md', 'different bytes\n');
    await second.git_('git add README.md');
    await second.git_('git commit -m "initial commit"');
    expect(await second.git_('git rev-parse HEAD')).not.toBe(await first.git_('git rev-parse HEAD'));
  });

  it('reproduces identical commit ids across independent runs (no wall clock in identity)', async () => {
    const build = async () => {
      const env = await harness();
      await env.git_('git init');
      await env.write('a.txt', 'alpha\n');
      await env.write('nested/b.txt', 'beta\n');
      await env.git_('git add .');
      await env.git_('git commit -m "first"');
      await env.write('a.txt', 'alpha two\n');
      await env.git_('git add a.txt');
      await env.git_('git commit -m "second"');
      return env.git_('git log --oneline');
    };
    const [runA, runB] = [await build(), await build()];
    expect(runA).toBe(runB);
    expect(runA.split('\n')).toHaveLength(2);
  });

  it('writes loose objects into the standard fan-out and records real parent pointers', async () => {
    const env = await seeded();
    await env.write('README.md', 'hello again\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "second"');
    const head = await env.git_('git rev-parse HEAD');
    expect(env.exists(`.git/objects/${head.slice(0, 2)}/${head.slice(2)}`)).toBe(true);
    const body = await env.git_('git cat-file -p HEAD');
    expect(body).toContain(`parent ${REAL_GIT.commit}`);
    expect(body).toMatch(/^tree [0-9a-f]{40}/);
    const record = env.git.listRepositories()[0]!.commits[0]!;
    expect(record.parents).toEqual([REAL_GIT.commit]);
  });

  it('exposes trees through cat-file and ls-tree', async () => {
    const env = await seeded();
    await env.write('src/main.ts', 'export const x = 1;\n');
    await env.git_('git add src');
    await env.git_('git commit -m "add source"');
    const tree = await env.git_('git cat-file -p HEAD^{tree}');
    expect(tree).toContain('blob');
    expect(tree).toContain('README.md');
    expect(tree).toContain('40000 tree');
    expect(await env.git_('git ls-tree -r HEAD')).toContain('src/main.ts');
  });
});

describe('git status', () => {
  it('reports a tracked file as modified after it is rewritten (regression: always reported clean)', async () => {
    const env = await seeded();
    expect(await env.git_('git status')).toContain('nothing to commit, working tree clean');
    await env.write('README.md', 'rewritten content\n');
    const status = await env.git_('git status');
    expect(status).toContain('Changes not staged for commit');
    expect(status).toContain('modified:   README.md');
    expect(status).not.toContain('nothing to commit, working tree clean');
    expect(await env.git_('git status --short')).toBe(' M README.md');
  });

  it('distinguishes staged, modified, untracked and deleted in one report', async () => {
    const env = await seeded();
    await env.write('staged.txt', 'staged\n');
    await env.git_('git add staged.txt');
    await env.write('README.md', 'changed\n');
    await env.write('untracked.txt', 'untracked\n');
    await env.git_('git rm --cached notes.txt').catch(() => undefined);
    const status = await env.git_('git status');
    expect(status).toContain('new file:   staged.txt');
    expect(status).toContain('modified:   README.md');
    expect(status).toContain('Untracked files');
    expect(status).toContain('untracked.txt');

    await env.vfs.remove(`${env.root}/README.md`);
    const afterDelete = await env.git_('git status');
    expect(afterDelete).toContain('deleted:    README.md');
    expect(await env.git_('git status -s')).toContain(' D README.md');
  });

  it('reports staged deletions and modifications against HEAD', async () => {
    const env = await seeded();
    await env.write('README.md', 'v2\n');
    await env.git_('git add README.md');
    expect(await env.git_('git status')).toContain('modified:   README.md');
    await env.git_('git rm README.md');
    expect(await env.git_('git status')).toContain('deleted:    README.md');
  });

  it('says "No commits yet" on an unborn branch and lists no branches', async () => {
    const env = await harness();
    await env.git_('git init');
    expect(await env.git_('git status')).toContain('No commits yet');
    // Regression: `branches: { main: undefined }` used to list an unborn branch.
    expect(await env.git_('git branch')).toBe('');
    expect(env.git.listRepositories()[0]!.branches).toEqual({});
  });
});

describe('git diff', () => {
  it('produces a real unified diff of worktree against index (regression: fake "new file mode" only)', async () => {
    const env = await seeded();
    await env.write('README.md', 'hello\nsecond line\n');
    const diff = await env.git_('git diff');
    expect(diff).toContain('diff --git a/README.md b/README.md');
    expect(diff).toContain('--- a/README.md');
    expect(diff).toContain('+++ b/README.md');
    expect(diff).toContain('@@ -1 +1,2 @@');
    expect(diff).toContain(' hello');
    expect(diff).toContain('+second line');
  });

  it('diffs index against HEAD with --staged and commit against commit', async () => {
    const env = await seeded();
    await env.write('README.md', 'hello\nstaged line\n');
    await env.git_('git add README.md');
    expect(await env.git_('git diff')).toBe('');
    const staged = await env.git_('git diff --staged');
    expect(staged).toContain('+staged line');
    await env.git_('git commit -m "second"');
    const range = await env.git_('git diff HEAD~1 HEAD');
    expect(range).toContain('+staged line');
    expect(await env.git_('git diff --name-only HEAD~1 HEAD')).toBe('README.md');
  });

  it('marks added and deleted files the way git does', async () => {
    const env = await seeded();
    await env.write('added.txt', 'brand new\n');
    await env.git_('git add added.txt');
    const staged = await env.git_('git diff --staged');
    expect(staged).toContain('new file mode 100644');
    expect(staged).toContain('--- /dev/null');
    expect(staged).toContain('+brand new');
    await env.git_('git commit -m "add file"');
    await env.git_('git rm added.txt');
    const removal = await env.git_('git diff --staged');
    expect(removal).toContain('deleted file mode 100644');
    expect(removal).toContain('+++ /dev/null');
    expect(removal).toContain('-brand new');
  });

  it('renders hunks with three lines of context', () => {
    const before = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'].join('\n') + '\n';
    const after = before.replace('5', 'five');
    const diff = unifiedDiff(before, after, { fromPath: 'f', toPath: 'f' });
    expect(diff).toContain('@@ -2,7 +2,7 @@');
    expect(diff).toContain('-5');
    expect(diff).toContain('+five');
    expect(diff).not.toContain(' 1\n');
  });
});

describe('git checkout and switch', () => {
  it('materializes the target tree: a file added later is GONE on an older branch', async () => {
    const env = await seeded();
    await env.git_('git branch legacy');
    await env.write('feature.txt', 'feature work\n');
    await env.git_('git add feature.txt');
    await env.git_('git commit -m "add feature"');
    expect(env.exists('feature.txt')).toBe(true);

    await env.git_('git switch legacy');
    // Regression: branches used to be labels attached to nothing.
    expect(env.exists('feature.txt')).toBe(false);
    expect(await env.git_('git status')).toContain('nothing to commit, working tree clean');

    await env.git_('git switch main');
    expect(env.exists('feature.txt')).toBe(true);
    expect(await env.read('feature.txt')).toBe('feature work\n');
  });

  it('restores changed content, not just presence, when switching branches', async () => {
    const env = await seeded();
    await env.git_('git checkout -b topic');
    await env.write('README.md', 'topic version\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "topic edit"');
    await env.git_('git checkout main');
    expect(await env.read('README.md')).toBe('hello\n');
    await env.git_('git checkout topic');
    expect(await env.read('README.md')).toBe('topic version\n');
  });

  it('refuses to clobber uncommitted changes, and --force discards them', async () => {
    const env = await seeded();
    await env.git_('git checkout -b topic');
    await env.write('README.md', 'committed on topic\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "topic"');
    await env.write('README.md', 'uncommitted local edit\n');
    await expect(env.git_('git checkout main')).rejects.toThrow(/local changes.*would be overwritten/s);
    expect(await env.read('README.md')).toBe('uncommitted local edit\n');
    await env.git_('git checkout -f main');
    expect(await env.read('README.md')).toBe('hello\n');
  });

  it('carries local edits across a switch when the file is identical in both trees', async () => {
    const env = await seeded();
    await env.write('shared.txt', 'base\n');
    await env.git_('git add shared.txt');
    await env.git_('git commit -m "shared"');
    await env.git_('git branch other');
    await env.write('shared.txt', 'work in progress\n');
    await env.git_('git checkout other');
    expect(await env.read('shared.txt')).toBe('work in progress\n');
  });

  it('detaches HEAD at a commit and reports it', async () => {
    const env = await seeded();
    await env.write('README.md', 'v2\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "second"');
    const output = await env.git_(`git checkout ${REAL_GIT.commit}`);
    expect(output).toContain('detached HEAD');
    expect(await env.read('README.md')).toBe('hello\n');
    expect(await env.git_('git rev-parse --abbrev-ref HEAD')).toBe('HEAD');
  });
});

describe('.gitignore', () => {
  it('keeps ignored paths out of `git add .`', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.write('.gitignore', 'node_modules/\n*.log\nbuild\n!keep.log\n');
    await env.write('app.ts', 'export {};\n');
    await env.write('node_modules/react/index.js', 'module.exports = {};\n');
    await env.write('debug.log', 'noise\n');
    await env.write('keep.log', 'kept\n');
    await env.write('build/out.js', 'compiled\n');
    await env.git_('git add .');
    const staged = (await env.git_('git ls-files')).split('\n').filter(Boolean);
    expect(staged).toContain('app.ts');
    expect(staged).toContain('.gitignore');
    expect(staged).toContain('keep.log');
    expect(staged.some((file) => file.startsWith('node_modules/'))).toBe(false);
    expect(staged).not.toContain('debug.log');
    expect(staged.some((file) => file.startsWith('build/'))).toBe(false);
    expect(await env.git_('git status')).not.toContain('node_modules');
  });

  it('refuses to add an explicitly named ignored path without -f', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.write('.gitignore', 'secret.txt\n');
    await env.write('secret.txt', 'shh\n');
    await expect(env.git_('git add secret.txt')).rejects.toThrow(/ignored by one of your .gitignore files/);
    await env.git_('git add -f secret.txt');
    expect(await env.git_('git ls-files')).toContain('secret.txt');
  });

  it('matches anchored, nested, negated and directory-only patterns', () => {
    const rules = [
      ...parseIgnoreFile('/root-only.txt\ndist/\n**/tmp/\n*.bak\n!important.bak\ndocs/*.pdf\n', '', '.gitignore'),
      ...parseIgnoreFile('local-noise.txt\n', 'packages', 'packages/.gitignore'),
    ];
    const matcher = new IgnoreMatcher(rules);
    expect(matcher.ignores('root-only.txt')).toBe(true);
    expect(matcher.ignores('nested/root-only.txt')).toBe(false);
    expect(matcher.ignores('dist', true)).toBe(true);
    expect(matcher.ignores('dist/app.js')).toBe(true);
    expect(matcher.ignores('packages/kernel/tmp/thing.txt')).toBe(true);
    expect(matcher.ignores('notes.bak')).toBe(true);
    expect(matcher.ignores('important.bak')).toBe(false);
    expect(matcher.ignores('docs/manual.pdf')).toBe(true);
    expect(matcher.ignores('docs/deep/manual.pdf')).toBe(false);
    expect(matcher.ignores('packages/local-noise.txt')).toBe(true);
    expect(matcher.ignores('local-noise.txt')).toBe(false);
  });
});

describe('git merge', () => {
  it('fast-forwards when the branch is strictly ahead', async () => {
    const env = await seeded();
    await env.git_('git checkout -b feature');
    await env.write('feature.txt', 'work\n');
    await env.git_('git add feature.txt');
    await env.git_('git commit -m "feature work"');
    await env.git_('git checkout main');
    expect(env.exists('feature.txt')).toBe(false);
    const output = await env.git_('git merge feature');
    expect(output).toContain('Fast-forward');
    expect(env.exists('feature.txt')).toBe(true);
  });

  it('creates a merge commit with two parents for divergent branches', async () => {
    const env = await seeded();
    await env.git_('git checkout -b feature');
    await env.write('feature.txt', 'feature side\n');
    await env.git_('git add feature.txt');
    await env.git_('git commit -m "feature"');
    await env.git_('git checkout main');
    await env.write('main.txt', 'main side\n');
    await env.git_('git add main.txt');
    await env.git_('git commit -m "main"');
    const output = await env.git_('git merge feature');
    expect(output).toContain("Merge made by the 'ort' strategy");
    expect(env.exists('feature.txt')).toBe(true);
    expect(env.exists('main.txt')).toBe(true);
    const merge = env.git.listRepositories()[0]!.commits[0]!;
    expect(merge.parents).toHaveLength(2);
    expect(await env.git_('git log')).toContain('Merge:');
  });

  it('writes conflict markers, records unmerged paths, and commits after resolution', async () => {
    const env = await seeded();
    await env.write('shared.txt', 'line one\nline two\nline three\n');
    await env.git_('git add shared.txt');
    await env.git_('git commit -m "base"');
    await env.git_('git checkout -b feature');
    await env.write('shared.txt', 'line one\nfeature change\nline three\n');
    await env.git_('git add shared.txt');
    await env.git_('git commit -m "feature edit"');
    await env.git_('git checkout main');
    await env.write('shared.txt', 'line one\nmain change\nline three\n');
    await env.git_('git add shared.txt');
    await env.git_('git commit -m "main edit"');

    await expect(env.git_('git merge feature')).rejects.toThrow(/CONFLICT \(content\): Merge conflict in shared.txt/);
    const conflicted = await env.read('shared.txt');
    expect(conflicted).toContain('<<<<<<< HEAD');
    expect(conflicted).toContain('main change');
    expect(conflicted).toContain('=======');
    expect(conflicted).toContain('feature change');
    expect(conflicted).toContain('>>>>>>> feature');
    const status = await env.git_('git status');
    expect(status).toContain('Unmerged paths');
    expect(status).toContain('both modified:   shared.txt');
    expect(await env.git_('git ls-files -u')).toContain('shared.txt');
    await expect(env.git_('git commit -m "premature"')).rejects.toThrow(/unmerged files/);

    await env.write('shared.txt', 'line one\nresolved\nline three\n');
    await env.git_('git add shared.txt');
    const committed = await env.git_('git commit -m "merge resolved"');
    expect(committed).toContain('merge resolved');
    expect(env.git.listRepositories()[0]!.commits[0]!.parents).toHaveLength(2);
    expect(await env.git_('git status')).toContain('nothing to commit');
  });

  it('aborts a conflicted merge back to HEAD', async () => {
    const env = await seeded();
    await env.git_('git checkout -b feature');
    await env.write('README.md', 'feature\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "feature"');
    await env.git_('git checkout main');
    await env.write('README.md', 'main\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "main"');
    await expect(env.git_('git merge feature')).rejects.toThrow(/CONFLICT/);
    await env.git_('git merge --abort');
    expect(await env.read('README.md')).toBe('main\n');
    expect(await env.git_('git status')).toContain('nothing to commit, working tree clean');
  });

  it('merges disjoint regions of the same file without conflicting', () => {
    const base = 'a\nb\nc\nd\ne\n';
    const ours = 'a CHANGED\nb\nc\nd\ne\n';
    const theirs = 'a\nb\nc\nd\ne CHANGED\n';
    const merged = mergeText(ours, base, theirs, { ours: 'HEAD', theirs: 'other' });
    expect(merged.conflicted).toBe(false);
    expect(merged.content).toBe('a CHANGED\nb\nc\nd\ne CHANGED\n');
  });
});

describe('git reset', () => {
  it('supports soft, mixed and hard', async () => {
    const env = await seeded();
    await env.write('second.txt', 'second\n');
    await env.git_('git add second.txt');
    await env.git_('git commit -m "second commit"');
    const first = REAL_GIT.commit;

    await env.git_(`git reset --soft ${first}`);
    expect(await env.git_('git rev-parse HEAD')).toBe(first);
    expect(await env.git_('git status')).toContain('new file:   second.txt');
    expect(env.exists('second.txt')).toBe(true);

    await env.git_(`git reset --mixed ${first}`);
    let status = await env.git_('git status');
    expect(status).not.toContain('new file:   second.txt');
    expect(status).toContain('second.txt');
    expect(status).toContain('Untracked files');
    expect(env.exists('second.txt')).toBe(true);

    await env.write('README.md', 'dirty\n');
    await env.git_(`git reset --hard ${first}`);
    expect(await env.read('README.md')).toBe('hello\n');
    status = await env.git_('git status');
    expect(status).toContain('Untracked files');
  });

  it('unstages a path without moving HEAD', async () => {
    const env = await seeded();
    await env.write('README.md', 'changed\n');
    await env.git_('git add README.md');
    expect(await env.git_('git status')).toContain('modified:   README.md');
    await env.git_('git reset -- README.md');
    const status = await env.git_('git status');
    expect(status).toContain('Changes not staged for commit');
    expect(status).not.toContain('Changes to be committed');
    expect(await env.git_('git rev-parse HEAD')).toBe(REAL_GIT.commit);
  });

  it('hard reset removes files that only exist in the discarded commit', async () => {
    const env = await seeded();
    await env.write('temp.txt', 'temporary\n');
    await env.git_('git add temp.txt');
    await env.git_('git commit -m "temp"');
    await env.git_('git reset --hard HEAD~1');
    expect(env.exists('temp.txt')).toBe(false);
  });
});

describe('git stash', () => {
  it('stashes working-tree changes and restores them on pop', async () => {
    const env = await seeded();
    await env.write('README.md', 'work in progress\n');
    await env.write('extra.txt', 'staged extra\n');
    await env.git_('git add extra.txt');
    const saved = await env.git_('git stash');
    expect(saved).toContain('Saved working directory');
    expect(await env.read('README.md')).toBe('hello\n');
    expect(await env.git_('git status')).toContain('nothing to commit, working tree clean');
    expect(await env.git_('git stash list')).toContain('stash@{0}');
    await env.git_('git stash pop');
    expect(await env.read('README.md')).toBe('work in progress\n');
    expect(await env.git_('git stash list')).toBe('');
  });

  it('reports nothing to save on a clean tree and drops entries', async () => {
    const env = await seeded();
    expect(await env.git_('git stash')).toBe('No local changes to save');
    await env.write('README.md', 'dirty\n');
    await env.git_('git stash -m "wip note"');
    expect(await env.git_('git stash list')).toContain('wip note');
    expect(await env.git_('git stash drop')).toContain('Dropped');
    expect(await env.git_('git stash list')).toBe('');
    expect(await env.read('README.md')).toBe('hello\n');
  });
});

describe('history commands', () => {
  it('tags, shows, cherry-picks and reverts', async () => {
    const env = await seeded();
    await env.git_('git tag v1.0');
    await env.git_('git tag -a v1.0-annotated -m "annotated release"');
    expect((await env.git_('git tag')).split('\n')).toEqual(['v1.0', 'v1.0-annotated']);
    expect(await env.git_('git rev-parse v1.0')).toBe(REAL_GIT.commit);
    expect(await env.git_('git show v1.0-annotated')).toContain('annotated release');

    await env.git_('git checkout -b topic');
    await env.write('picked.txt', 'cherry\n');
    await env.git_('git add picked.txt');
    await env.git_('git commit -m "pick me"');
    const picked = await env.git_('git rev-parse HEAD');
    await env.git_('git checkout main');
    expect(env.exists('picked.txt')).toBe(false);
    await env.git_(`git cherry-pick ${picked}`);
    expect(env.exists('picked.txt')).toBe(true);
    expect(await env.git_('git log --oneline')).toContain('pick me');

    await env.git_('git revert HEAD');
    expect(env.exists('picked.txt')).toBe(false);
    expect(await env.git_('git log --oneline')).toContain('Revert "pick me"');
  });

  it('shows a commit with its diff and a file at a revision', async () => {
    const env = await seeded();
    await env.write('README.md', 'hello\nmore\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "extend readme"');
    const show = await env.git_('git show HEAD');
    expect(show).toContain('Author: agent <agent@seed.local>');
    expect(show).toContain('extend readme');
    expect(show).toContain('+more');
    expect(await env.git_('git show HEAD~1:README.md')).toBe('hello\n');
  });

  it('blames lines to the commit that introduced them', async () => {
    const env = await seeded();
    await env.write('README.md', 'hello\nsecond author line\n');
    await env.git_('git add README.md');
    await env.git_('git commit -m "second"');
    const head = await env.git_('git rev-parse HEAD');
    const blame = await env.git_('git blame README.md');
    const [firstLine, secondLine] = blame.split('\n');
    expect(firstLine).toContain(REAL_GIT.commit.slice(0, 7));
    expect(firstLine).toContain('hello');
    expect(secondLine).toContain(head.slice(0, 7));
    expect(secondLine).toContain('second author line');
  });

  it('keeps a reflog of HEAD movements', async () => {
    const env = await seeded();
    await env.git_('git checkout -b topic');
    await env.write('a.txt', 'a\n');
    await env.git_('git add a.txt');
    await env.git_('git commit -m "topic commit"');
    const reflog = await env.git_('git reflog');
    expect(reflog).toContain('HEAD@{0}: commit: topic commit');
    expect(reflog).toContain('checkout: moving from main to topic');
    expect(await env.read('.git/logs/HEAD')).toContain('topic commit');
  });

  it('filters log by revision, count and path', async () => {
    const env = await seeded();
    await env.write('other.txt', 'other\n');
    await env.git_('git add other.txt');
    await env.git_('git commit -m "add other"');
    expect((await env.git_('git log --oneline -n 1')).split('\n')).toHaveLength(1);
    expect(await env.git_('git log --oneline -- other.txt')).toContain('add other');
    expect(await env.git_('git log --oneline -- other.txt')).not.toContain('initial commit');
    expect(await env.git_('git log --oneline HEAD~1')).toContain('initial commit');
  });
});

describe('worktree commands', () => {
  it('rm, mv and restore keep index and worktree in agreement', async () => {
    const env = await seeded();
    await env.write('move-me.txt', 'contents\n');
    await env.git_('git add move-me.txt');
    await env.git_('git commit -m "add file"');

    await env.git_('git mv move-me.txt moved.txt');
    expect(env.exists('move-me.txt')).toBe(false);
    expect(env.exists('moved.txt')).toBe(true);
    expect(await env.git_('git ls-files')).toContain('moved.txt');
    await env.git_('git commit -m "move file"');

    await env.write('moved.txt', 'oops\n');
    await env.git_('git restore moved.txt');
    expect(await env.read('moved.txt')).toBe('contents\n');

    await env.write('moved.txt', 'staged mistake\n');
    await env.git_('git add moved.txt');
    await env.git_('git restore --staged moved.txt');
    expect(await env.git_('git status')).toContain('Changes not staged for commit');

    await env.git_('git restore moved.txt');
    await env.git_('git rm moved.txt');
    expect(env.exists('moved.txt')).toBe(false);
    expect(await env.git_('git ls-files')).not.toContain('moved.txt');
  });

  it('lists index entries with stage information', async () => {
    const env = await seeded();
    const listing = await env.git_('git ls-files --stage');
    expect(listing).toBe(`100644 ${REAL_GIT.blobHello} 0\tREADME.md`);
  });
});

describe('git config', () => {
  it('persists configuration to .git/config and uses it for commit identity', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.git_('git config user.name "Jacob Valdez"');
    await env.git_('git config user.email jacob@seed.local');
    expect(await env.git_('git config user.name')).toBe('Jacob Valdez');
    const config = await env.read('.git/config');
    expect(config).toContain('[user]');
    expect(config).toContain('name = Jacob Valdez');
    expect(await env.git_('git config --list')).toContain('user.email=jacob@seed.local');

    await env.write('a.txt', 'a\n');
    await env.git_('git add a.txt');
    await env.git_('git commit -m "identity"');
    expect(await env.git_('git log')).toContain('Author: Jacob Valdez <jacob@seed.local>');
  });

  it('stores global configuration in ~/.gitconfig', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.git_('git config --global user.name globaluser');
    expect(await env.vfs.readFile('/home/agent/.gitconfig')).toContain('globaluser');
    expect(await env.git_('git config user.name')).toBe('globaluser');
  });
});

describe('remotes', () => {
  /** Remote that persists everything, including objects — a full Git server. */
  function objectRemote() {
    const state: GitRemoteSnapshot = { branches: {}, commits: [], objects: [] };
    const transport: GitRemoteTransport = {
      fetch: async () => structuredClone(state),
      push: async (_url, branch, commits, _expected, objects) => {
        for (const commit of commits) if (!state.commits.some((item) => item.hash === commit.hash)) state.commits.push(commit);
        for (const object of objects ?? []) if (!state.objects!.some((item) => item.hash === object.hash)) state.objects!.push(object);
        if (commits[0]) state.branches[branch] = commits[0].hash;
        return structuredClone(state);
      },
    };
    return { state, transport };
  }

  /** The v0.3 wire shape: branches + commits only, objects dropped in transit. */
  function metadataOnlyRemote() {
    const state: GitRemoteSnapshot = { branches: {}, commits: [] };
    const transport: GitRemoteTransport = {
      fetch: async () => structuredClone(state),
      push: async (_url, branch, commits) => {
        for (const commit of commits) if (!state.commits.some((item) => item.hash === commit.hash)) state.commits.push(commit);
        if (commits[0]) state.branches[branch] = commits[0].hash;
        return structuredClone(state);
      },
    };
    return { state, transport };
  }

  it('clones the remote tree into the working directory (regression: fabricated README)', async () => {
    const remote = objectRemote();
    const origin = await seeded(remote.transport);
    await origin.write('src/app.ts', 'export const app = true;\n');
    await origin.git_('git add src/app.ts');
    await origin.git_('git commit -m "add app"');
    await origin.git_('git remote add origin https://git.seed.local/seed/example.git');
    const pushed = await origin.git_('git push origin main');
    expect(pushed).toContain('main -> main');

    const clone = await harness(remote.transport);
    await clone.git_('git clone https://git.seed.local/seed/example.git copy', '/home/agent');
    const cloneRoot = '/home/agent/copy';
    expect(await clone.vfs.readFile(`${cloneRoot}/README.md`)).toBe('hello\n');
    expect(await clone.vfs.readFile(`${cloneRoot}/src/app.ts`)).toBe('export const app = true;\n');
    // The fabricated "cloned from ... through the seed git transport" file is gone.
    expect(await clone.vfs.readFile(`${cloneRoot}/README.md`)).not.toContain('seed git transport');
    expect(await clone.git.command(['log', '--oneline'], cloneRoot)).toContain('add app');
    expect(await clone.git.command(['status'], cloneRoot)).toContain('nothing to commit, working tree clean');
    expect(await clone.git.command(['rev-parse', 'HEAD'], cloneRoot)).toBe(await origin.git_('git rev-parse HEAD'));
  });

  it('keeps history visible when the remote only carries metadata, without inventing files', async () => {
    const remote = metadataOnlyRemote();
    const origin = await seeded(remote.transport);
    await origin.git_('git remote add origin https://git.seed.local/seed/example.git');
    await origin.git_('git push origin main');

    const clone = await harness(remote.transport);
    await clone.git_('git clone https://git.seed.local/seed/example.git copy', '/home/agent');
    const cloneRoot = '/home/agent/copy';
    expect(await clone.git.command(['log', '--oneline'], cloneRoot)).toContain('initial commit');
    expect(clone.vfs.statSync(`${cloneRoot}/README.md`)).toBeUndefined();
  });

  it('fetches into remote-tracking refs and fast-forwards on pull', async () => {
    const remote = objectRemote();
    const origin = await seeded(remote.transport);
    await origin.git_('git remote add origin https://git.seed.local/seed/example.git');
    await origin.git_('git push origin main');

    const clone = await harness(remote.transport);
    await clone.git_('git clone https://git.seed.local/seed/example.git copy', '/home/agent');
    const cloneRoot = '/home/agent/copy';

    await origin.write('later.txt', 'later work\n');
    await origin.git_('git add later.txt');
    await origin.git_('git commit -m "later"');
    await origin.git_('git push origin main');

    const fetched = await clone.git.command(['fetch', 'origin'], cloneRoot);
    expect(fetched).toContain('main');
    expect(clone.vfs.statSync(`${cloneRoot}/later.txt`)).toBeUndefined();
    const pulled = await clone.git.command(['pull', 'origin'], cloneRoot);
    expect(pulled).toContain('Fast-forward');
    expect(await clone.vfs.readFile(`${cloneRoot}/later.txt`)).toBe('later work\n');
  });

  it('degrades to offline output when the transport cannot reach the host', async () => {
    const transport: GitRemoteTransport = {
      fetch: async () => { throw new Error('connection refused: git.example.test:443'); },
      push: async () => { throw new Error('connection refused: git.example.test:443'); },
    };
    const env = await seeded(transport);
    await env.git_('git remote add origin https://git.example.test/team/app.git');
    expect(await env.git_('git push origin main')).toContain('main -> main');
    expect(await env.git_('git fetch origin')).toContain('Already up to date.');
  });

  it('surfaces a real server rejection instead of pretending it succeeded', async () => {
    const transport: GitRemoteTransport = {
      fetch: async () => ({ branches: {}, commits: [] }),
      push: async () => { throw new Error('git push failed: {"error":"non-fast-forward"}'); },
    };
    const env = await seeded(transport);
    await env.git_('git remote add origin https://git.seed.local/seed/example.git');
    await expect(env.git_('git push origin main')).rejects.toThrow(/non-fast-forward/);
  });

  it('records remote urls in .git/config and lists them', async () => {
    const env = await seeded();
    await env.git_('git remote add origin https://git.seed.local/seed/example.git');
    expect(await env.git_('git remote -v')).toContain('origin\thttps://git.seed.local/seed/example.git (push)');
    expect(await env.read('.git/config')).toContain('[remote "origin"]');
    expect(env.git.listRepositories()[0]!.remotes.origin).toBe('https://git.seed.local/seed/example.git');
  });
});

describe('on-disk layout', () => {
  it('writes standard refs, HEAD and a binary DIRC index', async () => {
    const env = await seeded();
    expect(await env.read('.git/HEAD')).toBe('ref: refs/heads/main\n');
    expect((await env.read('.git/refs/heads/main')).trim()).toBe(REAL_GIT.commit);
    expect(env.exists('.git/index')).toBe(true);
    // The bespoke v0.3 index is gone.
    expect(env.exists('.git/index.seed.json')).toBe(false);

    const encoded = encodeIndexFile([
      { path: 'README.md', hash: REAL_GIT.blobHello, mode: '100644', size: 6, mtimeMs: 0, stage: 0 },
    ]);
    expect(encoded.subarray(0, 4).toString('ascii')).toBe('DIRC');
    expect(encoded.readUInt32BE(4)).toBe(2);
    expect(encoded.readUInt32BE(8)).toBe(1);
    // 12-byte header + one 8-aligned entry (62 fixed + 9 path + padding) + SHA-1 trailer.
    expect(encoded.length).toBe(12 + 72 + 20);
    expect(encoded.subarray(40 + 12, 60 + 12).toString('hex')).toBe(REAL_GIT.blobHello);
    expect(createHash('sha1').update(encoded.subarray(0, encoded.length - 20)).digest('hex'))
      .toBe(encoded.subarray(encoded.length - 20).toString('hex'));
  });

  it('deletes ref files when a branch is deleted', async () => {
    const env = await seeded();
    await env.git_('git branch scratch');
    expect(env.exists('.git/refs/heads/scratch')).toBe(true);
    await env.git_('git branch -d scratch');
    expect(env.exists('.git/refs/heads/scratch')).toBe(false);
    expect(await env.git_('git branch')).toBe('* main');
  });

  it('refuses to delete an unmerged branch without -D', async () => {
    const env = await seeded();
    await env.git_('git checkout -b unmerged');
    await env.write('x.txt', 'x\n');
    await env.git_('git add x.txt');
    await env.git_('git commit -m "unmerged work"');
    await env.git_('git checkout main');
    await expect(env.git_('git branch -d unmerged')).rejects.toThrow(/not fully merged/);
    expect(await env.git_('git branch -D unmerged')).toContain('Deleted branch unmerged');
  });

  it('re-adopts a repository that exists only on disk', async () => {
    const env = await seeded();
    const rediscovered = new GitEnvironment(env.vfs);
    expect(await rediscovered.command(['log', '--oneline'], env.root)).toContain('initial commit');
    expect(await rediscovered.command(['rev-parse', 'HEAD'], env.root)).toBe(REAL_GIT.commit);
    expect(await rediscovered.command(['status'], env.root)).toContain('nothing to commit, working tree clean');
  });
});

describe('pathspecs and commit options', () => {
  it('scopes `git add .` to the current directory', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.write('top.txt', 'top\n');
    await env.write('sub/inner.txt', 'inner\n');
    await env.write('other/skip.txt', 'skip\n');
    await env.git_('git add .', `${env.root}/sub`);
    expect(await env.git_('git ls-files')).toBe('sub/inner.txt');
    await env.git_('git add .');
    expect((await env.git_('git ls-files')).split('\n').sort()).toEqual(['other/skip.txt', 'sub/inner.txt', 'top.txt']);
  });

  it('resolves relative pathspecs and diffs from a subdirectory', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.write('sub/inner.txt', 'one\n');
    await env.git_('git add sub/inner.txt');
    await env.git_('git commit -m "add inner"');
    await env.write('sub/inner.txt', 'two\n');
    const diff = await env.git_('git diff -- inner.txt', `${env.root}/sub`);
    expect(diff).toContain('a/sub/inner.txt');
    expect(diff).toContain('+two');
    await env.git_('git add inner.txt', `${env.root}/sub`);
    expect(await env.git_('git status')).toContain('modified:   sub/inner.txt');
  });

  it('stages tracked modifications with commit -a and rewrites history with --amend', async () => {
    const env = await seeded();
    await env.write('README.md', 'auto staged\n');
    await env.write('ignored-by-commit-a.txt', 'untracked\n');
    const committed = await env.git_('git commit -a -m "auto stage"');
    expect(committed).toContain('auto stage');
    expect(await env.git_('git status')).toContain('untracked');
    expect(await env.git_('git diff HEAD')).toBe('');

    const before = await env.git_('git rev-parse HEAD');
    const amended = await env.git_('git commit --amend -m "auto stage, reworded"');
    expect(amended).toContain('auto stage, reworded');
    expect(await env.git_('git rev-parse HEAD')).not.toBe(before);
    expect((await env.git_('git log --oneline')).split('\n')).toHaveLength(2);
    expect(await env.git_('git log --oneline')).toContain('auto stage, reworded');
  });

  it('tracks upstream state after `push -u`', async () => {
    const state: GitRemoteSnapshot = { branches: {}, commits: [], objects: [] };
    const transport: GitRemoteTransport = {
      fetch: async () => structuredClone(state),
      push: async (_url, branch, commits: GitCommitRecord[], _expected, objects?: GitObjectPayload[]) => {
        for (const commit of commits) if (!state.commits.some((item) => item.hash === commit.hash)) state.commits.push(commit);
        for (const object of objects ?? []) state.objects!.push(object);
        if (commits[0]) state.branches[branch] = commits[0].hash;
        return structuredClone(state);
      },
    };
    const env = await seeded(transport);
    await env.git_('git remote add origin https://git.seed.local/seed/example.git');
    await env.git_('git push -u origin main');
    expect(await env.git_('git status')).toContain("Your branch is up to date with 'origin/main'.");
    await env.write('next.txt', 'next\n');
    await env.git_('git add next.txt');
    await env.git_('git commit -m "next"');
    expect(await env.git_('git status')).toContain("Your branch is ahead of 'origin/main' by 1 commit.");
  });
});

describe('secondary command surface', () => {
  it('renames branches, unsets config, edits remotes and applies a stash without dropping it', async () => {
    const env = await seeded();
    await env.git_('git branch -m main trunk');
    expect(await env.git_('git branch')).toBe('* trunk');
    expect(await env.git_('git rev-parse --abbrev-ref HEAD')).toBe('trunk');

    await env.git_('git config core.ignorecase true');
    expect(await env.git_('git config core.ignorecase')).toBe('true');
    await env.git_('git config --unset core.ignorecase');
    expect(await env.git_('git config --list')).not.toContain('core.ignorecase');

    await env.git_('git remote add origin https://git.seed.local/seed/one.git');
    await env.git_('git remote set-url origin https://git.seed.local/seed/two.git');
    expect(await env.git_('git remote get-url origin')).toBe('https://git.seed.local/seed/two.git');
    await env.git_('git remote remove origin');
    expect(await env.git_('git remote')).toBe('');

    await env.write('README.md', 'stash me\n');
    await env.git_('git stash');
    expect(await env.git_('git stash show')).toContain('README.md');
    await env.git_('git stash apply');
    expect(await env.read('README.md')).toBe('stash me\n');
    expect(await env.git_('git stash list')).toContain('stash@{0}');
  });

  it('answers plumbing queries', async () => {
    const env = await seeded();
    expect(await env.git_('git rev-parse --show-toplevel')).toBe(env.root);
    expect(await env.git_('git rev-parse --git-dir')).toBe(`${env.root}/.git`);
    expect(await env.git_('git rev-parse --is-inside-work-tree')).toBe('true');
    expect(await env.git_('git rev-parse --short HEAD')).toBe(REAL_GIT.commit.slice(0, 7));
    expect(await env.git_('git ls-tree HEAD')).toBe(`100644 blob ${REAL_GIT.blobHello}\tREADME.md`);
    expect(await env.git_('git ls-files -o')).toBe('');
    await env.write('loose.txt', 'loose\n');
    expect(await env.git_('git ls-files --others')).toBe('loose.txt');
    await env.write('README.md', 'edited\n');
    expect(await env.git_('git ls-files -m')).toBe('README.md');
    expect(await env.git_('git help')).toContain('cherry-pick');
  });

  it('reverts without committing when asked', async () => {
    const env = await seeded();
    await env.write('temp.txt', 'temporary\n');
    await env.git_('git add temp.txt');
    await env.git_('git commit -m "add temp"');
    const head = await env.git_('git rev-parse HEAD');
    await env.git_('git revert -n HEAD');
    expect(env.exists('temp.txt')).toBe(false);
    expect(await env.git_('git rev-parse HEAD')).toBe(head);
    expect(await env.git_('git status')).toContain('deleted:    temp.txt');
  });
});

describe('bootstrap compatibility', () => {
  it('supports the synchronous stage() hand-off used before the first commit', async () => {
    const env = await harness();
    await env.write('README.md', 'hello\n');
    await env.write('package.json', '{}\n');
    await env.git.initRepository(env.root);
    env.git.stage(env.root, ['README.md', 'package.json']);
    const output = await env.git.command(['commit', '-m', 'bootstrap seed ecosystem'], env.root);
    expect(output).toContain('bootstrap seed ecosystem');
    expect(output).toContain('2 files changed');
    const record = env.git.listRepositories()[0]!;
    expect(record.commits).toHaveLength(1);
    expect(record.staged).toEqual([]);
    expect(record.head).toBeDefined();
    expect(env.exists(`.git/objects/${record.head!.slice(0, 2)}/${record.head!.slice(2)}`)).toBe(true);
  });

  it('reports an empty tree hash for a repository with no files', async () => {
    const env = await harness();
    await env.git_('git init');
    await env.git_('git commit --allow-empty -m "empty"');
    expect(env.git.listRepositories()[0]!.commits[0]!.treeDigest).toBe(EMPTY_TREE_HASH);
  });

  it('fails outside a repository the way git does', async () => {
    const env = await harness();
    await expect(env.git_('git status')).rejects.toThrow('fatal: not a git repository (or any parent up to mount point)');
  });
});
