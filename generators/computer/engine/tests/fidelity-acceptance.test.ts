/**
 * Fidelity acceptance suite.
 *
 * Every assertion here encodes a defect confirmed by executing the kernel during
 * the 2026-08-11 fidelity audit. Each one FAILED against commit 89f2fe8. They are
 * the standing gate on that remediation: if one regresses, the corresponding
 * fidelity claim in the README and the technical report is no longer true.
 *
 * Findings are labelled with their audit ids (SH-*, VFS-*, GIT-*, PKG-*, PROC-*,
 * SEC-*, NET-*) so a failure points straight at the claim it invalidates.
 */
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { seed2026Blueprint } from '@tcn-computer/ecosystem-seed-2026';
import { SimulationRuntime } from '@tcn-computer/kernel';

const MAC = 'mac-studio';
const UBUNTU = 'ubuntu-dev';
const WINDOWS = 'win-workstation';
let rt: SimulationRuntime;

const run = (computer: string, command: string) => rt.execute(computer, command);
const out = async (computer: string, command: string) => (await run(computer, command)).stdout.trim();

beforeAll(async () => {
  const stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-acceptance-'));
  rt = new SimulationRuntime({ topology: seed2026Blueprint, stateRoot, runId: 'acceptance' });
  await rt.initialize();
  await run(MAC, 'mkdir -p ~/acc');
}, 180_000);

describe('SH-1..5 the shell does not silently destroy data', () => {
  it('touch preserves an existing file instead of truncating it', async () => {
    await run(MAC, 'echo important-data > ~/acc/keep.txt');
    await run(MAC, 'touch ~/acc/keep.txt');
    expect(await out(MAC, 'cat ~/acc/keep.txt')).toBe('important-data');
  });

  it('appends with >> rather than writing to a phantom path', async () => {
    await run(MAC, 'echo one > ~/acc/app.txt');
    await run(MAC, 'echo two >> ~/acc/app.txt');
    expect(await out(MAC, 'cat ~/acc/app.txt')).toBe('one\ntwo');
  });

  it('keeps operators inside quotes intact', async () => {
    const semicolon = await run(MAC, 'echo "a; b"');
    expect(semicolon.stdout.trim()).toBe('a; b');
    expect(semicolon.stderr).toBe('');
    expect(semicolon.exitCode).toBe(0);
    expect(await out(MAC, 'echo "a > b"')).toBe('a > b');
  });

  it('refuses to delete a populated directory without -r', async () => {
    await run(MAC, 'mkdir -p ~/acc/tree/nested');
    await run(MAC, 'echo x > ~/acc/tree/nested/f.txt');
    expect((await run(MAC, 'rm ~/acc/tree')).exitCode).not.toBe(0);
    expect(await out(MAC, 'cat ~/acc/tree/nested/f.txt')).toBe('x');
    expect((await run(MAC, 'rm -r ~/acc/tree')).exitCode).toBe(0);
  });

  it('grep reads named files, uses real regexes, and reports no-match with exit 1', async () => {
    await run(MAC, 'echo important-data > ~/acc/g.txt');
    expect(await out(MAC, 'grep important ~/acc/g.txt')).toContain('important-data');
    expect(await out(MAC, 'grep -E "^imp.*data$" ~/acc/g.txt')).toContain('important-data');
    expect((await run(MAC, 'grep nomatchhere ~/acc/g.txt')).exitCode).toBe(1);
    expect((await run(MAC, 'grep IMPORTANT ~/acc/g.txt')).exitCode).toBe(1);
    expect(await out(MAC, 'grep -i IMPORTANT ~/acc/g.txt')).toContain('important-data');
  });
});

describe('SH-6..8 the shell is a real language with a real platform identity', () => {
  it('provides the core utilities', async () => {
    const missing: string[] = [];
    for (const command of ['mv', 'cp', 'find', 'head', 'tail', 'sed', 'awk', 'chmod', 'which', 'sort', 'df', 'tar', 'cut', 'tr', 'diff', 'tee']) {
      if ((await run(MAC, `${command} --version`)).exitCode === 127) missing.push(command);
    }
    expect(missing).toEqual([]);
  });

  it('accepts the POSIX numeric shorthand for head and tail', async () => {
    await run(MAC, 'printf "1\\n2\\n3\\n4\\n5\\n" > ~/acc/n.txt');
    expect(await out(MAC, 'head -2 ~/acc/n.txt')).toBe('1\n2');
    expect(await out(MAC, 'tail -2 ~/acc/n.txt')).toBe('4\n5');
  });

  it('supports ||, globbing, command substitution and variables', async () => {
    expect(await out(MAC, 'nosuchcmd || echo fallback')).toContain('fallback');
    await run(MAC, 'mkdir -p ~/acc/glob && echo a > ~/acc/glob/a.txt && echo b > ~/acc/glob/b.txt');
    const listed = await out(MAC, 'ls ~/acc/glob/*.txt');
    expect(listed).toContain('a.txt');
    expect(listed).toContain('b.txt');
    expect(await out(MAC, 'echo $(hostname)')).toBe(await out(MAC, 'hostname'));
    expect(await out(MAC, 'X=hello; echo $X')).toBe('hello');
  });

  it('evaluates arithmetic expansion instead of silently producing nothing', async () => {
    expect(await out(MAC, 'echo $((2 + 3))')).toBe('5');
    expect(await out(MAC, 'echo $(((2 + 3) * 4))')).toBe('20');
    expect(await out(MAC, 'X=5; echo $((X + 1))')).toBe('6');
    expect(await out(MAC, 'echo pre$((1+1))post')).toBe('pre2post');
    expect(await out(MAC, 'echo $((7 / 2))')).toBe('3');
    // A malformed expression must fail loudly rather than expand to nothing.
    expect((await run(MAC, 'echo $((1 +))')).exitCode).not.toBe(0);
  });

  it('keeps the shell dialects separate', async () => {
    expect((await run(MAC, 'Get-ChildItem ~/acc')).exitCode).toBe(127);
    expect((await run(MAC, 'LS ~/acc')).exitCode).toBe(127);
    expect((await run(WINDOWS, 'uname -a')).exitCode).toBe(127);
    expect((await run(WINDOWS, 'Get-ChildItem ~')).exitCode).toBe(0);
  });

  it('reports simulated identity rather than the host', async () => {
    const mac = await out(MAC, 'uname -a');
    const ubuntu = await out(UBUNTU, 'uname -a');
    expect(mac).toContain('Darwin');
    expect(ubuntu).toContain('Linux');
    // Distinct per computer, which is only possible if it is topology-derived.
    expect(mac).not.toBe(ubuntu);
  });
});

describe('VFS-1 binary content survives a round trip', () => {
  it('preserves arbitrary bytes exactly', async () => {
    const vfs = rt.getVfs(MAC);
    const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0xd8, 0x00, 0x01, 0xfe]);
    await vfs.writeFile('/Users/agent/acc/img.png', bytes);
    const back = await vfs.readBytes('/Users/agent/acc/img.png');
    expect(Buffer.compare(Buffer.from(back), Buffer.from(bytes))).toBe(0);
  });

  it('moves a file without losing its inode identity', async () => {
    await run(MAC, 'echo move-me > ~/acc/from.txt');
    const before = rt.statFile(MAC, '/Users/agent/acc/from.txt');
    await rt.moveFile(MAC, '/Users/agent/acc/from.txt', '/Users/agent/acc/to.txt');
    expect(rt.statFile(MAC, '/Users/agent/acc/to.txt').id).toBe(before.id);
  });
});

describe('GIT-1..3 git has a real object model', () => {
  beforeAll(async () => {
    await run(MAC, 'mkdir -p ~/acc/repo');
    await run(MAC, 'cd ~/acc/repo; git init; echo v1 > a.txt; git add .; git commit -m first');
  });

  it('sees a content change to a tracked file', async () => {
    await run(MAC, 'cd ~/acc/repo; echo COMPLETELY-DIFFERENT > a.txt');
    const status = await out(MAC, 'cd ~/acc/repo; git status');
    expect(status).not.toContain('nothing to commit, working tree clean');
    expect(status.toLowerCase()).toMatch(/modified/);
  });

  it('produces a real unified diff', async () => {
    const diff = await out(MAC, 'cd ~/acc/repo; git diff');
    expect(diff).toContain('-v1');
    expect(diff).toContain('+COMPLETELY-DIFFERENT');
  });

  it('reports untracked files', async () => {
    await run(MAC, 'cd ~/acc/repo; echo new > untracked.txt');
    expect((await out(MAC, 'cd ~/acc/repo; git status')).toLowerCase()).toContain('untracked');
  });

  it('materializes the working tree on checkout', async () => {
    await run(MAC, 'cd ~/acc/repo; git add .; git commit -m second');
    await run(MAC, 'cd ~/acc/repo; git branch other');
    await run(MAC, 'cd ~/acc/repo; echo ONLY-MAIN > main-only.txt; git add .; git commit -m third');
    await run(MAC, 'cd ~/acc/repo; git checkout other');
    expect((await run(MAC, 'cd ~/acc/repo; cat main-only.txt')).exitCode).not.toBe(0);
    await run(MAC, 'cd ~/acc/repo; git checkout main');
    expect(await out(MAC, 'cd ~/acc/repo; cat main-only.txt')).toBe('ONLY-MAIN');
  });

  it('honours .gitignore', async () => {
    await run(MAC, 'mkdir -p ~/acc/ig/node_modules; cd ~/acc/ig; git init');
    await run(MAC, 'cd ~/acc/ig; echo junk > node_modules/x.js; echo keep > keep.txt; echo node_modules > .gitignore');
    await run(MAC, 'cd ~/acc/ig; git add .');
    const staged = await out(MAC, 'cd ~/acc/ig; git status --short');
    expect(staged).not.toContain('node_modules');
    expect(staged).toContain('keep.txt');
  });

  it('accepts -C so a client need not mutate a shared working directory', async () => {
    expect(await out(MAC, 'git -C ~/acc/repo rev-parse --show-toplevel')).toContain('/acc/repo');
  });
});

describe('PKG-1 installed software is reachable and honestly versioned', () => {
  it('parses a version specifier instead of folding it into the package name', async () => {
    await run(UBUNTU, 'mkdir -p ~/proj; cd ~/proj; npm install react@18.2.0');
    const packages = rt.snapshot().computers.find((computer) => computer.spec.id === UBUNTU)!.packages;
    expect(packages.some((item) => item.name === 'react@18.2.0')).toBe(false);
    expect(packages.some((item) => item.name === 'react')).toBe(true);
  });

  it('puts an installed executable on PATH and runs it', async () => {
    await run(MAC, 'brew install ripgrep');
    expect(await out(MAC, 'which rg')).toBe('/opt/homebrew/bin/rg');
    expect((await run(MAC, 'rg --version')).exitCode).toBe(0);
  });

  it('errors on an unknown verb rather than silently listing', async () => {
    expect((await run(UBUNTU, 'apt frobnicate')).exitCode).not.toBe(0);
  });

  it('reports plausible upstream versions rather than hash-derived ones', async () => {
    const info = await out(MAC, 'brew info ffmpeg');
    // The old model produced versions bounded by 12.22.10 from a sha256 slice.
    expect(info).not.toMatch(/\b(?:[0-9]|1[0-2])\.\d{1,2}\.\d{1,2}\b\s*$/m);
    expect(info.length).toBeGreaterThan(0);
  });
});

describe('PROC-1 process identity and lifecycle', () => {
  it('never reissues PID 1 to a non-init process', async () => {
    for (let index = 0; index < 20; index++) await run(MAC, 'echo x');
    const processes = rt.snapshot().computers.find((computer) => computer.spec.id === MAC)!.processes;
    expect(processes.filter((process) => process.pid === 1)).toHaveLength(1);
  });

  it('derives the boot service tree from the OS profile', async () => {
    const processes = rt.snapshot().computers.find((computer) => computer.spec.id === MAC)!.processes;
    expect(processes.find((process) => process.pid === 1)?.executable).toBe('launchd');
    expect(processes.some((process) => process.executable === 'WindowServer')).toBe(true);
  });
});

describe('OS profiles are enforced rather than decorative', () => {
  it('uses the platform-correct home directory', async () => {
    expect(await out(MAC, 'cd ~; pwd')).toBe('/Users/agent');
    expect(await out(UBUNTU, 'cd ~; pwd')).toBe('/home/agent');
  });

  it('renders network state from the real interface model', async () => {
    const ifconfig = await out(MAC, 'ifconfig');
    expect(ifconfig).toContain('10.42.0.10');
    expect(ifconfig).toMatch(/ether ([0-9a-f]{2}:){5}[0-9a-f]{2}/);
  });
});
