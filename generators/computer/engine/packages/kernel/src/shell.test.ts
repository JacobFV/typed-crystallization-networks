import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import type { ComputerSpec, OSRuntimeProfile } from '@tcn-computer/protocol';
import { InternetFabric } from './network.js';
import { ProcessManager } from './processes.js';
import { ShellSession } from './shell.js';
import { SoftwareEnvironment } from './software.js';
import { VirtualFileSystem } from './vfs.js';

/* -------------------------------------------------------------------------- *
 * Harness: a real VFS, process table and network fabric behind one shell.
 * -------------------------------------------------------------------------- */

const macProfile: OSRuntimeProfile = {
  shell: { default: 'zsh', executable: '/bin/zsh', promptDialect: 'posix', startupFiles: ['~/.zshrc'] },
  filesystem: { root: '/', home: '/Users/agent', applications: '/Applications', userData: '/Users/agent/Library', temporary: '/private/tmp', caseSensitive: false, pathSeparator: '/' },
  packageManagers: { native: ['brew', 'mas'], language: ['npm', 'pip', 'cargo'], receiptRoots: ['/var/db/receipts'] },
  bootServices: [{ id: 'launchd', executable: '/sbin/launchd', role: 'init', parent: null, required: true }],
  conventions: { executableSuffix: '', sharedLibrarySuffix: '.dylib', environmentPathKey: 'PATH', localhostNames: ['localhost'] },
};

const ubuntuProfile: OSRuntimeProfile = {
  ...macProfile,
  shell: { default: 'bash', executable: '/bin/bash', promptDialect: 'posix', startupFiles: ['~/.bashrc'] },
  filesystem: { root: '/', home: '/home/agent', applications: '/opt', userData: '/home/agent/.local/share', temporary: '/tmp', caseSensitive: true, pathSeparator: '/' },
  packageManagers: { native: ['apt', 'dpkg', 'snap'], language: ['npm', 'pip', 'cargo'], receiptRoots: ['/var/lib/dpkg'] },
};

const windowsProfile: OSRuntimeProfile = {
  shell: { default: 'powershell', executable: 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe', promptDialect: 'powershell', startupFiles: ['~/Documents/profile.ps1'] },
  filesystem: { root: 'C:\\', home: '/C/Users/agent', applications: '/C/Program Files', userData: '/C/Users/agent/AppData', temporary: '/C/Users/agent/AppData/Local/Temp', caseSensitive: false, pathSeparator: '\\' },
  packageManagers: { native: ['winget', 'choco'], language: ['npm', 'pip'], receiptRoots: ['/C/ProgramData'] },
  bootServices: [{ id: 'wininit', executable: 'C:\\Windows\\System32\\wininit.exe', role: 'init', parent: null, required: true }],
  conventions: { executableSuffix: '.exe', sharedLibrarySuffix: '.dll', environmentPathKey: 'Path', localhostNames: ['localhost'] },
};

function specFor(os: ComputerSpec['os'], id: string): ComputerSpec {
  return {
    id, hostname: id, os, shell: os === 'windows' ? 'powershell' : os === 'macos' ? 'zsh' : 'bash',
    ipv4: os === 'macos' ? '10.42.0.10' : os === 'windows' ? '10.42.0.20' : '10.42.0.30',
    memoryBytes: 16 * 1024 ** 3, cpuCores: 8,
    disks: [{ id: 'disk0', label: 'Seed', mount: os === 'windows' ? 'C:' : '/', capacityBytes: 512 * 1024 ** 3 }],
    displays: [{ id: 'main', name: 'Seed Display', width: 1920, height: 1080, scale: 2 }],
  };
}

interface Harness { shell: ShellSession; vfs: VirtualFileSystem; processes: ProcessManager; network: InternetFabric; spec: ComputerSpec; software: SoftwareEnvironment; profile: OSRuntimeProfile }

let stateRoot = '';
let counter = 0;

async function harness(os: ComputerSpec['os'] = 'ubuntu'): Promise<Harness> {
  const profile = os === 'windows' ? windowsProfile : os === 'macos' ? macProfile : ubuntuProfile;
  const spec = specFor(os, `${os}-${++counter}`);
  const vfs = new VirtualFileSystem(stateRoot, `run-${counter}`, spec, { caseSensitive: profile.filesystem.caseSensitive });
  await vfs.initialize();
  await vfs.mkdir(profile.filesystem.home);
  await vfs.mkdir('/usr/local/bin');
  const processes = new ProcessManager(spec.id, { cpuCores: spec.cpuCores });
  processes.boot('init', profile.filesystem.home, {});
  const network = new InternetFabric();
  network.attach(spec, 'seed.local');
  const software = new SoftwareEnvironment(spec, vfs, processes, profile);
  const shell = new ShellSession({
    spec, profile, vfs, processes, network, software,
    listApps: () => [], catalog: () => [], install: () => { throw new Error('no registry in this harness'); }, onAction: () => undefined,
  });
  return { shell, vfs, processes, network, spec, software, profile };
}

beforeAll(async () => { stateRoot = await mkdtemp(path.join(tmpdir(), 'seed-shell-')); });

/** Convenience: run a line and return stdout. */
const out = async (harnessed: Harness, line: string): Promise<string> => (await harnessed.shell.execute(line)).stdout;

describe('data loss and correctness', () => {
  it('1. touch creates a file but never truncates an existing one', async () => {
    const box = await harness();
    await box.shell.execute('echo original-content > note.txt');
    expect(await box.vfs.readFile('/home/agent/note.txt')).toBe('original-content');
    const before = box.vfs.statSync('/home/agent/note.txt')!;
    const result = await box.shell.execute('touch note.txt');
    expect(result.exitCode).toBe(0);
    expect(await box.vfs.readFile('/home/agent/note.txt')).toBe('original-content');
    expect(box.vfs.statSync('/home/agent/note.txt')!.modifiedAt >= before.modifiedAt).toBe(true);
    expect((await box.shell.execute('touch fresh.txt')).exitCode).toBe(0);
    expect(box.vfs.statSync('/home/agent/fresh.txt')?.kind).toBe('file');
  });

  it('2. >> appends instead of writing a file literally named "> file"', async () => {
    const box = await harness();
    await box.shell.execute('echo one > log.txt');
    const appended = await box.shell.execute('echo two >> log.txt');
    expect(appended.exitCode).toBe(0);
    expect(appended.stdout).toBe('');
    expect(await box.vfs.readFile('/home/agent/log.txt')).toBe('one\ntwo');
    expect(box.vfs.statSync('/home/agent/> log.txt')).toBeUndefined();
    expect(await out(box, 'cat log.txt')).toBe('one\ntwo');
  });

  it('2b. supports <, 2>, &> and here-documents', async () => {
    const box = await harness();
    await box.shell.execute('printf "beta\\nalpha\\n" > unsorted.txt');
    expect(await out(box, 'sort < unsorted.txt')).toBe('alpha\nbeta');
    const redirected = await box.shell.execute('cat missing-file.txt 2> errors.txt');
    expect(redirected.stderr).toBe('');
    expect(await box.vfs.readFile('/home/agent/errors.txt')).toContain('No such file or directory');
    await box.shell.execute('cat missing-file.txt &> both.txt');
    expect(await box.vfs.readFile('/home/agent/both.txt')).toContain('No such file');
    const heredoc = await box.shell.execute('cat <<EOF\nfirst line\nsecond line\nEOF');
    expect(heredoc.stdout).toBe('first line\nsecond line');
  });

  it('3. statement splitting and redirection are quote-aware', async () => {
    const box = await harness();
    const semicolon = await box.shell.execute('echo "a; b"');
    expect(semicolon.stdout).toBe('a; b');
    expect(semicolon.stderr).toBe('');
    expect(semicolon.exitCode).toBe(0);
    const angle = await box.shell.execute('echo "a > b"');
    expect(angle.stdout).toBe('a > b');
    expect(box.vfs.statSync('/home/agent/b')).toBeUndefined();
    expect(await out(box, "echo 'single | quoted && text'")).toBe('single | quoted && text');
    expect(await out(box, 'echo one; echo two')).toBe('one\ntwo');
  });

  it('4. rm refuses a populated directory without -r', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p project/src; echo code > project/src/main.ts');
    const refused = await box.shell.execute('rm project');
    expect(refused.exitCode).toBe(1);
    expect(refused.stderr).toContain('is a directory');
    expect(box.vfs.statSync('/home/agent/project/src/main.ts')?.kind).toBe('file');
    const removed = await box.shell.execute('rm -r project');
    expect(removed.exitCode).toBe(0);
    expect(box.vfs.statSync('/home/agent/project')).toBeUndefined();
  });

  it('4b. rmdir surfaces the VFS "directory not empty" error', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p full/inner; mkdir empty');
    const refused = await box.shell.execute('rmdir full');
    expect(refused.exitCode).toBe(1);
    expect(refused.stderr).toContain('Directory not empty');
    expect((await box.shell.execute('rmdir empty')).exitCode).toBe(0);
  });

  it('5. grep opens named files, uses real regular expressions and exits 1 on no match', async () => {
    const box = await harness();
    await box.shell.execute('printf "alpha\\nBeta\\ngamma\\n" > words.txt');
    const found = await box.shell.execute('grep beta words.txt');
    expect(found.exitCode).toBe(1);
    expect(found.stdout).toBe('');
    const insensitive = await box.shell.execute('grep -i beta words.txt');
    expect(insensitive.exitCode).toBe(0);
    expect(insensitive.stdout).toBe('Beta');
    expect(await out(box, 'grep "^g" words.txt')).toBe('gamma');
    expect(await out(box, 'grep -n mm words.txt')).toBe('3:gamma');
    expect(await out(box, 'grep -c a words.txt')).toBe('3');
    expect(await out(box, 'grep -v a words.txt')).toBe('');
    expect(await out(box, 'grep -E "al(pha|ien)" words.txt')).toBe('alpha');
    const missing = await box.shell.execute('grep alpha nothing-here.txt');
    expect(missing.exitCode).toBe(2);
    expect(missing.stderr).toContain('No such file or directory');
    await box.shell.execute('mkdir tree; echo needle > tree/found.txt');
    expect(await out(box, 'grep -rl needle tree')).toBe('tree/found.txt');
  });

  it('6. wc honours -l/-w/-c and echo honours -n/-e', async () => {
    const box = await harness();
    await box.shell.execute('printf "one two\\nthree\\n" > counts.txt');
    expect(await out(box, 'wc -l counts.txt')).toBe('2 counts.txt');
    expect(await out(box, 'wc -w counts.txt')).toBe('3 counts.txt');
    // Redirection stores the payload without the stream's terminating newline,
    // so the file holds `one two\nthree` — 13 bytes.
    expect(await out(box, 'wc -c counts.txt')).toBe('13 counts.txt');
    expect((await out(box, 'wc counts.txt')).split(/\s+/).filter(Boolean).slice(0, 3)).toEqual(['2', '3', '13']);
    await box.shell.execute('echo -n no-newline > partial.txt');
    expect(await box.vfs.readFile('/home/agent/partial.txt')).toBe('no-newline');
    expect(await out(box, 'echo -n abc | wc -c')).toBe('3');
    expect(await out(box, 'echo abc | wc -c')).toBe('4');
    expect(await out(box, 'echo -e "a\\tb"')).toBe('a\tb');
    expect(await out(box, 'echo -E "a\\tb"')).toBe('a\\tb');
  });

  it('6b. rejects unknown flags rather than ignoring them', async () => {
    const box = await harness();
    const result = await box.shell.execute('wc -Z');
    expect(result.exitCode).toBe(2);
    expect(result.stderr).toContain("invalid option -- 'Z'");
    const longOption = await box.shell.execute('ls --nonsense');
    expect(longOption.exitCode).toBe(2);
    expect(longOption.stderr).toContain("unrecognized option '--nonsense'");
  });
});

describe('shell language', () => {
  it('7. runs || fallbacks, && chains and honours exit codes', async () => {
    const box = await harness();
    const fallback = await box.shell.execute('false || echo fallback-ran');
    expect(fallback.stdout).toBe('fallback-ran');
    expect(fallback.exitCode).toBe(0);
    expect(await out(box, 'true && echo chained')).toBe('chained');
    expect(await out(box, 'true || echo never-runs')).toBe('');
    expect(await out(box, 'grep nothing /home/agent || echo grep-failed')).toBe('grep-failed');
    expect(await out(box, 'echo first && echo second')).toBe('first\nsecond');
    expect((await box.shell.execute('false && echo skipped')).exitCode).toBe(1);
  });

  it('7b. expands globs', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p globs; cd globs; touch a.txt b.txt c.md');
    expect(await out(box, 'ls *.txt')).toBe('a.txt\nb.txt');
    expect(await out(box, 'echo *.md')).toBe('c.md');
    expect(await out(box, 'echo ?.txt')).toBe('a.txt b.txt');
    expect(await out(box, 'echo [ab].txt')).toBe('a.txt b.txt');
    expect(await out(box, 'echo no-match-*.zzz')).toBe('no-match-*.zzz');
    await box.shell.execute('mkdir -p deep/nested; touch deep/nested/found.txt');
    expect(await out(box, 'echo deep/*/found.txt')).toBe('deep/nested/found.txt');
    expect(await out(box, 'echo **/found.txt')).toBe('deep/nested/found.txt');
  });

  it('7c. performs command substitution with $() and backticks', async () => {
    const box = await harness();
    expect(await out(box, 'echo "host is $(hostname)"')).toBe(`host is ${box.spec.hostname}`);
    expect(await out(box, 'echo `echo nested`')).toBe('nested');
    await box.shell.execute('echo alpha > subject.txt');
    expect(await out(box, 'echo "file says: $(cat subject.txt)"')).toBe('file says: alpha');
    expect(await out(box, 'echo $(echo one; echo two)')).toBe('one two');
  });

  it('7d. assigns, exports and expands variables', async () => {
    const box = await harness();
    expect(await out(box, 'NAME=seed; echo $NAME')).toBe('seed');
    expect(await out(box, 'echo ${NAME}-suffix')).toBe('seed-suffix');
    expect(await out(box, 'echo ${MISSING:-default-value}')).toBe('default-value');
    expect(await out(box, 'echo ${MISSING:+set}')).toBe('');
    expect(await out(box, 'echo ${NAME:+present}')).toBe('present');
    expect(await out(box, 'echo ${#NAME}')).toBe('4');
    await box.shell.execute('export GREETING=hello');
    expect(await out(box, 'env | grep GREETING')).toBe('GREETING=hello');
    expect(await out(box, 'echo "quoted $NAME"')).toBe('quoted seed');
    expect(await out(box, "echo 'literal $NAME'")).toBe('literal $NAME');
    await box.shell.execute('unset NAME');
    expect(await out(box, 'echo "[$NAME]"')).toBe('[]');
    expect(await out(box, 'false; echo $?')).toBe('1');
    // Tilde expansion happens during expansion, so argv carries the real path.
    expect(await out(box, 'echo ~')).toBe('/home/agent');
    expect(await out(box, 'echo ~/Documents')).toBe('/home/agent/Documents');
    expect(await out(box, "echo '~'")).toBe('~');
  });

  it('7e. pipes through real processes and supports subshells and background jobs', async () => {
    const box = await harness();
    await box.shell.execute('printf "delta\\nalpha\\ncharlie\\n" > names.txt');
    expect(await out(box, 'cat names.txt | sort | head -n 2')).toBe('alpha\ncharlie');
    expect(await out(box, 'cat names.txt | wc -l')).toBe('3');
    expect(await out(box, '(cd /usr; pwd)')).toBe('/usr');
    expect(box.shell.cwd).toBe('/home/agent');
    const background = await box.shell.execute('echo backgrounded &');
    expect(background.exitCode).toBe(0);
    expect(background.stderr).toMatch(/^\[1\] \d+$/);
    expect(await out(box, 'jobs')).toContain('Done');
  });

  it('7f. reports syntax errors instead of guessing', async () => {
    const box = await harness();
    const unterminated = await box.shell.execute('echo "unbalanced');
    expect(unterminated.exitCode).toBe(2);
    expect(unterminated.stderr).toContain('unterminated double quote');
    const dangling = await box.shell.execute('echo hello |');
    expect(dangling.exitCode).toBe(2);
  });
});

describe('coreutils', () => {
  it('moves and copies through the VFS rename/copy primitives', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p src dest; echo payload > src/file.txt');
    expect((await box.shell.execute('mv src/file.txt dest/renamed.txt')).exitCode).toBe(0);
    expect(box.vfs.statSync('/home/agent/src/file.txt')).toBeUndefined();
    expect(await box.vfs.readFile('/home/agent/dest/renamed.txt')).toBe('payload');
    const noRecurse = await box.shell.execute('cp dest copies');
    expect(noRecurse.exitCode).toBe(1);
    expect(noRecurse.stderr).toContain('-r not specified');
    expect((await box.shell.execute('cp -r dest copies')).exitCode).toBe(0);
    expect(await box.vfs.readFile('/home/agent/copies/renamed.txt')).toBe('payload');
  });

  it('implements head, tail, cut, tr, sort, uniq, sed, awk, diff, tee, xargs', async () => {
    const box = await harness();
    await box.shell.execute('printf "1\\n2\\n3\\n4\\n5\\n" > numbers.txt');
    expect(await out(box, 'head -n 2 numbers.txt')).toBe('1\n2');
    expect(await out(box, 'tail -n 2 numbers.txt')).toBe('4\n5');
    await box.shell.execute('printf "a:b:c\\nd:e:f\\n" > fields.txt');
    expect(await out(box, 'cut -d: -f2 fields.txt')).toBe('b\ne');
    expect(await out(box, 'cat fields.txt | tr ":" "-"')).toBe('a-b-c\nd-e-f');
    expect(await out(box, 'printf "b\\na\\nb\\n" | sort | uniq -c')).toBe('      1 a\n      2 b');
    expect(await out(box, 'sed "s/a/A/g" fields.txt')).toBe('A:b:c\nd:e:f');
    expect(await out(box, "awk -F: '{print $3}' fields.txt")).toBe('c\nf');
    expect(await out(box, "awk 'END {print NR}' numbers.txt")).toBe('5');
    await box.shell.execute('printf "1\\n2\\n" > left.txt; printf "1\\n3\\n" > right.txt');
    const diff = await box.shell.execute('diff -u left.txt right.txt');
    expect(diff.exitCode).toBe(1);
    expect(diff.stdout).toContain('-2');
    expect(diff.stdout).toContain('+3');
    expect((await box.shell.execute('diff left.txt left.txt')).exitCode).toBe(0);
    expect(await out(box, 'echo teed | tee copy.txt')).toBe('teed');
    expect(await box.vfs.readFile('/home/agent/copy.txt')).toBe('teed');
    expect(await out(box, 'echo "a b" | xargs echo prefix')).toBe('prefix a b');
  });

  it('implements find, chmod, chown, ln, stat, file, du, df, basename, dirname, which', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p find/nested; touch find/one.txt find/nested/two.txt');
    expect(await out(box, 'find find -name "*.txt"')).toBe('find/nested/two.txt\nfind/one.txt');
    expect(await out(box, 'find find -type d')).toBe('find\nfind/nested');
    expect(await out(box, 'find find -name one.txt -exec cat {} ;')).toBe('');
    await box.shell.execute('echo script > run.sh');
    expect((await box.shell.execute('chmod 755 run.sh')).exitCode).toBe(0);
    expect(box.vfs.statSync('/home/agent/run.sh')!.mode).toBe(0o755);
    await box.shell.execute('chmod u-x run.sh');
    expect(box.vfs.statSync('/home/agent/run.sh')!.mode & 0o100).toBe(0);
    expect((await box.shell.execute('chown root:root run.sh')).exitCode).toBe(0);
    expect(box.vfs.owner('/home/agent/run.sh').uid).toBe(0);
    expect((await box.shell.execute('ln -s run.sh link.sh')).exitCode).toBe(0);
    expect(box.vfs.lstatSync('/home/agent/link.sh')?.kind).toBe('symlink');
    expect(await out(box, 'readlink link.sh')).toBe('run.sh');
    expect(await out(box, 'stat run.sh')).toContain('File: run.sh');
    expect(await out(box, 'file run.sh')).toContain('ASCII text');
    expect(await out(box, 'basename /a/b/c.txt .txt')).toBe('c');
    expect(await out(box, 'dirname /a/b/c.txt')).toBe('/a/b');
    expect(await out(box, 'df')).toContain('Filesystem');
    expect(await out(box, 'du -s find')).toMatch(/\tfind$/);
    expect(await out(box, 'which ls')).toContain('shell built-in');
  });

  it('round-trips archives through tar and zip', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p bundle; echo archived > bundle/inner.txt');
    expect((await box.shell.execute('tar -c -f bundle.tar bundle')).exitCode).toBe(0);
    expect(await out(box, 'tar -t -f bundle.tar')).toContain('bundle/inner.txt');
    await box.shell.execute('mkdir -p restored; tar -x -f bundle.tar -C restored');
    expect(await box.vfs.readFile('/home/agent/restored/bundle/inner.txt')).toBe('archived');
    await box.shell.execute('zip -r bundle.zip bundle');
    expect(await out(box, 'unzip -l bundle.zip')).toContain('bundle/inner.txt');
  });

  it('implements sudo, kill by name, ps, top and man', async () => {
    const box = await harness();
    expect(await out(box, 'sudo whoami')).toBe('root');
    expect(await out(box, 'whoami')).toBe('agent');
    expect(await out(box, 'ps')).toContain('PID');
    expect(await out(box, 'top')).toContain('Processes:');
    const record = box.processes.spawn({ executable: 'seed-daemon', cwd: '/', env: {} });
    expect(record.executable).toBe('seed-daemon');
    expect((await box.shell.execute('pkill seed-daemon')).exitCode).toBe(0);
    expect(box.processes.list().some((entry) => entry.executable === 'seed-daemon')).toBe(false);
    expect(await out(box, 'man grep')).toContain('search files with regular expressions');
    const missing = await box.shell.execute('man not-a-command');
    expect(missing.exitCode).toBe(16);
  });

  it('is honest about what it does not implement', async () => {
    const box = await harness();
    const follow = await box.shell.execute('tail -f numbers.txt');
    expect(follow.exitCode).toBe(1);
    expect(follow.stderr).toContain('not implemented in this simulation');
    const interactive = await box.shell.execute('rm -i anything');
    expect(interactive.exitCode).toBe(1);
    expect(interactive.stderr).toContain('not implemented in this simulation');
    const unknown = await box.shell.execute('definitely-not-a-command');
    expect(unknown.exitCode).toBe(127);
    expect(unknown.stderr).toContain('command not found');
  });
});

describe('platform identity', () => {
  it('8. rejects PowerShell cmdlets on macOS and is case-sensitive on a POSIX shell', async () => {
    const mac = await harness('macos');
    const cmdlet = await mac.shell.execute('Get-ChildItem');
    expect(cmdlet.exitCode).toBe(127);
    expect(cmdlet.stderr).toContain('command not found: Get-ChildItem');
    const shouting = await mac.shell.execute('LS');
    expect(shouting.exitCode).toBe(127);
    expect(shouting.stderr).toContain('command not found: LS');
    expect((await mac.shell.execute('ls')).exitCode).toBe(0);
    const wrongCase = await mac.shell.execute('Get-Process');
    expect(wrongCase.exitCode).toBe(127);
  });

  it('8b. resolves cmdlets and their aliases case-insensitively on PowerShell, and rejects POSIX-only tools', async () => {
    const windows = await harness('windows');
    expect((await windows.shell.execute('Get-ChildItem')).exitCode).toBe(0);
    expect((await windows.shell.execute('get-childitem')).exitCode).toBe(0);
    expect((await windows.shell.execute('LS')).exitCode).toBe(0);
    expect((await windows.shell.execute('dir')).exitCode).toBe(0);
    const posixOnly = await windows.shell.execute('grep pattern');
    expect(posixOnly.exitCode).toBe(127);
    expect(posixOnly.stderr).toContain('is not recognized');
    expect((await windows.shell.execute('uname -a')).exitCode).toBe(127);
    expect(await out(windows, 'Get-Location')).toBe('C:\\Users\\agent');
    expect(windows.shell.prompt()).toBe('PS C:\\Users\\agent> ');
    expect(await out(windows, 'Write-Output hello')).toBe('hello');
    await windows.shell.execute('Set-Content -Path note.txt -Value windows-payload');
    expect(await windows.vfs.readFile('/C/Users/agent/note.txt')).toBe('windows-payload');
    expect(await out(windows, 'Get-Content note.txt')).toBe('windows-payload');
    expect(await out(windows, 'Get-Process | findstr init')).toContain('init');
    expect(await out(windows, 'Test-Path note.txt')).toBe('True');
    // PowerShell escapes with a backtick, so a backslash path stays a path.
    expect(await out(windows, 'Get-Content C:\\Users\\agent\\note.txt')).toBe('windows-payload');
    expect(await out(windows, 'echo $env:USERNAME')).toBe('agent');
    const bashism = await windows.shell.execute('Remove-Item -rf note.txt');
    expect(bashism.exitCode).toBe(2);
  });

  it('9. derives date and uname from the simulation, never the host', async () => {
    const box = await harness();
    const first = await out(box, 'date');
    expect(first).toContain('2026');
    expect(first).toContain('UTC');
    expect(await out(box, 'date +%Y-%m-%d')).toMatch(/^2026-01-01$/);
    const second = await out(box, 'date +%s');
    await box.shell.execute('sleep 5');
    const third = await out(box, 'date +%s');
    expect(Number(third) - Number(second)).toBe(7); // 5 slept + one tick per command line
    // A second session over the same computer replays the same clock: the boot
    // instant is derived from the computer id, never from the host.
    const replay = new ShellSession({
      spec: box.spec, profile: box.profile, vfs: box.vfs, processes: box.processes, network: box.network, software: box.software,
      listApps: () => [], catalog: () => [], install: () => { throw new Error('no registry'); }, onAction: () => undefined,
    });
    expect((await replay.execute('date')).stdout).toBe(first);
    expect(await out(await harness(), 'uname -r')).toBe(await out(box, 'uname -r'));
    const uname = await out(box, 'uname -a');
    expect(uname).toContain('Linux');
    expect(uname).toContain('x86_64');
    expect(uname).not.toContain(process.arch);
    expect(await out(await harness('macos'), 'uname -a')).toContain('Darwin');
  });

  it('10. honours case sensitivity and the executable suffix from the profile', async () => {
    const insensitive = await harness('macos');
    await insensitive.shell.execute('echo payload > Case.txt');
    expect(await out(insensitive, 'cat case.txt')).toBe('payload');
    expect(await out(insensitive, 'ls CASE.TXT')).toBe('CASE.TXT');
    const sensitive = await harness('ubuntu');
    await sensitive.shell.execute('echo payload > Case.txt');
    const missing = await sensitive.shell.execute('cat case.txt');
    expect(missing.exitCode).toBe(1);
    const windows = await harness('windows');
    await windows.vfs.writeFile('/C/Windows/System32/demo.exe', JSON.stringify({ seedExecutable: 1, name: 'demo', package: 'demo', manager: 'winget', version: '1.0.0', provides: ['demo'], behavior: 'tree' }));
    await windows.vfs.chmod('/C/Windows/System32/demo.exe', 0o755);
    expect((await windows.shell.execute('demo --version')).exitCode).toBe(0);
  });
});

describe('PATH resolution and installed programs', () => {
  async function install(box: Harness, directory: string, name: string, behavior: string, extra: Record<string, unknown> = {}): Promise<void> {
    await box.vfs.writeFile(`${directory}/${name}`, JSON.stringify({
      seedExecutable: 1, name, package: name, manager: 'apt', version: '14.1.1', provides: [name], behavior, ...extra,
    }, null, 2));
    await box.vfs.chmod(`${directory}/${name}`, 0o755);
  }

  it('resolves an installed executable descriptor from PATH and runs it', async () => {
    const box = await harness();
    expect((await box.shell.execute('rg pattern')).exitCode).toBe(127);
    await install(box, '/usr/bin', 'rg', 'rg');
    await box.shell.execute('mkdir -p search/inner');
    await box.shell.execute('printf "hello world\\nsecond line\\n" > search/one.txt');
    await box.shell.execute('echo hello again > search/inner/two.txt');
    const result = await box.shell.execute('rg hello search');
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain('search/one.txt:1:hello world');
    expect(result.stdout).toContain('search/inner/two.txt:1:hello again');
    expect(await out(box, 'rg -l hello search')).toContain('search/one.txt');
    expect((await box.shell.execute('rg nothing-matches search')).exitCode).toBe(1);
    expect(await out(box, 'which rg')).toBe('/usr/bin/rg');
  });

  it('fails honestly when an installed executable has no implemented behavior', async () => {
    const box = await harness();
    await install(box, '/usr/bin', 'ffmpeg', 'ffmpeg');
    const result = await box.shell.execute('ffmpeg -i input.mov');
    expect(result.exitCode).toBe(127);
    expect(result.stderr).toContain("behavior 'ffmpeg' is not implemented in this simulation");
    expect(result.stderr).toContain('installed from apt package');
    expect(result.stderr).not.toContain('command not found');
  });

  it('resolves project-local node_modules/.bin tools', async () => {
    const box = await harness();
    await box.shell.execute('mkdir -p app/node_modules/.bin; cd app');
    await install(box, '/home/agent/app/node_modules/.bin', 'jq', 'jq');
    await box.shell.execute('echo \'{"name":"seed","tags":["a","b"]}\' > data.json');
    expect(await out(box, 'jq -r .name data.json')).toBe('seed');
    expect(await out(box, 'jq -c .tags data.json')).toBe('["a","b"]');
    expect(await out(box, 'cat data.json | jq ".tags | length"')).toBe('2');
  });

  it('runs tree, bat, fd and gh from descriptors', async () => {
    const box = await harness();
    for (const [name, behavior] of [['tree', 'tree'], ['bat', 'bat'], ['fdfind', 'fd']]) await install(box, '/usr/bin', name!, behavior!);
    await box.shell.execute('mkdir -p demo/nested; echo alpha > demo/alpha.txt; echo beta > demo/nested/beta.txt');
    const tree = await box.shell.execute('tree demo');
    expect(tree.stdout).toContain('└── ');
    expect(tree.stdout).toContain('1 directory, 2 files');
    expect(await out(box, 'bat -p demo/alpha.txt')).toBe('alpha');
    expect(await out(box, 'fdfind beta demo')).toBe('nested/beta.txt');
  });

  it('nginx registers a real HTTP service on the virtual fabric', async () => {
    const box = await harness();
    await box.vfs.writeFile('/usr/sbin/nginx', JSON.stringify({ seedExecutable: 1, name: 'nginx', package: 'nginx', manager: 'apt', version: '1.27.3', provides: ['nginx'], behavior: 'nginx' }));
    await box.vfs.chmod('/usr/sbin/nginx', 0o755);
    await box.vfs.mkdir('/usr/share/nginx/html');
    await box.vfs.writeFile('/usr/share/nginx/html/index.html', '<h1>served by nginx</h1>');
    const started = await box.shell.execute('nginx');
    expect(started.exitCode).toBe(0);
    expect(box.processes.list().some((record) => record.executable === 'nginx')).toBe(true);
    const response = await box.network.request(box.spec.id, `http://${box.spec.hostname}.seed.local/`);
    expect(response.status).toBe(200);
    expect(response.body).toContain('served by nginx');
    expect((await box.shell.execute('nginx -s stop')).exitCode).toBe(0);
    await expect(box.network.request(box.spec.id, `http://${box.spec.hostname}.seed.local/`)).rejects.toThrow();
  });
});

describe('processes, jobs and the network fabric', () => {
  it('models pipelines and background jobs with real process records', async () => {
    const box = await harness();
    const before = box.processes.list().length;
    await box.shell.execute('echo one | cat | cat');
    expect(box.processes.list().length).toBeGreaterThanOrEqual(before);
    await box.shell.execute('sleep 1 &');
    const jobs = await out(box, 'jobs');
    expect(jobs).toContain('[1]');
    const waited = await box.shell.execute('wait');
    expect(waited.exitCode).toBe(0);
  });

  it('serves files over the fabric and reports a duplicate bind', async () => {
    const box = await harness();
    await box.shell.execute('echo served-body > page.txt');
    const served = await box.shell.execute('serve 9300 page.txt localhost');
    expect(served.exitCode).toBe(0);
    expect(served.stdout).toMatch(/pid \d+/);
    expect(await out(box, 'curl http://localhost:9300/')).toBe('served-body');
    const duplicate = await box.shell.execute('serve 9300 page.txt localhost');
    expect(duplicate.exitCode).toBe(1);
    expect(duplicate.stderr).toContain('cannot bind');
    const refused = await box.shell.execute('curl http://localhost:9999/');
    expect(refused.exitCode).toBe(1);
    expect(refused.stderr).toContain('connection refused');
  });

  it('renders interfaces, routes and DNS from the fabric instead of literals', async () => {
    const box = await harness();
    const ifconfig = await out(box, 'ifconfig');
    expect(ifconfig).toContain(box.spec.ipv4);
    expect(ifconfig).toContain('ether');
    expect(await out(box, 'ip route')).toContain('default');
    expect(await out(box, 'nslookup localhost')).toContain('Address: 127.0.0.1');
    const unknown = await box.shell.execute('nslookup does-not-exist.invalid');
    expect(unknown.exitCode).toBe(1);
    expect(await out(box, 'ping -c 2 ' + box.spec.hostname + '.seed.local')).toContain('2 packets transmitted');
  });
});
