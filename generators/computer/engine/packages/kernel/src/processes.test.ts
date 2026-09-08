import { describe, expect, it } from 'vitest';
import { ProcessManager, SIGNAL_NUMBERS, type ProcessExitStatus, type SeedSignal } from './processes.js';

function booted(options?: ConstructorParameters<typeof ProcessManager>[1]) {
  const processes = new ProcessManager('test-box', options);
  const init = processes.boot('systemd', '/home/agent', { HOSTNAME: 'test-box' });
  return { processes, init };
}

describe('pid allocation', () => {
  it('gives init pid 1 and never issues it again, even after the table drains', () => {
    const { processes, init } = booted();
    expect(init.pid).toBe(1);

    const first = processes.spawn({ executable: 'a' });
    const second = processes.spawn({ executable: 'b' });
    expect([first.pid, second.pid]).toEqual([100, 101]);

    // Drain everything reachable: the old table deleted records outright, so the next spawn
    // collided with init.
    expect(processes.kill(first.pid)).toBe(true);
    expect(processes.kill(second.pid)).toBe(true);
    expect(processes.list().map((record) => record.pid)).toEqual([1]);

    const third = processes.spawn({ executable: 'c' });
    expect(third.pid).not.toBe(1);
    expect(third.pid).toBeGreaterThan(1);
    expect(processes.get(1)?.executable).toBe('systemd');
  });

  it('refuses to signal init and refuses unknown pids', () => {
    const { processes } = booted();
    expect(processes.kill(1)).toBe(false);
    expect(processes.kill(1, 'SIGKILL')).toBe(false);
    expect(processes.kill(4242)).toBe(false);
    expect(processes.get(1)?.state).toBe('running');
  });

  it('wraps the pid space without ever reusing a live pid or pid 1', () => {
    const { processes } = booted({ pidMin: 100, pidMax: 104 });
    const held = processes.spawn({ executable: 'held' });
    expect(held.pid).toBe(100);
    const pids = [held.pid];
    for (let index = 0; index < 4; index += 1) {
      const record = processes.spawn({ executable: `p${index}` });
      pids.push(record.pid);
      if (record.pid !== 101) processes.kill(record.pid);
    }
    expect(new Set(pids).size).toBe(pids.length);
    expect(pids).not.toContain(1);
    expect(processes.get(100)?.executable).toBe('held');
  });

  it('refuses to boot twice', () => {
    const { processes } = booted();
    expect(() => processes.boot('systemd', '/', {})).toThrow('already booted');
  });
});

describe('signals', () => {
  it('applies the default disposition and records a signal exit status', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const child = processes.spawn({ executable: 'worker', ppid: parent.pid });

    expect(processes.kill(child.pid, 'SIGTERM')).toBe(true);
    const zombie = processes.get(child.pid)!;
    expect(zombie.state).toBe('zombie');
    expect(zombie.exitSignal).toBe('SIGTERM');
    expect(zombie.exitCode).toBe(128 + SIGNAL_NUMBERS.SIGTERM);
  });

  it('lets a process catch SIGTERM but never SIGKILL', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const catcher = processes.spawn({ executable: 'catcher', ppid: parent.pid });
    const seen: SeedSignal[] = [];
    expect(processes.onSignal(catcher.pid, 'SIGTERM', (signal) => { seen.push(signal); return 'handled'; })).toBe(true);
    expect(processes.onSignal(catcher.pid, 'SIGKILL', () => 'handled')).toBe(false);

    processes.kill(catcher.pid, 'SIGTERM');
    expect(seen).toEqual(['SIGTERM']);
    expect(processes.get(catcher.pid)?.state).toBe('running');

    processes.kill(catcher.pid, 'SIGKILL');
    expect(processes.get(catcher.pid)?.state).toBe('zombie');
    expect(processes.get(catcher.pid)?.exitCode).toBe(128 + SIGNAL_NUMBERS.SIGKILL);
  });

  it('honours an ignore disposition and a handler that defers to the default action', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const ignorer = processes.spawn({ executable: 'ignorer', ppid: parent.pid });
    processes.setDisposition(ignorer.pid, 'SIGTERM', 'ignore');
    processes.kill(ignorer.pid, 'SIGTERM');
    expect(processes.get(ignorer.pid)?.state).toBe('running');

    const deferrer = processes.spawn({ executable: 'deferrer', ppid: parent.pid });
    processes.onSignal(deferrer.pid, 'SIGTERM', () => 'default');
    processes.kill(deferrer.pid, 'SIGTERM');
    expect(processes.get(deferrer.pid)?.state).toBe('zombie');
  });

  it('stops and continues a process, queueing signals while it is stopped', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const worker = processes.spawn({ executable: 'worker', ppid: parent.pid });

    processes.sendSignal(worker.pid, 'SIGSTOP');
    expect(processes.get(worker.pid)?.state).toBe('stopped');
    processes.tick(1_000);
    expect(processes.get(worker.pid)?.cpuTimeMs).toBe(0);

    processes.sendSignal(worker.pid, 'SIGUSR1');
    expect(processes.get(worker.pid)?.pendingSignals).toEqual(['SIGUSR1']);
    expect(processes.get(worker.pid)?.state).toBe('stopped');

    // SIGCONT resumes the process and replays what was queued; SIGUSR1 then terminates it.
    processes.sendSignal(worker.pid, 'SIGCONT');
    expect(processes.get(worker.pid)?.state).toBe('zombie');
    expect(processes.get(worker.pid)?.exitSignal).toBe('SIGUSR1');
  });

  it('kills a process group and a process tree', () => {
    const { processes } = booted();
    const leader = processes.spawn({ executable: 'leader' });
    processes.setpgid(leader.pid, leader.pid);
    const members = [1, 2].map(() => processes.spawn({ executable: 'member', ppid: leader.pid, pgid: leader.pid }));
    const outsider = processes.spawn({ executable: 'outsider' });

    expect(processes.killGroup(leader.pid, 'SIGKILL')).toBe(3);
    expect(processes.get(outsider.pid)?.state).toBe('running');
    for (const member of members) expect(processes.get(member.pid)).toBeUndefined();

    const root = processes.spawn({ executable: 'root' });
    const child = processes.spawn({ executable: 'child', ppid: root.pid });
    const grandchild = processes.spawn({ executable: 'grandchild', ppid: child.pid });
    expect(processes.killTree(root.pid, 'SIGKILL')).toEqual([grandchild.pid, child.pid, root.pid]);
    expect(processes.list().some((record) => record.executable.includes('child'))).toBe(false);
  });

  it('hangs up the rest of the session when a session leader dies', () => {
    const { processes } = booted();
    const shell = processes.spawn({ executable: 'bash' });
    processes.setsid(shell.pid);
    const job = processes.spawn({ executable: 'long-job', ppid: shell.pid, sid: shell.pid });
    const unrelated = processes.spawn({ executable: 'daemon' });

    processes.kill(shell.pid, 'SIGKILL');
    expect(processes.get(job.pid)).toBeUndefined();
    expect(processes.get(unrelated.pid)?.state).toBe('running');
  });
});

describe('exit statuses, zombies and reparenting', () => {
  it('keeps an exited child as a zombie until the parent waits for it', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const child = processes.spawn({ executable: 'worker', ppid: parent.pid });

    expect(processes.exit(child.pid, 3)).toMatchObject({ pid: child.pid, exitCode: 3, signalled: false });
    expect(processes.get(child.pid)?.state).toBe('zombie');
    expect(processes.stats().zombies).toBe(1);

    const status = processes.wait(parent.pid);
    expect(status).toMatchObject({ pid: child.pid, exitCode: 3, signal: null });
    expect(processes.get(child.pid)).toBeUndefined();
    expect(processes.wait(parent.pid)).toBeUndefined();
  });

  it('delivers SIGCHLD to the parent when a child exits', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const child = processes.spawn({ executable: 'worker', ppid: parent.pid });
    let notified = 0;
    processes.onSignal(parent.pid, 'SIGCHLD', () => { notified += 1; return 'handled'; });
    processes.kill(child.pid);
    expect(notified).toBe(1);
    expect(processes.get(parent.pid)?.state).toBe('running');
  });

  it('reparents orphans to pid 1 instead of leaking them out of the tree', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const child = processes.spawn({ executable: 'worker', ppid: parent.pid });
    const grandchild = processes.spawn({ executable: 'grandchild', ppid: child.pid });

    processes.kill(parent.pid, 'SIGKILL');
    expect(processes.get(child.pid)?.ppid).toBe(1);
    expect(processes.get(grandchild.pid)?.ppid).toBe(child.pid);
    // Every surviving process is reachable from init.
    for (const record of processes.list()) {
      let cursor = record;
      const guard = new Set<number>();
      while (cursor.pid !== 1) {
        expect(guard.has(cursor.pid)).toBe(false);
        guard.add(cursor.pid);
        cursor = processes.get(cursor.ppid)!;
        expect(cursor).toBeDefined();
      }
    }
  });

  it('lets init reap adopted zombies automatically so the table cannot fill with them', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const child = processes.spawn({ executable: 'worker', ppid: parent.pid });
    processes.exit(child.pid, 0);
    expect(processes.stats().zombies).toBe(1);

    processes.kill(parent.pid, 'SIGKILL');
    expect(processes.stats().zombies).toBe(0);
    expect(processes.list().map((record) => record.pid)).toEqual([1]);
    // init keeps the statuses of what it reaped, oldest exit first.
    expect(processes.waitAll(1).map((status) => status.pid)).toEqual([child.pid, parent.pid]);
  });

  it('notifies exit listeners so owners of external resources can release them', () => {
    const { processes } = booted();
    const observed: ProcessExitStatus[] = [];
    const unsubscribe = processes.onExit((status) => observed.push(status));
    const server = processes.spawn({ executable: 'seed-httpd', listeningPorts: [9100] });
    processes.kill(server.pid);
    expect(observed).toHaveLength(1);
    expect(observed[0]).toMatchObject({ pid: server.pid, executable: 'seed-httpd', signalled: true });
    unsubscribe();
    processes.kill(processes.spawn({ executable: 'other' }).pid);
    expect(observed).toHaveLength(1);
  });

  it('releases listening ports and descriptors on exit', () => {
    const { processes } = booted();
    const parent = processes.spawn({ executable: 'supervisor' });
    const server = processes.spawn({ executable: 'seed-httpd', ppid: parent.pid, listeningPorts: [9100] });
    processes.openFd(server.pid, '/var/log/httpd.log', 'w');
    processes.kill(server.pid);
    expect(processes.get(server.pid)?.listeningPorts).toEqual([]);
    expect(processes.listFds(server.pid)).toEqual([]);
  });
});

describe('file descriptors', () => {
  it('starts with the three standard descriptors and allocates the lowest free number', () => {
    const { processes } = booted();
    const worker = processes.spawn({ executable: 'worker' });
    expect(processes.listFds(worker.pid).map((descriptor) => descriptor.fd)).toEqual([0, 1, 2]);

    const first = processes.openFd(worker.pid, '/home/agent/a.txt', 'rw');
    const second = processes.openFd(worker.pid, '/home/agent/b.txt', 'r');
    expect([first, second]).toEqual([3, 4]);

    expect(processes.closeFd(worker.pid, 3)).toBe(true);
    expect(processes.closeFd(worker.pid, 3)).toBe(false);
    expect(processes.openFd(worker.pid, '/home/agent/c.txt', 'r')).toBe(3);

    const duplicated = processes.dupFd(worker.pid, 4);
    expect(duplicated).toBe(5);
    expect(processes.listFds(worker.pid).find((descriptor) => descriptor.fd === 5)?.path).toBe('/home/agent/b.txt');
  });
});

describe('scheduling', () => {
  it('shares the tick budget across cores and skips processes that are not runnable', () => {
    const { processes } = booted({ cpuCores: 2 });
    const busy = processes.spawn({ executable: 'busy' });
    const asleep = processes.spawn({ executable: 'asleep', runState: 'sleeping' });

    processes.tick(100);
    const total = processes.list().reduce((sum, record) => sum + record.cpuTimeMs, 0);
    expect(total).toBeCloseTo(200, -1);
    expect(processes.get(asleep.pid)?.cpuTimeMs).toBe(0);
    expect(processes.get(busy.pid)?.cpuTimeMs).toBeGreaterThan(0);
    expect(processes.get(asleep.pid)?.state).toBe('sleeping');
  });

  it('gives a higher-priority process a larger share of the cpu', () => {
    const { processes } = booted();
    const nice = processes.spawn({ executable: 'nice', priority: 10 });
    const greedy = processes.spawn({ executable: 'greedy', priority: -10 });
    for (let index = 0; index < 10; index += 1) processes.tick(100);
    expect(processes.get(greedy.pid)!.cpuTimeMs).toBeGreaterThan(processes.get(nice.pid)!.cpuTimeMs * 3);
  });

  it('marks exactly one process per core as on-cpu and rotates between them', () => {
    const { processes } = booted({ cpuCores: 1 });
    const first = processes.spawn({ executable: 'first' });
    const second = processes.spawn({ executable: 'second' });
    const onCpu = new Set<number>();
    for (let index = 0; index < 4; index += 1) {
      processes.tick(10);
      const running = processes.list().filter((record) => record.runState === 'running');
      expect(running).toHaveLength(1);
      onCpu.add(running[0]!.pid);
    }
    expect(onCpu.has(first.pid) && onCpu.has(second.pid)).toBe(true);
  });
});

describe('compatibility with the original surface', () => {
  it('keeps spawn, kill, list, get and tick behaving as callers expect', () => {
    const { processes } = booted();
    const record = processes.spawn({ executable: 'seed-httpd', argv: ['9100'], cwd: '/srv', env: { PORT: '9100' }, ppid: 1, listeningPorts: [9100], memoryBytes: 6 * 1024 * 1024 });
    expect(record).toMatchObject({ ppid: 1, computerId: 'test-box', state: 'running', cwd: '/srv', memoryBytes: 6 * 1024 * 1024 });
    expect(processes.get(record.pid)?.env).toEqual({ PORT: '9100' });
    expect(processes.list().map((item) => item.pid)).toEqual([1, record.pid]);

    processes.tick(500);
    expect(processes.get(record.pid)!.cpuTimeMs).toBeGreaterThan(0);

    // Children of init are reaped immediately, so `ps` never shows a leftover entry.
    expect(processes.kill(record.pid)).toBe(true);
    expect(processes.get(record.pid)).toBeUndefined();
    expect(processes.list().some((item) => item.pid === record.pid)).toBe(false);
    expect(processes.kill(record.pid)).toBe(false);
  });

  it('returns defensive copies so callers cannot mutate the table', () => {
    const { processes } = booted();
    const record = processes.spawn({ executable: 'worker' });
    const copy = processes.get(record.pid)!;
    copy.state = 'zombie';
    copy.argv.push('injected');
    copy.env.INJECTED = '1';
    expect(processes.get(record.pid)?.state).toBe('running');
    expect(processes.get(record.pid)?.argv).toEqual([]);
    expect(processes.get(record.pid)?.env).toEqual({});
  });
});
