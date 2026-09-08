import type { ProcessRecord, ProcessState } from '@tcn-computer/protocol';

/**
 * Process model for a single simulated computer.
 *
 * The externally consumed surface (`boot`, `spawn`, `kill`, `list`, `get`, `tick`) keeps its
 * original signatures; everything else is additive: lifecycle states, signal delivery with default
 * dispositions and catchable handlers, exit statuses, `wait()` with zombie reaping, orphan
 * reparenting to PID 1, process groups and sessions, per-process file-descriptor tables, and a
 * weighted round-robin scheduler that replaces the fabricated `deltaMs * 0.002` CPU accounting.
 */

export type SeedSignal =
  | 'SIGHUP' | 'SIGINT' | 'SIGQUIT' | 'SIGKILL' | 'SIGUSR1' | 'SIGUSR2'
  | 'SIGTERM' | 'SIGCHLD' | 'SIGCONT' | 'SIGSTOP';

export const SIGNAL_NUMBERS: Record<SeedSignal, number> = {
  SIGHUP: 1, SIGINT: 2, SIGQUIT: 3, SIGKILL: 9, SIGUSR1: 10,
  SIGUSR2: 12, SIGTERM: 15, SIGCHLD: 17, SIGCONT: 18, SIGSTOP: 19,
};

export type SignalAction = 'terminate' | 'ignore' | 'stop' | 'continue';

/** Default disposition of each signal when the process installs no handler. */
export const DEFAULT_DISPOSITIONS: Record<SeedSignal, SignalAction> = {
  SIGHUP: 'terminate', SIGINT: 'terminate', SIGQUIT: 'terminate', SIGKILL: 'terminate',
  SIGUSR1: 'terminate', SIGUSR2: 'terminate', SIGTERM: 'terminate', SIGCHLD: 'ignore',
  SIGCONT: 'continue', SIGSTOP: 'stop',
};

/** SIGKILL and SIGSTOP can be neither caught nor ignored, exactly as on a real kernel. */
export const UNCATCHABLE_SIGNALS: SeedSignal[] = ['SIGKILL', 'SIGSTOP'];

export type SeedRunState = 'ready' | 'running' | 'sleeping' | 'stopped' | 'zombie';

export type SignalHandler = (signal: SeedSignal, record: SeedProcessRecord) => 'handled' | 'default' | void;

export interface FileDescriptorRecord {
  fd: number;
  path: string;
  flags: 'r' | 'w' | 'rw';
  kind: 'file' | 'tty' | 'pipe' | 'socket';
  position: number;
  openedAt: string;
}

export interface SeedProcessRecord extends ProcessRecord {
  /** Process group id; a group leader has `pgid === pid`. */
  pgid: number;
  /** Session id; a session leader has `sid === pid`. */
  sid: number;
  /** Nice value, -20 (most CPU share) through 19 (least). */
  priority: number;
  /** Finer-grained scheduler state; `state` stays within the protocol's four values. */
  runState: SeedRunState;
  exitCode: number | null;
  exitSignal: SeedSignal | null;
  exitedAt: string | null;
  /** Signals delivered while the process was stopped, replayed on SIGCONT. */
  pendingSignals: SeedSignal[];
  /** Signals for which the process installed a handler or an explicit disposition. */
  caughtSignals: SeedSignal[];
  /** Signals actually delivered, in order, for trajectory evidence. */
  signalLog: SeedSignal[];
  fds: FileDescriptorRecord[];
  threads: number;
}

export interface ProcessExitStatus {
  pid: number;
  ppid: number;
  executable: string;
  exitCode: number;
  signal: SeedSignal | null;
  cpuTimeMs: number;
  exitedAt: string;
  /** True when the process ended because a signal's default action terminated it. */
  signalled: boolean;
}

export interface SpawnInput {
  executable: string;
  argv?: string[];
  cwd?: string;
  env?: Record<string, string>;
  ppid?: number;
  memoryBytes?: number;
  listeningPorts?: number[];
  /** Join an existing process group; defaults to a new group led by this process. */
  pgid?: number;
  /** Join an existing session; defaults to the parent's session. */
  sid?: number;
  priority?: number;
  /** Start the process stopped or sleeping instead of runnable. */
  runState?: Exclude<SeedRunState, 'zombie'>;
  threads?: number;
  /** Reap children automatically instead of leaving zombies (PID 1 does this by default). */
  autoReap?: boolean;
}

export interface ProcessManagerOptions {
  /** Used by the scheduler to size each tick's CPU budget. */
  cpuCores?: number;
  /** Highest PID before allocation wraps back to `pidMin`. */
  pidMax?: number;
  /** Lowest PID handed to a non-init process. */
  pidMin?: number;
  clock?: () => number;
}

export type ProcessExitListener = (status: ProcessExitStatus, record: SeedProcessRecord) => void;

const INIT_PID = 1;

function nowIso(): string { return new Date().toISOString(); }

function protocolState(runState: SeedRunState): ProcessState {
  if (runState === 'ready' || runState === 'running') return 'running';
  if (runState === 'sleeping') return 'sleeping';
  if (runState === 'stopped') return 'stopped';
  return 'zombie';
}

export class ProcessManager {
  private readonly records = new Map<number, SeedProcessRecord>();
  private readonly handlers = new Map<number, Map<SeedSignal, SignalHandler | 'ignore' | 'default'>>();
  private readonly autoReapers = new Set<number>();
  private readonly reaped = new Map<number, ProcessExitStatus[]>();
  private readonly exitListeners = new Set<ProcessExitListener>();
  private readonly pidMin: number;
  private readonly pidMax: number;
  private readonly cpuCores: number;
  private nextPid: number;
  /** PID 1 is issued exactly once for the lifetime of the table, even if the table drains. */
  private initIssued = false;
  private scheduleCursor = 0;

  constructor(private readonly computerId: string, options: ProcessManagerOptions = {}) {
    this.pidMin = Math.max(2, options.pidMin ?? 100);
    this.pidMax = Math.max(this.pidMin + 1, options.pidMax ?? 32_768);
    this.cpuCores = Math.max(1, options.cpuCores ?? 1);
    this.nextPid = this.pidMin;
  }

  boot(initExecutable: string, cwd: string, env: Record<string, string>): SeedProcessRecord {
    if (this.initIssued) throw new Error(`${this.computerId} has already booted`);
    const init = this.spawn({ executable: initExecutable, argv: [], cwd, env, ppid: 0, memoryBytes: 16 * 1024 * 1024, autoReap: true });
    this.nextPid = Math.max(this.nextPid, init.pid + 99);
    return init;
  }

  spawn(input: SpawnInput): SeedProcessRecord {
    const pid = this.allocatePid();
    const parent = input.ppid !== undefined ? this.records.get(input.ppid) : this.records.get(INIT_PID);
    const ppid = pid === INIT_PID ? (input.ppid ?? 0) : (parent ? parent.pid : INIT_PID);
    const runState = input.runState ?? 'ready';
    const record: SeedProcessRecord = {
      pid,
      ppid,
      computerId: this.computerId,
      executable: input.executable,
      argv: input.argv ?? [],
      cwd: input.cwd ?? '/',
      env: input.env ?? {},
      state: protocolState(runState),
      startedAt: nowIso(),
      cpuTimeMs: 0,
      memoryBytes: input.memoryBytes ?? 4 * 1024 * 1024,
      listeningPorts: input.listeningPorts ?? [],
      pgid: input.pgid ?? parent?.pgid ?? pid,
      sid: input.sid ?? parent?.sid ?? pid,
      priority: Math.min(19, Math.max(-20, input.priority ?? 0)),
      runState,
      exitCode: null,
      exitSignal: null,
      exitedAt: null,
      pendingSignals: [],
      caughtSignals: [],
      signalLog: [],
      fds: standardDescriptors(),
      threads: Math.max(1, input.threads ?? 1),
    };
    this.records.set(pid, record);
    if (input.autoReap ?? pid === INIT_PID) this.autoReapers.add(pid);
    return this.clone(record);
  }

  /**
   * Signal a process. Retains the original boolean contract: `false` for an unknown process, for
   * PID 1, and for a process that has already exited.
   */
  kill(pid: number, signal: SeedSignal = 'SIGTERM'): boolean {
    if (pid === INIT_PID) return false;
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return false;
    return this.deliver(record, signal).delivered;
  }

  /** Signal delivery with the full outcome, for callers that need more than a boolean. */
  sendSignal(pid: number, signal: SeedSignal): { delivered: boolean; action: SignalAction | 'handled' | 'queued' } {
    if (pid === INIT_PID && signal !== 'SIGCHLD') return { delivered: false, action: 'ignore' };
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return { delivered: false, action: 'ignore' };
    return this.deliver(record, signal);
  }

  /** Signal every member of a process group. Returns the number of processes signalled. */
  killGroup(pgid: number, signal: SeedSignal = 'SIGTERM'): number {
    let count = 0;
    for (const record of [...this.records.values()]) {
      if (record.pgid !== pgid || record.runState === 'zombie' || record.pid === INIT_PID) continue;
      if (this.deliver(record, signal).delivered) count += 1;
    }
    return count;
  }

  /** Signal a process and every descendant, deepest first. Returns the pids that were signalled. */
  killTree(pid: number, signal: SeedSignal = 'SIGTERM'): number[] {
    const order: number[] = [];
    const walk = (current: number): void => {
      for (const child of this.records.values()) if (child.ppid === current) walk(child.pid);
      order.push(current);
    };
    walk(pid);
    const signalled: number[] = [];
    for (const target of order) {
      if (target === INIT_PID) continue;
      const record = this.records.get(target);
      if (!record || record.runState === 'zombie') continue;
      if (this.deliver(record, signal).delivered) signalled.push(target);
    }
    return signalled;
  }

  /** Voluntary exit. Returns the status the parent will observe. */
  exit(pid: number, exitCode = 0): ProcessExitStatus | undefined {
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return undefined;
    return this.terminate(record, exitCode, null);
  }

  /**
   * Reap one exited child. Without `options.pid` any zombie child is reaped, oldest exit first.
   * Returns `undefined` when the process has no reapable child.
   */
  wait(ppid: number, options: { pid?: number } = {}): ProcessExitStatus | undefined {
    const buffered = this.reaped.get(ppid);
    if (buffered?.length) {
      const index = options.pid === undefined ? 0 : buffered.findIndex((status) => status.pid === options.pid);
      if (index >= 0) {
        const [status] = buffered.splice(index, 1);
        if (!buffered.length) this.reaped.delete(ppid);
        return status;
      }
    }
    const zombies = [...this.records.values()]
      .filter((record) => record.ppid === ppid && record.runState === 'zombie')
      .filter((record) => options.pid === undefined || record.pid === options.pid)
      .sort((left, right) => String(left.exitedAt).localeCompare(String(right.exitedAt)));
    const child = zombies[0];
    if (!child) return undefined;
    this.records.delete(child.pid);
    this.handlers.delete(child.pid);
    this.autoReapers.delete(child.pid);
    return statusOf(child);
  }

  /** Reap every exited child of `ppid`. */
  waitAll(ppid: number): ProcessExitStatus[] {
    const statuses: ProcessExitStatus[] = [];
    for (;;) {
      const status = this.wait(ppid);
      if (!status) return statuses;
      statuses.push(status);
    }
  }

  /** Install a handler for a catchable signal. Returning `'default'` lets the default action run. */
  onSignal(pid: number, signal: SeedSignal, handler: SignalHandler): boolean {
    if (UNCATCHABLE_SIGNALS.includes(signal)) return false;
    const record = this.records.get(pid);
    if (!record) return false;
    this.dispositionsFor(pid).set(signal, handler);
    if (!record.caughtSignals.includes(signal)) record.caughtSignals.push(signal);
    return true;
  }

  /** Ignore a signal, or restore its default disposition. */
  setDisposition(pid: number, signal: SeedSignal, disposition: 'ignore' | 'default'): boolean {
    if (UNCATCHABLE_SIGNALS.includes(signal)) return false;
    const record = this.records.get(pid);
    if (!record) return false;
    const table = this.dispositionsFor(pid);
    if (disposition === 'default') {
      table.delete(signal);
      record.caughtSignals = record.caughtSignals.filter((value) => value !== signal);
    } else {
      table.set(signal, 'ignore');
      if (!record.caughtSignals.includes(signal)) record.caughtSignals.push(signal);
    }
    return true;
  }

  /** Whether a process reaps its children automatically instead of accumulating zombies. */
  setAutoReap(pid: number, autoReap: boolean): void {
    if (autoReap) this.autoReapers.add(pid); else this.autoReapers.delete(pid);
  }

  setpgid(pid: number, pgid: number): boolean {
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return false;
    if (pgid !== pid && ![...this.records.values()].some((candidate) => candidate.pgid === pgid)) return false;
    record.pgid = pgid;
    return true;
  }

  /** Start a new session; the caller becomes both session and group leader. */
  setsid(pid: number): number | undefined {
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return undefined;
    record.sid = pid;
    record.pgid = pid;
    return pid;
  }

  /** Allocate the lowest free descriptor at or above 3. */
  openFd(pid: number, filePath: string, flags: FileDescriptorRecord['flags'] = 'r', kind: FileDescriptorRecord['kind'] = 'file'): number | undefined {
    const record = this.records.get(pid);
    if (!record || record.runState === 'zombie') return undefined;
    const used = new Set(record.fds.map((descriptor) => descriptor.fd));
    let fd = 3;
    while (used.has(fd)) fd += 1;
    record.fds.push({ fd, path: filePath, flags, kind, position: 0, openedAt: nowIso() });
    return fd;
  }

  closeFd(pid: number, fd: number): boolean {
    const record = this.records.get(pid);
    if (!record) return false;
    const index = record.fds.findIndex((descriptor) => descriptor.fd === fd);
    if (index < 0) return false;
    record.fds.splice(index, 1);
    return true;
  }

  /** Duplicate a descriptor onto the lowest free number, as `dup(2)` does. */
  dupFd(pid: number, fd: number): number | undefined {
    const record = this.records.get(pid);
    const source = record?.fds.find((descriptor) => descriptor.fd === fd);
    if (!record || !source) return undefined;
    const duplicated = this.openFd(pid, source.path, source.flags, source.kind);
    return duplicated;
  }

  listFds(pid: number): FileDescriptorRecord[] {
    return (this.records.get(pid)?.fds ?? []).map((descriptor) => ({ ...descriptor }));
  }

  children(pid: number): SeedProcessRecord[] {
    return [...this.records.values()].filter((record) => record.ppid === pid).map((record) => this.clone(record));
  }

  /** Notified whenever a process exits, so owners of external resources can release them. */
  onExit(listener: ProcessExitListener): () => void {
    this.exitListeners.add(listener);
    return () => this.exitListeners.delete(listener);
  }

  list(): SeedProcessRecord[] {
    return [...this.records.values()].map((record) => this.clone(record)).sort((left, right) => left.pid - right.pid);
  }

  get(pid: number): SeedProcessRecord | undefined {
    const record = this.records.get(pid);
    return record ? this.clone(record) : undefined;
  }

  /**
   * Advance the scheduler by `deltaMs`. The CPU budget for the interval is `deltaMs * cores`, shared
   * between runnable processes by nice-weighted round robin; stopped, sleeping and zombie processes
   * accrue nothing.
   */
  tick(deltaMs: number): void {
    if (deltaMs <= 0) return;
    const runnable = [...this.records.values()].filter((record) => record.runState === 'ready' || record.runState === 'running');
    if (!runnable.length) return;
    const ordered = [...runnable].sort((left, right) => left.pid - right.pid);
    const weights = ordered.map((record) => 1.25 ** -record.priority);
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    const budget = deltaMs * this.cpuCores;
    ordered.forEach((record, index) => {
      record.cpuTimeMs += Math.max(1, Math.round((budget * weights[index]!) / totalWeight));
      record.runState = 'ready';
      record.state = 'running';
    });
    this.scheduleCursor = (this.scheduleCursor + 1) % ordered.length;
    const onCpu = ordered.slice(this.scheduleCursor).concat(ordered.slice(0, this.scheduleCursor)).slice(0, this.cpuCores);
    for (const record of onCpu) record.runState = 'running';
  }

  stats(): { total: number; running: number; sleeping: number; stopped: number; zombies: number; cpuTimeMs: number } {
    const records = [...this.records.values()];
    return {
      total: records.length,
      running: records.filter((record) => record.runState === 'running' || record.runState === 'ready').length,
      sleeping: records.filter((record) => record.runState === 'sleeping').length,
      stopped: records.filter((record) => record.runState === 'stopped').length,
      zombies: records.filter((record) => record.runState === 'zombie').length,
      cpuTimeMs: records.reduce((sum, record) => sum + record.cpuTimeMs, 0),
    };
  }

  private deliver(record: SeedProcessRecord, signal: SeedSignal): { delivered: boolean; action: SignalAction | 'handled' | 'queued' } {
    if (record.runState === 'zombie') return { delivered: false, action: 'ignore' };
    record.signalLog.push(signal);
    if (record.signalLog.length > 50) record.signalLog.shift();

    // A stopped process only observes SIGKILL and SIGCONT; everything else waits for it to resume.
    if (record.runState === 'stopped' && signal !== 'SIGKILL' && signal !== 'SIGCONT') {
      record.pendingSignals.push(signal);
      return { delivered: true, action: 'queued' };
    }

    if (!UNCATCHABLE_SIGNALS.includes(signal)) {
      const disposition = this.handlers.get(record.pid)?.get(signal);
      if (disposition === 'ignore') return { delivered: true, action: 'ignore' };
      if (typeof disposition === 'function') {
        const outcome = disposition(signal, this.clone(record));
        if (outcome !== 'default') return { delivered: true, action: 'handled' };
      }
    }

    const action = DEFAULT_DISPOSITIONS[signal];
    if (action === 'ignore') return { delivered: true, action };
    if (action === 'stop') {
      record.runState = 'stopped';
      record.state = 'stopped';
      return { delivered: true, action };
    }
    if (action === 'continue') {
      if (record.runState === 'stopped') {
        record.runState = 'ready';
        record.state = 'running';
        const queued = record.pendingSignals.splice(0, record.pendingSignals.length);
        for (const pending of queued) {
          if (record.exitedAt !== null) break;
          this.deliver(record, pending);
        }
      }
      return { delivered: true, action };
    }
    this.terminate(record, 128 + SIGNAL_NUMBERS[signal], signal);
    return { delivered: true, action: 'terminate' };
  }

  private terminate(record: SeedProcessRecord, exitCode: number, signal: SeedSignal | null): ProcessExitStatus {
    record.runState = 'zombie';
    record.state = 'zombie';
    record.exitCode = exitCode;
    record.exitSignal = signal;
    record.exitedAt = nowIso();
    record.listeningPorts = [];
    record.fds = [];
    record.pendingSignals = [];
    const status = statusOf(record);

    // A session leader taking a fatal signal hangs up the rest of its session, as a terminal does.
    if (record.sid === record.pid) {
      for (const member of [...this.records.values()]) {
        if (member.pid === record.pid || member.sid !== record.sid || member.runState === 'zombie' || member.pid === INIT_PID) continue;
        this.deliver(member, 'SIGHUP');
      }
    }

    // Orphans are reparented to init rather than leaking out of the tree.
    for (const child of [...this.records.values()]) {
      if (child.ppid !== record.pid) continue;
      child.ppid = INIT_PID;
      if (child.runState === 'zombie' && this.autoReapers.has(INIT_PID)) this.reapInto(INIT_PID, child);
    }

    for (const listener of this.exitListeners) listener({ ...status }, this.clone(record));

    const parent = this.records.get(record.ppid);
    if (parent) this.deliver(parent, 'SIGCHLD');
    if (!parent || this.autoReapers.has(record.ppid)) this.reapInto(record.ppid, record);
    return status;
  }

  /** Remove a zombie from the table on behalf of an auto-reaping parent such as init. */
  private reapInto(ppid: number, record: SeedProcessRecord): void {
    this.records.delete(record.pid);
    this.handlers.delete(record.pid);
    this.autoReapers.delete(record.pid);
    if (!this.records.has(ppid)) return;
    const buffered = this.reaped.get(ppid) ?? [];
    buffered.push(statusOf(record));
    if (buffered.length > 100) buffered.shift();
    this.reaped.set(ppid, buffered);
  }

  private dispositionsFor(pid: number): Map<SeedSignal, SignalHandler | 'ignore' | 'default'> {
    const existing = this.handlers.get(pid);
    if (existing) return existing;
    const created = new Map<SeedSignal, SignalHandler | 'ignore' | 'default'>();
    this.handlers.set(pid, created);
    return created;
  }

  /**
   * PID 1 is issued exactly once. Afterwards PIDs advance monotonically and wrap at `pidMax`,
   * skipping live entries, so draining the table can never re-issue init's PID.
   */
  private allocatePid(): number {
    if (!this.initIssued) { this.initIssued = true; return INIT_PID; }
    const span = this.pidMax - this.pidMin + 1;
    for (let attempt = 0; attempt < span; attempt += 1) {
      const candidate = this.nextPid;
      this.nextPid = this.nextPid >= this.pidMax ? this.pidMin : this.nextPid + 1;
      if (candidate !== INIT_PID && !this.records.has(candidate)) return candidate;
    }
    throw new Error(`${this.computerId}: pid space exhausted`);
  }

  private clone(record: SeedProcessRecord): SeedProcessRecord {
    return {
      ...record,
      argv: [...record.argv],
      env: { ...record.env },
      listeningPorts: [...record.listeningPorts],
      pendingSignals: [...record.pendingSignals],
      caughtSignals: [...record.caughtSignals],
      signalLog: [...record.signalLog],
      fds: record.fds.map((descriptor) => ({ ...descriptor })),
    };
  }
}

function standardDescriptors(): FileDescriptorRecord[] {
  const openedAt = nowIso();
  return [
    { fd: 0, path: '/dev/stdin', flags: 'r', kind: 'tty', position: 0, openedAt },
    { fd: 1, path: '/dev/stdout', flags: 'w', kind: 'tty', position: 0, openedAt },
    { fd: 2, path: '/dev/stderr', flags: 'w', kind: 'tty', position: 0, openedAt },
  ];
}

function statusOf(record: SeedProcessRecord): ProcessExitStatus {
  return {
    pid: record.pid,
    ppid: record.ppid,
    executable: record.executable,
    exitCode: record.exitCode ?? 0,
    signal: record.exitSignal,
    cpuTimeMs: record.cpuTimeMs,
    exitedAt: record.exitedAt ?? nowIso(),
    signalled: record.exitSignal !== null,
  };
}
