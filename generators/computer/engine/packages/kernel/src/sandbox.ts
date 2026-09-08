import { Worker } from 'node:worker_threads';

/**
 * Execution sandbox for untrusted application bundles.
 *
 * Two properties matter here, and neither is achievable with `vm.runInContext` alone:
 *
 * 1. **Realm hygiene.** A `vm` context has its own intrinsics, but *any* host object handed into it
 *    re-exports the host realm: `hostObject.constructor` is the host `Function`, and
 *    `codeGeneration.strings: false` only restrains the sandbox realm's own compiler, not the host's.
 *    So nothing from the host realm crosses the boundary — not `Object`/`JSON`, not the `module`
 *    record, not the SDK object, not even a rejected promise (a host `Error` leaks
 *    `error.constructor.constructor` just as well). The module record, the `seed` SDK and the
 *    console are all constructed *inside* the context by a bootstrap script. The only host value the
 *    context ever holds is a single bridge function whose prototype is re-pointed at the sandbox
 *    realm's `Function.prototype`, and it speaks exclusively in JSON strings and sandbox callbacks.
 *
 * 2. **A budget that survives synchronous code.** `runInContext({ timeout })` only bounds top-level
 *    evaluation, and a `Promise.race` cannot interrupt `while (Date.now() < end);`. The bundle
 *    therefore runs in a `worker_threads` Worker, which can be terminated mid-loop, and the budget
 *    covers the whole execution including the exported function.
 */

export interface SandboxAppIdentity {
  id: string;
  version: string;
  capabilities: string[];
}

export interface SandboxOptions {
  /** Bundle source, evaluated as a CommonJS-style script that assigns `module.exports`. */
  source: string;
  /** Filename used for stack traces and error messages. */
  filename: string;
  /** Request object handed to the exported function; must be JSON-serializable. */
  request: unknown;
  /** Identity surfaced to the bundle as `seed.app`. */
  app?: SandboxAppIdentity;
  /** Serviced on the host side; this is the sandbox's only channel to the simulation. */
  dispatch(operation: string, payload: Record<string, unknown>): Promise<unknown>;
  /**
   * Budget for time the bundle itself is in control. The clock pauses while the kernel services a
   * syscall so that a slow virtual network request cannot consume the bundle's allowance.
   */
  timeoutMs?: number;
  /** Absolute wall-clock ceiling for the whole execution, syscall service time included. */
  hardTimeoutMs?: number;
  /** Heap ceiling for the worker; exceeding it terminates the bundle. */
  memoryLimitMb?: number;
  /** Diagnostic name for the sandbox realm. */
  contextName?: string;
}

export interface SandboxOutcome {
  value: unknown;
  logs: string[];
  syscalls: number;
  /** Time the bundle itself was in control, excluding kernel syscall service time. */
  durationMs: number;
}

export const SANDBOX_DEFAULT_TIMEOUT_MS = 250;
export const SANDBOX_DEFAULT_MEMORY_MB = 128;

export class SandboxTimeoutError extends Error {
  constructor(readonly budgetMs: number, readonly kind: 'budget' | 'wall-clock') {
    super(`sandbox execution exceeded its ${kind} budget of ${budgetMs}ms`);
    this.name = 'SandboxTimeoutError';
  }
}

/**
 * Runs entirely inside the sandbox realm. It only ever touches intrinsics of that realm plus the
 * `bridge` argument, and hands the bridge nothing but strings and its own closures.
 */
const BOOTSTRAP_SOURCE = `(function (bridge, appJson) {
  'use strict';
  var module = { exports: undefined };
  globalThis.module = module;
  globalThis.exports = module.exports;
  var logs = [];
  var pending = new Map();
  var nextId = 1;
  function render(value) {
    if (typeof value === 'string') return value;
    try { return JSON.stringify(value); } catch (error) { return String(value); }
  }
  function log(level) {
    return function () {
      var parts = [];
      for (var index = 0; index < arguments.length; index += 1) parts.push(render(arguments[index]));
      if (logs.length < 200) logs.push(level + ': ' + parts.join(' '));
    };
  }
  globalThis.console = Object.freeze({ log: log('log'), info: log('info'), warn: log('warn'), error: log('error'), debug: log('debug') });
  globalThis.__seedSettle = function (id, json) {
    var entry = pending.get(id);
    if (!entry) return;
    pending.delete(id);
    var parsed = JSON.parse(json);
    if (parsed.ok) entry.resolve(parsed.value);
    else entry.reject(new Error(String(parsed.error)));
  };
  var seed = Object.freeze({
    apiVersion: 1,
    app: Object.freeze(JSON.parse(appJson)),
    dispatch: function (operation, payload) {
      var id = nextId++;
      return new Promise(function (resolve, reject) {
        pending.set(id, { resolve: resolve, reject: reject });
        var encoded;
        try { encoded = JSON.stringify({ operation: String(operation), payload: payload === undefined || payload === null ? {} : payload }); }
        catch (error) { pending.delete(id); reject(new Error('syscall arguments must be JSON-serializable')); return; }
        bridge(id, encoded);
      });
    },
  });
  globalThis.__seedInvoke = function (requestJson, done) {
    function finish(ok, value, error) {
      var payload;
      try { payload = JSON.stringify({ ok: ok, value: value === undefined ? null : value, error: error, logs: logs }); }
      catch (failure) { payload = JSON.stringify({ ok: false, error: 'application result is not serializable', logs: logs }); }
      done(payload);
    }
    var entry = module.exports;
    if (typeof entry !== 'function') { finish(false, null, 'ENTRY_NOT_A_FUNCTION'); return; }
    var outcome;
    try { outcome = entry(seed, JSON.parse(requestJson)); }
    catch (error) { finish(false, null, String((error && error.message) || error)); return; }
    Promise.resolve(outcome).then(
      function (value) { finish(true, value, undefined); },
      function (error) { finish(false, null, String((error && error.message) || error)); }
    );
  };
})`;

/** Evaluated inside the worker thread (not inside the sandbox realm). */
const WORKER_SOURCE = `(function main() {
  'use strict';
  var vm = require('node:vm');
  var threads = require('node:worker_threads');
  var port = threads.parentPort;
  var data = threads.workerData;

  function report(json) { port.postMessage({ type: 'done', json: json }); }

  var context = vm.createContext(Object.create(null), {
    name: data.contextName,
    codeGeneration: { strings: false, wasm: false },
  });
  var sandboxFunctionPrototype = vm.runInContext('Function.prototype', context);

  function bridge(id, json) { port.postMessage({ type: 'syscall', id: id, json: json }); }
  function done(json) { report(json); }
  // Re-point host callables at the sandbox realm so even a leaked reference cannot reach the host
  // Function constructor; the sandbox realm's compiler is disabled.
  Object.setPrototypeOf(bridge, sandboxFunctionPrototype);
  Object.setPrototypeOf(done, sandboxFunctionPrototype);

  try {
    var bootstrap = new vm.Script(data.bootstrap, { filename: 'seed:sandbox-bootstrap' });
    bootstrap.runInContext(context, { timeout: data.evaluateTimeoutMs })(bridge, data.appJson);
  } catch (error) {
    report(JSON.stringify({ ok: false, error: 'sandbox bootstrap failed: ' + String((error && error.message) || error), logs: [] }));
    return;
  }

  try {
    new vm.Script(data.source, { filename: data.filename }).runInContext(context, { timeout: data.evaluateTimeoutMs });
  } catch (error) {
    report(JSON.stringify({ ok: false, error: String((error && error.message) || error), logs: [] }));
    return;
  }

  port.on('message', function (message) {
    if (message && message.type === 'syscall-result') {
      try { context.__seedSettle(message.id, message.json); }
      catch (error) { report(JSON.stringify({ ok: false, error: String((error && error.message) || error), logs: [] })); }
    }
  });

  try { context.__seedInvoke(JSON.stringify(data.request), done); }
  catch (error) { report(JSON.stringify({ ok: false, error: String((error && error.message) || error), logs: [] })); }
})();`;

interface SandboxMessage {
  type: 'done' | 'syscall';
  id?: number;
  json: string;
}

interface SandboxReport {
  ok: boolean;
  value?: unknown;
  error?: string;
  logs?: string[];
}

export async function runSandboxedBundle(options: SandboxOptions): Promise<SandboxOutcome> {
  const budgetMs = Math.max(1, options.timeoutMs ?? SANDBOX_DEFAULT_TIMEOUT_MS);
  const hardBudgetMs = Math.max(budgetMs, options.hardTimeoutMs ?? Math.max(budgetMs * 10, 5_000));
  const worker = new Worker(WORKER_SOURCE, {
    eval: true,
    workerData: {
      source: options.source,
      bootstrap: BOOTSTRAP_SOURCE,
      filename: options.filename,
      request: options.request ?? {},
      appJson: JSON.stringify(options.app ?? { id: 'anonymous', version: '0.0.0', capabilities: [] }),
      contextName: options.contextName ?? `seed-sandbox:${options.filename}`,
      evaluateTimeoutMs: budgetMs,
    },
    // Nothing from the host environment is inherited: no env, no argv, no stdio passthrough.
    env: {},
    argv: [],
    execArgv: [],
    stdout: true,
    stderr: true,
    resourceLimits: {
      maxOldGenerationSizeMb: options.memoryLimitMb ?? SANDBOX_DEFAULT_MEMORY_MB,
      maxYoungGenerationSizeMb: 32,
      stackSizeMb: 4,
    },
  });

  let settled = false;
  let paused = false;
  let outstandingSyscalls = 0;
  let syscalls = 0;
  let consumedMs = 0;
  let runningSince = Date.now();
  let budgetTimer: NodeJS.Timeout | undefined;
  let hardTimer: NodeJS.Timeout | undefined;

  return await new Promise<SandboxOutcome>((resolve, reject) => {
    const finish = (error: Error | undefined, outcome?: SandboxOutcome): void => {
      if (settled) return;
      settled = true;
      if (budgetTimer) clearTimeout(budgetTimer);
      if (hardTimer) clearTimeout(hardTimer);
      void worker.terminate();
      if (error) reject(error); else resolve(outcome!);
    };

    const pauseBudget = (): void => {
      if (paused) return;
      paused = true;
      if (budgetTimer) { clearTimeout(budgetTimer); budgetTimer = undefined; }
      consumedMs += Date.now() - runningSince;
    };
    const resumeBudget = (): void => {
      if (settled || !paused && budgetTimer) return;
      paused = false;
      runningSince = Date.now();
      const remaining = budgetMs - consumedMs;
      if (remaining <= 0) { finish(new SandboxTimeoutError(budgetMs, 'budget')); return; }
      budgetTimer = setTimeout(() => finish(new SandboxTimeoutError(budgetMs, 'budget')), remaining);
      budgetTimer.unref?.();
    };

    hardTimer = setTimeout(() => finish(new SandboxTimeoutError(hardBudgetMs, 'wall-clock')), hardBudgetMs);
    hardTimer.unref?.();
    resumeBudget();

    worker.on('message', (raw: SandboxMessage) => {
      if (settled || !raw || typeof raw.json !== 'string') return;
      if (raw.type === 'done') {
        pauseBudget();
        let report: SandboxReport;
        try { report = JSON.parse(raw.json) as SandboxReport; }
        catch { finish(new Error('sandbox returned a malformed result')); return; }
        if (!report.ok) {
          const message = report.error === 'ENTRY_NOT_A_FUNCTION'
            ? `${options.filename} must export a function`
            : report.error ?? 'sandbox execution failed';
          finish(new Error(message));
          return;
        }
        finish(undefined, { value: report.value ?? null, logs: report.logs ?? [], syscalls, durationMs: consumedMs });
        return;
      }
      if (raw.type !== 'syscall' || typeof raw.id !== 'number') return;
      syscalls += 1;
      // The budget only runs while the bundle itself is in control: it stops for the first
      // outstanding syscall and restarts once the kernel has answered all of them.
      outstandingSyscalls += 1;
      pauseBudget();
      const id = raw.id;
      void (async () => {
        let response: string;
        try {
          const parsed = JSON.parse(raw.json) as { operation?: unknown; payload?: unknown };
          const operation = typeof parsed.operation === 'string' ? parsed.operation : '';
          const payload = (parsed.payload && typeof parsed.payload === 'object' && !Array.isArray(parsed.payload)
            ? parsed.payload
            : {}) as Record<string, unknown>;
          const value = await options.dispatch(operation, payload);
          response = JSON.stringify({ ok: true, value: value === undefined ? null : value });
        } catch (error) {
          response = JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) });
        }
        if (settled) return;
        outstandingSyscalls -= 1;
        if (outstandingSyscalls === 0) resumeBudget();
        worker.postMessage({ type: 'syscall-result', id, json: response });
      })();
    });

    worker.on('error', (error: Error) => finish(error));
    worker.on('exit', () => finish(new Error('sandbox worker exited before returning a result')));
  });
}
