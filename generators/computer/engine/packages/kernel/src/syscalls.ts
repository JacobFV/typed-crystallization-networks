/**
 * The syscall boundary: a validated request carrying a PID, a per-process security context checked
 * before dispatch, and a registry of handlers that never see the caller's own claims about identity.
 *
 * Adapted from the browser-os project (MIT licensed) — `packages/kernel/src/SyscallRouter.ts` and
 * `packages/kernel/src/types.ts`. Seed replaces the zod schema with a hand-rolled validator (the
 * kernel package carries no runtime dependencies), adds an audit trail so denials are observable in
 * a trajectory, and adds `invoke()` for in-kernel callers that prefer exceptions to result objects.
 */

import type { PermissionManager, SecurityContext } from './permissions.js';

export interface SyscallRequest {
  id: string;
  pid: number;
  syscall: string;
  args: Record<string, unknown>;
}

export interface SyscallResponse {
  id: string;
  success: boolean;
  data?: unknown;
  error?: string;
}

export type SyscallHandler = (args: Record<string, unknown>, context: SecurityContext) => Promise<unknown>;

export interface SyscallAuditEntry {
  at: string;
  pid: number;
  subject: string;
  syscall: string;
  outcome: 'completed' | 'denied' | 'failed' | 'unknown-syscall' | 'no-context';
  error?: string;
}

/** Validates an untrusted request object before anything else touches it. */
export function parseSyscallRequest(value: unknown): SyscallRequest {
  if (typeof value !== 'object' || value === null) throw new Error('syscall request must be an object');
  const candidate = value as Partial<SyscallRequest>;
  if (typeof candidate.syscall !== 'string' || !candidate.syscall.trim()) throw new Error('syscall name is required');
  if (!/^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+$/.test(candidate.syscall)) {
    throw new Error(`malformed syscall name: ${candidate.syscall}`);
  }
  if (typeof candidate.pid !== 'number' || !Number.isInteger(candidate.pid) || candidate.pid < 1) {
    throw new Error('syscall request requires an integer pid');
  }
  if (candidate.args !== undefined && (typeof candidate.args !== 'object' || candidate.args === null || Array.isArray(candidate.args))) {
    throw new Error('syscall args must be a plain object');
  }
  return {
    id: typeof candidate.id === 'string' && candidate.id ? candidate.id : `sc-${Math.random().toString(16).slice(2)}`,
    pid: candidate.pid,
    syscall: candidate.syscall,
    args: (candidate.args as Record<string, unknown> | undefined) ?? {},
  };
}

export class SyscallRouter {
  private readonly handlers = new Map<string, SyscallHandler>();
  private readonly auditTrail: SyscallAuditEntry[] = [];

  constructor(private readonly permissions: PermissionManager, private readonly auditLimit = 500) {}

  register(name: string, handler: SyscallHandler): this {
    this.handlers.set(name, handler);
    return this;
  }

  registerAll(handlers: Record<string, SyscallHandler>): this {
    for (const [name, handler] of Object.entries(handlers)) this.register(name, handler);
    return this;
  }

  has(syscall: string): boolean { return this.handlers.has(syscall); }
  names(): string[] { return [...this.handlers.keys()].sort(); }
  audit(): SyscallAuditEntry[] { return this.auditTrail.map((entry) => ({ ...entry })); }

  /** Result-object form. Never throws for a denied or failing call. */
  async handle(request: unknown): Promise<SyscallResponse> {
    let validated: SyscallRequest;
    try { validated = parseSyscallRequest(request); }
    catch (error) { return { id: 'invalid', success: false, error: error instanceof Error ? error.message : String(error) }; }

    const context = this.permissions.getSecurityContext(validated.pid);
    if (!context) {
      this.record({ pid: validated.pid, subject: 'unknown', syscall: validated.syscall, outcome: 'no-context' });
      return { id: validated.id, success: false, error: `no security context for pid ${validated.pid}` };
    }
    if (!context.canSyscall(validated.syscall)) {
      const error = `permission denied: ${context.subject} may not call ${validated.syscall}`;
      this.record({ pid: validated.pid, subject: context.subject, syscall: validated.syscall, outcome: 'denied', error });
      return { id: validated.id, success: false, error };
    }
    const handler = this.handlers.get(validated.syscall);
    if (!handler) {
      const error = `unknown syscall: ${validated.syscall}`;
      this.record({ pid: validated.pid, subject: context.subject, syscall: validated.syscall, outcome: 'unknown-syscall', error });
      return { id: validated.id, success: false, error };
    }
    try {
      const data = await handler(validated.args, context);
      this.record({ pid: validated.pid, subject: context.subject, syscall: validated.syscall, outcome: 'completed' });
      return { id: validated.id, success: true, data };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.record({ pid: validated.pid, subject: context.subject, syscall: validated.syscall, outcome: 'failed', error: message });
      return { id: validated.id, success: false, error: message };
    }
  }

  /** Exception form for in-kernel callers; preserves the handler's own error message verbatim. */
  async invoke(request: Omit<SyscallRequest, 'id'> & { id?: string }): Promise<unknown> {
    const response = await this.handle(request);
    if (!response.success) throw new Error(response.error ?? 'syscall failed');
    return response.data;
  }

  private record(entry: Omit<SyscallAuditEntry, 'at'>): void {
    this.auditTrail.push({ at: new Date().toISOString(), ...entry });
    if (this.auditTrail.length > this.auditLimit) this.auditTrail.shift();
  }
}
