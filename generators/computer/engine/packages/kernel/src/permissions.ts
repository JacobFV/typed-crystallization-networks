/**
 * Per-process security contexts and the permission table that backs them.
 *
 * The shape of this module (a `PermissionManager` that stores a `Permission` per PID and hands out
 * a `SecurityContext` that answers `canSyscall` / `canAccessPath`) is adapted from the browser-os
 * project (MIT licensed) — `packages/kernel/src/PermissionManager.ts` and
 * `packages/kernel/src/SecurityContext.ts`. Seed extends it with host matching, a read/write split
 * on filesystem access, an explicit subject label so handlers can never be tricked into acting for
 * a different application, and a corrected glob matcher (browser-os expands `**` to `.*` and then
 * rewrites the `*` it just produced, so `/tmp/**` never matched a nested path).
 */

export type FilesystemAccess = 'read' | 'write' | 'execute';

export interface SeedPermission {
  /** Stable identity of the subject the permission belongs to (an app id for application processes). */
  label: string;
  /** Default-deny allowlist of syscall names. */
  allowedSyscalls: string[];
  /** Explicit denials; evaluated before the allowlist. */
  deniedSyscalls?: string[];
  /** Glob patterns the subject may read. `execute` is checked against this list too. */
  fsRead: string[];
  /** Glob patterns the subject may write. */
  fsWrite: string[];
  /** Hostname globs the subject may reach over the virtual network. */
  network: string[];
}

export function emptyPermission(label: string): SeedPermission {
  return { label, allowedSyscalls: [], deniedSyscalls: [], fsRead: [], fsWrite: [], network: [] };
}

/** Escape everything a RegExp treats specially except the glob wildcards we handle ourselves. */
function globToRegExp(pattern: string): RegExp {
  let source = '';
  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index]!;
    if (character === '*') {
      if (pattern[index + 1] === '*') {
        // `prefix/**` also matches `prefix` itself, mirroring how a directory grant reads.
        if (source.endsWith('/') && index + 2 === pattern.length) source = `${source.slice(0, -1)}(?:/.*)?`;
        else source += '.*';
        index += 1;
      } else source += '[^/]*';
      continue;
    }
    if (character === '?') { source += '[^/]'; continue; }
    source += character.replace(/[.+^${}()|[\]\\]/g, '\\$&');
  }
  return new RegExp(`^${source}$`);
}

const globCache = new Map<string, RegExp>();

export function globMatches(value: string, pattern: string): boolean {
  if (pattern === '*' || pattern === '**') return true;
  let regex = globCache.get(pattern);
  if (!regex) { regex = globToRegExp(pattern); globCache.set(pattern, regex); }
  return regex.test(value);
}

/**
 * The capability handle a syscall handler receives. It is the only thing a handler learns about its
 * caller, which keeps handlers from trusting attacker-supplied arguments for identity.
 */
export class SecurityContext {
  constructor(readonly pid: number, readonly permissions: SeedPermission) {}

  /** Identity of the subject this context speaks for (never taken from syscall arguments). */
  get subject(): string { return this.permissions.label; }

  canSyscall(syscall: string): boolean {
    if (this.permissions.deniedSyscalls?.includes(syscall)) return false;
    return this.permissions.allowedSyscalls.includes(syscall);
  }

  canAccessPath(filePath: string, operation: FilesystemAccess): boolean {
    const patterns = operation === 'write' ? this.permissions.fsWrite : this.permissions.fsRead;
    return patterns.some((pattern) => globMatches(filePath, pattern));
  }

  canReachHost(host: string): boolean {
    return this.permissions.network.some((pattern) => globMatches(host, pattern));
  }

  requireSyscall(syscall: string): void {
    if (!this.canSyscall(syscall)) throw new Error(`permission denied: ${this.subject} may not call ${syscall}`);
  }

  requirePath(filePath: string, operation: FilesystemAccess): void {
    if (!this.canAccessPath(filePath, operation)) {
      throw new Error(`permission denied: ${this.subject} may not ${operation} ${filePath}`);
    }
  }

  requireHost(host: string): void {
    if (!this.canReachHost(host)) throw new Error(`permission denied: ${this.subject} may not reach ${host}`);
  }
}

export class PermissionManager {
  private readonly permissions = new Map<number, SeedPermission>();

  setPermission(pid: number, permission: SeedPermission): void {
    this.permissions.set(pid, {
      ...permission,
      allowedSyscalls: [...permission.allowedSyscalls],
      deniedSyscalls: [...(permission.deniedSyscalls ?? [])],
      fsRead: [...permission.fsRead],
      fsWrite: [...permission.fsWrite],
      network: [...permission.network],
    });
  }

  getPermission(pid: number): SeedPermission | null {
    return this.permissions.get(pid) ?? null;
  }

  getSecurityContext(pid: number): SecurityContext | null {
    const permission = this.permissions.get(pid);
    return permission ? new SecurityContext(pid, permission) : null;
  }

  removePermission(pid: number): void { this.permissions.delete(pid); }

  grantSyscall(pid: number, syscall: string): void {
    const permission = this.permissions.get(pid);
    if (permission && !permission.allowedSyscalls.includes(syscall)) permission.allowedSyscalls.push(syscall);
  }

  revokeSyscall(pid: number, syscall: string): void {
    const permission = this.permissions.get(pid);
    if (!permission) return;
    permission.allowedSyscalls = permission.allowedSyscalls.filter((value) => value !== syscall);
    permission.deniedSyscalls = [...new Set([...(permission.deniedSyscalls ?? []), syscall])];
  }

  grantPath(pid: number, pattern: string, operation: FilesystemAccess = 'read'): void {
    const permission = this.permissions.get(pid);
    if (!permission) return;
    const target = operation === 'write' ? permission.fsWrite : permission.fsRead;
    if (!target.includes(pattern)) target.push(pattern);
  }

  pids(): number[] { return [...this.permissions.keys()]; }
}
