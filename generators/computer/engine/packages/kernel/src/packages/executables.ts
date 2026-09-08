import type { PackageManagerKind } from '@tcn-computer/protocol';
import type { VirtualFileSystem } from '../vfs.js';
import { binDirectoryForBinary, executableFileName, serviceUnitPath } from './layout.js';
import type { CatalogPackage, ManagerContext, PackageScope, SeedExecutableDescriptor, ServiceDefinition } from './types.js';

/**
 * The PATH contract.
 *
 * An executable is a VFS file with mode 0o755 whose content is a JSON
 * descriptor. The shell's program registry resolves `behavior` to a real
 * implementation, which is what makes `brew install ripgrep` followed by `rg`
 * work instead of exiting 127. Installing writes one descriptor per binary the
 * package provides; removing deletes exactly the descriptors it wrote.
 */

export const EXECUTABLE_MODE = 0o755;
export const DESCRIPTOR_MAGIC = 1;

export interface ExecutablePlan {
  path: string;
  descriptor: SeedExecutableDescriptor;
}

/** Serialized form written into the VFS. Stable key order keeps diffs readable. */
export function serializeDescriptor(descriptor: SeedExecutableDescriptor): string {
  return `${JSON.stringify(descriptor, null, 2)}\n`;
}

/** Parses a descriptor, returning `undefined` for any file that is not one. */
export function parseDescriptor(content: string): SeedExecutableDescriptor | undefined {
  try {
    const value = JSON.parse(content) as Partial<SeedExecutableDescriptor>;
    if (value?.seedExecutable !== DESCRIPTOR_MAGIC || typeof value.name !== 'string' || typeof value.behavior !== 'string') return undefined;
    return value as SeedExecutableDescriptor;
  } catch { return undefined; }
}

/**
 * Descriptors a package should install. `provides` lists every binary in the
 * package so a shell resolving one of them can see its siblings; `behavior` is
 * the tool's canonical name unless the entry overrides it (Debian ships ripgrep's
 * `fd` as `fdfind`, but the behavior is still `fd`).
 */
export function planExecutables(
  context: ManagerContext,
  entry: CatalogPackage,
  version: string,
  scope: PackageScope,
): ExecutablePlan[] {
  const binaries = entry.binaries ?? [entry.name];
  if (!binaries.length) return [];
  const provides = [...binaries];
  return binaries.map((binary) => ({
    path: `${binDirectoryForBinary(context, scope, entry, binary)}/${executableFileName(context, binary)}`,
    descriptor: {
      seedExecutable: DESCRIPTOR_MAGIC,
      name: binary,
      package: entry.name,
      manager: context.manager,
      version,
      provides,
      behavior: entry.behaviors?.[binary] ?? binary,
      ...(entry.service && entry.service.executable === binary ? { service: entry.service } : {}),
    },
  }));
}

/** Writes every descriptor at mode 0o755 and returns the paths written. */
export async function writeExecutables(vfs: VirtualFileSystem, plans: readonly ExecutablePlan[]): Promise<string[]> {
  const written: string[] = [];
  for (const plan of plans) {
    await vfs.writeFile(plan.path, serializeDescriptor(plan.descriptor));
    await vfs.chmod(plan.path, EXECUTABLE_MODE);
    written.push(plan.path);
  }
  return written;
}

/**
 * Removes only descriptors this package owns. A descriptor overwritten by a
 * different package (two managers shipping `docker`) is left alone, which is
 * what keeps `brew uninstall docker` from breaking a Docker Desktop install.
 */
export async function removeExecutables(
  vfs: VirtualFileSystem,
  paths: readonly string[],
  owner: { package: string; manager: PackageManagerKind },
): Promise<string[]> {
  const removed: string[] = [];
  for (const path of paths) {
    let descriptor: SeedExecutableDescriptor | undefined;
    try { descriptor = parseDescriptor(await vfs.readFile(path)); } catch { continue; }
    if (!descriptor) continue;
    if (descriptor.package !== owner.package || descriptor.manager !== owner.manager) continue;
    await vfs.remove(path);
    removed.push(path);
  }
  return removed;
}

/**
 * Registers a startable daemon: a platform-native unit file plus a machine
 * readable record the shell/process layer reads to bring up a real listener.
 */
export async function writeServiceUnit(
  vfs: VirtualFileSystem,
  context: ManagerContext,
  service: ServiceDefinition,
  packageName: string,
  version: string,
): Promise<string[]> {
  const unit = serviceUnitPath(context, service.name);
  const exec = `${binDirectoryForBinary(context, 'system', { name: packageName, version, description: '', service }, service.executable)}/${executableFileName(context, service.executable)}`;
  const argv = [exec, ...(service.args ?? [])].join(' ');
  if (context.os === 'ubuntu') {
    await vfs.writeFile(unit, [
      '[Unit]',
      `Description=${service.displayName}`,
      'After=network.target',
      '',
      '[Service]',
      'Type=simple',
      `ExecStart=${argv}`,
      'Restart=on-failure',
      '',
      '[Install]',
      'WantedBy=multi-user.target',
      '',
    ].join('\n'));
  } else if (context.os === 'macos') {
    await vfs.writeFile(unit, [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<plist version="1.0">',
      '  <dict>',
      `    <key>Label</key><string>homebrew.mxcl.${service.name}</string>`,
      `    <key>ProgramArguments</key><array><string>${exec}</string>${(service.args ?? []).map((arg) => `<string>${arg}</string>`).join('')}</array>`,
      '    <key>RunAtLoad</key><true/>',
      '  </dict>',
      '</plist>',
      '',
    ].join('\n'));
  } else {
    await vfs.writeFile(unit, `${JSON.stringify({ name: service.name, displayName: service.displayName, imagePath: exec, startType: 'Automatic' }, null, 2)}\n`);
  }
  // Machine-readable index so a service can be started without parsing units.
  const registry = context.os === 'windows' ? '/C/ProgramData/Seed/Services/index.json' : '/var/lib/seed/services.json';
  let index: Record<string, unknown> = {};
  try { index = JSON.parse(await vfs.readFile(registry)) as Record<string, unknown>; } catch { /* first service */ }
  index[service.name] = {
    ...service, unit, executablePath: exec, package: packageName, manager: context.manager, version,
    registeredAt: new Date().toISOString(),
  };
  await vfs.writeFile(registry, `${JSON.stringify(index, null, 2)}\n`);
  return [unit, registry];
}

export async function removeServiceUnit(vfs: VirtualFileSystem, context: ManagerContext, service: ServiceDefinition): Promise<void> {
  await vfs.remove(serviceUnitPath(context, service.name));
  const registry = context.os === 'windows' ? '/C/ProgramData/Seed/Services/index.json' : '/var/lib/seed/services.json';
  try {
    const index = JSON.parse(await vfs.readFile(registry)) as Record<string, unknown>;
    delete index[service.name];
    await vfs.writeFile(registry, `${JSON.stringify(index, null, 2)}\n`);
  } catch { /* nothing registered */ }
}
