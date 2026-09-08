import type { PackageManagerKind } from '@tcn-computer/protocol';
import { aptManager, dpkgManager } from './apt.js';
import { brewManager, masManager } from './brew.js';
import { cargoManager, goManager } from './cargo.js';
import { javascriptManagers } from './javascript.js';
import { pythonManagers } from './python.js';
import { runtimeManagers } from './runtimes.js';
import type { ManagerSpec } from './shared.js';
import { flatpakManager, snapManager } from './snap.js';
import { wingetManager } from './winget.js';
import { chocoManager, scoopManager } from './windows-stores.js';

export * from './shared.js';

/** Every manager module, in the order they are documented. */
export const managerModules: readonly ManagerSpec[] = [
  brewManager, masManager,
  aptManager, dpkgManager, snapManager, flatpakManager,
  wingetManager, chocoManager, scoopManager,
  ...javascriptManagers,
  ...pythonManagers,
  cargoManager, goManager,
  ...runtimeManagers,
];

const byId = new Map<PackageManagerKind, ManagerSpec>(managerModules.map((module) => [module.id, module]));

/** Command name (including aliases) → manager module. */
export const managerByCommand = new Map<string, ManagerSpec>(
  managerModules.flatMap((module) => module.commands.map((command) => [command, module] as const)),
);

export function managerModule(id: PackageManagerKind): ManagerSpec {
  const module = byId.get(id);
  if (!module) throw new Error(`no module registered for package manager ${id}`);
  return module;
}

export function resolveCommand(command: string): ManagerSpec | undefined {
  return managerByCommand.get(command.trim().toLowerCase());
}
