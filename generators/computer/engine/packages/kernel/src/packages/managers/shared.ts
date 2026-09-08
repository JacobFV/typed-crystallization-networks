import type { PackageManagerKind } from '@tcn-computer/protocol';
import type { CatalogPackage, InstalledRecord, ManagerContext, OperationKind, PackageScope } from '../types.js';

/** One package as the install/remove renderers see it. */
export interface PackageView {
  name: string;
  version: string;
  direct: boolean;
  entry: CatalogPackage;
  installPath: string;
  /** Absolute descriptor paths written for this package. */
  binaries: string[];
  /** Set when this install replaced an older version. */
  previousVersion?: string;
}

export interface InstallRender {
  context: ManagerContext;
  scope: PackageScope;
  /** Newly written packages, dependencies first. */
  installed: PackageView[];
  /** Already present at a satisfying version. */
  reused: PackageView[];
  dryRun: boolean;
  cask: boolean;
  dev: boolean;
}

export interface RemoveRender {
  context: ManagerContext;
  removed: PackageView[];
  /** Transitive packages left behind that nothing depends on any more. */
  orphans: string[];
  /** Orphans this command actually deleted. */
  autoremoved: PackageView[];
  purge: boolean;
}

export interface ListRender {
  context: ManagerContext;
  records: InstalledRecord[];
  /** Manifest name of the project being listed, when the manager has one. */
  projectName?: string;
}

export interface SearchRender {
  context: ManagerContext;
  query: string;
  matches: CatalogPackage[];
  installed: Set<string>;
}

export interface InfoRender {
  context: ManagerContext;
  name: string;
  entry?: CatalogPackage;
  record?: InstalledRecord;
}

export interface OutdatedRender {
  context: ManagerContext;
  entries: { record: InstalledRecord; latest: string }[];
}

export interface Renderers {
  install(view: InstallRender): string;
  remove(view: RemoveRender): string;
  list(view: ListRender): string;
  search(view: SearchRender): string;
  info(view: InfoRender): string;
  refresh(context: ManagerContext): string;
  outdated(view: OutdatedRender): string;
  upgrade?(view: InstallRender): string;
}

export interface ManagerSpec {
  id: PackageManagerKind;
  /** Command names routed here, including aliases (`apt-get`, `pip3`, `mamba`). */
  commands: readonly string[];
  family: 'native' | 'language';
  /** Installs into the current project directory rather than the machine. */
  projectScoped: boolean;
  /** Reads a bare `1.2.3` as `^1.2.3` (Cargo, Composer). */
  bareCaret?: boolean;
  /** Verb → operation. Anything not listed is an unknown subcommand and errors. */
  verbs: Readonly<Record<string, OperationKind>>;
  /** Operation used when the command line has no verb at all. */
  defaultVerb?: OperationKind;
  /** Leading words to drop before the verb (`dotnet tool install`, `uv pip install`). */
  prefixes?: readonly (readonly string[])[];
  /** Flags carrying an out-of-band version pin. */
  versionFlags?: readonly string[];
  renderers: Renderers;
  /** Text for `<manager> --help` and unknown-verb hints. */
  helpVerbs?: readonly string[];
}

export function defineManager(spec: ManagerSpec): ManagerSpec { return spec; }

/* ------------------------------------------------------------------ helpers */

/** apt's "1,234 kB" convention. */
export function aptSize(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1).replace(/\.0$/, '')} MB`;
  return `${Math.round(bytes / 1000).toLocaleString('en-US')} kB`;
}

export function humanBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

/** Wraps a package list the way apt does: two leading spaces, wrapped at 76 columns. */
export function wrapNames(names: readonly string[], indent = '  ', width = 76): string {
  if (!names.length) return '';
  const lines: string[] = [];
  let current = indent;
  for (const name of names) {
    if (current.length > indent.length && current.length + name.length + 1 > width) {
      lines.push(current);
      current = indent;
    }
    current += current.length > indent.length ? ` ${name}` : name;
  }
  lines.push(current);
  return lines.join('\n');
}

/** Left-aligned fixed-width table used by winget/dpkg/scoop listings. */
export function table(headers: readonly string[], rows: readonly (readonly string[])[], separator?: string): string {
  const widths = headers.map((header, index) => Math.max(header.length, ...rows.map((row) => (row[index] ?? '').length)));
  const render = (cells: readonly string[]): string => cells.map((cell, index) => (index === cells.length - 1 ? cell : cell.padEnd(widths[index]!))).join(' ').trimEnd();
  const lines = [render(headers)];
  if (separator) lines.push(widths.map((width) => separator.repeat(width)).join(' '));
  for (const row of rows) lines.push(render(row));
  return lines.join('\n');
}

export const packageWord = (count: number): string => (count === 1 ? 'package' : 'packages');

/** Total installed footprint of a plan, used for apt/dnf-style disk lines. */
export function totalSize(views: readonly PackageView[], key: 'sizeBytes' | 'downloadBytes'): number {
  return views.reduce((sum, view) => sum + (view.entry[key] ?? (key === 'sizeBytes' ? 262_144 : 98_304)), 0);
}

/** Deterministic-looking archive URL for transcripts that echo a fetch. */
export function archiveUrl(manager: PackageManagerKind, entry: CatalogPackage, version: string): string {
  if (manager === 'apt' || manager === 'dpkg') return `https://packages.seed.local/ubuntu noble/${entry.section ?? 'main'} amd64 ${entry.name} amd64 ${version}`;
  if (manager === 'brew') return `https://ghcr.seed.local/v2/homebrew/core/${entry.name}/blobs/sha256-${entry.name}`;
  return `https://registry.seed.local/${manager}/${entry.name}/-/${entry.name}-${version}.tgz`;
}
