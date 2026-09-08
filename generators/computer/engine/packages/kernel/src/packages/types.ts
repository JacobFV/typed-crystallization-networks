import type { OSKind, OSRuntimeProfile, PackageManagerKind, PackageRecord } from '@tcn-computer/protocol';

/** Scope a manager installs into. Mirrors `PackageRecord['scope']`. */
export type PackageScope = PackageRecord['scope'];

/**
 * A daemon a package can start. Registered as a unit file plus a startable
 * executable descriptor so a listener can be brought up on the virtual fabric
 * after installation.
 */
export interface ServiceDefinition {
  /** Unit/service name, e.g. `nginx`, `postgresql`, `redis-server`. */
  name: string;
  displayName: string;
  /** Program that hosts the service; must be one of the package's binaries. */
  executable: string;
  args?: readonly string[];
  port: number;
  protocol: 'http' | 'tcp';
  /** Directory the daemon serves from, relative to the filesystem root. */
  documentRoot?: string;
}

/** A curated registry entry: what the manager would actually ship. */
export interface CatalogPackage {
  name: string;
  /** Newest published version. Always present in `versions`. */
  version: string;
  description: string;
  /** Every version the resolver may pick, oldest first. */
  versions?: readonly string[];
  /** Executables installed onto PATH. Defaults to `[name]` for native managers. */
  binaries?: readonly string[];
  /** `binary -> behavior key`, when the shell's behavior name differs from the binary. */
  behaviors?: Readonly<Record<string, string>>;
  /** Runtime dependency ranges resolved transitively at install time. */
  dependencies?: Readonly<Record<string, string>>;
  /** Packages that cannot be co-installed, with the offending range. */
  conflicts?: Readonly<Record<string, string>>;
  /** Alternative names the manager accepts (virtual packages, transitional names). */
  aliases?: readonly string[];
  service?: ServiceDefinition;
  /** Homebrew casks and Chocolatey GUI apps install as applications, not formulae. */
  cask?: boolean;
  /** Debian section / Homebrew tap / winget moniker source. */
  section?: string;
  license?: string;
  homepage?: string;
  /** Installed footprint in bytes; drives apt's "After this operation" line. */
  sizeBytes?: number;
  /** Download size in bytes; drives apt's "Need to get" line. */
  downloadBytes?: number;
  keywords?: readonly string[];
  /** Publisher shown by winget/choco/mas listings. */
  publisher?: string;
}

/** A parsed command-line package reference. */
export interface PackageSpecifier {
  raw: string;
  name: string;
  /** Version range in the shared dialect. `*` when unconstrained. */
  range: string;
  /** Dist-tag or channel requested instead of a range (`latest`, `next`, `beta`). */
  tag?: string;
  /** PEP 508 extras, e.g. `fastapi[all]`. */
  extras?: string[];
  /** Non-registry sources keep their locator so the receipt can record it. */
  source?: { kind: 'registry' | 'file' | 'git' | 'url' | 'path'; locator: string };
  /** Debian architecture qualifier (`pkg:amd64`) or Go module suffix. */
  architecture?: string;
}

export type OperationKind =
  | 'install' | 'remove' | 'autoremove' | 'list' | 'search' | 'info' | 'upgrade'
  | 'refresh' | 'outdated' | 'run' | 'exec' | 'build' | 'test' | 'start' | 'services' | 'help' | 'clean';

/** Everything a manager module needs to compute paths and render output. */
export interface ManagerContext {
  manager: PackageManagerKind;
  os: OSKind;
  profile: OSRuntimeProfile;
  home: string;
  cwd: string;
  /** `.exe` on Windows, empty elsewhere. From `conventions.executableSuffix`. */
  executableSuffix: string;
  hostname: string;
}

/** The on-disk executable contract shared with the shell's program registry. */
export interface SeedExecutableDescriptor {
  seedExecutable: 1;
  name: string;
  package: string;
  manager: PackageManagerKind;
  version: string;
  provides: string[];
  behavior: string;
  /** Present only for daemons; lets the shell start a real listener. */
  service?: ServiceDefinition;
}

/** Internal installed-package record. Superset of the protocol's `PackageRecord`. */
export interface InstalledRecord extends PackageRecord {
  /** Name as the manager displays it, before any path-safe mangling. */
  displayName: string;
  /** Extra identifiers accepted by `remove`/`info` (short names, deb filenames). */
  aliases: string[];
  /** Absolute paths of every executable descriptor written for this package. */
  binaries: string[];
  /** Binary basenames, without the platform suffix. */
  provides: string[];
  /** Range the user (or a dependent) asked for. */
  requestedRange: string;
  /** Dependency name → range, the constraint form of `dependencies`. */
  dependencyRanges: Record<string, string>;
  /** Project root for project-scoped installs; the home directory otherwise. */
  root: string;
  service?: ServiceDefinition;
  cask?: boolean;
  section?: string;
  description: string;
}

/** Strips the internal-only fields so snapshots carry exactly the protocol shape. */
export function toPackageRecord(record: InstalledRecord): PackageRecord {
  return {
    id: record.id, name: record.name, version: record.version, manager: record.manager, scope: record.scope,
    installPath: record.installPath, installedAt: record.installedAt, files: [...record.files], source: record.source,
    integrity: record.integrity, dependencies: [...record.dependencies], dependencyType: record.dependencyType,
  };
}

