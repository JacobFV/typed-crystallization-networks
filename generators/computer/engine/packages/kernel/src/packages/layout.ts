import type { OSRuntimeProfile, PackageManagerKind } from '@tcn-computer/protocol';
import type { CatalogPackage, ManagerContext, PackageScope } from './types.js';

/**
 * Where each manager actually puts things.
 *
 * Three independent questions, answered separately because managers answer them
 * differently: the payload root (a Cellar keg, a `node_modules` folder, a crate
 * source checkout), the bin directory that ends up on PATH, and the receipt
 * location — which must live under one of `profile.packageManagers.receiptRoots`
 * so the OS profile stays the single source of truth for package bookkeeping.
 */

const safe = (value: string): string => value.replaceAll('\\', '/').replace(/^\/+/, '').replaceAll(':', '_') || 'package';

/** Directory-safe form of a package name; scoped npm names keep their nesting. */
export function safeName(manager: PackageManagerKind, name: string): string {
  if (['npm', 'pnpm', 'yarn', 'bun'].includes(manager) && name.startsWith('@')) return safe(name);
  if (manager === 'composer' || manager === 'go') return safe(name);
  return safe(name).replaceAll('/', '__');
}

/** Windows keeps `.exe`; POSIX keeps the bare name. From `conventions.executableSuffix`. */
export function executableFileName(context: ManagerContext, binary: string): string {
  const suffix = context.executableSuffix;
  return suffix && !binary.toLowerCase().endsWith(suffix.toLowerCase()) ? `${binary}${suffix}` : binary;
}

/** `visual-studio-code` → `Visual Studio Code`: the bundle name a cask installs as. */
export function applicationName(entry: CatalogPackage): string {
  if (!entry.cask) return entry.description;
  return entry.name.split(/[-_.]/).filter(Boolean)
    .map((word) => (word.length <= 2 ? word.toUpperCase() : word[0]!.toUpperCase() + word.slice(1)))
    .join(' ');
}

const windowsRoamingPython = (home: string): string => `${home}/AppData/Roaming/Python/Scripts`;

/** Bin directory receiving executable descriptors for a manager/scope pair. */
export function binDirectory(context: ManagerContext, scope: PackageScope): string {
  const { manager, home, cwd, os } = context;
  const windows = os === 'windows';
  switch (manager) {
    case 'brew': return os === 'macos' ? '/opt/homebrew/bin' : '/home/linuxbrew/.linuxbrew/bin';
    case 'mas': return '/usr/local/bin';
    case 'apt': case 'dpkg': return '/usr/bin';
    case 'snap': return '/snap/bin';
    case 'flatpak': return '/var/lib/flatpak/exports/bin';
    case 'winget': return `${home}/AppData/Local/Microsoft/WinGet/Links`;
    case 'choco': return '/C/ProgramData/chocolatey/bin';
    case 'scoop': return `${home}/scoop/shims`;
    case 'npm': case 'yarn': case 'bun': case 'pnpm': {
      if (scope === 'project') return `${cwd}/node_modules/.bin`;
      if (windows) return `${home}/AppData/Roaming/npm`;
      if (manager === 'bun') return `${home}/.bun/bin`;
      if (manager === 'pnpm') return `${home}/.local/share/pnpm`;
      if (manager === 'yarn') return `${home}/.yarn/bin`;
      return `${home}/.local/bin`;
    }
    case 'pip': return windows ? windowsRoamingPython(home) : `${home}/.local/bin`;
    case 'pipx': case 'uv': return windows ? `${home}/AppData/Roaming/Python/Scripts` : `${home}/.local/bin`;
    case 'poetry': return windows ? `${cwd}/.venv/Scripts` : `${cwd}/.venv/bin`;
    case 'conda': return windows ? `${home}/miniconda3/Scripts` : `${home}/miniconda3/bin`;
    case 'cargo': return `${home}/.cargo/bin`;
    case 'go': return `${home}/go/bin`;
    case 'gem': return windows ? `${home}/.gem/ruby/3.4.0/bin` : `${home}/.gem/ruby/3.4.0/bin`;
    case 'composer': return scope === 'project' ? `${cwd}/vendor/bin` : `${home}/.composer/vendor/bin`;
    case 'dotnet': return `${home}/.dotnet/tools`;
    case 'nuget': return `${cwd}/packages/tools`;
    case 'vcpkg': return `${cwd}/vcpkg_installed/${windows ? 'x64-windows' : 'x64-linux'}/tools`;
    default: return `${home}/.local/bin`;
  }
}

/**
 * Bin directory for one specific binary. Debian puts daemons in `/usr/sbin`,
 * so `apt install nginx` has to land the same place a real one would.
 */
export function binDirectoryForBinary(context: ManagerContext, scope: PackageScope, entry: CatalogPackage, binary: string): string {
  const daemon = entry.service?.executable === binary;
  if (daemon && (context.manager === 'apt' || context.manager === 'dpkg')) return '/usr/sbin';
  return binDirectory(context, scope);
}

/** Payload root for one resolved package. */
export function installRoot(context: ManagerContext, entry: CatalogPackage, version: string, scope: PackageScope): string {
  const { manager, home, cwd, os } = context;
  const name = safeName(manager, entry.name);
  switch (manager) {
    case 'brew': return entry.cask
      ? `/opt/homebrew/Caskroom/${name}/${version}`
      : `${os === 'macos' ? '/opt/homebrew' : '/home/linuxbrew/.linuxbrew'}/Cellar/${name}/${version}`;
    case 'mas': return `/Applications/${safe(entry.description)}.app`;
    case 'apt': return `/usr/share/${name}`;
    case 'dpkg': return `/var/lib/dpkg/info/${name}`;
    case 'snap': return `/snap/${name}/current`;
    case 'flatpak': return `/var/lib/flatpak/app/${name}/current/active`;
    case 'winget': return `/C/Program Files/${name}`;
    case 'choco': return `/C/ProgramData/chocolatey/lib/${name}`;
    case 'scoop': return `${home}/scoop/apps/${name}/${version}`;
    case 'npm': case 'bun': case 'yarn': return scope === 'project'
      ? `${cwd}/node_modules/${name}`
      : `${home}/.local/lib/node_modules/${name}`;
    case 'pnpm': return scope === 'project'
      ? `${cwd}/node_modules/.pnpm/${name.replaceAll('/', '+')}@${version}/node_modules/${name}`
      : `${home}/.local/share/pnpm/global/5/node_modules/${name}`;
    case 'pip': return `${home}/.local/lib/python3.13/site-packages/${name}`;
    case 'poetry': return `${cwd}/.venv/lib/python3.13/site-packages/${name}`;
    case 'pipx': return `${home}/.local/share/pipx/venvs/${name}`;
    case 'uv': return `${home}/.local/share/uv/tools/${name}`;
    case 'conda': return `${home}/miniconda3/pkgs/${name}-${version}`;
    case 'cargo': return `${home}/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/${name}-${version}`;
    case 'go': return `${home}/go/pkg/mod/${name}@${version}`;
    case 'gem': return `${home}/.gem/ruby/3.4.0/gems/${name}-${version}`;
    case 'composer': return `${cwd}/vendor/${name}`;
    case 'dotnet': return `${home}/.dotnet/tools/.store/${name}/${version}`;
    case 'nuget': return `${cwd}/packages/${name}.${version}`;
    case 'vcpkg': return `${cwd}/vcpkg_installed/${os === 'windows' ? 'x64-windows' : 'x64-linux'}/${name}`;
    default: return `${home}/.local/share/packages/${name}`;
  }
}

/**
 * Receipt path, always rooted at one of `profile.packageManagers.receiptRoots`.
 * The root is chosen by prefix match against the install root so brew receipts
 * land in the Cellar and dpkg receipts land in `/var/lib/dpkg`; managers with no
 * matching root fall back to the profile's last (generic) root.
 */
export function receiptPath(profile: OSRuntimeProfile, context: ManagerContext, entry: CatalogPackage, version: string, root: string): string {
  const roots = profile.packageManagers.receiptRoots;
  const fallback = roots.at(-1) ?? `${context.home}/.local/share/seed/receipts`;
  const matched = roots.find((candidate) => root === candidate || root.startsWith(`${candidate}/`));
  const name = safeName(context.manager, entry.name);
  switch (context.manager) {
    // Homebrew really does write INSTALL_RECEIPT.json into the keg.
    case 'brew': return matched ? `${root}/INSTALL_RECEIPT.json` : `${fallback}/brew/${name}-${version}.json`;
    case 'dpkg': case 'apt': return `${roots.find((candidate) => candidate.endsWith('/dpkg')) ?? fallback}/info/${name}.list`;
    case 'snap': return `${roots.find((candidate) => candidate.endsWith('snapd')) ?? fallback}/seed/${name}_${version}.json`;
    case 'flatpak': return `${roots.find((candidate) => candidate.endsWith('flatpak')) ?? fallback}/app/${name}/current/active/deploy.json`;
    case 'mas': return `${roots.find((candidate) => candidate.endsWith('receipts')) ?? fallback}/com.apple.mas.${name}.plist.json`;
    case 'choco': return `${roots.find((candidate) => candidate.includes('chocolatey')) ?? fallback}/lib/${name}/${name}.nuspec.json`;
    // Scoop's own install.json, in the versioned app directory.
    case 'scoop': return matched ? `${root}/install.json` : `${fallback}/${name}/${version}/install.json`;
    case 'winget': return `${roots[0] ?? fallback}/${name}_${version}.json`;
    default: return `${fallback}/${context.manager}/${name}-${version}.json`;
  }
}

/** Systemd/launchd/services unit written for a package that ships a daemon. */
export function serviceUnitPath(context: ManagerContext, service: string): string {
  if (context.os === 'ubuntu') return `/lib/systemd/system/${service}.service`;
  if (context.os === 'macos') return `/opt/homebrew/opt/${service}/homebrew.mxcl.${service}.plist`;
  return `/C/ProgramData/Seed/Services/${service}.json`;
}
