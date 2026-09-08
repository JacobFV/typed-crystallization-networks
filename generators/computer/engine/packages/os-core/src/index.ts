import type { OSKind, PackageManagerKind, ShellKind } from '@tcn-computer/protocol';

export interface OperatingSystemProfile {
  id: OSKind;
  productName: string;
  release: string;
  kernel: {
    family: 'xnu' | 'nt' | 'linux';
    version: string;
    init: string;
    serviceManager: string;
  };
  desktop: {
    shell: string;
    windowManager: string;
    compositor: string;
    displayServer: string;
    launcher: string;
    settingsApp: string;
  };
  shell: {
    default: ShellKind;
    executable: string;
    promptDialect: 'posix' | 'powershell';
    startupFiles: readonly string[];
  };
  filesystem: {
    root: string;
    home: string;
    applications: string;
    userData: string;
    temporary: string;
    caseSensitive: boolean;
    pathSeparator: '/' | '\\';
    nativeFormats: readonly string[];
  };
  packageManagers: {
    native: readonly PackageManagerKind[];
    language: readonly PackageManagerKind[];
    receiptRoots: readonly string[];
  };
  bootServices: readonly BootServiceProfile[];
  peripherals: readonly PeripheralProfile[];
  systemAppIds: readonly string[];
  conventions: {
    executableSuffix: string;
    sharedLibrarySuffix: string;
    environmentPathKey: 'PATH' | 'Path';
    localhostNames: readonly string[];
  };
}

export interface BootServiceProfile {
  id: string;
  executable: string;
  role: 'init' | 'session' | 'display' | 'network' | 'audio' | 'device' | 'security' | 'indexing' | 'updates';
  parent: string | null;
  required: boolean;
}

export interface PeripheralProfile {
  kind: 'display' | 'keyboard' | 'pointer' | 'camera' | 'microphone' | 'speaker' | 'storage' | 'network';
  driver: string;
  hotPluggable: boolean;
}

export function defineOperatingSystem(profile: OperatingSystemProfile): Readonly<OperatingSystemProfile> {
  validateOperatingSystem(profile);
  return Object.freeze(profile);
}

export function validateOperatingSystem(profile: OperatingSystemProfile): void {
  if (!profile.productName || !profile.release) throw new Error(`${profile.id}: product name and release are required`);
  if (!profile.bootServices.some((service) => service.role === 'init' && service.parent === null)) throw new Error(`${profile.id}: an init service is required`);
  if (!profile.bootServices.some((service) => service.role === 'display')) throw new Error(`${profile.id}: a display service is required`);
  if (!profile.bootServices.some((service) => service.role === 'network')) throw new Error(`${profile.id}: a network service is required`);
  if (!profile.packageManagers.native.length) throw new Error(`${profile.id}: a native package manager is required`);
  if (!profile.systemAppIds.length) throw new Error(`${profile.id}: system application set is empty`);
  const appIds = new Set(profile.systemAppIds);
  if (appIds.size !== profile.systemAppIds.length) throw new Error(`${profile.id}: duplicate system application ids`);
}
