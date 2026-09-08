import type { AppManifest, OSKind } from '@tcn-computer/protocol';
import { ecosystemApplications } from './definitions/ecosystem.js';
import { systemApplications } from './definitions/system.js';

export { ecosystemApplications } from './definitions/ecosystem.js';
export { systemApplications } from './definitions/system.js';
export { app, allOperatingSystems, type CatalogAppInput } from './factory.js';

export const appCatalog: AppManifest[] = [...systemApplications, ...ecosystemApplications];

export function appsForOS(os: OSKind): AppManifest[] {
  return appCatalog.filter((manifest) => manifest.supportedOS.includes(os));
}

export function systemAppsForOS(os: OSKind): AppManifest[] {
  return appsForOS(os).filter((manifest) => manifest.system);
}

export function catalogApp(id: string): AppManifest | undefined {
  return appCatalog.find((manifest) => manifest.id === id);
}

export function fileExtension(filePath: string): string {
  const name = filePath.split(/[\\/]/).at(-1) ?? '';
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

/**
 * Applications that declare themselves able to open `filePath`, most specific
 * first. This is the only consumer of `AppManifest.fileAssociations`, so a
 * declared association is a behavioral claim rather than documentation.
 */
export function fileHandlersFor(filePath: string, os: OSKind, installedAppIds?: readonly string[]): AppManifest[] {
  const extension = fileExtension(filePath);
  if (!extension) return [];
  const installed = installedAppIds ? new Set(installedAppIds) : undefined;
  return appsForOS(os)
    .filter((manifest) => manifest.fileAssociations?.includes(extension))
    .filter((manifest) => !installed || installed.has(manifest.id))
    // A system default beats a third-party handler when both claim the type.
    .sort((left, right) => Number(Boolean(right.system)) - Number(Boolean(left.system)));
}

/** The application that should open `filePath`, or undefined when nothing claims it. */
export function defaultFileHandler(filePath: string, os: OSKind, installedAppIds?: readonly string[]): AppManifest | undefined {
  return fileHandlersFor(filePath, os, installedAppIds)[0];
}
