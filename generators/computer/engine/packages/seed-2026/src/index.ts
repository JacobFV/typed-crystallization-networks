import { appCatalog } from '@tcn-computer/catalog';
import type { OperatingSystemProfile } from '@tcn-computer/os-core';
import { macOSProfile } from '@tcn-computer/os-macos';
import { ubuntuProfile } from '@tcn-computer/os-ubuntu';
import { windowsProfile } from '@tcn-computer/os-windows';
import type {
  OSKind,
  SimulationComputerTemplate,
  SimulationServiceSpec,
  SimulationTopology,
} from '@tcn-computer/protocol';
import { applicationSurfaces, validateSurfaceCoverage } from '@tcn-computer/ui-surfaces';

const GIB = 1024 ** 3;

export interface EcosystemComputerTemplate extends SimulationComputerTemplate {
  profileId: OSKind;
}

export interface EcosystemServiceNode extends SimulationServiceSpec {}

export interface EcosystemBlueprint extends SimulationTopology {
  operatingSystems: Readonly<Record<OSKind, Readonly<OperatingSystemProfile>>>;
  computers: readonly EcosystemComputerTemplate[];
  services: readonly EcosystemServiceNode[];
  evidenceContract: {
    everyActionIsRecorded: boolean;
    packetTracesAreCausal: boolean;
    filesystemIsInodeBacked: boolean;
    appStateIsVfsBacked: boolean;
    surfaceCoverageRequired: boolean;
  };
}

/** Backward-compatible name for the seed distribution's blueprint contract. */
export type SeedEcosystemBlueprint = EcosystemBlueprint;

const thirdParty: Record<OSKind, readonly string[]> = {
  macos: ['chromium', 'firefox', 'slack', 'teams', 'chatgpt', 'vscode', 'cursor', 'wireshark', 'package-center', 'github-desktop', 'gitkraken', 'docker-desktop', 'postman', 'figma', 'notion', 'linear', 'zoom', 'spotify', 'obsidian', 'vlc', 'blender', 'gimp', 'libreoffice', 'bitwarden'],
  windows: ['chromium', 'firefox', 'slack', 'teams', 'vscode', 'cursor', 'wireshark', 'package-center', 'github-desktop', 'gitkraken', 'docker-desktop', 'postman', 'figma', 'notion', 'linear', 'discord', 'zoom', 'spotify', 'obsidian', 'vlc', 'audacity', 'steam', 'bitwarden', 'onepassword'],
  ubuntu: ['chromium', 'firefox', 'slack', 'vscode', 'cursor', 'wireshark', 'package-center', 'gitkraken', 'docker-desktop', 'postman', 'discord', 'zoom', 'spotify', 'obsidian', 'vlc', 'blender', 'gimp', 'libreoffice', 'audacity', 'dbeaver', 'steam', 'onepassword'],
};

export const seed2026Blueprint = Object.freeze({
  id: 'seed-2026', version: '0.3.0',
  network: { cidr: '10.42.0.0/24', dns: '10.42.0.2', domain: 'seed.local' },
  operatingSystems: { macos: macOSProfile, windows: windowsProfile, ubuntu: ubuntuProfile },
  computers: [
    {
      profileId: 'macos', roles: ['desktop', 'creative', 'agent-client'], systemAppIds: macOSProfile.systemAppIds, thirdPartyAppIds: thirdParty.macos,
      spec: { id: 'mac-studio', hostname: 'mac-studio', os: 'macos', shell: 'zsh', ipv4: '10.42.0.10', memoryBytes: 16 * GIB, cpuCores: 10, disks: [{ id: 'Macintosh-HD', label: 'Macintosh HD', mount: '/', capacityBytes: 256 * GIB }], displays: [{ id: 'main', name: 'Studio Display', width: 1512, height: 982, scale: 2 }] },
    },
    {
      profileId: 'windows', roles: ['desktop', 'enterprise', 'agent-client'], systemAppIds: windowsProfile.systemAppIds, thirdPartyAppIds: thirdParty.windows,
      spec: { id: 'win-workstation', hostname: 'win-workstation', os: 'windows', shell: 'powershell', ipv4: '10.42.0.20', memoryBytes: 16 * GIB, cpuCores: 8, disks: [{ id: 'C', label: 'Windows', mount: 'C:', capacityBytes: 256 * GIB }], displays: [{ id: 'main', name: 'Generic PnP Monitor', width: 1440, height: 900, scale: 1.25 }] },
    },
    {
      profileId: 'ubuntu', roles: ['desktop', 'developer', 'server-host', 'agent-client'], systemAppIds: ubuntuProfile.systemAppIds, thirdPartyAppIds: thirdParty.ubuntu,
      spec: { id: 'ubuntu-dev', hostname: 'ubuntu-dev', os: 'ubuntu', shell: 'bash', ipv4: '10.42.0.30', memoryBytes: 8 * GIB, cpuCores: 8, disks: [{ id: 'root', label: 'Ubuntu', mount: '/', capacityBytes: 128 * GIB }], displays: [{ id: 'main', name: 'VirtIO Display', width: 1440, height: 900, scale: 1 }] },
    },
    {
      profileId: 'ubuntu', roles: ['service-node', 'registry', 'dns'], systemAppIds: [], thirdPartyAppIds: [],
      spec: { id: 'seed-registry', hostname: 'registry', os: 'ubuntu', shell: 'bash', ipv4: '10.42.0.2', memoryBytes: 2 * GIB, cpuCores: 2, disks: [{ id: 'root', label: 'Registry', mount: '/', capacityBytes: 32 * GIB }], displays: [] },
    },
  ],
  services: [
    { id: 'dns', host: 'dns.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 53, protocol: 'udp', kind: 'dns', isolationDomain: 'seed-core' },
    { id: 'app-store-registry', host: 'appstore.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'app-registry', targetOS: 'macos', isolationDomain: 'apple-registry' },
    { id: 'microsoft-store-registry', host: 'store.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'app-registry', targetOS: 'windows', isolationDomain: 'microsoft-registry' },
    { id: 'ubuntu-package-registry', host: 'packages.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'app-registry', targetOS: 'ubuntu', isolationDomain: 'ubuntu-registry' },
    { id: 'slack', host: 'slack.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'collaboration', isolationDomain: 'slack' },
    { id: 'teams', host: 'teams.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'collaboration', isolationDomain: 'teams' },
    { id: 'git', host: 'git.seed.local', ipv4: '10.42.0.2', computerId: 'seed-registry', port: 443, protocol: 'https', kind: 'git', isolationDomain: 'git' },
    { id: 'intranet', host: 'intranet.seed.local', ipv4: '10.42.0.30', computerId: 'ubuntu-dev', port: 8080, protocol: 'http', kind: 'intranet', isolationDomain: 'ubuntu-dev' },
  ],
  gateways: [
    { id: 'internet-egress', name: 'open internet egress', enabled: true, direction: 'egress', protocols: ['tcp', 'http', 'https'], cidrs: [], hostnames: ['*'], ports: '*', audit: true },
    { id: 'docs-egress', name: 'documentation egress', enabled: true, direction: 'egress', protocols: ['https'], cidrs: [], hostnames: ['developer.mozilla.org', 'docs.python.org', 'platform.openai.com'], ports: [443], audit: true },
    { id: 'default-deny', name: 'default deny', enabled: false, direction: 'egress', protocols: ['tcp', 'udp', 'http', 'https'], cidrs: ['0.0.0.0/0'], hostnames: ['*'], ports: '*', audit: true },
  ],
  evidenceContract: {
    everyActionIsRecorded: true, packetTracesAreCausal: true, filesystemIsInodeBacked: true,
    appStateIsVfsBacked: true, surfaceCoverageRequired: true,
  },
} satisfies SeedEcosystemBlueprint);

export function validateSeed2026Blueprint(blueprint: SeedEcosystemBlueprint = seed2026Blueprint): string[] {
  const findings = validateSurfaceCoverage(appCatalog);
  const appById = new Map(appCatalog.map((app) => [app.id, app]));
  const computerIds = new Set<string>();
  const addresses = new Set<string>();
  for (const computer of blueprint.computers) {
    if (computerIds.has(computer.spec.id)) findings.push(`duplicate computer id: ${computer.spec.id}`);
    if (addresses.has(computer.spec.ipv4)) findings.push(`duplicate computer address: ${computer.spec.ipv4}`);
    computerIds.add(computer.spec.id); addresses.add(computer.spec.ipv4);
    if (computer.spec.os !== computer.profileId) findings.push(`${computer.spec.id}: profile and ComputerSpec OS disagree`);
    const profile = blueprint.operatingSystems[computer.profileId];
    if (computer.spec.shell !== profile.shell.default) findings.push(`${computer.spec.id}: shell disagrees with ${profile.id} profile`);
    if (computer.spec.displays.length && computer.systemAppIds.join('\0') !== profile.systemAppIds.join('\0')) {
      findings.push(`${computer.spec.id}: materialized system apps disagree with ${profile.id} profile`);
    }
    const installed = [...computer.systemAppIds, ...computer.thirdPartyAppIds];
    if (new Set(installed).size !== installed.length) findings.push(`${computer.spec.id}: duplicate installed application id`);
    for (const id of installed) {
      const app = appById.get(id);
      if (!app) findings.push(`${computer.spec.id}: unknown application ${id}`);
      else if (!app.supportedOS.includes(computer.spec.os)) findings.push(`${computer.spec.id}: ${id} does not support ${computer.spec.os}`);
      else if (computer.systemAppIds.includes(id) !== Boolean(app.system)) findings.push(`${computer.spec.id}: ${id} system/third-party classification disagrees with catalog`);
    }
  }
  for (const [os, profile] of Object.entries(blueprint.operatingSystems) as Array<[OSKind, OperatingSystemProfile]>) {
    for (const id of profile.systemAppIds) if (!appById.get(id)?.system) findings.push(`${os}: ${id} is not declared as a system app`);
  }
  const slack = blueprint.services.find((service) => service.id === 'slack');
  const teams = blueprint.services.find((service) => service.id === 'teams');
  if (!slack || !teams || slack.host === teams.host || slack.isolationDomain === teams.isolationDomain) findings.push('Slack and Teams must have separate hosts and isolation domains');
  const serviceIds = new Set<string>();
  const serviceOrigins = new Set<string>();
  for (const service of blueprint.services) {
    if (serviceIds.has(service.id)) findings.push(`duplicate service id: ${service.id}`);
    serviceIds.add(service.id);
    const origin = `${service.protocol}://${service.host}:${service.port}`;
    if (serviceOrigins.has(origin)) findings.push(`duplicate service origin: ${origin}`);
    serviceOrigins.add(origin);
    const hostComputer = blueprint.computers.find((computer) => computer.spec.id === service.computerId);
    if (!hostComputer) findings.push(`${service.id}: unknown host computer ${service.computerId}`);
    else if (hostComputer.spec.ipv4 !== service.ipv4) findings.push(`${service.id}: service address disagrees with ${service.computerId}`);
    if (service.kind === 'app-registry' && !service.targetOS) findings.push(`${service.id}: app registry has no target OS`);
  }
  for (const os of Object.keys(blueprint.operatingSystems) as OSKind[]) {
    if (!blueprint.services.some((service) => service.kind === 'app-registry' && service.targetOS === os)) findings.push(`${os}: no application registry service`);
  }
  const dns = blueprint.services.find((service) => service.kind === 'dns');
  if (!dns || dns.ipv4 !== blueprint.network.dns) findings.push('network DNS address has no matching DNS service');
  if (new Set(blueprint.gateways.map((gateway) => gateway.id)).size !== blueprint.gateways.length) findings.push('duplicate gateway id');
  if (new Set(applicationSurfaces.flatMap((surface) => surface.appIds)).size !== appCatalog.length) findings.push('surface/application cardinality mismatch');
  return findings;
}

export function assertSeed2026Blueprint(): void {
  const findings = validateSeed2026Blueprint();
  if (findings.length) throw new Error(`invalid seed-2026 blueprint:\n- ${findings.join('\n- ')}`);
}
