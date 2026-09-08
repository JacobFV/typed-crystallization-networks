import { defineApplication, type ApplicationDefinitionInput } from '@tcn-computer/app-sdk';
import type { AppManifest, AppServiceContract, OSKind } from '@tcn-computer/protocol';

export const allOperatingSystems: OSKind[] = ['macos', 'windows', 'ubuntu'];
export type CatalogAppInput = ApplicationDefinitionInput;

const cloudBackends: Record<string, { kind: AppServiceContract['kind']; host: string; auth: AppServiceContract['auth']; operations: string[] }> = {
  slack: { kind: 'collaboration', host: 'slack.seed.local', auth: 'session', operations: ['list-channels', 'poll-messages', 'send-message'] },
  teams: { kind: 'collaboration', host: 'teams.seed.local', auth: 'oauth', operations: ['list-teams', 'list-channels', 'poll-messages', 'send-message'] },
  chatgpt: { kind: 'model-api', host: 'chatgpt.seed.local', auth: 'session', operations: ['chat', 'projects', 'tools'] },
  'app-store': { kind: 'app-registry', host: 'appstore.seed.local', auth: 'session', operations: ['catalog', 'package', 'receipt'] },
  store: { kind: 'app-registry', host: 'store.seed.local', auth: 'session', operations: ['catalog', 'package', 'receipt'] },
  'app-center': { kind: 'app-registry', host: 'packages.seed.local', auth: 'none', operations: ['catalog', 'package', 'receipt'] },
  mail: { kind: 'mail', host: 'mail.seed.local', auth: 'session', operations: ['mailboxes', 'messages', 'send'] },
  outlook: { kind: 'mail', host: 'outlook.seed.local', auth: 'oauth', operations: ['mailboxes', 'messages', 'send', 'calendar'] },
  calendar: { kind: 'calendar', host: 'calendar.seed.local', auth: 'session', operations: ['events', 'create', 'update'] },
  messages: { kind: 'collaboration', host: 'messages.seed.local', auth: 'session', operations: ['conversations', 'send'] },
  facetime: { kind: 'collaboration', host: 'facetime.seed.local', auth: 'session', operations: ['calls', 'presence'] },
  music: { kind: 'media-catalog', host: 'music.seed.local', auth: 'session', operations: ['library', 'stream'] },
  maps: { kind: 'cloud-data', host: 'maps.seed.local', auth: 'none', operations: ['places', 'routes'] },
  figma: { kind: 'cloud-data', host: 'figma.seed.local', auth: 'oauth', operations: ['files', 'documents', 'presence'] },
  notion: { kind: 'cloud-data', host: 'notion.seed.local', auth: 'oauth', operations: ['pages', 'databases', 'search'] },
  linear: { kind: 'cloud-data', host: 'linear.seed.local', auth: 'oauth', operations: ['issues', 'projects', 'cycles'] },
  discord: { kind: 'collaboration', host: 'discord.seed.local', auth: 'session', operations: ['guilds', 'channels', 'messages'] },
  zoom: { kind: 'collaboration', host: 'zoom.seed.local', auth: 'oauth', operations: ['meetings', 'calls'] },
  spotify: { kind: 'media-catalog', host: 'spotify.seed.local', auth: 'oauth', operations: ['library', 'search', 'stream'] },
  steam: { kind: 'cloud-data', host: 'steam.seed.local', auth: 'session', operations: ['library', 'downloads', 'friends'] },
  bitwarden: { kind: 'identity', host: 'bitwarden.seed.local', auth: 'session', operations: ['sync', 'unlock'] },
  onepassword: { kind: 'identity', host: 'onepassword.seed.local', auth: 'session', operations: ['sync', 'unlock'] },
};

function serviceContractsFor(value: CatalogAppInput): AppServiceContract[] {
  const backend = cloudBackends[value.id];
  if (backend) return [{
    id: `${value.id}-backend`, kind: backend.kind, host: backend.host, protocol: 'https', port: 443,
    auth: backend.auth, required: true, operations: [...backend.operations],
  }];
  if (value.capabilities.includes('network')) return [{
    id: `${value.id}-network-client`, kind: 'virtual-network-client', host: '*', protocol: 'virtual', port: 0,
    auth: 'none', required: true, operations: ['dns-resolve', 'connect', 'request'],
  }];
  return [];
}

export function app(value: CatalogAppInput): AppManifest {
  return defineApplication({ ...value, serviceContracts: value.serviceContracts ?? serviceContractsFor(value) });
}
