import type {
  AppLaunchRequest,
  AppManifest,
  AppRuntimeDescriptor,
  AppServiceContract,
  OSKind,
} from '@tcn-computer/protocol';

export type ApplicationDefinitionInput =
  Omit<AppManifest, 'operations' | 'serviceContracts' | 'runtime'> &
  Partial<Pick<AppManifest, 'operations' | 'serviceContracts' | 'runtime'>>;

export interface ApplicationPackageDescriptor {
  format: 'seed-app-package-v1';
  manifest: AppManifest;
  entryFile: string;
  stateFile: string;
  receiptNamespace: string;
}

const operationProfiles: Record<string, readonly string[]> = {
  'system://files': ['list', 'open', 'copy', 'move', 'rename', 'trash', 'restore', 'search'],
  'system://terminal': ['execute', 'interrupt', 'clear', 'history'],
  'system://settings': ['inspect', 'set-preference', 'set-network-policy'],
  'system://editor': ['open', 'edit', 'save', 'save-as', 'find'],
  'system://preview': ['open', 'zoom', 'rotate', 'annotate', 'export'],
  'system://photos': ['browse', 'import', 'favorite', 'edit', 'export'],
  'system://calculator': ['calculate', 'clear', 'copy-result'],
  'system://calendar': ['list-events', 'create-event', 'update-event', 'delete-event'],
  'system://mail': ['list-mailboxes', 'list-messages', 'read-message', 'compose', 'send', 'archive'],
  'system://app-store': ['browse', 'search', 'inspect', 'install', 'update', 'uninstall'],
  'app://browser': ['navigate', 'back', 'forward', 'reload', 'new-tab', 'close-tab', 'bookmark', 'download'],
  'app://slack': ['list-channels', 'poll-messages', 'send-message', 'reply-thread', 'add-reaction'],
  'app://teams': ['list-teams', 'list-channels', 'poll-messages', 'send-message', 'reply-thread', 'start-meeting'],
  'app://chatgpt': ['new-chat', 'send-message', 'open-project', 'attach-file', 'run-tool'],
  'app://vscode': ['open-folder', 'open-file', 'edit', 'save', 'search', 'run-task', 'source-control'],
  'app://wireshark': ['capture', 'stop-capture', 'filter', 'inspect-packet', 'export-capture'],
  'app://messages': ['list-conversations', 'send-message', 'add-reaction'],
  'app://calls': ['list-meetings', 'start-call', 'join-call', 'mute', 'set-video', 'share-screen', 'end-call'],
  'app://media': ['browse', 'open', 'play', 'pause', 'seek', 'queue', 'record', 'stop-recording'],
  'app://maps': ['search-place', 'route', 'save-place'],
  'app://tasks': ['list', 'create', 'update', 'complete', 'delete'],
  'app://canvas': ['new-document', 'open', 'draw', 'undo', 'redo', 'save', 'export'],
  'app://capture': ['capture-region', 'capture-window', 'annotate', 'save'],
  'app://processes': ['list-processes', 'inspect-process', 'terminate-process'],
  'app://packages': ['list', 'search', 'inspect', 'install', 'upgrade', 'remove'],
  'app://git': ['open-repository', 'status', 'stage', 'commit', 'branch', 'fetch', 'pull', 'push'],
  'app://containers': ['list-containers', 'list-images', 'start', 'stop', 'inspect', 'build'],
  'app://api-client': ['new-request', 'send-request', 'save-request', 'set-environment'],
  'app://design': ['new-document', 'open', 'select', 'transform', 'edit-properties', 'save', 'export'],
  'app://documents': ['new-document', 'open', 'edit', 'save', 'search', 'export'],
  'app://library': ['browse', 'install', 'launch', 'stop', 'update', 'uninstall'],
  'app://vault': ['unlock', 'lock', 'list-items', 'create-item', 'copy-field'],
  'app://database': ['connect', 'browse-schema', 'query', 'commit', 'rollback', 'export'],
};

export function operationsForEntrypoint(entrypoint: string): string[] {
  return [...(operationProfiles[entrypoint] ?? ['open', 'close'])];
}

export function defineApplication(input: ApplicationDefinitionInput): AppManifest {
  const runtime: AppRuntimeDescriptor = input.runtime ?? {
    kind: input.entrypoint.startsWith('system://') ? 'system-component' : 'seed-js',
    apiVersion: 1,
    entryFile: 'main.seed.js',
    stateSchema: `seed.app.${input.id}.v1`,
  };
  const manifest: AppManifest = {
    ...input,
    supportedOS: [...input.supportedOS],
    capabilities: [...input.capabilities],
    operations: input.operations ? [...input.operations] : operationsForEntrypoint(input.entrypoint),
    serviceContracts: input.serviceContracts?.map(cloneServiceContract) ?? [],
    runtime,
  };
  assertApplicationManifest(manifest);
  return Object.freeze(manifest);
}

export function defineApplicationSet(inputs: readonly ApplicationDefinitionInput[]): AppManifest[] {
  const manifests = inputs.map(defineApplication);
  const ids = new Set<string>();
  for (const manifest of manifests) {
    if (ids.has(manifest.id)) throw new Error(`duplicate application id: ${manifest.id}`);
    ids.add(manifest.id);
  }
  return manifests;
}

export function packageDescriptor(manifest: AppManifest): ApplicationPackageDescriptor {
  return Object.freeze({
    format: 'seed-app-package-v1',
    manifest,
    entryFile: manifest.runtime.entryFile,
    stateFile: 'state.json',
    receiptNamespace: `seed.app.${manifest.id}`,
  });
}

export function launchRequest(manifest: AppManifest, operation: string, payload: Record<string, unknown> = {}): AppLaunchRequest {
  if (!manifest.operations.includes(operation)) throw new Error(`${manifest.id} does not expose ${operation}`);
  return { operation, payload: structuredClone(payload) };
}

export function supportsOS(manifest: AppManifest, os: OSKind): boolean {
  return manifest.supportedOS.includes(os);
}

export function assertApplicationManifest(manifest: AppManifest): void {
  if (!/^[a-z0-9][a-z0-9-]*$/.test(manifest.id)) throw new Error(`invalid application id: ${manifest.id}`);
  if (!manifest.name.trim()) throw new Error(`${manifest.id}: name is required`);
  if (!manifest.entrypoint.includes('://')) throw new Error(`${manifest.id}: invalid entrypoint`);
  if (!manifest.supportedOS.length) throw new Error(`${manifest.id}: at least one OS is required`);
  if (!manifest.operations.length) throw new Error(`${manifest.id}: at least one operation is required`);
  const operations = new Set(manifest.operations);
  if (operations.size !== manifest.operations.length) throw new Error(`${manifest.id}: duplicate operations`);
  for (const contract of manifest.serviceContracts) {
    if (contract.required && contract.host === '') throw new Error(`${manifest.id}: required service host is empty`);
  }
}

function cloneServiceContract(contract: AppServiceContract): AppServiceContract {
  return { ...contract, operations: [...contract.operations] };
}
