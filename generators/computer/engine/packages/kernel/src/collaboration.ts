import { randomUUID } from './determinism.js';
import type {
  CollaborationChannel,
  CollaborationMessage,
  CollaborationPollResult,
  CollaborationServiceId,
  CollaborationServiceSnapshot,
} from '@tcn-computer/protocol';

interface CollaborationServiceOptions {
  id: CollaborationServiceId;
  host: string;
  workspaceId: string;
  workspaceName: string;
  channels: CollaborationChannel[];
}

export interface NewCollaborationMessage {
  author: string;
  computerId: string;
  text: string;
  threadId?: string;
}

/**
 * Server-authoritative state for one collaboration product.
 *
 * Instances deliberately share no storage. Slack and Teams may happen to have
 * channels with the same human-readable name, but their workspaces, revisions,
 * messages, HTTP origins, and polling cursors remain independent.
 */
export class CollaborationService {
  readonly id: CollaborationServiceId;
  readonly host: string;
  readonly workspaceId: string;
  readonly workspaceName: string;
  private readonly channels = new Map<string, CollaborationChannel>();
  private readonly messages: CollaborationMessage[] = [];
  private revision = 0;

  constructor(options: CollaborationServiceOptions) {
    this.id = options.id;
    this.host = options.host;
    this.workspaceId = options.workspaceId;
    this.workspaceName = options.workspaceName;
    for (const channel of options.channels) this.channels.set(channel.id, structuredClone(channel));
  }

  seed(channelId: string, author: string, computerId: string, text: string): CollaborationMessage {
    return this.post(channelId, { author, computerId, text });
  }

  post(channelId: string, input: NewCollaborationMessage): CollaborationMessage {
    this.requireChannel(channelId);
    const text = input.text.trim();
    if (!text) throw new Error('message text must not be empty');
    if (text.length > 40_000) throw new Error('message exceeds 40000 characters');
    const message: CollaborationMessage = {
      id: randomUUID(),
      serviceId: this.id,
      workspaceId: this.workspaceId,
      channelId,
      sequence: ++this.revision,
      author: input.author.trim() || 'agent',
      computerId: input.computerId,
      text,
      at: new Date().toISOString(),
      threadId: input.threadId,
    };
    this.messages.push(message);
    return structuredClone(message);
  }

  poll(channelId: string, afterRevision = 0): CollaborationPollResult {
    this.requireChannel(channelId);
    return {
      serviceId: this.id,
      workspaceId: this.workspaceId,
      channelId,
      revision: this.revision,
      messages: this.messages
        .filter((message) => message.channelId === channelId && message.sequence > afterRevision)
        .map((message) => structuredClone(message)),
    };
  }

  snapshot(): CollaborationServiceSnapshot {
    return {
      id: this.id,
      productName: this.id === 'slack' ? 'Slack' : 'Microsoft Teams',
      host: this.host,
      workspaceId: this.workspaceId,
      workspaceName: this.workspaceName,
      revision: this.revision,
      channels: [...this.channels.values()].map((channel) => structuredClone(channel)),
      messages: this.messages.map((message) => structuredClone(message)),
    };
  }

  private requireChannel(channelId: string): CollaborationChannel {
    const channel = this.channels.get(channelId);
    if (!channel) throw new Error(`${this.id}: unknown channel ${channelId}`);
    return channel;
  }
}

export function createSeedCollaborationServices(hosts: Partial<Record<CollaborationServiceId, string>> = {}): Map<CollaborationServiceId, CollaborationService> {
  const sharedShape = [
    { id: 'general', name: 'general', displayName: 'General', memberCount: 12 },
    { id: 'agent-runs', name: 'agent-runs', displayName: 'Agent Runs', memberCount: 9 },
    { id: 'factory-floor', name: 'factory-floor', displayName: 'Factory Floor', memberCount: 7 },
  ];
  return new Map([
    ['slack', new CollaborationService({
      id: 'slack', host: hosts.slack ?? 'slack.seed.local', workspaceId: 'T-SEEDLAB', workspaceName: 'Seed Lab',
      channels: [...sharedShape, { id: 'random', name: 'random', displayName: 'Random', memberCount: 12 }],
    })],
    ['teams', new CollaborationService({
      id: 'teams', host: hosts.teams ?? 'teams.seed.local', workspaceId: 'team-seed-engineering', workspaceName: 'Seed Engineering',
      channels: sharedShape,
    })],
  ]);
}
