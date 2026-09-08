import { createHash } from 'node:crypto';
import type { TrajectoryEvent } from '@tcn-computer/protocol';

export class TrajectoryRecorder {
  private sequence = 0;
  private readonly events: TrajectoryEvent[] = [];
  constructor(readonly runId: string) {}

  record(event: Omit<TrajectoryEvent, 'sequence' | 'at' | 'runId'>): TrajectoryEvent {
    const complete: TrajectoryEvent = { ...event, sequence: ++this.sequence, at: new Date().toISOString(), runId: this.runId };
    this.events.push(complete);
    return structuredClone(complete);
  }

  snapshot(): TrajectoryEvent[] { return structuredClone(this.events); }
  jsonl(): string { return this.events.map((event) => JSON.stringify(event)).join('\n'); }
  digest(): string { return createHash('sha256').update(this.jsonl()).digest('hex'); }
  get length(): number { return this.events.length; }
}
