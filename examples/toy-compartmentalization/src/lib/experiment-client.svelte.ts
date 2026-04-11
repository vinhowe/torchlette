/**
 * Reactive client for the compartmentalization-server WebSocket protocol.
 *
 * This is the browser-side counterpart to compartmentalization-server's
 * /ws endpoint. It speaks newline-delimited JSON over a single long-lived
 * WebSocket and exposes everything as Svelte 5 reactive state — components
 * just read `client.experiments[id]`, `client.connected`, `client.scripts`,
 * etc. and rerender when the server pushes events.
 *
 * Subscription model:
 *   - call `client.subscribe(id)` from a component's `onMount`
 *   - call `client.unsubscribe(id)` in the cleanup return
 *   - the per-experiment state (`client.experiments[id]`) is populated
 *     with the snapshot the server sends in the subscribe reply, then
 *     updated as live `metric` / `status` / `checkpoint` events arrive
 *   - the component reads the same `client.experiments[id]` whether the
 *     data came from history backfill or the live stream
 *
 * RPC pattern: each request gets a monotonically-increasing request_id,
 * the response carries the same id, we resolve a per-id pending Promise.
 * Events that aren't responses (metric, status, log, etc.) get routed by
 * `channel` field — `_global` events update the experiments map directly,
 * per-experiment events update `experiments[id]`.
 *
 * The client is a CLASS rather than module-level state because there are
 * a few places we want to instantiate it independently (testing, multiple
 * server endpoints) and Svelte 5 runes work fine inside a class as long
 * as the instance is created in a .svelte / .svelte.ts file.
 */

// ──────────────────────────────────────────────────────────────────────────
// Wire types — must match compartmentalization_server.{ipc, manager, server}
// ──────────────────────────────────────────────────────────────────────────

export type ParamSpec = {
  type?: 'number' | 'boolean' | 'select';
  default?: number | boolean | string;
  min?: number;
  max?: number;
  scale?: 'linear' | 'log';
  live?: boolean;
  choices?: string[];
  description?: string;
};

export type ScriptInfo = {
  name: string;
  description?: string;
  params: Record<string, ParamSpec>;
};

export type ExperimentStatus =
  | 'created'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'stopped'
  | 'failed';

export type ExperimentSummary = {
  id: string;
  script: string;
  description?: string;
  params: Record<string, number | string | boolean>;
  total_steps: number;
  step_count: number;
  status: ExperimentStatus;
  gpu: number | null;
  pid: number | null;
  created_at: string;
  updated_at: string;
  last_checkpoint_step: number;
  latest_metrics: Record<string, number>;
};

export type MetricEntry = {
  step: number;
  metrics: Record<string, number>;
  ts?: string;
};

/** What the client tracks per experiment. Mirrors `ExperimentSummary` but
 *  is reactive ($state) and adds the in-memory metric history that gets
 *  populated by the subscribe history-backfill plus the live stream. */
export type ExperimentRecord = ExperimentSummary & {
  history: MetricEntry[];
};

// ──────────────────────────────────────────────────────────────────────────
// Internal RPC bookkeeping
// ──────────────────────────────────────────────────────────────────────────

type Pending = {
  resolve: (payload: any) => void;
  reject: (error: Error) => void;
  // The expected reply type, e.g. "create_result" — we match it against
  // incoming messages alongside the request_id so a misrouted reply
  // (which shouldn't happen) gets a useful error.
  expectType: string;
};

// ──────────────────────────────────────────────────────────────────────────
// Client
// ──────────────────────────────────────────────────────────────────────────

export class ExperimentClient {
  /** Map of experiment id → reactive record. Use $state.raw on the map and
   *  $state on records: writes happen by reassigning the map (cheap, since
   *  it's a small dict) so Svelte triggers re-renders, and per-record
   *  updates use Object.assign on the value. */
  experiments = $state<Record<string, ExperimentRecord>>({});

  /** All registered scripts on the server. Populated lazily on first call
   *  to listScripts(); the create-experiment form depends on this. */
  scripts = $state<ScriptInfo[]>([]);

  /** Server connection status. UI uses this to show "connecting…", retry, etc. */
  connected = $state(false);
  lastError = $state<string | null>(null);

  private ws: WebSocket | null = null;
  private url: string;
  private nextRequestId = 1;
  private pending = new Map<number, Pending>();
  /** Set of experiment ids we want to be subscribed to. The client
   *  re-establishes these subscriptions automatically after a reconnect. */
  private wantedSubscriptions = new Set<string>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionallyClosed = false;

  constructor(url: string) {
    this.url = url;
  }

  // ── lifecycle ──

  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.intentionallyClosed = false;
    try {
      this.ws = new WebSocket(this.url);
    } catch (e: any) {
      this.lastError = e?.message ?? String(e);
      this.scheduleReconnect();
      return;
    }

    this.ws.addEventListener('open', () => {
      this.connected = true;
      this.lastError = null;
      // Refresh server state on every (re)connect — list_scripts is
      // basically free and the experiments list may have changed while
      // we were disconnected.
      void this.listScripts().catch(() => {});
      void this.list().catch(() => {});
      // Re-subscribe to anything the user was watching before the drop.
      for (const id of this.wantedSubscriptions) {
        void this.subscribe(id, /* re-establish */ true).catch(() => {});
      }
    });
    this.ws.addEventListener('message', (ev) => this.onMessage(ev));
    this.ws.addEventListener('close', () => {
      this.connected = false;
      this.failAllPending(new Error('connection closed'));
      if (!this.intentionallyClosed) this.scheduleReconnect();
    });
    this.ws.addEventListener('error', () => {
      // Browsers don't give us useful error info from the WS error event.
      // Surface a generic message; the close handler will follow up.
      this.lastError = 'websocket error';
    });
  }

  close(): void {
    this.intentionallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.failAllPending(new Error('client closed'));
    this.ws?.close();
    this.ws = null;
    this.connected = false;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.intentionallyClosed) return;
    // Constant 2s — fancier backoff is overkill for a single dev server.
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  private failAllPending(err: Error): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  // ── RPC dispatch ──

  /** Send a request and await its matching reply. Reply matching is by
   *  request_id; the type field is checked as a sanity guard. */
  private send<T>(type: string, payload: Record<string, any> = {}): Promise<T> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('not connected'));
    }
    const requestId = this.nextRequestId++;
    const expectType = `${type}_result`;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject, expectType });
      try {
        this.ws!.send(JSON.stringify({ type, request_id: requestId, ...payload }));
      } catch (e: any) {
        this.pending.delete(requestId);
        reject(e);
      }
    });
  }

  private onMessage(ev: MessageEvent): void {
    let msg: any;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    // RPC reply path: has a request_id we're awaiting.
    if (typeof msg.request_id === 'number') {
      const pending = this.pending.get(msg.request_id);
      if (pending) {
        this.pending.delete(msg.request_id);
        if (msg.error) {
          pending.reject(new Error(msg.error));
        } else if (msg.type !== pending.expectType) {
          pending.reject(new Error(`expected ${pending.expectType}, got ${msg.type}`));
        } else {
          pending.resolve(msg);
        }
        return;
      }
    }
    // Streaming path: events tagged with a `channel` field.
    if (typeof msg.channel === 'string') {
      this.handleStreamEvent(msg.channel, msg);
      return;
    }
    // Unprompted error.
    if (msg.type === 'error') {
      this.lastError = msg.message ?? 'unknown error';
    }
  }

  private handleStreamEvent(channel: string, ev: any): void {
    if (channel === '_global') {
      // Lifecycle events for the experiment list view.
      if (ev.type === 'created' || ev.type === 'updated') {
        const summary: ExperimentSummary = ev.experiment;
        const existing = this.experiments[summary.id];
        if (existing) {
          // Preserve any locally-accumulated history; just sync metadata.
          Object.assign(existing, summary);
          this.experiments = { ...this.experiments };
        } else {
          this.experiments = {
            ...this.experiments,
            [summary.id]: { ...summary, history: [] },
          };
        }
      } else if (ev.type === 'deleted') {
        const next = { ...this.experiments };
        delete next[ev.id];
        this.experiments = next;
      }
      return;
    }
    // Per-experiment event.
    const rec = this.experiments[channel];
    if (!rec) return;
    if (ev.type === 'metric') {
      rec.history.push({ step: ev.step, metrics: ev.metrics });
      rec.step_count = ev.step;
      rec.latest_metrics = ev.metrics;
      // No need to reassign the map: history is a $state array on a $state
      // record. But to be safe — Svelte's deep reactivity tracks this fine
      // when the parent is $state and we mutate in place, but mixing with
      // top-level map reassignment confuses the DX. Reassign for clarity.
      this.experiments = { ...this.experiments };
    } else if (ev.type === 'checkpoint') {
      rec.last_checkpoint_step = ev.step;
      this.experiments = { ...this.experiments };
    } else if (ev.type === 'status') {
      rec.status = ev.status;
      this.experiments = { ...this.experiments };
    } else if (ev.type === 'ready') {
      rec.status = 'running';
      rec.step_count = ev.step;
      this.experiments = { ...this.experiments };
    } else if (ev.type === 'done') {
      rec.status = 'stopped';
      rec.step_count = ev.step;
      this.experiments = { ...this.experiments };
    } else if (ev.type === 'log') {
      // Logs aren't persisted client-side in v0.1; could push to a ring
      // buffer here if we want a log pane.
    } else if (ev.type === 'error') {
      this.lastError = `${channel}: ${ev.message}`;
      rec.status = 'failed';
      this.experiments = { ...this.experiments };
    }
  }

  // ── public API: thin wrappers around send() ──

  async list(): Promise<ExperimentSummary[]> {
    const reply = await this.send<{ experiments: ExperimentSummary[] }>('list');
    // Seed the reactive map (preserving any history we might already have).
    const next: Record<string, ExperimentRecord> = {};
    for (const exp of reply.experiments) {
      const existing = this.experiments[exp.id];
      next[exp.id] = existing
        ? { ...exp, history: existing.history }
        : { ...exp, history: [] };
    }
    this.experiments = next;
    return reply.experiments;
  }

  async listScripts(): Promise<ScriptInfo[]> {
    const reply = await this.send<{ scripts: ScriptInfo[] }>('list_scripts');
    this.scripts = reply.scripts;
    return reply.scripts;
  }

  async create(opts: {
    script: string;
    params?: Record<string, number | string | boolean>;
    total_steps: number;
    description?: string;
  }): Promise<string> {
    const reply = await this.send<{ id: string }>('create', opts);
    return reply.id;
  }

  async subscribe(id: string, _reestablish = false): Promise<ExperimentRecord> {
    this.wantedSubscriptions.add(id);
    const reply = await this.send<{
      experiment: ExperimentSummary;
      history: MetricEntry[];
    }>('subscribe', { id });
    const existing = this.experiments[id];
    const rec: ExperimentRecord = {
      ...reply.experiment,
      // Replace history with the server's authoritative version. Live
      // events that arrived between the subscribe RPC and the reply are
      // appended afterward by handleStreamEvent.
      history: reply.history,
    };
    this.experiments = { ...this.experiments, [id]: rec };
    return rec;
  }

  async unsubscribe(id: string): Promise<void> {
    this.wantedSubscriptions.delete(id);
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        await this.send('unsubscribe', { id });
      } catch {
        // tolerate — connection may have dropped
      }
    }
  }

  async setParam(
    id: string,
    key: string,
    value: number | boolean | string,
  ): Promise<void> {
    await this.send('set_param', { id, key, value });
    // Optimistically update local state so sliders feel snappy. The server
    // will eventually echo back via an `updated` global event with the
    // canonical params dict; that overwrites this.
    const rec = this.experiments[id];
    if (rec) {
      rec.params = { ...rec.params, [key]: value };
      this.experiments = { ...this.experiments };
    }
  }

  async stop(id: string): Promise<void> {
    await this.send('stop', { id });
  }

  async deleteExperiment(id: string): Promise<void> {
    await this.send('delete', { id });
  }
}
