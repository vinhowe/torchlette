/**
 * Transport abstraction for the DiLoCo barrier protocol.
 *
 * Production transports (WebRTC-via-relay, WebSocket) and test transports
 * (in-process bus) both implement this interface. The state machine and
 * agent glue depend only on Transport, never on a specific network layer.
 *
 * Routing semantics (what each SendTarget means) are defined by the
 * production relay's behavior; test transports must mimic them faithfully
 * or tests will lie about real behavior.
 */

import type { PeerId, ProtocolMessage, SendTarget } from "./messages.ts";

export interface Transport {
  /** The peer-id this transport is bound to (assigned by the coordinator). */
  readonly peerId: PeerId;

  /**
   * Send `message`, routed according to `target`:
   *   - broadcast      → all currently-connected peers except self
   *   - peer:id        → just that peer (no-op if disconnected)
   *   - cluster:id     → all peers in the named cluster except self
   *   - heads          → all cluster heads except self (used for
   *                      inter-cluster CH-to-CH exchange)
   *
   * Delivery is asynchronous; receivers see the message in a later
   * microtask. Multiple sends from the same call site arrive at any given
   * receiver in send order.
   */
  send(target: SendTarget, message: ProtocolMessage): void;

  /**
   * Subscribe to incoming messages. The same transport supports multiple
   * subscribers (e.g., state machine + debug logger). Returns an unsubscribe
   * function.
   */
  onReceive(handler: (message: ProtocolMessage) => void): () => void;

  /** Disconnect; subsequent sends are no-ops, no further receives fire. */
  close(): void;
}
