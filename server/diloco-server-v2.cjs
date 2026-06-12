/**
 * DiLoCo Server v2 — envelope-routed, cluster-aware.
 *
 * Differences from v1 (diloco-server.cjs):
 *   - Cluster topology: assigns each peer a clusterId + isHead on register;
 *     fills clusters of size CLUSTER_SIZE (env), starts a new cluster when
 *     full. First peer in each cluster is head; on head departure the next
 *     peer in that cluster is promoted.
 *   - Native broadcast / peer / cluster / heads routing via envelope frames.
 *     The protocol's barrier semantics work directly against this server —
 *     no client-side fanout for cluster/heads.
 *   - peer-list broadcasts on every membership change carry the full
 *     authoritative swarm view + (added, removed) diff hints.
 *   - Listens on V2_PORT (default 8443) so it can run alongside the v1
 *     server on :443 during the transition.
 *
 * Wire format:
 *   - Text frames (small protocol messages): JSON {from, target, msg}.
 *     Also accepts plain JSON for legacy control: register, ping.
 *   - Binary frames (grad, F16W): [u32 envelope_len][envelope JSON utf-8]
 *     [opaque payload bytes]. Server reads only the envelope to route;
 *     forwards the original frame as-is.
 *
 * The server NEVER inspects the tensor payload. The envelope's `msg.type`
 * and `tensorShapes` belong to the client protocol.
 */

const { WebSocketServer } = require("ws");
const PORT = parseInt(process.env.V2_PORT || "8443", 10);
const CLUSTER_SIZE = parseInt(process.env.CLUSTER_SIZE || "8", 10);
const KEEPALIVE_MS = parseInt(process.env.KEEPALIVE_MS || "120000", 10);

const wss = new WebSocketServer({ port: PORT, maxPayload: 500 * 1024 * 1024 });

/** @type {Map<string, {ws: any, clusterId: number, isHead: boolean, model: string, lastSeen: number}>} */
const peers = new Map();

const stats = {
  totalEnvelopes: 0,
  totalBytes: 0,
  maxRound: 0,
  lossHistory: [],
  startTime: Date.now(),
};

function peerView() {
  return [...peers.entries()].map(([peerId, p]) => ({
    peerId,
    clusterId: p.clusterId,
    isHead: p.isHead,
  }));
}

function assignCluster() {
  // Count peers per cluster, find smallest non-full cluster; if all full,
  // start a fresh one. First peer assigned to a cluster becomes its head.
  const counts = new Map();
  let maxClusterId = -1;
  for (const p of peers.values()) {
    counts.set(p.clusterId, (counts.get(p.clusterId) || 0) + 1);
    if (p.clusterId > maxClusterId) maxClusterId = p.clusterId;
  }
  let clusterId = -1;
  let smallest = Infinity;
  for (const [cid, count] of counts) {
    if (count < CLUSTER_SIZE && count < smallest) {
      clusterId = cid;
      smallest = count;
    }
  }
  if (clusterId === -1) clusterId = maxClusterId + 1;
  const existingHead = [...peers.values()].some(
    (p) => p.clusterId === clusterId && p.isHead,
  );
  return { clusterId, isHead: !existingHead };
}

function broadcastJson(obj, excludePeerId) {
  const str = JSON.stringify(obj);
  for (const [pid, p] of peers) {
    if (pid !== excludePeerId && p.ws.readyState === 1) p.ws.send(str);
  }
}

function broadcastPeerList(added, removed) {
  broadcastJson({
    type: "peer-list",
    peers: peerView(),
    added,
    removed,
  });
}

function recipientsFor(target, senderPeerId) {
  /** @type {{ws: any, peerId: string}[]} */
  const out = [];
  switch (target?.kind) {
    case "broadcast":
      for (const [pid, p] of peers) {
        if (pid !== senderPeerId) out.push({ ws: p.ws, peerId: pid });
      }
      break;
    case "peer": {
      const p = peers.get(target.peerId);
      if (p && target.peerId !== senderPeerId) {
        out.push({ ws: p.ws, peerId: target.peerId });
      }
      break;
    }
    case "cluster":
      for (const [pid, p] of peers) {
        if (pid !== senderPeerId && p.clusterId === target.clusterId) {
          out.push({ ws: p.ws, peerId: pid });
        }
      }
      break;
    case "heads":
      for (const [pid, p] of peers) {
        if (pid !== senderPeerId && p.isHead) {
          out.push({ ws: p.ws, peerId: pid });
        }
      }
      break;
  }
  return out;
}

function forwardFrame(raw, target, senderPeerId, isBinary) {
  const recipients = recipientsFor(target, senderPeerId);
  for (const r of recipients) {
    if (r.ws.readyState === 1) {
      // Preserve the original frame's text/binary type. The ws library
      // treats Buffer as binary by default; for text frames we send the
      // UTF-8 string so the recipient sees a text frame.
      if (isBinary) r.ws.send(raw);
      else r.ws.send(raw.toString("utf8"));
    }
  }
  return recipients.length;
}

function trackStatsFromMsg(msg, senderPeerId) {
  if (!msg || typeof msg !== "object") return;
  stats.totalEnvelopes++;
  if (msg.type === "grad" && typeof msg.round === "number") {
    if (msg.round > stats.maxRound) stats.maxRound = msg.round;
  }
}

wss.on("connection", (ws) => {
  let peerId = null;

  ws.on("message", (raw, isBinary) => {
    // Bookkeeping: refresh lastSeen on every message.
    if (peerId) {
      const me = peers.get(peerId);
      if (me) me.lastSeen = Date.now();
    }

    // Binary path: envelope-prefixed frame. `isBinary` is the RFC 6455
    // opcode-derived flag from the ws library — text frames arrive as
    // Buffer too, so we rely on this rather than guessing by content.
    if (isBinary && Buffer.isBuffer(raw) && raw.length >= 4) {
      const envLen = raw.readUInt32LE(0);
      const headerEnd = 4 + envLen;
      if (envLen > 0 && envLen <= raw.length - 4) {
        let envelope = null;
        try {
          envelope = JSON.parse(raw.toString("utf8", 4, headerEnd));
        } catch {
          envelope = null;
        }
        if (envelope && envelope.target && peerId) {
          stats.totalBytes += raw.length;
          trackStatsFromMsg(envelope.msg, peerId);
          forwardFrame(raw, envelope.target, peerId, true);
        }
      }
      return;
    }

    // Text path: JSON. Could be (a) register, (b) ping, (c) protocol
    // envelope, (d) anything else (ignored).
    let text;
    try {
      text = JSON.parse(raw.toString());
    } catch {
      return;
    }

    // Registration must come first.
    if (!peerId) {
      if (text.type !== "register") return;
      peerId =
        text.peerId || "peer-" + Math.random().toString(36).slice(2, 10);
      // SUPERSEDE duplicate registrations: a second connection claiming an
      // existing peerId is either a legitimate reconnect (the transport
      // retries with the same id) or an operator error (two processes
      // launched with one identity — observed: both submit grads under one
      // name and the round accounting silently corrupts, loss=0 stats).
      // Either way the OLD socket must die, loudly.
      const existing = peers.get(peerId);
      if (existing) {
        console.log(
          `! ${peerId} re-registered — superseding previous connection (reconnect or duplicate launch)`,
        );
        try {
          existing.ws.close(4000, "superseded by new registration");
        } catch {}
        peers.delete(peerId);
      }
      const { clusterId, isHead } = assignCluster();
      const needsSync = peers.size > 0;
      peers.set(peerId, {
        ws,
        clusterId,
        isHead,
        model: text.model || "unknown",
        lastSeen: Date.now(),
      });
      ws.send(
        JSON.stringify({
          type: "registered",
          peerId,
          clusterId,
          isHead,
          peers: peerView(),
          needsSync,
        }),
      );
      broadcastPeerList([peerId], []);
      console.log(
        `+ ${peerId} cluster=${clusterId}${isHead ? "/head" : ""} (${peers.size} peers, ${needsSync ? "needsSync" : "first"})`,
      );
      return;
    }

    // Protocol envelope: {from, target, msg}.
    if (text.target && text.msg) {
      trackStatsFromMsg(text.msg, peerId);
      stats.totalBytes += raw.length;
      forwardFrame(raw, text.target, peerId, false);
      return;
    }

    // Control: ping.
    if (text.type === "ping") {
      ws.send(
        JSON.stringify({
          type: "pong",
          peers: peers.size,
          uptime_ms: Date.now() - stats.startTime,
          envelopes: stats.totalEnvelopes,
          bytes: stats.totalBytes,
          max_round: stats.maxRound,
        }),
      );
      return;
    }
  });

  ws.on("close", () => {
    if (!peerId) return;
    const peer = peers.get(peerId);
    if (!peer) return;
    peers.delete(peerId);
    // If this peer was the head of its cluster, promote a survivor.
    if (peer.isHead) {
      const survivors = [...peers.entries()]
        .filter(([, p]) => p.clusterId === peer.clusterId)
        .sort((a, b) => a[0].localeCompare(b[0]));
      if (survivors.length > 0) {
        survivors[0][1].isHead = true;
      }
    }
    broadcastPeerList([], [peerId]);
    console.log(`- ${peerId} (${peers.size} peers)`);
  });
});

setInterval(() => {
  const now = Date.now();
  for (const [pid, p] of peers) {
    if (now - p.lastSeen > KEEPALIVE_MS) {
      try {
        p.ws.terminate();
      } catch {}
      peers.delete(pid);
      if (p.isHead) {
        const survivors = [...peers.entries()]
          .filter(([, q]) => q.clusterId === p.clusterId)
          .sort((a, b) => a[0].localeCompare(b[0]));
        if (survivors.length > 0) survivors[0][1].isHead = true;
      }
      broadcastPeerList([], [pid]);
      console.log(`x ${pid} (stale)`);
    }
  }
}, Math.max(10000, KEEPALIVE_MS / 4));

console.log(
  `DiLoCo v2 server on :${PORT} (cluster_size=${CLUSTER_SIZE}, keepalive=${KEEPALIVE_MS}ms)`,
);
