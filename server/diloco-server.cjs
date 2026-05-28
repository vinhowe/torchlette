process.on("uncaughtException", (e) => console.log("UNCAUGHT:", e.message));
process.on("unhandledRejection", (e) => console.log("UNHANDLED:", e));

const { WebSocketServer } = require("ws");
const PORT = 443;
const K = 2;
const wss = new WebSocketServer({ port: PORT, maxPayload: 500 * 1024 * 1024 });

const peers = new Map();

// Global training stats (persisted across peer connections)
const globalStats = {
  totalTokens: 0,
  totalExchanges: 0,
  maxRound: 0,
  startTime: Date.now(),
  totalPeersEver: 0,
  lossHistory: [],  // [{round, loss, peerId, time}]
};

function broadcast(msg, exclude) {
  const str = JSON.stringify(msg);
  for (const [id, p] of peers) {
    if (id !== exclude && p.ws.readyState === 1) p.ws.send(str);
  }
}

function pickNeighbors(peerId, k) {
  const candidates = [...peers.keys()].filter(id => id !== peerId);
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  return candidates.slice(0, k);
}

wss.on("connection", (ws) => {
  let peerId = null;

  ws.on("message", (raw) => {
    // Registration
    if (!peerId) {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.type === "register") {
          peerId = msg.peerId || ("peer-" + Math.random().toString(36).slice(2, 8));
          const needsSync = peers.size > 0; // need weights if joining late
          peers.set(peerId, { ws, model: msg.model || "gpt2-124m", round: 0, lastSeen: Date.now() });
          
          ws.send(JSON.stringify({ type: "registered", peerId, peers: peers.size, needsSync }));
          broadcast({ type: "peer-joined", peerId, peers: peers.size }, peerId);
          globalStats.totalPeersEver++;
          console.log("+ " + peerId + " (" + peers.size + " peers)" + (needsSync ? " [needs sync]" : ""));
          
          // If new peer needs weights, ask the oldest peer to send them
          if (needsSync) {
            const source = [...peers.entries()].find(([id]) => id !== peerId);
            if (source) {
              source[1].ws.send(JSON.stringify({ type: "send-weights", target: peerId }));
              console.log("  asking " + source[0] + " to send weights to " + peerId);
            }
          }
          return;
        }
      } catch {}
      // Legacy plain string registration
      peerId = raw.toString();
      peers.set(peerId, { ws, model: "gpt2-124m", round: 0, lastSeen: Date.now() });
      ws.send(JSON.stringify({ type: "registered", peerId, peers: peers.size, needsSync: false }));
      broadcast({ type: "peer-joined", peerId, peers: peers.size }, peerId);
      console.log("+ " + peerId + " (" + peers.size + ")");
      return;
    }

    const peer = peers.get(peerId);
    if (peer) peer.lastSeen = Date.now();

    // Binary data — check for weight sync prefix or gradient data
    if (Buffer.isBuffer(raw) && raw.length > 1000) {
      // Check if this is a weight sync (first 4 bytes = "WGHT")
      if (raw.length > 4 && (raw.toString("utf8", 0, 4) === "WGHT" || raw.toString("utf8", 0, 4) === "DLTA" || raw.toString("utf8", 0, 4) === "F16W")) {
        // Weight sync — forward to specific target
        const targetIdLen = raw.readUInt16LE(4);
        const targetId = raw.toString("utf8", 6, 6 + targetIdLen);
        const weightData = raw.slice(6 + targetIdLen);
        const target = peers.get(targetId);
        if (target && target.ws.readyState === 1) {
          // Send with WGHT prefix so receiver knows it's weights
          const header = Buffer.alloc(4);
          header.write(raw.toString("utf8", 0, 4));
          target.ws.send(Buffer.concat([header, weightData]));
          console.log("  weights " + peerId + " → " + targetId + " (" + (weightData.length / 1024 / 1024).toFixed(1) + "MB)");
        }
        return;
      }
      
      // Regular gradient blob — forward to neighbors
      const neighbors = peer._neighbors || [];
      let sent = 0;
      for (const nid of neighbors) {
        const n = peers.get(nid);
        if (n && n.ws.readyState === 1) { n.ws.send(raw); sent++; }
      }
      console.log("  " + peerId + " → " + sent + "/" + neighbors.length + " (" + (raw.length / 1024 / 1024).toFixed(1) + "MB)");
      // Track global stats from GRAD header (12 bytes: GRAD + u32 tokens + u32 round)
      if (raw.length > 12 && raw.toString("utf8", 0, 4) === "GRAD") {
        const toks = raw.readUInt32LE(4);
        const rnd = raw.readUInt32LE(8);
        globalStats.totalTokens += toks;
        globalStats.totalExchanges++;
        if (rnd > globalStats.maxRound) globalStats.maxRound = rnd;
        if (raw.length > 16) {
          const lossVal = raw.readFloatLE(12);
          if (isFinite(lossVal)) {
            globalStats.lossHistory.push({ round: rnd, loss: +lossVal.toFixed(4), peer: peerId, t: Date.now() });
            if (globalStats.lossHistory.length > 500) globalStats.lossHistory = globalStats.lossHistory.slice(-500);
          }
        }
      }
      return;
    }

    // JSON control messages
    try {
      const msg = JSON.parse(raw.toString());
      if (msg.type === "request-neighbors") {
        const neighbors = pickNeighbors(peerId, Math.min(K, peers.size - 1));
        if (peer) peer._neighbors = neighbors;
        ws.send(JSON.stringify({ type: "neighbors", round: msg.round, neighbors, total: peers.size }));
        console.log("  " + peerId + " round " + msg.round + ": [" + neighbors.join(", ") + "]");
      }
      if (msg.type === "request-weights") {
        // Browser explicitly requesting weights (connected before agent)
        const source = [...peers.entries()].find(([id]) => id !== peerId);
        if (source) {
          source[1].ws.send(JSON.stringify({ type: "send-weights", target: peerId }));
          console.log("  " + peerId + " requested weights from " + source[0]);
        }
      }
            if (msg.type === "signal") {
        // Forward WebRTC signaling to target peer
        const target = peers.get(msg.target);
        if (target && target.ws.readyState === 1) {
          target.ws.send(JSON.stringify({ type: "signal", from: peerId, signal: msg.signal }));
        }
      }
      if (msg.type === "ping") {
        ws.send(JSON.stringify({ type: "pong", peers: peers.size, stats: globalStats }));
      }
    } catch {
      const neighbors = peer?._neighbors || [];
      for (const nid of neighbors) {
        const n = peers.get(nid);
        if (n && n.ws.readyState === 1) n.ws.send(raw);
      }
    }
  });

  ws.on("close", () => {
    if (peerId) {
      peers.delete(peerId);
      broadcast({ type: "peer-left", peerId, peers: peers.size });
      console.log("- " + peerId + " (" + peers.size + ")");
    }
  });
});

setInterval(() => {
  const now = Date.now();
  for (const [id, p] of peers) {
    if (now - p.lastSeen > 120000) {
      p.ws.terminate();
      peers.delete(id);
      console.log("x " + id + " (stale)");
    }
  }
}, 60000);

console.log("DiLoCo server on :" + PORT + " (K=" + K + ")");
