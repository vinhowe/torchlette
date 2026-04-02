// Minimal WebSocket relay for DiLoCo gradient exchange.
// Browser and Node agent both connect here. Messages are broadcast to all others.
const { WebSocketServer } = require("ws");
const wss = new WebSocketServer({ port: 9876 });
const clients = new Map();

wss.on("connection", (ws) => {
  let id = null;
  ws.on("message", (raw) => {
    const str = raw.toString();
    // First message = registration: just a string ID
    if (!id) {
      id = str;
      clients.set(id, ws);
      console.log(`+ ${id} (${clients.size} clients)`);
      ws.send(JSON.stringify({ type: "peers", peers: [...clients.keys()].filter(k => k !== id) }));
      // Notify others
      for (const [cid, c] of clients) {
        if (cid !== id && c.readyState === 1) {
          c.send(JSON.stringify({ type: "join", peer: id }));
        }
      }
      return;
    }
    // Subsequent messages: broadcast binary gradient data to all others
    for (const [cid, c] of clients) {
      if (cid !== id && c.readyState === 1) {
        c.send(raw);
      }
    }
  });
  ws.on("close", () => {
    if (id) {
      clients.delete(id);
      console.log(`- ${id} (${clients.size} clients)`);
      for (const [, c] of clients) {
        if (c.readyState === 1) c.send(JSON.stringify({ type: "leave", peer: id }));
      }
    }
  });
});
console.log("Gradient relay on ws://localhost:9876");
