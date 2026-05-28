/**
 * WebSocket relay v2 + Transport adapter integration test.
 *
 * Spawns the relay server in-process on a random localhost port, opens
 * two transports against it, and exercises every routing target as
 * well as a binary-frame round-trip with a Float32Array payload.
 */

import { spawn, type ChildProcess } from "node:child_process";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type {
  GradMessage,
  PeerListUpdateMessage,
  ProtocolMessage,
} from "../../src/distributed/protocol/messages.ts";
import { WebSocketRelayTransport } from "../../src/distributed/transports/websocket-relay.ts";

const REPO_ROOT = path.resolve(__dirname, "../..");
const SERVER_SCRIPT = path.join(REPO_ROOT, "server/diloco-server-v2.cjs");

let serverProc: ChildProcess | null = null;
let serverUrl = "";

beforeAll(async () => {
  // Pick a free-ish high port; if it clashes the test fails fast.
  const port = 18443 + Math.floor(Math.random() * 100);
  serverUrl = `ws://127.0.0.1:${port}`;
  serverProc = spawn("node", [SERVER_SCRIPT], {
    env: { ...process.env, V2_PORT: String(port), CLUSTER_SIZE: "2" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  // Wait for the server to log its listen banner.
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("server start timeout")),
      5_000,
    );
    serverProc!.stdout!.on("data", (buf: Buffer) => {
      if (buf.toString().includes(`v2 server on :${port}`)) {
        clearTimeout(timer);
        resolve();
      }
    });
    serverProc!.on("error", reject);
  });
}, 10_000);

afterAll(() => {
  if (serverProc) {
    serverProc.kill("SIGTERM");
    serverProc = null;
  }
});

function drain(ms = 50): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

describe("WebSocketRelayTransport against diloco-server-v2", () => {
  it("two peers register and exchange a broadcast text message", async () => {
    const txA = await WebSocketRelayTransport.create({
      serverUrl,
      peerId: "a",
      log: () => {},
    });
    const txB = await WebSocketRelayTransport.create({
      serverUrl,
      peerId: "b",
      log: () => {},
    });

    const recvA: ProtocolMessage[] = [];
    const recvB: ProtocolMessage[] = [];
    txA.onReceive((m) => recvA.push(m));
    txB.onReceive((m) => recvB.push(m));

    await drain(100);

    // Both peers should have received a JoinAck.
    const ackA = recvA.find((m) => m.type === "join-ack");
    const ackB = recvB.find((m) => m.type === "join-ack");
    expect(ackA).toBeDefined();
    expect(ackB).toBeDefined();
    expect(ackA?.type === "join-ack" && ackA.peerId).toBe("a");
    expect(ackB?.type === "join-ack" && ackB.peerId).toBe("b");
    // CLUSTER_SIZE=2 → both in cluster 0; a is head, b is not.
    expect(ackA?.type === "join-ack" && ackA.isHead).toBe(true);
    expect(ackB?.type === "join-ack" && ackB.isHead).toBe(false);

    // A broadcasts. B should receive (A should not loop back).
    recvA.length = 0;
    recvB.length = 0;
    txA.send(
      { kind: "broadcast" },
      {
        type: "round-ready",
        peerId: "a",
        round: 7,
        anchor: 3,
        clusterId: 0,
      },
    );
    await drain(50);
    expect(recvA.length).toBe(0);
    expect(recvB.length).toBe(1);
    expect(recvB[0].type).toBe("round-ready");

    txA.close();
    txB.close();
  });

  it("binary frame with tensor payload roundtrips intact", async () => {
    const txA = await WebSocketRelayTransport.create({
      serverUrl,
      peerId: "ga",
      log: () => {},
    });
    const txB = await WebSocketRelayTransport.create({
      serverUrl,
      peerId: "gb",
      log: () => {},
    });

    const recvB: ProtocolMessage[] = [];
    txB.onReceive((m) => recvB.push(m));
    await drain(100);
    recvB.length = 0;

    // Build a payload with two tensors of different sizes.
    const t1 = new Float32Array([1, 2, 3, 4]);
    const t2 = new Float32Array(8);
    for (let i = 0; i < t2.length; i++) t2[i] = i * 0.5;

    const grad: GradMessage = {
      type: "grad",
      fromPeerId: "ga",
      round: 9,
      anchor: 4,
      kind: "peer-grad",
      peerCount: 1,
      payload: [t1, t2],
    };
    txA.send({ kind: "peer", peerId: "gb" }, grad);

    await drain(100);

    const got = recvB.find((m) => m.type === "grad") as GradMessage | undefined;
    expect(got).toBeDefined();
    if (!got) throw new Error("no grad received");
    expect(got.fromPeerId).toBe("ga");
    expect(got.round).toBe(9);
    expect(got.anchor).toBe(4);
    expect(got.kind).toBe("peer-grad");
    expect(got.peerCount).toBe(1);
    expect(got.payload.length).toBe(2);
    expect(Array.from(got.payload[0])).toEqual(Array.from(t1));
    expect(Array.from(got.payload[1])).toEqual(Array.from(t2));

    txA.close();
    txB.close();
  });

  it("cluster routing — message reaches same-cluster only", async () => {
    // CLUSTER_SIZE=2, so 4 peers form 2 clusters of 2.
    const txs = await Promise.all([
      WebSocketRelayTransport.create({ serverUrl, peerId: "c0a", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "c0b", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "c1a", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "c1b", log: () => {} }),
    ]);
    const recv = txs.map(() => [] as ProtocolMessage[]);
    txs.forEach((t, i) => t.onReceive((m) => recv[i].push(m)));
    await drain(150);

    // Wait for the most-recent peer-list to settle on all peers.
    const lastView = txs.map(
      (_, i) =>
        recv[i]
          .filter((m): m is PeerListUpdateMessage => m.type === "peer-list")
          .at(-1) ??
        (recv[i].find((m) => m.type === "join-ack") as
          | { peers?: { peerId: string; clusterId: number }[] }
          | undefined),
    );
    expect(lastView.every((v) => v && v.peers && v.peers.length === 4)).toBe(
      true,
    );

    recv.forEach((r) => (r.length = 0));

    // c0a sends to its cluster — c0b should receive, c1a/c1b should not.
    const c0aClusterId = (
      lastView[0]!.peers!.find((p) => p.peerId === "c0a")!
    ).clusterId;
    txs[0].send(
      { kind: "cluster", clusterId: c0aClusterId },
      {
        type: "round-ready",
        peerId: "c0a",
        round: 1,
        anchor: 0,
        clusterId: c0aClusterId,
      },
    );
    await drain(80);

    const c0bIdx = lastView[1]!.peers!.findIndex((p) => p.peerId === "c0b");
    const c0bClusterId = lastView[1]!.peers![c0bIdx].clusterId;

    if (c0bClusterId === c0aClusterId) {
      expect(recv[1].some((m) => m.type === "round-ready")).toBe(true);
    }
    // c1a and c1b should NOT have received it.
    for (let i = 2; i < 4; i++) {
      const c1Cluster = lastView[i]!.peers!.find(
        (p) => p.peerId === txs[i].peerId,
      )?.clusterId;
      if (c1Cluster !== c0aClusterId) {
        expect(recv[i].some((m) => m.type === "round-ready")).toBe(false);
      }
    }

    for (const t of txs) t.close();
  });

  it("heads routing — message reaches cluster heads only", async () => {
    const txs = await Promise.all([
      WebSocketRelayTransport.create({ serverUrl, peerId: "ha0", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "ha1", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "ha2", log: () => {} }),
      WebSocketRelayTransport.create({ serverUrl, peerId: "ha3", log: () => {} }),
    ]);
    const recv = txs.map(() => [] as ProtocolMessage[]);
    txs.forEach((t, i) => t.onReceive((m) => recv[i].push(m)));
    await drain(150);

    const peerList = recv[0]
      .filter((m): m is PeerListUpdateMessage => m.type === "peer-list")
      .at(-1);
    expect(peerList).toBeDefined();
    if (!peerList) throw new Error("no peer-list update");
    const heads = peerList.peers.filter((p) => p.isHead).map((p) => p.peerId);
    expect(heads.length).toBeGreaterThanOrEqual(2);

    // Pick a head and have it send to "heads"; other head should receive.
    const sender = txs.find((t) => heads.includes(t.peerId))!;
    recv.forEach((r) => (r.length = 0));
    sender.send(
      { kind: "heads" },
      {
        type: "round-ready",
        peerId: sender.peerId,
        round: 0,
        anchor: 0,
        clusterId: 0,
      },
    );
    await drain(80);

    for (let i = 0; i < txs.length; i++) {
      const isOtherHead =
        heads.includes(txs[i].peerId) && txs[i].peerId !== sender.peerId;
      const gotMsg = recv[i].some((m) => m.type === "round-ready");
      expect(gotMsg).toBe(isOtherHead);
    }

    for (const t of txs) t.close();
  });
});
