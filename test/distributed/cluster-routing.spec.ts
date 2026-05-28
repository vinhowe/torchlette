/**
 * Cluster assignment + routing semantics for the in-process bus.
 *
 * The in-process bus is the only place where cluster routing rules are
 * actually executed in the test path (the production WebRTC relay
 * mirrors them server-side). If these tests pass, hierarchical
 * aggregation built on top of SendTarget will route correctly.
 */

import { describe, expect, it } from "vitest";
import type {
  PeerListUpdateMessage,
  ProtocolMessage,
} from "../../src/distributed/protocol/messages.ts";
import {
  FixedClusterAssigner,
  InProcessBus,
} from "../../src/distributed/transports/in-process.ts";

// Drain pending microtasks so the bus's queueMicrotask deliveries fire.
const drain = () => new Promise<void>((r) => setTimeout(r, 0));

describe("FixedClusterAssigner", () => {
  it("fills clusters in order up to size, then opens a new cluster", () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    bus.connect("a");
    bus.connect("b");
    bus.connect("c");
    bus.connect("d");
    const view = bus.peerView();
    const byId = Object.fromEntries(view.map((p) => [p.peerId, p]));
    expect(byId["a"].clusterId).toBe(0);
    expect(byId["b"].clusterId).toBe(0);
    expect(byId["c"].clusterId).toBe(1);
    expect(byId["d"].clusterId).toBe(1);
    expect(byId["a"].isHead).toBe(true);
    expect(byId["b"].isHead).toBe(false);
    expect(byId["c"].isHead).toBe(true);
    expect(byId["d"].isHead).toBe(false);
  });

  it("flat mode (cluster size Infinity) puts everyone in cluster 0", () => {
    const bus = new InProcessBus(new FixedClusterAssigner(Infinity));
    bus.connect("a");
    bus.connect("b");
    bus.connect("c");
    const view = bus.peerView();
    expect(view.every((p) => p.clusterId === 0)).toBe(true);
    expect(view.filter((p) => p.isHead).map((p) => p.peerId)).toEqual(["a"]);
  });
});

describe("InProcessBus routing", () => {
  it("delivers broadcast to all peers except sender", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    const tx = ["a", "b", "c", "d"].map((id) => bus.connect(id));
    const received: Record<string, ProtocolMessage[]> = {};
    for (const t of tx) {
      received[t.peerId] = [];
      t.onReceive((m) => received[t.peerId].push(m));
    }
    await drain();

    // Clear join-ack + peer-list deliveries that fired during setup.
    for (const id of Object.keys(received)) received[id] = [];

    tx[0].send(
      { kind: "broadcast" },
      { type: "leave", peerId: "a" },
    );
    await drain();

    expect(received["a"]).toEqual([]); // sender doesn't loop back
    expect(received["b"].length).toBe(1);
    expect(received["c"].length).toBe(1);
    expect(received["d"].length).toBe(1);
  });

  it("cluster: delivers only to same-cluster peers (not sender)", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    const tx = ["a", "b", "c", "d"].map((id) => bus.connect(id));
    const received: Record<string, ProtocolMessage[]> = {};
    for (const t of tx) {
      received[t.peerId] = [];
      t.onReceive((m) => received[t.peerId].push(m));
    }
    await drain();
    for (const id of Object.keys(received)) received[id] = [];

    // a is in cluster 0 with b; c, d are in cluster 1.
    tx[0].send(
      { kind: "cluster", clusterId: 0 },
      { type: "leave", peerId: "a" },
    );
    await drain();

    expect(received["a"]).toEqual([]);
    expect(received["b"].length).toBe(1);
    expect(received["c"]).toEqual([]);
    expect(received["d"]).toEqual([]);
  });

  it("heads: delivers only to cluster heads (not sender, not non-heads)", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    const tx = ["a", "b", "c", "d"].map((id) => bus.connect(id));
    const received: Record<string, ProtocolMessage[]> = {};
    for (const t of tx) {
      received[t.peerId] = [];
      t.onReceive((m) => received[t.peerId].push(m));
    }
    await drain();
    for (const id of Object.keys(received)) received[id] = [];

    // a (head of cluster 0) sends to all heads — should reach c (head of
    // cluster 1) only.
    tx[0].send(
      { kind: "heads" },
      { type: "leave", peerId: "a" },
    );
    await drain();

    expect(received["a"]).toEqual([]); // sender excluded
    expect(received["b"]).toEqual([]); // non-head
    expect(received["c"].length).toBe(1); // other head
    expect(received["d"]).toEqual([]); // non-head
  });

  it("peer: delivers only to the target", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    const tx = ["a", "b", "c"].map((id) => bus.connect(id));
    const received: Record<string, ProtocolMessage[]> = {};
    for (const t of tx) {
      received[t.peerId] = [];
      t.onReceive((m) => received[t.peerId].push(m));
    }
    await drain();
    for (const id of Object.keys(received)) received[id] = [];

    tx[0].send(
      { kind: "peer", peerId: "c" },
      { type: "leave", peerId: "a" },
    );
    await drain();

    expect(received["a"]).toEqual([]);
    expect(received["b"]).toEqual([]);
    expect(received["c"].length).toBe(1);
  });

  it("promotes a new head when the existing head disconnects", async () => {
    const bus = new InProcessBus(new FixedClusterAssigner(2));
    const txA = bus.connect("a");
    const txB = bus.connect("b");
    await drain();

    // Initially a is head of cluster 0.
    expect(bus.peerView().find((p) => p.peerId === "a")?.isHead).toBe(true);
    expect(bus.peerView().find((p) => p.peerId === "b")?.isHead).toBe(false);

    // Collect peer-list updates B receives after a disconnects.
    const updates: PeerListUpdateMessage[] = [];
    txB.onReceive((m) => {
      if (m.type === "peer-list") updates.push(m);
    });

    txA.close();
    await drain();

    // b should now be the head.
    const view = bus.peerView();
    expect(view.length).toBe(1);
    expect(view[0].peerId).toBe("b");
    expect(view[0].isHead).toBe(true);

    // And b received an update saying so.
    expect(updates.length).toBe(1);
    const lastUpdate = updates[updates.length - 1];
    expect(lastUpdate.removed).toContain("a");
    expect(lastUpdate.peers[0].isHead).toBe(true);
  });
});
