/**
 * Tests for storage tracker (src/engine/storage-tracker.ts)
 *
 * Covers: register/unregister, reachability tracking, GC via WeakRef,
 * scoped destruction, view aliasing, canSafelyRelease, stats/debugCounters.
 */

import { beforeEach, describe, expect, it } from "vitest";
import type { BackendTensor } from "../src/backend/types";
import type { StorageHandle } from "../src/graph/types";
import {
  canSafelyRelease,
  releaseBufferImmediate,
  storageTracker,
} from "../src/graph/storage-tracker";

/** Create a mock StorageHandle with a trackable destroy(). */
function mockStorage(
  id: number,
  opts?: { ownsBuffer?: boolean; baseStorageId?: number },
): StorageHandle & { destroyed: boolean } {
  const destroyed = { value: false };
  const tensor: BackendTensor = {
    dtype: "f32",
    shape: [4],
    ownsBuffer: opts?.ownsBuffer ?? true,
    destroy() {
      destroyed.value = true;
    },
  };
  const storage: StorageHandle & { destroyed: boolean } = {
    id,
    device: "cpu",
    backendTensor: tensor,
    baseStorageId: opts?.baseStorageId,
    get destroyed() {
      return destroyed.value;
    },
  };
  return storage;
}

describe("StorageTracker", () => {
  beforeEach(() => {
    storageTracker.reset();
    storageTracker.debugCounters(); // flush accumulated counters
  });

  // ========================================================================
  // register / unregister
  // ========================================================================
  describe("register / unregister", () => {
    it("tracks registered storages in stats", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      storageTracker.register(s1);
      storageTracker.register(s2);

      const stats = storageTracker.stats();
      expect(stats.totalStorages).toBe(2);
      expect(stats.reachableStorages).toBe(0);
      expect(stats.unreachableStorages).toBe(2);
    });

    it("unregister removes from all tracking", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      storageTracker.markReachable(1);
      storageTracker.unregister(1);

      const stats = storageTracker.stats();
      expect(stats.totalStorages).toBe(0);
      expect(stats.reachableStorages).toBe(0);
      expect(storageTracker.isReachable(1)).toBe(false);
    });
  });

  // ========================================================================
  // markReachable / markUnreachable / isReachable
  // ========================================================================
  describe("reachability", () => {
    it("markReachable makes a storage reachable", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      expect(storageTracker.isReachable(1)).toBe(false);

      storageTracker.markReachable(1);
      expect(storageTracker.isReachable(1)).toBe(true);
    });

    it("markUnreachable makes a storage unreachable", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      storageTracker.markReachable(1);
      expect(storageTracker.isReachable(1)).toBe(true);

      storageTracker.markUnreachable(1);
      expect(storageTracker.isReachable(1)).toBe(false);
    });

    it("getReachableIds returns correct set", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      const s3 = mockStorage(3);
      storageTracker.register(s1);
      storageTracker.register(s2);
      storageTracker.register(s3);
      storageTracker.markReachable(1);
      storageTracker.markReachable(3);

      const ids = storageTracker.getReachableIds();
      expect(ids).toEqual(new Set([1, 3]));
    });
  });

  // ========================================================================
  // destroyUnreachable
  // ========================================================================
  describe("destroyUnreachable", () => {
    it("destroys unreachable storages and returns count", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      const s3 = mockStorage(3);
      storageTracker.register(s1);
      storageTracker.register(s2);
      storageTracker.register(s3);
      storageTracker.markReachable(1); // keep alive

      // Mark all as recently unreachable to trigger scan
      storageTracker.markReachable(2);
      storageTracker.markUnreachable(2);
      storageTracker.markReachable(3);
      storageTracker.markUnreachable(3);

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(2);
      expect(s1.destroyed).toBe(false);
      expect(s2.destroyed).toBe(true);
      expect(s3.destroyed).toBe(true);
    });

    it("does not destroy reachable storages", () => {
      const s1 = mockStorage(1);
      storageTracker.register(s1);
      storageTracker.markReachable(1);

      // Need a recently unreachable entry to trigger scan
      const s2 = mockStorage(2);
      storageTracker.register(s2);
      storageTracker.markReachable(2);
      storageTracker.markUnreachable(2);

      storageTracker.destroyUnreachable();
      expect(s1.destroyed).toBe(false);
    });

    it("does not destroy view storages (ownsBuffer=false)", () => {
      const s = mockStorage(1, { ownsBuffer: false });
      storageTracker.register(s);

      // Trigger scan
      storageTracker.markReachable(1);
      storageTracker.markUnreachable(1);

      storageTracker.destroyUnreachable();
      // View's destroy() should not be called since ownsBuffer=false
      expect(s.destroyed).toBe(false);
    });
  });

  // ========================================================================
  // View aliasing: base storage kept alive
  // ========================================================================
  describe("view aliasing", () => {
    it("keeps base storage alive when a reachable view references it", () => {
      const base = mockStorage(10);
      const view = mockStorage(20, { ownsBuffer: false, baseStorageId: 10 });
      storageTracker.register(base);
      storageTracker.register(view);

      // Only the view is reachable, but base should be kept alive
      storageTracker.markReachable(20);

      // Trigger scan path
      storageTracker.markReachable(10);
      storageTracker.markUnreachable(10);

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(0); // base kept alive by view
      expect(base.destroyed).toBe(false);
    });

    it("destroys base storage when no reachable view references it", () => {
      const base = mockStorage(10);
      const view = mockStorage(20, { ownsBuffer: false, baseStorageId: 10 });
      storageTracker.register(base);
      storageTracker.register(view);

      // Neither is reachable → both should be destroyed
      // Trigger scan
      storageTracker.markReachable(10);
      storageTracker.markUnreachable(10);
      storageTracker.markReachable(20);
      storageTracker.markUnreachable(20);

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(2);
    });

    it("handles transitive view chains: view → view → base", () => {
      const base = mockStorage(10);
      const view1 = mockStorage(20, { ownsBuffer: false, baseStorageId: 10 });
      const view2 = mockStorage(30, { ownsBuffer: false, baseStorageId: 20 });
      storageTracker.register(base);
      storageTracker.register(view1);
      storageTracker.register(view2);

      // Only view2 is reachable — should transitively keep base alive
      storageTracker.markReachable(30);
      // Trigger scan on unreachable entries
      storageTracker.markReachable(10);
      storageTracker.markUnreachable(10);

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(0); // all kept alive transitively
    });
  });

  // ========================================================================
  // destroyUnreachableSince (scoped destruction)
  // ========================================================================
  describe("destroyUnreachableSince", () => {
    it("only destroys storages with id >= sinceId", () => {
      const old = mockStorage(5);
      const new1 = mockStorage(10);
      const new2 = mockStorage(15);
      storageTracker.register(old);
      storageTracker.register(new1);
      storageTracker.register(new2);

      // None are reachable
      const count = storageTracker.destroyUnreachableSince(10);
      expect(count).toBe(2);
      expect(old.destroyed).toBe(false);
      expect(new1.destroyed).toBe(true);
      expect(new2.destroyed).toBe(true);
    });

    it("respects reachability for storages >= sinceId", () => {
      const s1 = mockStorage(10);
      const s2 = mockStorage(20);
      storageTracker.register(s1);
      storageTracker.register(s2);
      storageTracker.markReachable(10);

      const count = storageTracker.destroyUnreachableSince(10);
      expect(count).toBe(1);
      expect(s1.destroyed).toBe(false);
      expect(s2.destroyed).toBe(true);
    });
  });

  // ========================================================================
  // canSafelyRelease
  // ========================================================================
  describe("canSafelyRelease", () => {
    it("returns true for unreachable storage with no views", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      const active = new Map<number, StorageHandle>([[1, s]]);

      expect(canSafelyRelease(s, active)).toBe(true);
    });

    it("returns false for reachable storage", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      storageTracker.markReachable(1);
      const active = new Map<number, StorageHandle>([[1, s]]);

      expect(canSafelyRelease(s, active)).toBe(false);
    });

    it("returns false when another active storage uses it as base", () => {
      const base = mockStorage(1);
      const view = mockStorage(2, { ownsBuffer: false, baseStorageId: 1 });
      storageTracker.register(base);
      storageTracker.register(view);
      const active = new Map<number, StorageHandle>([
        [1, base],
        [2, view],
      ]);

      expect(canSafelyRelease(base, active)).toBe(false);
    });

    it("returns true for base when view is no longer in active set", () => {
      const base = mockStorage(1);
      storageTracker.register(base);
      const active = new Map<number, StorageHandle>([[1, base]]);

      expect(canSafelyRelease(base, active)).toBe(true);
    });
  });

  // ========================================================================
  // releaseBufferImmediate
  // ========================================================================
  describe("releaseBufferImmediate", () => {
    it("destroys owned buffers and unregisters", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      expect(storageTracker.stats().totalStorages).toBe(1);

      releaseBufferImmediate(s);

      expect(s.destroyed).toBe(true);
      expect(storageTracker.stats().totalStorages).toBe(0);
    });

    it("does not destroy views (ownsBuffer=false)", () => {
      const s = mockStorage(1, { ownsBuffer: false });
      storageTracker.register(s);

      releaseBufferImmediate(s);

      expect(s.destroyed).toBe(false);
      // Still unregistered even for views
    });
  });

  // ========================================================================
  // debugCounters
  // ========================================================================
  describe("debugCounters", () => {
    it("tracks register, reachable, unreachable, destroy counts", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      storageTracker.register(s1);
      storageTracker.register(s2);
      storageTracker.markReachable(1);
      storageTracker.markReachable(2);
      storageTracker.markUnreachable(2);

      const counters = storageTracker.debugCounters();
      expect(counters.registered).toBe(2);
      expect(counters.reachable).toBe(2);
      expect(counters.unreachable).toBe(1);
    });

    it("resets counters after reading", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      storageTracker.debugCounters(); // consume

      const counters = storageTracker.debugCounters();
      expect(counters.registered).toBe(0);
      expect(counters.reachable).toBe(0);
      expect(counters.unreachable).toBe(0);
      expect(counters.destroyed).toBe(0);
    });
  });

  // ========================================================================
  // getStorage
  // ========================================================================
  describe("getStorage", () => {
    it("returns registered storage by ID", () => {
      const s = mockStorage(42);
      storageTracker.register(s);
      expect(storageTracker.getStorage(42)).toBe(s);
    });

    it("returns undefined for unregistered ID", () => {
      expect(storageTracker.getStorage(999)).toBeUndefined();
    });
  });
});
