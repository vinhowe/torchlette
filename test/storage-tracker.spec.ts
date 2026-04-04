/**
 * Tests for storage tracker (src/graph/storage-tracker.ts)
 *
 * With the rc-based system, liveness is determined by rcGet(id) > 0.
 * Views keep their base alive via rcRetain on the base; no separate
 * "needed by views" walk is required.
 */

import { beforeEach, describe, expect, it } from "vitest";
import type { BackendTensor } from "../src/backend/types";
import { rcRelease, rcReset, rcRetain } from "../src/graph/refcount";
import {
  canSafelyRelease,
  releaseBufferImmediate,
  storageTracker,
} from "../src/graph/storage-tracker";
import type { StorageHandle } from "../src/graph/types";

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
    rcReset();
  });

  describe("register / unregister", () => {
    it("tracks registered storages in stats", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      storageTracker.register(s1);
      storageTracker.register(s2);
      // Simulate tensor retain
      rcRetain(1, "test");
      expect(storageTracker.stats().totalStorages).toBe(2);
      expect(storageTracker.stats().reachableStorages).toBe(1);
    });

    it("unregister removes from tracking", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      storageTracker.unregister(1);
      expect(storageTracker.stats().totalStorages).toBe(0);
    });
  });

  describe("destroyUnreachable", () => {
    it("destroys storages with rc <= 0 and returns count", () => {
      const s1 = mockStorage(1);
      const s2 = mockStorage(2);
      const s3 = mockStorage(3);
      storageTracker.register(s1);
      storageTracker.register(s2);
      storageTracker.register(s3);
      rcRetain(1, "live"); // s1 is live, others stay at rc=0

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(2);
      expect(s1.destroyed).toBe(false);
      expect(s2.destroyed).toBe(true);
      expect(s3.destroyed).toBe(true);
    });

    it("does not destroy live storages", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      rcRetain(1, "live");
      storageTracker.destroyUnreachable();
      expect(s.destroyed).toBe(false);
    });

    it("does not destroy view storages (ownsBuffer=false) even with rc<=0", () => {
      const s = mockStorage(1, { ownsBuffer: false });
      storageTracker.register(s);
      storageTracker.destroyUnreachable();
      // View is unregistered but destroy() not called (doesn't own buffer)
      expect(s.destroyed).toBe(false);
      expect(storageTracker.stats().totalStorages).toBe(0);
    });
  });

  describe("view aliasing via refcount", () => {
    it("keeps base alive when a view retains it", () => {
      const base = mockStorage(10);
      const view = mockStorage(20, { ownsBuffer: false, baseStorageId: 10 });
      storageTracker.register(base);
      storageTracker.register(view);
      // Simulate wrapResultAsStorage: view retains base
      rcRetain(10, "view.baseStorageId");
      // View is retained by a tensor
      rcRetain(20, "tensor");

      const count = storageTracker.destroyUnreachable();
      expect(count).toBe(0);
      expect(base.destroyed).toBe(false);
    });

    it("destroys base on next call after view releases its ref", () => {
      const base = mockStorage(10);
      const view = mockStorage(20, { ownsBuffer: false, baseStorageId: 10 });
      storageTracker.register(base);
      storageTracker.register(view);
      rcRetain(10, "view.baseStorageId");
      // View has rc=0, base has rc=1 (from view retain)

      // First call: destroys view, which releases base → base rc=0
      // But base wasn't in the toDestroy list (collected before view was processed)
      expect(storageTracker.destroyUnreachable()).toBe(1);
      expect(base.destroyed).toBe(false);
      expect(storageTracker.stats().totalStorages).toBe(1); // only base remains

      // Second call: base now has rc=0 and is destroyed
      expect(storageTracker.destroyUnreachable()).toBe(1);
      expect(base.destroyed).toBe(true);
    });
  });

  describe("destroyUnreachableSince", () => {
    it("only destroys storages with id >= sinceId", () => {
      const old = mockStorage(5);
      const new1 = mockStorage(10);
      const new2 = mockStorage(15);
      storageTracker.register(old);
      storageTracker.register(new1);
      storageTracker.register(new2);
      // All at rc=0 → all dead
      const count = storageTracker.destroyUnreachableSince(10);
      expect(count).toBe(2);
      expect(old.destroyed).toBe(false);
      expect(new1.destroyed).toBe(true);
      expect(new2.destroyed).toBe(true);
    });

    it("respects rc for storages >= sinceId", () => {
      const s1 = mockStorage(10);
      const s2 = mockStorage(20);
      storageTracker.register(s1);
      storageTracker.register(s2);
      rcRetain(10, "live");
      const count = storageTracker.destroyUnreachableSince(10);
      expect(count).toBe(1);
      expect(s1.destroyed).toBe(false);
      expect(s2.destroyed).toBe(true);
    });
  });

  describe("canSafelyRelease", () => {
    it("returns true for dead storage (rc <= 0) with no views", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      const active = new Map<number, StorageHandle>([[1, s]]);
      expect(canSafelyRelease(s, active)).toBe(true);
    });

    it("returns false for live storage (rc > 0)", () => {
      const s = mockStorage(1);
      storageTracker.register(s);
      rcRetain(1, "live");
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

    it("returns true for base when view is no longer active", () => {
      const base = mockStorage(1);
      storageTracker.register(base);
      const active = new Map<number, StorageHandle>([[1, base]]);
      expect(canSafelyRelease(base, active)).toBe(true);
    });
  });

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
    });

    it("releases base ref when releasing an owned view (ownsBuffer=true, baseStorageId set)", () => {
      const base = mockStorage(1);
      const viewOwned = mockStorage(2, { baseStorageId: 1 });
      storageTracker.register(base);
      storageTracker.register(viewOwned);
      rcRetain(1, "view.baseStorageId");
      releaseBufferImmediate(viewOwned);
      // After releasing viewOwned, base's rc should be back to 0
      expect(viewOwned.destroyed).toBe(true);
    });
  });

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
