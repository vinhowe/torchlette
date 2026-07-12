/**
 * Dropped-submit guard gate (task #94, item 3).
 *
 * A memory-pressured device (VkOOM) drops the submit an uncaptured error occurs
 * in — downstream reads then see stale / all-zero data and training silently
 * continues on garbage (the VULKAN_DEVICE_INDEX=1 incident). The onuncapturederror
 * handler counts these, but its in-callback throw does not reach the training
 * loop's control flow. assertNoDroppedSubmits() is the in-band detector called at
 * fence / readback points: under TORCHLETTE_STRICT_GPU it THROWS (naming device
 * pressure) when the uncaptured-error count grew since the last check — turning a
 * silent all-zero corruption into a loud, deterministic failure. It rides the
 * existing STRICT_GPU flag (no new env flag) and is inert otherwise.
 *
 * We simulate a dropped submit by bumping the counter (a real VkOOM is not
 * reproducible on demand) and assert the guard's throw/inert behavior.
 */

import { afterEach, describe, expect, it } from "vitest";
import {
  _simulateDroppedSubmitForTest,
  assertNoDroppedSubmits,
} from "../../src/backend/webgpu";

describe("dropped-submit guard (task #94, item 3)", () => {
  const prev = process.env.TORCHLETTE_STRICT_GPU;
  afterEach(() => {
    if (prev === undefined) delete process.env.TORCHLETTE_STRICT_GPU;
    else process.env.TORCHLETTE_STRICT_GPU = prev;
  });

  it("is inert when no submit was dropped (count unchanged)", () => {
    process.env.TORCHLETTE_STRICT_GPU = "1";
    // Sync the high-water mark first (any earlier test noise), then check clean.
    assertNoDroppedSubmits("sync");
    expect(() => assertNoDroppedSubmits("clean fence")).not.toThrow();
  });

  it("does NOT throw on a dropped submit without STRICT_GPU (default behavior)", () => {
    delete process.env.TORCHLETTE_STRICT_GPU;
    assertNoDroppedSubmits("sync");
    _simulateDroppedSubmitForTest();
    // Non-strict: silently advances the high-water mark (the existing
    // console.error already logged it), never throws.
    expect(() => assertNoDroppedSubmits("readback")).not.toThrow();
  });

  it("THROWS under STRICT_GPU when a submit was dropped, naming device pressure", () => {
    process.env.TORCHLETTE_STRICT_GPU = "1";
    assertNoDroppedSubmits("sync"); // clear high-water mark
    _simulateDroppedSubmitForTest();
    expect(() => assertNoDroppedSubmits("tensor readback")).toThrow(
      /DROPPED.*tensor readback|device (memory )?pressure|VkOOM/i,
    );
    // The high-water mark advanced — a subsequent clean check does not re-throw.
    expect(() => assertNoDroppedSubmits("next fence")).not.toThrow();
  });
});
