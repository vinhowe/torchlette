import { describe, expect, it } from "vitest";
import {
  Engine,
  RngReplayExhaustedError,
  RngReplayMismatchError,
} from "../src";

describe("rng semantics", () => {
  it("assigns unique draw nonces for repeated random ops", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 1, seed: 123 });

    const first = engine._debug_random(7);
    const second = engine._debug_random(7);

    expect(first.drawNonce).toBe(1);
    expect(second.drawNonce).toBe(2);
    expect(first.value).not.toBe(second.value);
  });

  it("is deterministic for explicit draw nonces and advances the counter", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 2, seed: 9 });

    const first = engine._debug_random(3, 10);
    const second = engine._debug_random(3, 10);
    expect(first.value).toBe(second.value);
    expect(engine._debug_getRngDrawNonce()).toBe(10);

    const third = engine._debug_random(3);
    expect(third.drawNonce).toBe(11);
  });

  it("replays checkpointed draws without advancing persistent RNG state", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 0, seed: 42 });

    engine._debug_startCheckpointRecord();
    const first = engine._debug_random(10);
    const second = engine._debug_random(11);
    const recorded = engine._debug_finishCheckpointRecord();

    const counterAfterRecord = engine._debug_getRngDrawNonce();

    engine._debug_startCheckpointReplay(recorded);
    const replayFirst = engine._debug_random(10);
    const replaySecond = engine._debug_random(11);
    engine._debug_finishCheckpointReplay();

    const counterAfterReplay = engine._debug_getRngDrawNonce();

    expect(replayFirst.value).toBe(first.value);
    expect(replaySecond.value).toBe(second.value);
    expect(counterAfterReplay).toBe(counterAfterRecord);

    const next = engine._debug_random(12);
    expect(next.drawNonce).toBe(counterAfterRecord + 1);
  });

  it("throws on replay mismatch or exhaustion", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 0, seed: 1 });

    engine._debug_startCheckpointRecord();
    engine._debug_random(2);
    const recorded = engine._debug_finishCheckpointRecord();

    engine._debug_startCheckpointReplay(recorded);
    expect(() => engine._debug_random(3)).toThrow(RngReplayMismatchError);
    expect(() => engine._debug_random(2, 999)).toThrow(RngReplayMismatchError);
    engine._debug_finishCheckpointReplay();

    engine._debug_startCheckpointReplay(recorded);
    engine._debug_random(2);
    expect(() => engine._debug_random(2)).toThrow(RngReplayExhaustedError);
  });
});
