/**
 * Test wrapper: launch the DiLoCo agent with a synthetic disconnect window
 * injected. While currentRound is in [DISCONNECT_FROM_ROUND, DISCONNECT_UNTIL_ROUND),
 * the agent silently drops every WebSocket message in both directions — as
 * if the relay went silent. Used to verify the principled fix recovers from
 * long disconnects without the production agent carrying any test-only code
 * on its critical path.
 *
 * Usage:
 *   DISCONNECT_FROM_ROUND=15 DISCONNECT_UNTIL_ROUND=35 \
 *   ROUNDS=50 STEPS=10 BATCH_SIZE=1 SEQ_LEN=256 \
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=...vk-shim \
 *   HF_DATASET=... HF_CONFIG=... HF_ROWS=... \
 *   npx tsx tools/diloco-test-disconnect.ts
 */
import { faultInject } from "./diloco-fault-inject";

const FROM = parseInt(process.env.DISCONNECT_FROM_ROUND ?? "-1", 10);
const UNTIL = parseInt(process.env.DISCONNECT_UNTIL_ROUND ?? "-1", 10);

if (FROM < 0 || UNTIL < 0 || UNTIL <= FROM) {
  console.error(
    `[diloco-test-disconnect] DISCONNECT_FROM_ROUND/DISCONNECT_UNTIL_ROUND must be set with UNTIL > FROM >= 0 (got FROM=${FROM}, UNTIL=${UNTIL})`,
  );
  process.exit(1);
}

const inWindow = (currentRound: number): boolean =>
  currentRound >= FROM && currentRound < UNTIL;

faultInject.shouldDropOut = inWindow;
faultInject.shouldDropIn = inWindow;

console.error(
  `[diloco-test-disconnect] Fault injection ACTIVE: dropping all messages for currentRound in [${FROM}, ${UNTIL})`,
);

// Dynamic import so the hook overrides above are in place before the agent's
// main() reads them when wiring conn.send and messageHandler.
await import("./diloco-webrtc-agent");
