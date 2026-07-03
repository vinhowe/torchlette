/**
 * Gate: torchlette on a CALLER-OWNED GPUDevice (renderer interop path).
 *
 * Simulates the host-app flow end to end: the app acquires its own adapter,
 * creates its own device with webgpuDeviceRequirements() merged in, installs
 * its OWN onuncapturederror handler, then hands the device to torchlette.
 * Asserts: (1) compute is correct on the external device, (2) torchlette
 * reports the same device (buffers are shareable), (3) the app's error
 * handler survives init (chained, not clobbered), (4) destroyWebGPU leaves
 * the device alive and usable by the app, with the handler restored.
 *
 * Run: npx tsx tools/t-external-device.ts   (Node/Dawn; exits 0 on pass)
 */

import { create } from "webgpu";
import {
  destroyWebGPU,
  getWebGPUDevice,
  initWebGPU,
  webgpuDeviceRequirements,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

function fail(msg: string): never {
  console.error(`FAIL: ${msg}`);
  process.exit(1);
}

async function main() {
  // --- The "renderer" side: own adapter, own device, own error handler.
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) fail("no adapter");
  const reqs = webgpuDeviceRequirements(adapter as never);
  console.log(
    `requirements: features=[${reqs.requiredFeatures.join(",")}] ` +
      `maxBuf=${reqs.requiredLimits.maxBufferSize}`,
  );
  const device = await adapter.requestDevice({
    requiredFeatures: reqs.requiredFeatures as never,
    requiredLimits: reqs.requiredLimits,
  });
  const appHandler = () => {};
  (device as { onuncapturederror: unknown }).onuncapturederror = appHandler;

  // --- Hand it to torchlette.
  const ok = await initWebGPU({ device: device as never });
  if (!ok) fail("initWebGPU({device}) returned false");
  const ctx = getWebGPUDevice();
  if (ctx?.device !== (device as never)) {
    fail("torchlette is not running on the provided device");
  }
  const installed = (device as { onuncapturederror: unknown })
    .onuncapturederror;
  if (installed === appHandler) fail("torchlette did not install its handler");

  // --- Compute correctness on the shared device.
  const api = new Torchlette("webgpu");
  const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
  const b = api.tensorFromArray([1, 0, 0, 1, 1, 1], [3, 2]);
  const out = new Float32Array(await api.matmul(a, b).cpu());
  const expect = [4, 5, 10, 11];
  for (let i = 0; i < 4; i++) {
    if (Math.abs(out[i] - expect[i]) > 1e-5) {
      fail(`matmul wrong at ${i}: ${out[i]} != ${expect[i]}`);
    }
  }
  console.log("matmul on external device: correct");

  // --- Double-init with a DIFFERENT device must throw loudly.
  let threw = false;
  try {
    await initWebGPU({ device: {} as never });
  } catch {
    threw = true;
  }
  if (!threw) fail("re-init with a different device did not throw");

  // --- Teardown must NOT destroy the app's device, and must restore handler.
  destroyWebGPU();
  const restored = (device as { onuncapturederror: unknown })
    .onuncapturederror;
  if (restored !== appHandler) fail("app error handler not restored");
  // Device still usable by the "renderer": create + map a buffer.
  const probe = device.createBuffer({ size: 16, usage: 0x0001 | 0x0008 }); // MAP_READ|COPY_DST
  await probe.mapAsync(1 /* READ */);
  probe.unmap();
  probe.destroy();
  console.log("teardown: device survived, handler restored");

  console.log("EXTERNAL DEVICE GATE PASS");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAIL:", e);
  process.exit(1);
});
