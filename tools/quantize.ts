/**
 * Weight-only int8 quantization utility — moved to src/backend/quantize.ts
 * (phase 2, task #93: it earned generality via the browser loader). This
 * re-export keeps the phase-1 gate scripts (probe-quant-gemv.ts,
 * quant-lmhead-realweight.ts) importing from tools/ working unchanged.
 */
export {
  dequantizeToF32,
  f16BitsToF32,
  f32ToF16Bits,
  quantizeLinearWeight,
  type QuantizedWeight,
} from "../src/backend/quantize";
