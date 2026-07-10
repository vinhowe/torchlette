/**
 * Node loader for the flat-.bin SAE format produced by convert-npz.py.
 * Reads the sae.json manifest + the five f32 .bin files into an SAEParams.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { SAEConfig, SAEParams } from "./sae";

export type SAEManifest = {
  layer: number;
  width: number;
  l0: number;
  dModel: number;
  numFeatures: number;
  neuronpediaSaeId: string;
  files: Record<"W_enc" | "b_enc" | "W_dec" | "b_dec" | "threshold", string>;
};

function readF32(file: string): Float32Array {
  const buf = fs.readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function loadSAEFromDir(dir: string): {
  config: SAEConfig;
  params: SAEParams;
  manifest: SAEManifest;
} {
  const manifest: SAEManifest = JSON.parse(
    fs.readFileSync(path.join(dir, "sae.json"), "utf-8"),
  );
  const params: SAEParams = {
    W_enc: readF32(path.join(dir, manifest.files.W_enc)),
    b_enc: readF32(path.join(dir, manifest.files.b_enc)),
    W_dec: readF32(path.join(dir, manifest.files.W_dec)),
    b_dec: readF32(path.join(dir, manifest.files.b_dec)),
    threshold: readF32(path.join(dir, manifest.files.threshold)),
  };
  const config: SAEConfig = {
    dModel: manifest.dModel,
    numFeatures: manifest.numFeatures,
    layer: manifest.layer,
    neuronpediaSaeId: manifest.neuronpediaSaeId,
  };
  return { config, params, manifest };
}
