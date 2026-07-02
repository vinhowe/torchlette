/**
 * Shim: the Qwen3 model moved to the shared workspace package
 * `packages/qwen3-browser` (consumed by both the Node harnesses here and the
 * browser chat app). Kept so the example scripts' imports stay stable.
 */
export * from "../../packages/qwen3-browser/src/model";
