import { createPageConfig } from './page-config.svelte';

export type RnnPageConfig = {
  world: { selfLoop: number; sharedFrac: number };
  model: { hiddenDim: number; seqLen: number };
  optim: { lr: number; batchSize: number; lambda: number };
  eval: { batchSize: number; seqLen: number; burnIn: number };
  viz: { stepsPerFrame: number; vizInterval: number };
};

const CONFIG_DEFAULTS: RnnPageConfig = {
  world: { selfLoop: 0.765, sharedFrac: 0.0 },
  model: { hiddenDim: 3, seqLen: 20 },
  optim: { lr: 1e-3, batchSize: 64, lambda: 0 },
  eval: { batchSize: 600, seqLen: 100, burnIn: 10 },
  viz: { stepsPerFrame: 200, vizInterval: 1000 },
};

const CONFIG_LABEL_ALIASES: Record<string, string> = {
  'world.selfLoop': 'loop',
  'world.sharedFrac': 'shared',
  'model.hiddenDim': 'h',
  'model.seqLen': 'sl',
  'optim.lr': 'lr',
  'optim.batchSize': 'bs',
  'optim.lambda': 'λ',
  'eval.batchSize': 'eb',
  'eval.seqLen': 'esl',
  'eval.burnIn': 'burn',
  'viz.stepsPerFrame': 'spf',
  'viz.vizInterval': 'viz',
};

const manager = createPageConfig(CONFIG_DEFAULTS, CONFIG_LABEL_ALIASES);
export const config = manager.config;
export const initConfigUrlSync = manager.initUrlSync;
export const resetConfigToDefaults = manager.reset;
export const describeConfigDelta = manager.describeDelta;
