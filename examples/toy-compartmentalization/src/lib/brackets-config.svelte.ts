import { createPageConfig } from './page-config.svelte';

export type PosEncoding = 'learned' | 'rope';

export type BracketsPageConfig = {
  world: { contentLen: number; maxDepth: number; nPairTypes: number; nCompartments: number };
  objective: { singlePct: number; mixedPct: number };
  init: { seed: number; weightScale: number };
  model: {
    embedDim: number; numLayers: number; numHeads: number; seqLen: number;
    posEncoding: PosEncoding;
  };
  optim: { lr: number; weightDecay: number; batchSize: number };
};

const CONFIG_DEFAULTS: BracketsPageConfig = {
  world: { contentLen: 8, maxDepth: 4, nPairTypes: 1, nCompartments: 2 },
  objective: { singlePct: 100, mixedPct: 0 },
  init: { seed: 1024, weightScale: 1.0 },
  model: { embedDim: 64, numLayers: 2, numHeads: 4, seqLen: 16, posEncoding: 'learned' },
  optim: { lr: 1e-3, weightDecay: 0, batchSize: 64 },
};

const CONFIG_LABEL_ALIASES: Record<string, string> = {
  'world.contentLen': 'len',
  'world.maxDepth': 'depth',
  'world.nPairTypes': 'types',
  'world.nCompartments': 'comp',
  'objective.singlePct': 'single',
  'objective.mixedPct': 'mix',
  'init.seed': 'seed',
  'init.weightScale': 'w×',
  'model.embedDim': 'dim',
  'model.numLayers': 'L',
  'model.numHeads': 'H',
  'model.seqLen': 'sl',
  'model.posEncoding': 'pe',
  'optim.lr': 'lr',
  'optim.weightDecay': 'wd',
  'optim.batchSize': 'bs',
};

const manager = createPageConfig(CONFIG_DEFAULTS, CONFIG_LABEL_ALIASES);
export const config = manager.config;
export const initConfigUrlSync = manager.initUrlSync;
export const resetConfigToDefaults = manager.reset;
export const describeConfigDelta = manager.describeDelta;
