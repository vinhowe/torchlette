import { createPageConfig } from './page-config.svelte';

export type Mess3PageConfig = {
  world: { selfLoop: number; nCompartments: number };
  objective: { translationPct: number };
  init: { seed: number; weightScale: number };
  model: { seqLen: number; embedDim: number; numLayers: number; mlpDim: number };
  optim: { lr: number; weightDecay: number; batchSize: number };
  probe: { batchSize: number };
  remote: { enabled: boolean; url: string };
};

const CONFIG_DEFAULTS: Mess3PageConfig = {
  world: { selfLoop: 0.765, nCompartments: 1 },
  objective: { translationPct: 0 },
  init: { seed: 1024, weightScale: 1.0 },
  model: { seqLen: 10, embedDim: 64, numLayers: 4, mlpDim: 256 },
  optim: { lr: 1e-2, weightDecay: 0, batchSize: 64 },
  probe: { batchSize: 256 },
  remote: { enabled: false, url: 'ws://localhost:9882/ws' },
};

const CONFIG_LABEL_ALIASES: Record<string, string> = {
  'world.selfLoop': 'loop',
  'world.nCompartments': 'comp',
  'objective.translationPct': 'tr',
  'init.seed': 'seed',
  'init.weightScale': 'w×',
  'model.seqLen': 'sl',
  'model.embedDim': 'dim',
  'model.numLayers': 'L',
  'model.mlpDim': 'mlp',
  'optim.lr': 'lr',
  'optim.weightDecay': 'wd',
  'optim.batchSize': 'bs',
  'probe.batchSize': 'probe',
  'remote.enabled': 'remote',
  'remote.url': 'ws',
};

const manager = createPageConfig(CONFIG_DEFAULTS, CONFIG_LABEL_ALIASES);
export const config = manager.config;
export const initConfigUrlSync = manager.initUrlSync;
export const resetConfigToDefaults = manager.reset;
export const describeConfigDelta = manager.describeDelta;
