import { createPageConfig } from './page-config.svelte';

export type PosEncoding = 'learned' | 'rope';

export type XorPageConfig = {
  world: {
    nEntities: number;
    nCompartments: number;
  };
  objective: {
    /** % of training batches that show only the XOR bit (single compartment A). */
    singleAPct: number;
    /** % of training batches that show only the F1 bit (single compartment B). */
    singleBPct: number;
    /** % of training batches that are mixed-obs sequences (both partial obs in one seq). */
    mixedPct: number;
  };
  init: {
    seed: number;
    weightScale: number;
  };
  model: {
    embedDim: number;
    numLayers: number;
    numHeads: number;
    seqLen: number;
    posEncoding: PosEncoding;
  };
  optim: {
    lr: number;
    weightDecay: number;
    batchSize: number;
  };
};

const CONFIG_DEFAULTS: XorPageConfig = {
  world: { nEntities: 100, nCompartments: 2 },
  objective: { singleAPct: 50, singleBPct: 50, mixedPct: 0 },
  init: { seed: 1024, weightScale: 1.0 },
  model: { embedDim: 64, numLayers: 2, numHeads: 4, seqLen: 8, posEncoding: 'learned' },
  optim: { lr: 1e-3, weightDecay: 0, batchSize: 32 },
};

const CONFIG_LABEL_ALIASES: Record<string, string> = {
  'world.nEntities': 'ent',
  'world.nCompartments': 'comp',
  'objective.singleAPct': 'A',
  'objective.singleBPct': 'B',
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
