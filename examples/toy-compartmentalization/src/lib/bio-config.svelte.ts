import { createPageConfig } from './page-config.svelte';

export type TranslationMode = 'mirror' | 'continuation' | 'dictionary';
export type PosEncoding = 'learned' | 'rope';

export type BioPageConfig = {
  world: {
    nEntities: number;
    nAttributes: number;
    nValues: number;
    nCompartments: number;
    tokensPerEntity: number;
    tokensPerValue: number;
    mixCompartments: boolean;
  };
  objective: {
    translationPct: number;
    translationMode: TranslationMode;
  };
  init: {
    seed: number;
    tieInit: boolean;
    weightScale: number;
    embedScale: number;
    headScale: number;
    residualZeroInit: boolean;
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

const CONFIG_DEFAULTS: BioPageConfig = {
  world: { nEntities: 100, nAttributes: 6, nValues: 20, nCompartments: 2, tokensPerEntity: 1, tokensPerValue: 1, mixCompartments: false },
  objective: { translationPct: 0, translationMode: 'mirror' },
  init: { seed: 1024, tieInit: false, weightScale: 1.0, embedScale: 1.0, headScale: 1.0, residualZeroInit: false },
  model: { embedDim: 64, numLayers: 2, numHeads: 4, seqLen: 64, posEncoding: 'learned' },
  optim: { lr: 1e-4, weightDecay: 0, batchSize: 32 },
};

const CONFIG_LABEL_ALIASES: Record<string, string> = {
  'world.nEntities': 'ent',
  'world.nAttributes': 'attr',
  'world.nValues': 'val',
  'world.nCompartments': 'comp',
  'world.tokensPerEntity': 'tok/e',
  'world.tokensPerValue': 'tok/v',
  'world.mixCompartments': 'mix',
  'objective.translationPct': 'tr',
  'objective.translationMode': 'mode',
  'init.seed': 'seed',
  'init.tieInit': 'tied',
  'init.weightScale': 'w×',
  'init.embedScale': 'emb×',
  'init.headScale': 'head×',
  'init.residualZeroInit': 'res=0',
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
