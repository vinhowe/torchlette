export interface HasLR {
  getLR(): number;
  setLR(lr: number): void;
}

export interface LRScheduler {
  step(): void;
  getLR(): number;
  getLastLR(): number;
}

export class StepLR implements LRScheduler {
  private readonly optimizer: HasLR;
  private readonly baseLR: number;
  private readonly stepSize: number;
  private readonly gamma: number;
  private epoch = 0;
  private lastLR: number;

  constructor(optimizer: HasLR, stepSize: number, gamma = 0.1) {
    this.optimizer = optimizer;
    this.baseLR = optimizer.getLR();
    this.stepSize = stepSize;
    this.gamma = gamma;
    this.lastLR = this.baseLR;
  }

  step(): void {
    this.epoch++;
    const lr =
      this.baseLR * this.gamma ** Math.floor(this.epoch / this.stepSize);
    this.lastLR = lr;
    this.optimizer.setLR(lr);
  }

  getLR(): number {
    return this.optimizer.getLR();
  }

  getLastLR(): number {
    return this.lastLR;
  }
}

export class ExponentialLR implements LRScheduler {
  private readonly optimizer: HasLR;
  private readonly baseLR: number;
  private readonly gamma: number;
  private epoch = 0;
  private lastLR: number;

  constructor(optimizer: HasLR, gamma: number) {
    this.optimizer = optimizer;
    this.baseLR = optimizer.getLR();
    this.gamma = gamma;
    this.lastLR = this.baseLR;
  }

  step(): void {
    this.epoch++;
    const lr = this.baseLR * this.gamma ** this.epoch;
    this.lastLR = lr;
    this.optimizer.setLR(lr);
  }

  getLR(): number {
    return this.optimizer.getLR();
  }

  getLastLR(): number {
    return this.lastLR;
  }
}

export class CosineAnnealingLR implements LRScheduler {
  private readonly optimizer: HasLR;
  private readonly baseLR: number;
  private readonly tMax: number;
  private readonly etaMin: number;
  private epoch = 0;
  private lastLR: number;

  constructor(optimizer: HasLR, tMax: number, etaMin = 0) {
    this.optimizer = optimizer;
    this.baseLR = optimizer.getLR();
    this.tMax = tMax;
    this.etaMin = etaMin;
    this.lastLR = this.baseLR;
  }

  step(): void {
    this.epoch++;
    const lr =
      this.etaMin +
      ((this.baseLR - this.etaMin) *
        (1 + Math.cos((Math.PI * this.epoch) / this.tMax))) /
        2;
    this.lastLR = lr;
    this.optimizer.setLR(lr);
  }

  getLR(): number {
    return this.optimizer.getLR();
  }

  getLastLR(): number {
    return this.lastLR;
  }
}

export class PolynomialLR implements LRScheduler {
  private readonly optimizer: HasLR;
  private readonly baseLR: number;
  private readonly totalIters: number;
  private readonly power: number;
  private epoch = 0;
  private lastLR: number;

  constructor(optimizer: HasLR, totalIters: number, power = 1.0) {
    this.optimizer = optimizer;
    this.baseLR = optimizer.getLR();
    this.totalIters = totalIters;
    this.power = power;
    this.lastLR = this.baseLR;
  }

  step(): void {
    this.epoch++;
    const t = Math.min(this.epoch, this.totalIters);
    const lr = this.baseLR * (1 - t / this.totalIters) ** this.power;
    this.lastLR = lr;
    this.optimizer.setLR(lr);
  }

  getLR(): number {
    return this.optimizer.getLR();
  }

  getLastLR(): number {
    return this.lastLR;
  }
}
