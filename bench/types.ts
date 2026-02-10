export type BenchCase = {
  name: string;
  run?: () => void | Promise<void>;
  flops?: number;
  bytes?: number;
  skip?: string;
};

export type BenchResult =
  | {
      name: string;
      status: "skipped";
      reason: string;
    }
  | {
      name: string;
      status: "ok";
      iterations: number;
      msMedian: number;
      flopsPerSec?: number;
      bytesPerSec?: number;
    };
