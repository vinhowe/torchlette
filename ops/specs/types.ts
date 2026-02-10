export type DType = "f16" | "f32" | "i32" | "u32" | "bool";

export type OrdType = number | "fro" | "nuc";
export type OrdDimType = number | number[] | null;

export type OpStatus = "implemented" | "planned";
export type OpCaseExpectation = "match" | "expected_failure";

export type OpInput = {
  values: number[];
  shape: number[];
  dtype?: DType;
};

export type OpCase = {
  name: string;
  inputs: OpInput[];
  options?: Record<string, unknown>;
  expectation: OpCaseExpectation;
  atol?: number;
  rtol?: number;
};

export type OpSpec = {
  name: string;
  torchOp: string;
  signature: string;
  optionsSpec?: string;
  optionsDefaults?: Record<string, unknown>;
  status: OpStatus;
  note?: string;
  cases?: OpCase[];
};
