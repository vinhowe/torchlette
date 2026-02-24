export class EngineBusyError extends Error {
  name = "EngineBusyError";
}

export class CheckpointImpureRegionError extends Error {
  name = "CheckpointImpureRegionError";
}

export class HostReadInCompileError extends Error {
  name = "HostReadInCompileError";
}

export class AsyncInCompileError extends Error {
  name = "AsyncInCompileError";
}

export class InvalidTraceTensorEscapeError extends Error {
  name = "InvalidTraceTensorEscapeError";
}

export class SavedTensorModifiedError extends Error {
  name = "SavedTensorModifiedError";
}

export class NonReentrantBackwardError extends Error {
  name = "NonReentrantBackwardError";
}

export class PoisonedEngineError extends Error {
  name = "PoisonedEngineError";
}

export class RngReplayExhaustedError extends Error {
  name = "RngReplayExhaustedError";
}

export class RngReplayMismatchError extends Error {
  name = "RngReplayMismatchError";
}
