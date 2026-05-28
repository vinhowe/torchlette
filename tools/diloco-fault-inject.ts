/**
 * Mutable fault-injection hooks for the DiLoCo agent.
 *
 * The agent reads `faultInject.shouldDropOut` and `faultInject.shouldDropIn`
 * around its WebSocket send and receive paths. In production deployments
 * nothing imports this module's mutators, so the hooks stay at their no-op
 * defaults and the agent's behavior is unchanged.
 *
 * Test wrappers (e.g. tools/diloco-test-disconnect.ts) overwrite the hooks
 * BEFORE dynamically importing the agent, then the wrapper's filters take
 * effect when the agent invokes them at runtime.
 *
 * Keep this module very small — it's a contract surface, not a kitchen sink.
 */
export type FaultHook = (currentRound: number) => boolean;

const noop: FaultHook = () => false;

export const faultInject: {
  shouldDropOut: FaultHook;
  shouldDropIn: FaultHook;
} = {
  shouldDropOut: noop,
  shouldDropIn: noop,
};
