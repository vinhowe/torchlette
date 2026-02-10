/**
 * Advanced compiled region features per spec §8.3-8.11
 *
 * This module implements:
 * - Arg alias groups (§8.3): Track which arguments alias each other
 * - StateIfaceSig (§8.4): Ordered state access signature
 * - State-slot alias patterns (§8.5): Bind-time alias tracking
 * - Token ABI reconciliation (§8.6): Entry/exit token handling
 * - Auto-externalize (§8.7): SSA to pending_loc conversion
 * - Null-state sentinels (§8.8): Stable missing state representation
 * - Functionalization (§8.10): Mutation to functional conversion
 * - Region-exit persistence (§8.11): Commit tracking
 */

import type { DType } from "../backend/types";
import type { BaseId, LocId } from "./engine";
import type { IRGraph, IRNode } from "./ir";
import type { Token } from "./tokens";

// ============================================================================
// §8.3 Canonical Arg Alias Groups
// ============================================================================

/**
 * An alias group represents a set of input arguments that share the same BaseId.
 * Arguments in the same group alias each other.
 */
export type AliasGroup = {
  groupId: number;
  argIndices: number[]; // Which input args are in this group
  baseId?: BaseId; // The shared BaseId (if known at bind time)
};

/**
 * Compute canonical alias groups from input arguments.
 * Arguments that share the same BaseId are grouped together.
 *
 * @param argBaseIds - BaseId for each input argument (undefined if not aliased)
 * @returns Canonical alias groups sorted by first argument index
 */
export function computeAliasGroups(
  argBaseIds: (BaseId | undefined)[],
): AliasGroup[] {
  const baseIdToGroup = new Map<BaseId, number[]>();
  const nonAliasedArgs: number[] = [];

  for (let i = 0; i < argBaseIds.length; i++) {
    const baseId = argBaseIds[i];
    if (baseId === undefined) {
      nonAliasedArgs.push(i);
    } else {
      const existing = baseIdToGroup.get(baseId);
      if (existing) {
        existing.push(i);
      } else {
        baseIdToGroup.set(baseId, [i]);
      }
    }
  }

  const groups: AliasGroup[] = [];
  let groupId = 0;

  // Add aliased groups (sorted by first arg index for determinism)
  const aliasedGroups = Array.from(baseIdToGroup.entries())
    .filter(([_, indices]) => indices.length > 1)
    .sort((a, b) => a[1][0] - b[1][0]);

  for (const [baseId, indices] of aliasedGroups) {
    groups.push({
      groupId: groupId++,
      argIndices: indices.sort((a, b) => a - b),
      baseId,
    });
  }

  // Single-element groups for non-aliased args
  for (const argIdx of nonAliasedArgs.sort((a, b) => a - b)) {
    groups.push({
      groupId: groupId++,
      argIndices: [argIdx],
      baseId: undefined,
    });
  }

  // Single-element groups for unique baseIds
  const singleBaseIds = Array.from(baseIdToGroup.entries())
    .filter(([_, indices]) => indices.length === 1)
    .sort((a, b) => a[1][0] - b[1][0]);

  for (const [baseId, indices] of singleBaseIds) {
    groups.push({
      groupId: groupId++,
      argIndices: indices,
      baseId,
    });
  }

  return groups;
}

/**
 * Compute a canonical string key for alias groups.
 * Used in cache key generation.
 */
export function aliasGroupsKey(groups: AliasGroup[]): string {
  return groups
    .map((g) => `g${g.groupId}:[${g.argIndices.join(",")}]`)
    .join("|");
}

// ============================================================================
// §8.4 Pinned Ordered State Access Signature (StateIfaceSig)
// ============================================================================

/**
 * Access target types for state interface signature.
 */
export type AccessTarget =
  | { kind: "state"; slot: number }
  | { kind: "arg_base"; groupId: number }
  | { kind: "global" };

/**
 * A single ordered access in the state interface signature.
 */
export type StateAccess = {
  target: AccessTarget;
  accessKind: "load" | "store" | "load_store";
  order: number; // Position in access sequence
};

/**
 * The complete ordered state interface signature for a compiled region.
 * This is pinned at compile time and enforced at bind time.
 */
export type StateIfaceSig = {
  accesses: StateAccess[];
  touchedTargets: AccessTarget[];
  usesGlobalToken: boolean;
};

/**
 * Build a state interface signature from IR analysis.
 */
export function buildStateIfaceSig(
  graph: IRGraph,
  stateSlotAccesses: Map<number, "load" | "store" | "load_store">,
  argGroupAccesses: Map<number, "load" | "store" | "load_store">,
  usesGlobal: boolean,
): StateIfaceSig {
  const accesses: StateAccess[] = [];
  let order = 0;

  // Add state slot accesses in slot order
  const sortedSlots = Array.from(stateSlotAccesses.entries()).sort(
    (a, b) => a[0] - b[0],
  );
  for (const [slot, accessKind] of sortedSlots) {
    accesses.push({
      target: { kind: "state", slot },
      accessKind,
      order: order++,
    });
  }

  // Add arg group accesses
  const sortedGroups = Array.from(argGroupAccesses.entries()).sort(
    (a, b) => a[0] - b[0],
  );
  for (const [groupId, accessKind] of sortedGroups) {
    accesses.push({
      target: { kind: "arg_base", groupId },
      accessKind,
      order: order++,
    });
  }

  // Add global if used
  if (usesGlobal) {
    accesses.push({
      target: { kind: "global" },
      accessKind: "load_store",
      order: order++,
    });
  }

  const touchedTargets = accesses.map((a) => a.target);

  return {
    accesses,
    touchedTargets,
    usesGlobalToken: usesGlobal,
  };
}

/**
 * Serialize state interface signature for cache key.
 */
export function stateIfaceSigKey(sig: StateIfaceSig): string {
  const accessKeys = sig.accesses.map((a) => {
    const targetKey =
      a.target.kind === "state"
        ? `s${a.target.slot}`
        : a.target.kind === "arg_base"
          ? `a${a.target.groupId}`
          : "g";
    return `${targetKey}:${a.accessKind}`;
  });
  return `[${accessKeys.join(",")}]${sig.usesGlobalToken ? "+g" : ""}`;
}

// ============================================================================
// §8.5 Bind-Time State-Slot Alias Pattern
// ============================================================================

/**
 * Pattern describing which state slots alias at bind time.
 */
export type StateSlotAliasPattern = {
  // Groups of slot indices that alias each other
  aliasGroups: number[][];
  // Universal may-alias flag (conservative fallback)
  universalMayAlias: boolean;
};

/**
 * Compute state slot alias pattern from bind-time information.
 */
export function computeStateSlotAliasPattern(
  slotBaseIds: (BaseId | undefined)[],
  forceUniversal = false,
): StateSlotAliasPattern {
  if (forceUniversal) {
    return {
      aliasGroups: [],
      universalMayAlias: true,
    };
  }

  const baseIdToSlots = new Map<BaseId, number[]>();

  for (let i = 0; i < slotBaseIds.length; i++) {
    const baseId = slotBaseIds[i];
    if (baseId !== undefined) {
      const existing = baseIdToSlots.get(baseId);
      if (existing) {
        existing.push(i);
      } else {
        baseIdToSlots.set(baseId, [i]);
      }
    }
  }

  const aliasGroups = Array.from(baseIdToSlots.values())
    .filter((slots) => slots.length > 1)
    .map((slots) => slots.sort((a, b) => a - b))
    .sort((a, b) => a[0] - b[0]);

  return {
    aliasGroups,
    universalMayAlias: false,
  };
}

/**
 * Serialize alias pattern for cache key.
 */
export function aliasPatternKey(pattern: StateSlotAliasPattern): string {
  if (pattern.universalMayAlias) {
    return "universal";
  }
  if (pattern.aliasGroups.length === 0) {
    return "none";
  }
  return pattern.aliasGroups.map((g) => `[${g.join(",")}]`).join(";");
}

// ============================================================================
// §8.6 CompiledCall Token ABI Reconciliation
// ============================================================================

/**
 * Token state for a compiled call entry/exit.
 */
export type CompiledCallTokenState = {
  tokGlobal: Token;
  tokLocs: Map<LocId, Token>;
};

/**
 * Result of token reconciliation at compiled call entry.
 */
export type EntryReconciliation = {
  entryToken: Token;
  locTokens: Map<LocId, Token>;
};

/**
 * Result of token reconciliation at compiled call exit.
 */
export type ExitReconciliation = {
  exitToken: Token;
  updatedLocTokens: Map<LocId, Token>;
};

// ============================================================================
// §8.7 Auto-Externalize
// ============================================================================

/**
 * Information about an SSA base that needs to be externalized.
 */
export type ExternalizeRequest = {
  baseId: BaseId;
  targetLocId?: LocId; // If already has a pending_loc
  needsMaterializer: boolean;
};

/**
 * Analyze which SSA bases need to be externalized for a compiled call.
 */
export function analyzeExternalizeNeeds(
  ssaBases: BaseId[],
  existingBindings: Map<BaseId, { kind: string; locId?: LocId }>,
): ExternalizeRequest[] {
  const requests: ExternalizeRequest[] = [];

  for (const baseId of ssaBases) {
    const binding = existingBindings.get(baseId);

    if (!binding || binding.kind === "ssa") {
      // Pure SSA - needs externalization
      requests.push({
        baseId,
        targetLocId: undefined,
        needsMaterializer: true,
      });
    } else if (binding.kind === "pending_loc") {
      // Already pending - may need materializer
      requests.push({
        baseId,
        targetLocId: binding.locId,
        needsMaterializer: true,
      });
    }
    // loc-backed bases don't need externalization
  }

  return requests;
}

// ============================================================================
// §8.8 Null-State Sentinels
// ============================================================================

/**
 * A stable sentinel for missing state.
 * Per spec: "absent state slots must not bind by aliasing another slot or real state"
 */
export type NullStateSentinel = {
  compiledFnId: number;
  stateSlot: number;
  sentinelId: number;
};

let nextSentinelId = 1;

/**
 * Create a stable null-state sentinel for a (compiledFn, stateSlot) pair.
 */
export function createNullStateSentinel(
  compiledFnId: number,
  stateSlot: number,
): NullStateSentinel {
  return {
    compiledFnId,
    stateSlot,
    sentinelId: nextSentinelId++,
  };
}

/**
 * Cache of null-state sentinels keyed by (compiledFnId, stateSlot).
 */
const nullStateSentinelCache = new Map<string, NullStateSentinel>();

/**
 * Get or create a stable null-state sentinel.
 */
export function getNullStateSentinel(
  compiledFnId: number,
  stateSlot: number,
): NullStateSentinel {
  const key = `${compiledFnId}:${stateSlot}`;
  let sentinel = nullStateSentinelCache.get(key);
  if (!sentinel) {
    sentinel = createNullStateSentinel(compiledFnId, stateSlot);
    nullStateSentinelCache.set(key, sentinel);
  }
  return sentinel;
}

/**
 * Reset sentinel cache (for testing).
 */
export function resetNullStateSentinels(): void {
  nullStateSentinelCache.clear();
  nextSentinelId = 1;
}

// ============================================================================
// §8.10 Functionalization
// ============================================================================

/**
 * A functionalized mutation: converts in-place op to out-of-place.
 */
export type FunctionalizedMutation = {
  originalNodeId: number;
  mutatedBaseId: BaseId;
  newValueNodeId: number;
  viewChain?: number[]; // For view mutations
};

/**
 * Result of functionalizing a compiled region's IR.
 */
export type FunctionalizationResult = {
  functionalizedGraph: IRGraph;
  mutations: FunctionalizedMutation[];
  requiresWriteback: BaseId[];
};

/**
 * Check if an op is an in-place mutation.
 */
export function isInPlaceMutation(op: string): boolean {
  return op.endsWith("_"); // PyTorch convention: add_, mul_, etc.
}

/**
 * Convert in-place op name to out-of-place equivalent.
 */
export function toOutOfPlaceOp(op: string): string {
  if (op.endsWith("_")) {
    return op.slice(0, -1);
  }
  return op;
}

// ============================================================================
// §8.11 Region-Exit Persistence
// ============================================================================

/**
 * Information about a base that needs to be committed at region exit.
 */
export type RegionExitCommit = {
  baseId: BaseId;
  mutId: number;
  locId?: LocId;
};

/**
 * SSA writeback requirement for region exit.
 */
export type SSAWriteback = {
  baseId: BaseId;
  valueNodeId: number;
  targetLocId: LocId;
};

/**
 * Complete region exit persistence plan.
 */
export type RegionExitPlan = {
  commits: RegionExitCommit[];
  writebacks: SSAWriteback[];
  usesGlobalToken: boolean;
};

/**
 * Analyze a compiled region's exit persistence requirements.
 */
export function analyzeRegionExit(
  mutations: FunctionalizedMutation[],
  outputBases: BaseId[],
  baseBindings: Map<BaseId, { kind: string; locId?: LocId }>,
): RegionExitPlan {
  const commits: RegionExitCommit[] = [];
  const writebacks: SSAWriteback[] = [];
  let usesGlobalToken = false;
  let nextMutId = 1;

  // Mutations need commits
  for (const mutation of mutations) {
    const binding = baseBindings.get(mutation.mutatedBaseId);
    commits.push({
      baseId: mutation.mutatedBaseId,
      mutId: nextMutId++,
      locId: binding?.locId,
    });
    usesGlobalToken = true;
  }

  // Outputs may need writeback
  for (const baseId of outputBases) {
    const binding = baseBindings.get(baseId);
    if (binding?.kind === "pending_loc" && binding.locId !== undefined) {
      writebacks.push({
        baseId,
        valueNodeId: -1, // Will be resolved during lowering
        targetLocId: binding.locId,
      });
    }
  }

  return {
    commits,
    writebacks,
    usesGlobalToken,
  };
}

// ============================================================================
// Enhanced Cache Key
// ============================================================================

/**
 * Extended cache key that includes all §8 features.
 */
export type ExtendedCompiledCacheKey = {
  irHash: string;
  inputShapes: number[][];
  inputDtypes: DType[];
  aliasGroups: string;
  stateIfaceSig: string;
  stateSlotAliasPattern: string;
};

/**
 * Generate an extended cache key for a compiled region.
 */
export function generateExtendedCacheKey(
  irHash: string,
  inputShapes: number[][],
  inputDtypes: DType[],
  aliasGroups: AliasGroup[],
  stateIfaceSig: StateIfaceSig,
  aliasPattern: StateSlotAliasPattern,
): ExtendedCompiledCacheKey {
  return {
    irHash,
    inputShapes,
    inputDtypes,
    aliasGroups: aliasGroupsKey(aliasGroups),
    stateIfaceSig: stateIfaceSigKey(stateIfaceSig),
    stateSlotAliasPattern: aliasPatternKey(aliasPattern),
  };
}

/**
 * Serialize extended cache key to string.
 */
export function serializeExtendedCacheKey(
  key: ExtendedCompiledCacheKey,
): string {
  const shapesKey = key.inputShapes.map((s) => s.join("x")).join(";");
  const dtypesKey = key.inputDtypes.join(",");
  return `${key.irHash}|${shapesKey}|${dtypesKey}|${key.aliasGroups}|${key.stateIfaceSig}|${key.stateSlotAliasPattern}`;
}
