import { SvelteURL, SvelteURLSearchParams } from 'svelte/reactivity';

/**
 * Generic page-config factory shared by all demo pages (mess3, bio, xor,
 * brackets, ...). Each page used to copy ~150 lines of identical helpers
 * for URL <-> $state sync. Now they pass their defaults + label aliases
 * here and get back a typed reactive config plus the standard helpers.
 *
 * Usage from a page-config module (must be `.svelte.ts` so $state works):
 *
 *   const DEFAULTS: MyConfig = { ... };
 *   const ALIASES = { 'optim.lr': 'lr', ... };
 *   export const { config, initUrlSync, describeDelta, reset } =
 *     createPageConfig(DEFAULTS, ALIASES);
 *
 * Then from +page.svelte:
 *
 *   import { config, initUrlSync, describeDelta, reset } from '$lib/...';
 *   initUrlSync();              // call once inside <script> top-level
 *   bind:value={config.optim.lr}
 */

function parseValueBasedOnDefault(valueStr: string, defaultValue: unknown): unknown {
  if (typeof defaultValue === 'boolean') return valueStr.toLowerCase() === 'true';
  if (typeof defaultValue === 'number') {
    const num = parseFloat(valueStr);
    return isNaN(num) ? defaultValue : num;
  }
  return valueStr;
}

function buildConfigFromUrlParams<T extends object>(
  params: URLSearchParams,
  defaults: T,
): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [path, valueStr] of params) {
    const keys = path.split('.');
    let level = out;
    let defaultLevel: unknown = defaults;
    try {
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        if (defaultLevel === undefined || typeof defaultLevel !== 'object' || defaultLevel === null) {
          throw new Error(`Invalid config path from URL: ${path}`);
        }
        defaultLevel = (defaultLevel as Record<string, unknown>)[key];
        if (i < keys.length - 1) {
          if (!level[key] || typeof level[key] !== 'object') level[key] = {};
          level = level[key] as Record<string, unknown>;
        } else {
          level[key] = parseValueBasedOnDefault(valueStr, defaultLevel);
        }
      }
    } catch (e) {
      console.warn((e as Error).message);
      continue;
    }
  }
  return out as Partial<T>;
}

function mergeDeep(target: Record<string, unknown>, source: Record<string, unknown>) {
  for (const key in source) {
    if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
    const sourceVal = source[key];
    if (sourceVal && typeof sourceVal === 'object' && !Array.isArray(sourceVal)) {
      let t = target[key] as Record<string, unknown>;
      if (!t || typeof t !== 'object' || Array.isArray(t)) { t = {}; target[key] = t; }
      mergeDeep(t, sourceVal as Record<string, unknown>);
    } else if (sourceVal !== undefined) {
      target[key] = sourceVal;
    }
  }
}

function valuesDeepEqual(a: unknown, b: unknown): boolean {
  try { return JSON.stringify(a) === JSON.stringify(b); } catch { return a === b; }
}

function flattenNonDefault(
  obj: Record<string, unknown>,
  defaults: Record<string, unknown>,
  prefix = '',
): Record<string, string> {
  const params: Record<string, string> = {};
  for (const key in obj) {
    if (!Object.prototype.hasOwnProperty.call(obj, key)) continue;
    const newPrefix = prefix ? `${prefix}.${key}` : key;
    const value = obj[key];
    const defaultValue = (defaults ?? {})[key];
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      const defaultChild = defaultValue !== null && typeof defaultValue === 'object' && !Array.isArray(defaultValue)
        ? (defaultValue as Record<string, unknown>) : ({} as Record<string, unknown>);
      Object.assign(params, flattenNonDefault(value as Record<string, unknown>, defaultChild, newPrefix));
    } else if (value !== undefined) {
      if (defaultValue === undefined || !valuesDeepEqual(value, defaultValue)) {
        params[newPrefix] = String(value);
      }
    }
  }
  return params;
}

function getInitialConfig<T extends object>(defaults: T): T {
  const base = JSON.parse(JSON.stringify(defaults)) as T;
  if (typeof window !== 'undefined' && window.location && window.URLSearchParams) {
    try {
      const params = new URLSearchParams(window.location.search);
      const overrides = buildConfigFromUrlParams(params, base);
      mergeDeep(base as unknown as Record<string, unknown>, overrides as Record<string, unknown>);
    } catch (e) {
      console.error('Error processing config from URL, using defaults:', e);
      return JSON.parse(JSON.stringify(defaults));
    }
  }
  return base;
}

export type PageConfigManager<T> = {
  /** Reactive ($state) config object — bind directly with `bind:value`. */
  config: T;
  /** Call once from inside a component <script> to keep the URL in sync. */
  initUrlSync: () => void;
  /** "lr=0.01, dim=128, tied" style summary of non-default values. */
  describeDelta: (cfg: T) => string;
  /** Reset the reactive config back to its defaults (in-place). */
  reset: () => void;
};

/**
 * Build a reactive page config with URL sync, default-diffing, and reset.
 *
 * `defaults` is treated as the canonical baseline. Anything that differs
 * from it ends up in the URL and in the delta string; anything that
 * matches stays out.
 *
 * `labelAliases` maps dotted paths (e.g. `'optim.lr'`) to short labels
 * used in `describeDelta` output. Paths without an alias fall through
 * to the dotted path itself.
 */
export function createPageConfig<T extends object>(
  defaults: T,
  labelAliases: Record<string, string>,
): PageConfigManager<T> {
  const config = $state(getInitialConfig(defaults));

  function initUrlSync() {
    if (typeof window === 'undefined' || !window.history || !window.URL) return;
    $effect(() => {
      const snapshot = $state.snapshot(config);
      const flat = flattenNonDefault(
        snapshot as unknown as Record<string, unknown>,
        defaults as unknown as Record<string, unknown>,
      );
      const searchParamsString = new SvelteURLSearchParams(flat).toString();
      const currentUrl = new SvelteURL(window.location.href);
      currentUrl.search = searchParamsString;
      if (window.location.href !== currentUrl.href) {
        window.history.replaceState({}, '', currentUrl.toString());
      }
    });
  }

  function describeDelta(cfg: T): string {
    const flat = flattenNonDefault(
      $state.snapshot(cfg) as unknown as Record<string, unknown>,
      defaults as unknown as Record<string, unknown>,
    );
    const parts: string[] = [];
    for (const [path, val] of Object.entries(flat)) {
      const name = labelAliases[path] ?? path;
      if (val === 'true') parts.push(name);
      else if (val === 'false') parts.push(`!${name}`);
      else parts.push(`${name}=${val}`);
    }
    return parts.length === 0 ? 'defaults' : parts.join(', ');
  }

  function reset() {
    const fresh = JSON.parse(JSON.stringify(defaults)) as T;
    // Re-assign each top-level section so $state proxies pick up the change.
    for (const key of Object.keys(fresh) as (keyof T)[]) {
      (config as T)[key] = fresh[key];
    }
  }

  return { config, initUrlSync, describeDelta, reset };
}
