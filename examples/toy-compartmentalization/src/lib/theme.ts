/**
 * Single source of truth for design tokens used by JS code (echarts options,
 * canvas drawing). Keep these in sync with the Tailwind utility classes used
 * in markup — both should refer to the same conceptual color.
 *
 * Light, distill.pub-inspired palette: warm off-white background, soft black
 * text, distill blue/green/red accents.
 */
export const THEME = {
  bg:      '#fffaf3', // warm off-white
  fg:      'rgba(0,0,0,0.84)',
  muted:   'rgba(0,0,0,0.54)',
  faint:   'rgba(0,0,0,0.32)',
  grid:    'rgba(0,0,0,0.08)',
  surface: '#ffffff',
  accent:  '#1f77b4', // distill blue
  accent2: '#2ca02c', // distill green
  warning: '#ff7f0e', // distill orange
  danger:  '#d62728', // distill red
  purple:  '#9467bd',
} as const;

/** Distill-style categorical palette for multi-series line charts. */
export const SERIES_PALETTE = [
  '#1f77b4', // blue
  '#ff7f0e', // orange
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
  '#e377c2', // pink
  '#7f7f7f', // gray
] as const;

/** Compartment-colored palette (used to distinguish compartments visually). */
export const COMP_COLORS = [
  '#d4a017', // gold (comp 0 — fitted/primary)
  '#c2185b', // crimson
  '#388e3c', // green
  '#7b1fa2', // purple
  '#e65100', // orange
  '#0097a7', // teal
  '#9c27b0', // magenta
  '#616161', // gray
] as const;

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.match(/^#([0-9a-fA-F]{6})$/);
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/** RGB triples for compartment colors (for use in canvas fillStyle). */
export const COMP_RGB: readonly [number, number, number][] = COMP_COLORS.map(hexToRgb);

/** Token-color triples used by mess3-style belief→RGB mappings. */
export const TOKEN_COLORS_RGB: readonly [number, number, number][] = [
  [232, 82, 74],   // red-ish
  [46, 204, 113],  // green-ish
  [74, 154, 222],  // blue-ish
];
