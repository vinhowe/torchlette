# Checkpoint D — canvas interaction quality

Review captures:

- [Persistent gesture palette and navigation hint](./checkpoint-d-palette.png)
- [Valid group-partition targets highlighted; incompatible targets receded](./checkpoint-d-valid-targets.png)
- [Keyboard and gesture reference opened with `?`](./checkpoint-d-help.png)

## Chosen desktop-canvas conventions

- Wheel/two-finger deltas pan in scroll direction by translating the world in
  the opposite direction; they never change scale without Ctrl/⌘.
- Shift converts the vertical wheel component to horizontal pan.
- Ctrl/⌘ wheel and browser pinch share cursor-anchored exponential zoom. The
  scale range is 0.36–2.5, with resistance near the edges and at most 0.45 log
  units applied per animation frame. Ten `-25px` gentle pinch ticks are roughly
  2×.
- Primary-button drag pans only from empty paper. Array regions keep their
  crosshair paint cursor; axes and boxes use copy cursors while accepting chips.
- Paint commits on release over the same region where it began. Escape or a
  release elsewhere cancels it.
- Empty-space double-click is inert. It does not zoom, fit, edit, or reset.
- Window resize keeps the world coordinate at the viewport center fixed.

## Automated aggressive-user protocol

`pnpm test` runs the original unit suite, builds the application, starts an
isolated preview, and executes `tests/ncd-interaction.mjs`. It asserts:

1. wheel without Ctrl pans with unchanged scale;
2. Ctrl-wheel preserves the pointer's world coordinate;
3. ten gentle pinch ticks are approximately 2×;
4. empty-space drag pans;
5. canvas label drag cannot create a document selection;
6. Escape restores the exact pre-pan transform;
7. 50 rapid wheel events coalesce to a finite clamped transform;
8. Shift-wheel pans horizontally only;
9. resize preserves the world-space view center;
10. Escape cancels paint before commit;
11. a palette drag released outside clears all drop state; and
12. empty-space double-click is inert.

The preview process is terminated in a `finally` block even when an assertion
fails.
