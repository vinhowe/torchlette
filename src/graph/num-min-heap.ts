/**
 * Minimal binary min-heap of numbers.
 *
 * The compile-path Kahn passes (reorderPlanForFusion, enforceWriteAfterReadOrder)
 * need a "pop the smallest ready node index" primitive. A sorted-array +
 * `splice` insert is O(n) per op → O(n²) over a whole-step graph whose ready
 * frontier is O(n) wide (the backward pass fans out one gradient branch per
 * parameter). This heap makes each push/pop O(log n). Callers that need lazy
 * deletion track an `emitted`/staleness flag externally and skip stale tops.
 */
export class NumMinHeap {
  private readonly a: number[] = [];

  get size(): number {
    return this.a.length;
  }

  peek(): number {
    return this.a[0];
  }

  push(x: number): void {
    const a = this.a;
    a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p] <= a[i]) break;
      const t = a[p];
      a[p] = a[i];
      a[i] = t;
      i = p;
    }
  }

  pop(): number {
    const a = this.a;
    const top = a[0];
    const last = a.pop() as number;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      const n = a.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < n && a[l] < a[m]) m = l;
        if (r < n && a[r] < a[m]) m = r;
        if (m === i) break;
        const t = a[m];
        a[m] = a[i];
        a[i] = t;
        i = m;
      }
    }
    return top;
  }
}
