import { THEME } from './theme';

/**
 * Common echarts option fragments. Use these to compose chart options
 * consistently across pages instead of duplicating the boilerplate.
 *
 *   const option = $derived({
 *     ...baseChartOpt(),
 *     ...lineChartAxes('log'),
 *     series: [...],
 *   });
 */

export function baseChartOpt() {
  return {
    backgroundColor: 'transparent',
    textStyle: { color: THEME.muted, fontFamily: 'system-ui', fontSize: 11 },
    grid: { top: 28, right: 12, bottom: 24, left: 44 },
    animation: false,
  };
}

type AxisType = 'value' | 'log' | 'category';

export function chartAxes(opts: {
  yType?: AxisType;
  yMin?: number;
  yMax?: number;
  xType?: AxisType;
  xData?: (number | string)[];
} = {}) {
  const yType = opts.yType ?? 'value';
  const xType = opts.xType ?? 'value';
  return {
    xAxis: {
      type: xType,
      ...(opts.xData ? { data: opts.xData } : {}),
      axisLine: { lineStyle: { color: THEME.grid } },
      axisLabel: { color: THEME.muted },
      splitLine: { show: false },
    },
    yAxis: {
      type: yType,
      ...(opts.yMin !== undefined ? { min: opts.yMin } : {}),
      ...(opts.yMax !== undefined ? { max: opts.yMax } : {}),
      axisLine: { show: false },
      splitLine: { lineStyle: { color: THEME.grid } },
      axisLabel: { color: THEME.muted },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: THEME.surface,
      borderColor: THEME.grid,
      textStyle: { color: THEME.fg, fontSize: 11 },
    },
  };
}

/** Standard horizontal markLine — useful for entropy floors / chance baselines. */
export function refLine(yValue: number, label: string, color: string = THEME.accent) {
  return {
    silent: true,
    symbol: 'none',
    animation: false,
    data: [{
      yAxis: yValue,
      lineStyle: { color, type: 'dotted' as const, width: 1, opacity: 0.5 },
      label: {
        formatter: label,
        position: 'insideEndTop' as const,
        color: THEME.muted,
        fontSize: 9,
      },
    }],
  };
}

/** Single-series line — common case. */
export function lineSeries(data: number[] | [number, number][], opts: {
  color?: string;
  width?: number;
  symbol?: boolean;
  symbolSize?: number;
  area?: boolean;
  name?: string;
  markLine?: ReturnType<typeof refLine>;
} = {}) {
  return {
    type: 'line' as const,
    data,
    showSymbol: opts.symbol ?? false,
    ...(opts.symbolSize !== undefined ? { symbolSize: opts.symbolSize } : {}),
    ...(opts.name ? { name: opts.name } : {}),
    lineStyle: { width: opts.width ?? 1.5, color: opts.color ?? THEME.accent },
    ...(opts.area ? { areaStyle: { color: opts.color ?? THEME.accent, opacity: 0.08 } } : {}),
    ...(opts.markLine ? { markLine: opts.markLine } : {}),
  };
}

/** Standard legend block (top-right, small). */
export function legendBlock() {
  return {
    top: 2,
    right: 12,
    textStyle: { color: THEME.muted, fontSize: 10 },
    icon: 'circle',
    itemWidth: 8,
    itemHeight: 8,
  };
}
