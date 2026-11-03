/** Plotly charts with the app's shared styling. */
import Plotly from "plotly.js-dist-min";
import { h } from "./widgets";

// Categorical slots 1-3 of the validated reference palette (as in the Python app)
export const C1 = "#2a78d6";
export const C2 = "#eb6834";
export const C3 = "#1baf7a";
export const MUTED = "rgba(128,128,128,0.25)";
export const GRAY = "#8a8a8a";

export type Trace = Partial<Plotly.ScatterData>;
export type Layout = Partial<Plotly.Layout>;

/** y-axis domains for stacked subplots sharing one x axis (top first). */
export function stacked(heights: number[], gap = 0.06): [number, number][] {
  const total = heights.reduce((a, b) => a + b, 0);
  const avail = 1 - gap * (heights.length - 1);
  const out: [number, number][] = [];
  let top = 1;
  for (const hgt of heights) {
    const span = (hgt / total) * avail;
    out.push([Math.max(top - span, 0), top]);
    top -= span + gap;
  }
  return out;
}

const AXIS = { showgrid: true, gridwidth: 1, gridcolor: MUTED, zeroline: false, automargin: true };

/** A chart that keeps its div and redraws in place. */
export class Chart {
  readonly el: HTMLDivElement;

  constructor(private readonly height = 360, private readonly toolbar = true) {
    this.el = h("div.chart");
    this.el.style.height = `${height}px`;
  }

  draw(traces: Trace[], layout: Layout, title = ""): void {
    const yaxes: Record<string, unknown> = {};
    for (const k of Object.keys(layout)) {
      if (!/^[xy]axis\d*$/.test(k)) continue;
      const ax = (layout as Record<string, Record<string, unknown>>)[k];
      // fewer labelled minor ticks on log axes
      yaxes[k] = { ...AXIS, ...(ax.type === "log" ? { nticks: 7 } : {}), ...ax };
    }
    const full: Layout = {
      height: this.height,
      margin: { l: 10, r: 10, t: title ? 70 : 30, b: 10 },
      title: title ? { text: title, y: 0.98, yanchor: "top", x: 0.01, xanchor: "left", font: { size: 15 } } : undefined,
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 12 },
      ...layout,
      ...yaxes,
    };
    if (!("xaxis" in layout)) (full as Record<string, unknown>).xaxis = AXIS;
    if (!("yaxis" in layout)) (full as Record<string, unknown>).yaxis = AXIS;
    void Plotly.react(this.el, traces as Plotly.Data[], full, { responsive: true, displaylogo: false, displayModeBar: this.toolbar ? "hover" : false });
  }
}

/** A vertical dotted marker line across the whole plot area. */
export function vline(x: number, color = C1): Partial<Plotly.Shape> {
  return { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color, dash: "dot", width: 1 } };
}

/** A horizontal dotted line on one y axis. */
export function hline(y: number, yref: string, color = C2): Partial<Plotly.Shape> {
  return { type: "line", xref: "paper", yref: yref as Plotly.YAxisName, x0: 0, x1: 1, y0: y, y1: y,
           line: { color, dash: "dot", width: 1 } };
}

/** NaN-safe copy for Plotly (gaps instead of NaN/inf). */
export const gaps = (a: number[]): (number | null)[] => a.map((v) => (Number.isFinite(v) ? v : null));
