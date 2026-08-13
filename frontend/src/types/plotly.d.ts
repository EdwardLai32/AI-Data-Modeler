/**
 * Minimal ambient types for `plotly.js-dist-min`, which ships no declarations.
 *
 * Only the four calls this app makes are declared. `data` and `layout` stay
 * `unknown`/`Record<string, unknown>` because the figures come from
 * `plotly.io.write_json` on the Python side — the frontend passes them through
 * untouched rather than re-deriving a schema for them here.
 */
declare module "plotly.js-dist-min" {
  interface PlotlyStatic {
    newPlot(
      root: HTMLElement,
      data: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<HTMLElement>;
    react(
      root: HTMLElement,
      data: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<HTMLElement>;
    relayout(root: HTMLElement, layout: Record<string, unknown>): Promise<HTMLElement>;
    purge(root: HTMLElement): void;
    Plots: { resize(root: HTMLElement): void };
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
