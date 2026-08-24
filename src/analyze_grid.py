"""
Phase 5 -- turns results/grid_results.csv into summary tables and figures.
Safe to run at any point while the grid is still filling in (run_grid.py is
resumable) -- every plot here degrades gracefully (skipped with a printed
note, never a crash) if the data it needs isn't in yet.

Palette follows the project's dataviz reference palette (categorical slots
1/2/3 = blue/orange/aqua) for model-colored charts, and the same
chain-blue / tree-amber / graph-crimson triad already used in
kill_chain_pipeline.png and the flow diagram for method-colored charts, so
color means the same thing across every artifact in this project.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CSV_PATH = RESULTS_DIR / "grid_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

SCENARIO_ORDER = ["WS12", "Ubuntu", "Sidewinder", "FIN6", "APT29"]
METHOD_ORDER = ["CoT", "ToT", "GoT"]
MODEL_ORDER = ["qwen2.5-7b", "llama3.1-8b"]

# -- palette --------------------------------------------------------------
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"

METHOD_COLOR = {"CoT": "#3f7cac", "ToT": "#b9791f", "GoT": "#a6373a"}
MODEL_COLOR = {"qwen2.5-7b": "#2a78d6", "llama3.1-8b": "#eb6834"}


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", colors=INK_MUTED, length=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def load_grid() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} doesn't exist yet -- run src/run_grid.py first")
    df = pd.read_csv(CSV_PATH)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def write_summaries(df: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["event_precision", "event_recall", "event_f1", "stage_precision", "stage_recall",
               "stage_f1", "stage_recall_partial", "hallucination_rate", "llm_calls", "total_tokens", "wall_clock_s"]

    def agg(group_cols, name):
        g = df.groupby(group_cols, observed=True)[metrics].agg(["mean", "std", "count"])
        g.columns = ["_".join(c) for c in g.columns]
        out_path = RESULTS_DIR / f"summary_{name}.csv"
        g.to_csv(out_path)
        print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({len(g)} rows)")
        return g

    agg(["scenario", "method", "model"], "by_scenario_method_model")
    agg(["method", "model"], "by_method_model")
    agg(["method"], "by_method")
    agg(["scenario", "method"], "by_scenario_method")


def plot_per_scenario_topology_x_model(df: pd.DataFrame, metric="stage_recall"):
    """The user's original request: for a given scenario, plot accuracy
    against the 3 topologies and 3 models. One panel per scenario present
    in the data, topology on x-axis, one bar per model."""
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    if not scenarios:
        print("skip plot_per_scenario_topology_x_model: no data")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 4.2), facecolor=PAGE, squeeze=False)
    axes = axes[0]
    width = 0.25

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        for i, model in enumerate(MODEL_ORDER):
            vals = [sub[(sub["method"] == m) & (sub["model"] == model)][metric].mean() for m in METHOD_ORDER]
            xs = [j + (i - 1) * width for j in range(len(METHOD_ORDER))]
            ax.bar(xs, vals, width=width, color=MODEL_COLOR[model], label=model, zorder=3)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(METHOD_ORDER, fontsize=9, color=INK)
        ax.set_title(scenario, fontsize=11, fontweight="bold", color=INK)
        ax.set_ylim(0, 1)
        _style_axis(ax)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.suptitle(f"{metric.replace('_', ' ').title()} by topology and model, per scenario", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 0.95, 0.94])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"per_scenario_topology_x_model_{metric}.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def plot_headline(df: pd.DataFrame, metric="stage_recall"):
    """The single figure the hypothesis rests on: metric by topology,
    averaged across every scenario/model/seed, with std-dev error bars."""
    means = [df[df["method"] == m][metric].mean() for m in METHOD_ORDER]
    stds = [df[df["method"] == m][metric].std() for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=PAGE)
    bars = ax.bar(METHOD_ORDER, means, yerr=stds, capsize=4,
                   color=[METHOD_COLOR[m] for m in METHOD_ORDER], zorder=3,
                   error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2})
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{mean:.2f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric.replace('_', ' ').title()}: CoT vs ToT vs GoT\n(mean ± std across all scenarios, models, seeds)",
                 fontsize=11, fontweight="bold", color=INK)
    _style_axis(ax)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"headline_{metric}.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def plot_cross_scenario(df: pd.DataFrame, metric="stage_recall"):
    """Cross-scenario comparison: scenario on x-axis, grouped by method --
    shows whether the topology ranking holds as scenario scale/difficulty
    changes (WS12's 6 nodes vs. APT29's 37), and which scenarios are
    generally easier or harder for the whole pipeline."""
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    if not scenarios:
        print("skip plot_cross_scenario: no data")
        return

    fig, ax = plt.subplots(figsize=(2.2 * len(scenarios) + 2, 4.5), facecolor=PAGE)
    width = 0.25
    for i, method in enumerate(METHOD_ORDER):
        vals = [df[(df["scenario"] == s) & (df["method"] == method)][metric].mean() for s in scenarios]
        xs = [j + (i - 1) * width for j in range(len(scenarios))]
        ax.bar(xs, vals, width=width, color=METHOD_COLOR[method], label=method, zorder=3)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.replace('_', ' ').title()} across scenarios, by topology", fontsize=12, fontweight="bold", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"cross_scenario_{metric}.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def plot_scenario_method_heatmap(df: pd.DataFrame, metric="stage_recall"):
    """Compact single-figure overview of the whole grid's main comparison:
    one cell per (scenario, method), averaged across models/seeds."""
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    if not scenarios:
        print("skip plot_scenario_method_heatmap: no data")
        return

    grid = df.pivot_table(index="method", columns="scenario", values=metric, aggfunc="mean", observed=True)
    grid = grid.reindex(index=METHOD_ORDER, columns=scenarios)

    fig, ax = plt.subplots(figsize=(1.4 * len(scenarios) + 2, 3.2), facecolor=PAGE)
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=9, color=INK)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=9, color=INK)
    for i in range(len(METHOD_ORDER)):
        for j in range(len(scenarios)):
            val = grid.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else INK, fontsize=9)
    ax.set_title(f"{metric.replace('_', ' ').title()} -- scenario × topology overview", fontsize=11, fontweight="bold", color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric.replace("_", " "))
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"heatmap_scenario_x_method_{metric}.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def plot_pareto(df: pd.DataFrame, metric="stage_recall"):
    """Accuracy-vs-cost: one point per (method, model), averaged across
    scenarios/seeds -- answers whether GoT's extra cost is worth it."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PAGE)
    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            sub = df[(df["method"] == method) & (df["model"] == model)]
            if sub.empty:
                continue
            ax.scatter(sub["total_tokens"].mean(), sub[metric].mean(), s=90,
                       color=METHOD_COLOR[method], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.annotate(f"{method}/{model}", (sub["total_tokens"].mean(), sub[metric].mean()),
                        fontsize=7, color=INK_SOFT, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean total tokens per run", fontsize=9, color=INK_MUTED)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9, color=INK_MUTED)
    ax.set_title("Accuracy vs. cost", fontsize=12, fontweight="bold", color=INK)
    _style_axis(ax)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"pareto_{metric}_vs_cost.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def main():
    df = load_grid()
    print(f"Loaded {len(df)} rows from {CSV_PATH.relative_to(REPO_ROOT)}\n")
    write_summaries(df)
    print()
    for metric in ["stage_recall", "event_f1"]:
        plot_per_scenario_topology_x_model(df, metric)
        plot_headline(df, metric)
        plot_cross_scenario(df, metric)
        plot_scenario_method_heatmap(df, metric)
    plot_pareto(df, "stage_recall")


if __name__ == "__main__":
    main()
