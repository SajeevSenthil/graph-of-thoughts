"""
Phase 1 -- visible-output artifact.

Renders a small-multiples summary of the parsed ground truth (data/ground_truth/
all_scenarios.json) so Phase 1's output is something that can be dropped
straight into the paper/report, not just a pile of JSON.

Colors follow the project's data-viz reference palette (dataviz skill,
references/palette.md): chart chrome + single sequential blue hue, since each
panel is a single series (a stage-count bar chart needs no categorical
identity color -- position on the x-axis already distinguishes the stages).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "all_scenarios.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "summary.png"

# Fixed, consistent stage order across every panel (a conventional
# kill-chain / MITRE-tactic ordering) so the same stage always lands in the
# same x position across scenarios -- makes the panels visually comparable.
STAGE_ORDER = [
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
]
STAGE_CODE = {
    "Initial Access": "IA",
    "Execution": "EX",
    "Persistence": "PE",
    "Privilege Escalation": "PR",
    "Defense Evasion": "DE",
    "Credential Access": "CA",
    "Discovery": "DI",
    "Lateral Movement": "LM",
    "Collection": "CO",
    "Command and Control": "C2",
}

# dataviz reference palette (light mode)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BAR_HUE = "#256abf"  # sequential blue, step 500


def load_counts():
    scenarios = json.loads(DATA_PATH.read_text())
    counts = {}
    for scenario, nodes in scenarios.items():
        c = {stage: 0 for stage in STAGE_ORDER}
        for n in nodes:
            c[n["stage"]] = c.get(n["stage"], 0) + 1
        counts[scenario] = c
    return counts


def main():
    counts = load_counts()
    scenario_order = ["WS12", "Ubuntu", "Sidewinder", "FIN6", "APT29"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=PAGE)
    fig.suptitle(
        "Ground-truth kill-chain nodes per scenario, by attack stage",
        fontsize=13, color=INK_PRIMARY, fontweight="bold", x=0.02, ha="left", y=1.04,
    )

    stage_labels = [STAGE_CODE[s] for s in STAGE_ORDER]

    for ax, scenario in zip(axes, scenario_order):
        c = counts[scenario]
        values = [c[s] for s in STAGE_ORDER]
        total = sum(values)

        ax.set_facecolor(SURFACE)
        bars = ax.bar(range(len(STAGE_ORDER)), values, color=BAR_HUE, width=0.62, zorder=3)

        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + max(values) * 0.03,
                    str(v), ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY,
                )

        ax.set_title(f"{scenario}  ({total} nodes)", fontsize=11, color=INK_PRIMARY, fontweight="bold", pad=10)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(stage_labels, fontsize=8, color=INK_MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)

    key = "  ".join(f"{STAGE_CODE[s]}={s}" for s in STAGE_ORDER)
    fig.text(0.02, -0.005, key, fontsize=7.5, color=INK_MUTED)
    fig.text(
        0.02, -0.045,
        "89 ground-truth nodes total across 5 scenarios. Every (stage, match_substring) pair hand-verified "
        "against attack_annotation/*.txt and attack_analysis.xls -- see src/parse_ground_truth.py.",
        fontsize=8, color=INK_MUTED,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=200, facecolor=PAGE, bbox_inches="tight")
    print(f"Saved {OUT_PATH.relative_to(OUT_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
