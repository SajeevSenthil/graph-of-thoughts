"""
Phase 5 -- interpretability follow-up, not part of the original topology
comparison. Aggregate significance tests on event_f1/stage_recall showed no
method-vs-method difference clears p<0.05 anywhere in the grid (see
IMPLEMENTATION.md) -- the more informative question turned out to be *why*
stage_recall floors near zero at all, independent of topology.

For every ground-truth node with a match_substring, checks whether any
flagged event in that run's output matched it (via scorer.py's own
_contains), and if so, what stage was predicted vs. the true stage. This is
strictly a read of results/raw/*.json + data/ground_truth/*.json -- no new
inference, no change to scorer.py's actual scoring logic.

Two outputs:
  results/stage_confusion_matrix.csv    -- (true_stage, pred_stage, count)
      for every matched event across the whole grid
  results/stage_prediction_distribution.csv -- (model, stage, count, frac)
      over ALL flagged events (matched, mislabeled, or pure false positive)
  results/plots/stage_confusion_heatmap_{model}.png
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "results" / "raw"
GT_DIR = REPO_ROOT / "data" / "ground_truth"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

from src.scorer import _contains  # noqa: E402
from src.methods.prompter import STAGES  # noqa: E402

STAGE_ORDER = [s.lower() for s in STAGES]

SURFACE, PAGE, INK, INK_MUTED = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#898781"


def load_ground_truth(scenario: str) -> list[dict]:
    return json.loads((GT_DIR / f"{scenario}.json").read_text())


def collect():
    confusion = collections.Counter()  # (true_stage, pred_stage) -> count
    pred_freq = collections.defaultdict(collections.Counter)  # model -> stage -> count
    missed = 0
    matched = 0
    node_correct = 0
    gt_cache = {}

    for f in sorted(RAW_DIR.glob("*.json")):
        scenario, method, model, seedtag = f.stem.split("__")
        output = json.loads(f.read_text())["output"]
        flagged = output.get("malicious_events", [])
        if scenario not in gt_cache:
            gt_cache[scenario] = load_ground_truth(scenario)
        gt = gt_cache[scenario]

        for e in flagged:
            stage = (e.get("stage") or "").strip().lower()
            pred_freq[model][stage] += 1

        for n in gt:
            if n["match_substring"] is None:
                continue
            true_stage = n["stage"].strip().lower()
            hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
            if not hits:
                missed += 1
                continue
            matched += 1
            if any((e.get("stage") or "").strip().lower() == true_stage for e in hits):
                node_correct += 1
            for e in hits:
                pred_stage = (e.get("stage") or "").strip().lower()
                confusion[(true_stage, pred_stage)] += 1

    return confusion, pred_freq, matched, missed, node_correct


def write_csvs(confusion, pred_freq):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"true_stage": t, "pred_stage": p, "count": c} for (t, p), c in confusion.items()]
    pd.DataFrame(rows).sort_values(["true_stage", "count"], ascending=[True, False]).to_csv(
        RESULTS_DIR / "stage_confusion_matrix.csv", index=False
    )
    print(f"Wrote {RESULTS_DIR / 'stage_confusion_matrix.csv'} ({len(rows)} rows)")

    rows = []
    for model, counter in pred_freq.items():
        total = sum(counter.values())
        for stage, c in counter.items():
            rows.append({"model": model, "stage": stage, "count": c, "frac": c / total})
    pd.DataFrame(rows).sort_values(["model", "count"], ascending=[True, False]).to_csv(
        RESULTS_DIR / "stage_prediction_distribution.csv", index=False
    )
    print(f"Wrote {RESULTS_DIR / 'stage_prediction_distribution.csv'} ({len(rows)} rows)")


def plot_heatmap(confusion, model_matched_subset, model_name, gt_stage_universe):
    stages = [s for s in STAGE_ORDER if s in gt_stage_universe or any(k[1] == s for k in confusion)]
    grid = pd.DataFrame(0, index=stages, columns=stages, dtype=int)
    for (t, p), c in confusion.items():
        if t in grid.index and p in grid.columns:
            grid.loc[t, p] += c

    fig, ax = plt.subplots(figsize=(0.55 * len(stages) + 3, 0.55 * len(stages) + 2), facecolor=PAGE)
    im = ax.imshow(grid.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=7.5, color=INK, rotation=45, ha="right")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=7.5, color=INK)
    ax.set_xlabel("Predicted stage", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("True stage", fontsize=9, color=INK_MUTED)
    for i in range(len(stages)):
        for j in range(len(stages)):
            val = grid.values[i, j]
            if val > 0:
                color = "white" if val > grid.values.max() * 0.5 else INK
                weight = "bold" if i == j else "normal"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color, fontweight=weight)
    # mark the diagonal (correct predictions) with an outline
    for i in range(len(stages)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#2a9d5c", linewidth=1.5))
    ax.set_title(f"Stage confusion -- {model_name}\n(matched events only; diagonal = correct)", fontsize=11, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"stage_confusion_heatmap_{model_name}.png"
    fig.savefig(out, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def main():
    confusion, pred_freq, matched, missed, node_correct = collect()
    write_csvs(confusion, pred_freq)

    total = matched + missed
    correct = node_correct
    print(f"\nMatched (event found at all): {matched}/{total} ({matched/total:.1%})")
    print(f"Of matched, exact stage correct: {correct}/{matched} ({correct/matched:.1%})")

    # Per-model confusion heatmaps: split raw files by model first.
    for model in ["qwen2.5-7b", "llama3.1-8b"]:
        model_confusion = collections.Counter()
        gt_stage_universe = set()
        for f in sorted(RAW_DIR.glob(f"*__{model}__*.json")):
            scenario = f.stem.split("__")[0]
            gt = load_ground_truth(scenario)
            gt_stage_universe.update(n["stage"].strip().lower() for n in gt if n["match_substring"])
            output = json.loads(f.read_text())["output"]
            flagged = output.get("malicious_events", [])
            for n in gt:
                if n["match_substring"] is None:
                    continue
                true_stage = n["stage"].strip().lower()
                hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
                for e in hits:
                    pred_stage = (e.get("stage") or "").strip().lower()
                    model_confusion[(true_stage, pred_stage)] += 1
        if model_confusion:
            plot_heatmap(model_confusion, None, model, gt_stage_universe)


if __name__ == "__main__":
    main()
