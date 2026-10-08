"""
Compares grid variants against each other (read-only over existing results):
  v1_baseline   results/grid_results.csv
  <name>        results/variants/<name>/grid_results.csv

Paired comparisons match runs on (scenario, method, model, seed), so each
variant is compared like-for-like rather than on pooled averages.

Usage: python src/compare_variants.py v1_baseline v2_maxtok4096 v3_stagedefs
Writes results/variants/comparison_<a>_vs_<b>.csv for each consecutive pair.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.scorer import _contains  # noqa: E402

METRICS = ["event_f1", "event_recall", "stage_recall", "stage_recall_partial", "hallucination_rate"]
KEY = ["scenario", "method", "model", "seed"]


def load(variant: str) -> pd.DataFrame:
    path = REPO / "results" / "grid_results.csv" if variant == "v1_baseline" else REPO / "results" / "variants" / variant / "grid_results.csv"
    df = pd.read_csv(path)
    df["parse_ok"] = df["parse_ok"].astype(bool)
    return df


def funnel(df: pd.DataFrame) -> dict:
    total = detected = correct = ia_pred = all_pred = 0
    for _, r in df.iterrows():
        flagged = json.loads((REPO / r.raw_path).read_text())["output"].get("malicious_events", [])
        all_pred += len(flagged)
        ia_pred += sum(1 for e in flagged if (e.get("stage") or "").strip().lower() == "initial access")
        for n in json.loads((REPO / "data" / "ground_truth" / f"{r.scenario}.json").read_text()):
            if n["match_substring"] is None:
                continue
            total += 1
            hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
            if hits:
                detected += 1
                if any((e.get("stage") or "").strip().lower() == n["stage"].strip().lower() for e in hits):
                    correct += 1
    return {
        "gt_events": total,
        "detected_pct": round(100 * detected / total, 1),
        "stage_correct_of_detected_pct": round(100 * correct / detected, 1) if detected else 0.0,
        "initial_access_pred_pct": round(100 * ia_pred / all_pred, 1) if all_pred else 0.0,
    }


def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name}: {len(df)} runs, parse failures {int((~df.parse_ok).sum())} ===")
    small = df[df.scenario.isin(["WS12", "Ubuntu"])]
    print(small.groupby(["method", "model"])[["event_f1", "stage_recall", "stage_recall_partial"]].mean().round(2).to_string())
    print("funnel (all runs):", funnel(df))


def paired(a_name: str, a: pd.DataFrame, b_name: str, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    rows = []
    for scope, sub in [("all", m), ("qwen2.5-7b", m[m.model == "qwen2.5-7b"]), ("llama3.1-8b", m[m.model == "llama3.1-8b"])]:
        for metric in METRICS:
            x, y = sub[f"{metric}_a"], sub[f"{metric}_b"]
            try:
                p = stats.wilcoxon(y, x).pvalue
            except ValueError:
                p = float("nan")
            rows.append([scope, metric, len(sub), round(x.mean(), 3), round(y.mean(), 3), round((y - x).mean(), 3), round(p, 4)])
    out = pd.DataFrame(rows, columns=["scope", "metric", "pairs", f"mean_{a_name}", f"mean_{b_name}", "mean_diff", "wilcoxon_p"])
    path = REPO / "results" / "variants" / f"comparison_{a_name}_vs_{b_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n=== {b_name} vs {a_name} (paired on {KEY}) -> {path.relative_to(REPO)}")
    print(out.to_string(index=False))
    return out


def main():
    names = sys.argv[1:] or ["v1_baseline", "v2_maxtok4096", "v3_stagedefs"]
    dfs = {n: load(n) for n in names}
    for n, df in dfs.items():
        summarize(n, df)
    for a, b in zip(names, names[1:]):
        paired(a, dfs[a], b, dfs[b])


if __name__ == "__main__":
    main()
