"""
Phase 5 -- full evaluation grid runner: 5 scenarios x 3 methods x 3 models x
N seeds. Resumable (skips any combo already present in results/grid_results.csv,
so an interrupted run picks back up instead of restarting) and stores results
at three levels of detail:

  results/raw/{scenario}__{method}__{model}__seed{N}.json  -- full output +
      cost + metadata for that one run (source of truth, append-only)
  results/grid_results.csv                                  -- one flat row
      per run, everything scorer.py + the method's cost snapshot produced
  results/summary_*.csv                                     -- built separately
      by src/analyze_grid.py once enough of the grid has run

One model is loaded per OS process, not per Python call within one process:
vLLM does not reliably release GPU memory when a model object is dropped and
a new one constructed in the same process, so `main()` with no --model given
re-execs this same script as a child process once per model. Each child gets
a clean process (and therefore guaranteed-clean GPU memory) on exit, while a
single top-level command still runs the entire grid unattended.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SERIALIZED_DIR = REPO_ROOT / "data" / "serialized"
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
CSV_PATH = RESULTS_DIR / "grid_results.csv"

SCENARIOS = ["WS12", "Ubuntu", "Sidewinder", "FIN6", "APT29"]
MODELS = ["qwen2.5-7b", "llama3.1-8b", "phi4"]
DEFAULT_SEEDS = [0, 1, 2]

CSV_FIELDS = [
    "scenario", "method", "model", "seed",
    "event_precision", "event_recall", "event_f1", "event_tp", "event_fp", "event_fn",
    "stage_precision", "stage_recall", "stage_f1", "stage_recall_partial", "stage_tp", "stage_fp", "stage_fn",
    "hallucination_rate",
    "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens", "wall_clock_s", "parse_retries",
    "parse_ok", "timestamp", "raw_path",
]


def _load_done_combos() -> set[tuple]:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {
            (row["scenario"], row["method"], row["model"], int(row["seed"]))
            for row in csv.DictReader(f)
        }


def _append_row(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flatten(scenario, method, model, seed, output, metrics, raw_path) -> dict:
    return {
        "scenario": scenario, "method": method, "model": model, "seed": seed,
        "event_precision": metrics["event"]["precision"],
        "event_recall": metrics["event"]["recall"],
        "event_f1": metrics["event"]["f1"],
        "event_tp": metrics["event"]["tp"], "event_fp": metrics["event"]["fp"], "event_fn": metrics["event"]["fn"],
        "stage_precision": metrics["stage"]["precision"],
        "stage_recall": metrics["stage"]["recall"],
        "stage_f1": metrics["stage"]["f1"],
        "stage_recall_partial": metrics["stage"]["recall_partial"],
        "stage_tp": metrics["stage"]["tp"], "stage_fp": metrics["stage"]["fp"], "stage_fn": metrics["stage"]["fn"],
        "hallucination_rate": metrics["hallucination_rate"],
        "llm_calls": output["_cost"]["llm_calls"],
        "prompt_tokens": output["_cost"]["prompt_tokens"],
        "completion_tokens": output["_cost"]["completion_tokens"],
        "total_tokens": output["_cost"]["total_tokens"],
        "wall_clock_s": output["_cost"]["wall_clock_s"],
        "parse_retries": output["_cost"]["parse_retries"],
        "parse_ok": output["_parse_ok"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path.relative_to(REPO_ROOT)),
    }


def run_for_one_model(model_key: str, scenarios: list[str], methods: list[str], seeds: list[int]):
    # Imports deferred to here: this function only ever runs inside a child
    # process dedicated to one model, so vLLM/torch only ever get imported
    # once per process, right before they're needed.
    from src.llm_client import VLLMLanguageModel
    from src.methods.cot import run_cot
    from src.methods.tot import run_tot
    from src.methods.got import run_got
    from src.scorer import score_scenario

    method_fns = {
        "CoT": run_cot,
        "ToT": lambda e, l: run_tot(e, l, k=3),
        "GoT": lambda e, l: run_got(e, l, n_segments=4),
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done_combos()

    todo = [
        (s, m, seed)
        for s in scenarios
        for m in methods
        for seed in seeds
        if (s, m, model_key, seed) not in done
    ]
    print(f"[{model_key}] {len(todo)} combos to run ({len(scenarios)*len(methods)*len(seeds) - len(todo)} already done)")
    if not todo:
        return

    lm = VLLMLanguageModel(model_key, temperature=0.7, max_tokens=1536, seed=seeds[0])

    evidence_cache = {}
    for i, (scenario, method, seed) in enumerate(todo, 1):
        if scenario not in evidence_cache:
            evidence_cache[scenario] = (SERIALIZED_DIR / f"{scenario}.txt").read_text()
        evidence_text = evidence_cache[scenario]

        lm.seed = seed
        t0 = time.time()
        try:
            output = method_fns[method](evidence_text, lm)
        except Exception as exc:  # noqa: BLE001 -- a single failed combo must not kill the whole grid
            print(f"  [{i}/{len(todo)}] {scenario}/{method}/seed{seed}: FAILED -- {exc!r}")
            continue

        metrics = score_scenario(scenario, output)

        raw_path = RAW_DIR / f"{scenario}__{method}__{model_key}__seed{seed}.json"
        raw_path.write_text(json.dumps({"output": output, "score": metrics}, indent=2))

        _append_row(_flatten(scenario, method, model_key, seed, output, metrics, raw_path))
        print(
            f"  [{i}/{len(todo)}] {scenario:11s} {method:4s} seed{seed} -> "
            f"stage_recall={metrics['stage']['recall']:.2f} event_f1={metrics['event']['f1']:.2f} "
            f"calls={output['_cost']['llm_calls']} tokens={output['_cost']['total_tokens']} "
            f"({time.time() - t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None, help="Run only this model (used internally by the per-model subprocess dispatch)")
    parser.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    parser.add_argument("--methods", nargs="*", default=["CoT", "ToT", "GoT"])
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    if args.model is not None:
        run_for_one_model(args.model, args.scenarios, args.methods, args.seeds)
        return

    # No --model: dispatch one child process per model so each gets a clean
    # GPU memory state, rather than trying to load/unload models in-process.
    for model_key in MODELS:
        print(f"\n{'=' * 70}\nStarting subprocess for model: {model_key}\n{'=' * 70}")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--model", model_key,
            "--scenarios", *args.scenarios,
            "--methods", *args.methods,
            "--seeds", *[str(s) for s in args.seeds],
        ]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"WARNING: subprocess for {model_key} exited with code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
