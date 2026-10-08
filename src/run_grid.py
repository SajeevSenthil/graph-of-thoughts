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
MODELS = ["qwen2.5-7b", "llama3.1-8b"]
# Scale test (larger siblings, 4-bit AWQ); never run by default, only via --models.
LARGE_MODELS = ["qwen2.5-32b-awq", "llama3.1-70b-awq"]
DEFAULT_SEEDS = [0, 1, 2]

# Qwen 2.5 7B's real, vLLM-verified native context ceiling is 32768 tokens
# (see llm_client.py's MODEL_MAX_MODEL_LEN). APT29/Sidewinder/FIN6 share one
# W10 log and serialize to ~58K tokens each -- well over that limit. CoT/ToT
# need the full evidence in a single call, so there is no safe way to run
# Qwen on those three scenarios: forcing a longer context risks NaN output
# on this RoPE model past its trained length, and truncating evidence would
# confound "the model reasoned badly" with "the model never saw the
# evidence." Decision (with user, 2026-08-24): Qwen runs on WS12 + Ubuntu
# only (both fit under 32768 tokens); Llama runs all 5 scenarios (native
# 131072 context safely covers even the largest scenario at the 65536 cap).
MODEL_SCENARIO_ALLOWLIST = {
    "qwen2.5-7b": ["WS12", "Ubuntu"],
    "llama3.1-8b": SCENARIOS,
    "qwen2.5-32b-awq": ["WS12", "Ubuntu"],  # same 32K native limit as the 7B
    "llama3.1-70b-awq": SCENARIOS,
}

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


BASELINE_MAX_TOKENS = 1536


def _set_variant(variant: str | None):
    """Point all result paths at results/variants/<variant>/ so a variant run
    can never append to, or overwrite, the v1 baseline under results/."""
    global RESULTS_DIR, RAW_DIR, CSV_PATH
    if variant:
        RESULTS_DIR = REPO_ROOT / "results" / "variants" / variant
        RAW_DIR = RESULTS_DIR / "raw"
        CSV_PATH = RESULTS_DIR / "grid_results.csv"


def run_for_one_model(
    model_key: str,
    scenarios: list[str],
    methods: list[str],
    seeds: list[int],
    variant: str | None = None,
    max_tokens: int = BASELINE_MAX_TOKENS,
    prompt_variant: str = "base",
    gpu_mem_util: float | None = None,
    triage: bool = False,
    got_link_hints: bool = False,
):
    # Imports deferred to here: this function only ever runs inside a child
    # process dedicated to one model, so vLLM/torch only ever get imported
    # once per process, right before they're needed.
    from src.llm_client import VLLMLanguageModel
    from src.methods.cot import run_cot
    from src.methods.tot import run_tot
    from src.methods.got import run_got
    from src.scorer import score_scenario, _contains
    from src.methods.triage import run_triage
    import src.methods.prompter as prompter_mod
    import torch

    prompter_mod.PROMPT_VARIANT = prompt_variant

    method_fns = {
        "CoT": run_cot,
        "ToT": lambda e, l: run_tot(e, l, k=3),
        "GoT": lambda e, l: run_got(e, l, n_segments=4, link_hints=got_link_hints),
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done_combos()

    allowed_scenarios = MODEL_SCENARIO_ALLOWLIST.get(model_key, scenarios)
    skipped_scenarios = [s for s in scenarios if s not in allowed_scenarios]
    if skipped_scenarios:
        print(
            f"[{model_key}] skipping {skipped_scenarios} -- exceeds this model's safe "
            f"context window (see MODEL_SCENARIO_ALLOWLIST)"
        )
    scenarios = [s for s in scenarios if s in allowed_scenarios]

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

    lm = VLLMLanguageModel(
        model_key, temperature=0.7, max_tokens=max_tokens, seed=seeds[0], gpu_memory_utilization=gpu_mem_util
    )
    meta = {
        "variant": variant or "v1_baseline",
        "max_tokens": max_tokens,
        "prompt_variant": prompt_variant,
        "gpu": torch.cuda.get_device_name(0),
        "triage": triage,
        "got_link_hints": got_link_hints,
    }

    def triaged_evidence(scenario: str, seed: int, full_text: str) -> str:
        # Once per (scenario, model, seed), persisted, so all three topologies
        # -- including after a job restart -- reason over the identical subset.
        path = RESULTS_DIR / "triage" / f"{scenario}__{model_key}__seed{seed}.json"
        if path.exists():
            return json.loads(path.read_text())["evidence_text"]
        lm.seed = seed
        result = run_triage(full_text, lm)
        gt = [n for n in json.loads((REPO_ROOT / "data" / "ground_truth" / f"{scenario}.json").read_text()) if n["match_substring"]]
        result["gt_in_full"] = sum(1 for n in gt if _contains(n["match_substring"], full_text))
        result["gt_after_triage"] = sum(1 for n in gt if _contains(n["match_substring"], result["evidence_text"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(
            f"  [triage] {scenario} seed{seed}: kept {result['n_flagged']}/{result['n_blocks']} processes, "
            f"{result['chars_after']:,}/{result['chars_before']:,} chars, "
            f"ground truth kept {result['gt_after_triage']}/{result['gt_in_full']} "
            f"({result['cost']['wall_clock_s']:.1f}s)"
        )
        return result["evidence_text"]

    evidence_cache = {}
    for i, (scenario, method, seed) in enumerate(todo, 1):
        if scenario not in evidence_cache:
            evidence_cache[scenario] = (SERIALIZED_DIR / f"{scenario}.txt").read_text()
        evidence_text = evidence_cache[scenario]
        if triage:
            evidence_text = triaged_evidence(scenario, seed, evidence_text)

        lm.seed = seed
        t0 = time.time()
        try:
            output = method_fns[method](evidence_text, lm)
        except Exception as exc:  # noqa: BLE001 -- a single failed combo must not kill the whole grid
            print(f"  [{i}/{len(todo)}] {scenario}/{method}/seed{seed}: FAILED -- {exc!r}")
            continue

        metrics = score_scenario(scenario, output)

        raw_path = RAW_DIR / f"{scenario}__{method}__{model_key}__seed{seed}.json"
        raw_path.write_text(json.dumps({"output": output, "score": metrics, "meta": meta}, indent=2))

        _append_row(_flatten(scenario, method, model_key, seed, output, metrics, raw_path))
        print(
            f"  [{i}/{len(todo)}] {scenario:11s} {method:4s} seed{seed} -> "
            f"stage_recall={metrics['stage']['recall']:.2f} event_f1={metrics['event']['f1']:.2f} "
            f"calls={output['_cost']['llm_calls']} tokens={output['_cost']['total_tokens']} "
            f"({time.time() - t0:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + LARGE_MODELS, default=None, help="Run only this model (used internally by the per-model subprocess dispatch)")
    parser.add_argument("--models", nargs="*", choices=MODELS + LARGE_MODELS, default=MODELS, help="Models to dispatch, one subprocess each")
    parser.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    parser.add_argument("--methods", nargs="*", default=["CoT", "ToT", "GoT"])
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variant", default=None, help="Write to results/variants/<name>/ instead of the v1 baseline")
    parser.add_argument("--max-tokens", type=int, default=BASELINE_MAX_TOKENS)
    parser.add_argument("--prompt-variant", choices=["base", "stagedefs"], default="base")
    parser.add_argument("--gpu-mem-util", type=float, default=None, help="Override per-model default (e.g. on an exclusive SLURM GPU)")
    parser.add_argument("--triage", action="store_true", help="Filter evidence to model-flagged processes first (variant v4)")
    parser.add_argument("--got-link-hints", action="store_true", help="Give GoT's Aggregate the cross-segment structural links (variant v5)")
    args = parser.parse_args()

    if not args.variant and (args.max_tokens != BASELINE_MAX_TOKENS or args.prompt_variant != "base" or args.triage or args.got_link_hints):
        parser.error("non-baseline settings require --variant, so they never mix into the v1 baseline results")
    _set_variant(args.variant)

    if args.model is not None:
        run_for_one_model(
            args.model, args.scenarios, args.methods, args.seeds,
            variant=args.variant, max_tokens=args.max_tokens,
            prompt_variant=args.prompt_variant, gpu_mem_util=args.gpu_mem_util, triage=args.triage,
            got_link_hints=args.got_link_hints,
        )
        return

    # No --model: dispatch one child process per model so each gets a clean
    # GPU memory state, rather than trying to load/unload models in-process.
    for model_key in args.models:
        print(f"\n{'=' * 70}\nStarting subprocess for model: {model_key}\n{'=' * 70}")
        cmd = [
            # -u: without it, the child's stdout is fully buffered (not a
            # TTY once redirected to a log file), so progress prints only
            # flush in one burst at process exit instead of incrementally --
            # made per-combo progress invisible mid-run on a real grid run.
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--model", model_key,
            "--scenarios", *args.scenarios,
            "--methods", *args.methods,
            "--seeds", *[str(s) for s in args.seeds],
            "--max-tokens", str(args.max_tokens),
            "--prompt-variant", args.prompt_variant,
        ]
        if args.variant:
            cmd += ["--variant", args.variant]
        if args.gpu_mem_util is not None:
            cmd += ["--gpu-mem-util", str(args.gpu_mem_util)]
        if args.triage:
            cmd += ["--triage"]
        if args.got_link_hints:
            cmd += ["--got-link-hints"]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"WARNING: subprocess for {model_key} exited with code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
