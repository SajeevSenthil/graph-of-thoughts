# Pitch: Does Reasoning Topology (CoT vs ToT vs GoT) Help LLMs Reconstruct Cyber-Attack Kill Chains?

## 1. The problem

Security analysts investigating an intrusion face a provenance log with tens of thousands of events. The attack is a small, scattered subset of them, and it spans several kill-chain stages (initial access, persistence, credential theft, discovery, lateral movement). Existing detectors (e.g. NODLINK-style systems) are trained models that flag anomalies. They do not produce a *narrative*, and they need training data.

**Question:** can an off-the-shelf, locally hosted LLM, with no training or fine-tuning, reconstruct the attack and name its stages? And does *how* we let it reason change the answer?

## 2. Reasoning topologies (brief)

All three use the same evidence, the same prompt and the same scorer. Only the shape of the reasoning differs.

| Topology | Shape | In this project |
|---|---|---|
| **Chain-of-Thought (CoT)** | One linear pass: read the evidence, reason step by step, answer. | One call over the full serialized evidence. |
| **Tree-of-Thoughts (ToT)** | Branch into several candidate analyses, score or prune them, keep the best. | Several independent candidate analyses, then pick or merge by score. |
| **Graph-of-Thoughts (GoT)** | Thoughts are graph nodes. Operations include Generate, **Aggregate** (merge several thoughts into one), Refine and Score. | The evidence is split into segments and each is analysed separately. **Aggregate** then merges them, so clues in different segments can be linked. Refine follows. Implemented with `spcl/graph-of-thoughts`. |

**Hypothesis:** attack evidence is scattered (a file dropped early, executed later, then used to move laterally). GoT's Aggregate can join those pieces, so it should give higher **stage-level recall** than CoT or ToT.

## 3. Experiment design (the invariant)

The pipeline has these phases, and **only the reasoning step varies**:

1. Ground truth: 89 hand-verified `(stage, match_substring)` nodes across 5 scenarios (WS12, Ubuntu, Sidewinder, FIN6, APT29) from NODLINK's Simulated-Data.
2. Evidence graph, reduction and serialization. A hard gate requires 100% of ground-truth events to survive reduction. Reduction uses structural rules only (event type, external IPs) and does no content-based judgment, so the experiment does not smuggle in a trained detector.
3. Scorer, locked before any method was written: event P/R/F1, stage P/R/F1 with partial credit, and hallucination rate. These are never blended into one composite.
4. Reasoning: CoT, ToT or GoT, with one shared prompt and parser.
5. Grid: scenario x method x model x 3 seeds, resumable, with paired significance tests.

## 4. Infrastructure

- GPU node: 2x RTX 6000 Ada 48 GB for development, then SLURM A100 80 GB jobs on `workq`.
- Inference: **vLLM** (continuous batching). It was chosen over Ollama because ToT/GoT make many calls per cell. No training or fine-tuning anywhere.
- Environment: `uv` venv, Python 3.10, vLLM 0.26, `graph-of-thoughts` 0.0.2.
- Practical lessons:
  - The home directory is GlusterFS, which breaks Triton/vLLM JIT caches, so those caches go on local disk.
  - The machine is shared, so vLLM memory use is capped per model, and SLURM jobs size it from the GPU's actual free memory.
  - `graph-of-thoughts` 0.0.2 imports an `openai<1.0` attribute, so a stub `openai.error` module avoids downgrading the SDK.
  - Gated Llama weights require a Hugging Face access request.

## 5. Why these models

| Model | Reason | Limit found |
|---|---|---|
| Llama 3.1 8B | Native 131K context, which covers the largest scenario (~58K tokens). | None. It runs all 5 scenarios. |
| Qwen 2.5 7B | A different model family, so results are not Llama-specific. | Native context is 32K, so it runs WS12 and Ubuntu only. Forcing longer context risks NaN output from RoPE extrapolation, so we refused to do it. |
| Qwen 2.5 32B, Llama 3.1 70B (4-bit AWQ) | Scale test: does a bigger model change the conclusion? | Needs the A100 80 GB. |
| Llama-Primus-Reasoning (Llama 3.1 8B, security-tuned, emits think traces) | Domain-specialised model on the same architecture as the baseline. | Done: significantly *worse* than Llama 8B (event F1 0.101 vs 0.295); see Section 7. |
| Dropped: Phi-4, Phi-3.5-mini | Phi-4's 16K context is too small for the evidence. Phi-3.5-mini produced unparseable JSON across four symmetric fix attempts. | The lineup stayed at 2 base models rather than chasing a fifth fix. |

## 6. Results and where it failed so far

**Headline (baseline, 63 runs, 3 seeds): topology differences are not statistically significant.** No method-vs-method comparison reached p<0.05.

**Why (diagnosed by reading raw outputs against ground truth):**
- Only 15.3% of real attack events are ever detected, regardless of topology.
- Of those detected, exact stage labelling is correct ~20% of the time (random baseline: 10%).
- Each model has a miscalibrated stage-label prior. Qwen predicts "Initial Access" in 3.9% of its output and "Lateral Movement" in 1.1%, but those two stages are 8 of the 14 (57%) combined WS12+Ubuntu ground-truth events, which explains its 0.000 stage recall under CoT and ToT.
- Sidewinder, FIN6 and APT29 share one byte-identical evidence file, so the model must pick among three narratives from the same evidence. This is a caveat for the write-up, not a bug.

**Improvement rounds (each isolates one change):**

| Round | Change | Effect |
|---|---|---|
| v3 | MITRE stage definitions in the prompt | Exact stage recall 0.056 -> 0.123 (Wilcoxon p = 0.0015, n=63). First significant result. |
| v4 | Per-process triage before labelling | Precision 0.245 -> 0.329 (p = 0.04). Mostly fewer false positives; detection barely moves. |
| v5 | Link-aware GoT Aggregate | Llama event recall 0.41 -> 0.54 (p = 0.055). GoT still does not separate from CoT/ToT. A pre-check found that in WS12 all attack events sit in one segment, so the scattered structure GoT targets is barely present in the data. |
| Scale | Qwen 7B -> 32B | Exact stage recall 0.15 -> 0.30 (p = 0.012). |
| Scale | Llama 8B -> 70B | Precision 0.28 -> 0.38 and hallucination 0.19 -> 0.06 (p = 0.001), with recall flat. GoT was the top method for both large models (best p = 0.073). |
| Scale | Llama 70B without triage | Worse: event F1 0.33 -> 0.23 (p = 0.0015). On ~58K-token inputs recall collapses to 0.09 (8B: 0.22). Anything that shortens the input helps, which may be part of what segmentation does for GoT. |

**Bugs caught along the way (evidence the pipeline is trustworthy):**
- PID reuse silently overwrote graph nodes.
- A dataset field (`is_warn`) leaks the ground-truth label, so it is never used.
- GoT segmentation split by block count rather than size, so one segment held 78% of the evidence.
- A backslash-escaping mismatch between model output and ground truth.
- A job reporting `COMPLETED` when its model had crashed (fixed today).

## 7. Latest run: Llama-Primus-Reasoning (job 13610, completed 45/45)

The first attempt (13606) died on an incomplete offline model cache. After re-downloading, the rerun finished in 1h26m.
- Versus Llama 3.1 8B on identical triaged evidence (45 matched runs): event F1 0.101 vs 0.295, event recall 0.096 vs 0.381, stage recall 0.034 vs 0.145. All significant (p<=0.002); hallucination lower (0.087 vs 0.194, p=0.005) only because it reports fewer events.
- 17 of 45 runs (38%) gave no parseable answer; among the 28 that did, event F1 is still only 0.126.
- Topology within Primus: GoT ~ ToT > CoT on recall, none significant (p>=0.12).
- Reading: domain pre-training did not fix detection or stage calibration; its reasoning-trace output format is handled imperfectly by our pipeline.

## 8. What I would pitch beyond the current results

1. **Reframe the contribution.** The honest finding is that at 7-8B scale, base-model detection recall and stage-vocabulary calibration dominate over reasoning topology. That is useful negative evidence for anyone assuming "fancier reasoning will fix it".
2. **Input length is a hidden confound in topology comparisons.** The 70B result shows segmentation helps by shortening context. The proposed next test is **segmented CoT** (same split, no LLM merge), to separate "GoT's Aggregate helps" from "GoT's splitting helps".
3. **Harder, more GoT-friendly data.** Add scenarios where attack events really are scattered across segments, since the pre-check shows the current ones are not.
4. **Calibration as a fix.** Few-shot stage examples, or a stage-prior correction learned on held-out scenarios, directly targets the diagnosed failure.
5. **Statistical power.** Report effect sizes and confidence intervals, and use more seeds only where effects are near the threshold.
6. **Practical value.** Everything runs locally on open models, with no data leaving the site, which matters for security logs. It needs no training data, unlike trained provenance detectors.

## 9. One-slide summary

> We hold evidence, prompt, and scorer fixed and vary only reasoning topology. Across 5 attack scenarios and 2 to 4 open LLMs, topology effects are not significant. Prompt stage definitions, triage and model scale are. The bottleneck is detection recall and stage calibration, not reasoning shape. Next, we test whether GoT's gains come from aggregation or merely from shorter segments.
