# Project Context — Read This First

This file is auto-loaded by Claude Code whenever a session starts in this repo (any machine, any SSH connection). It exists so a fresh session has full context without needing the original conversation transcript, which is stored locally on whichever machine started it and does not sync across machines.

## What this project is

Full detail lives in `idea.md`, `phases.md`, `workflow.md`, and `proposal_document.md` in this repo — **read all four before doing anything else.** One-line summary: we compare Chain-of-Thought, Tree-of-Thoughts, and Graph-of-Thoughts reasoning (via `spcl/graph-of-thoughts`, over local LLMs) on reconstructing multi-stage cyber-attack kill chains from the NODLINK Simulated-Data dataset, holding everything except reasoning topology fixed. Hypothesis: GoT's Aggregate operation gives higher stage-level recall than CoT/ToT. See `proposal_document.md` Section 6 for the formal hypothesis and Section 8 for the pipeline diagram.

**The one invariant that must never break** (from `workflow.md`): Phases 1–4 and 6 (ground truth, evidence graph, reduction, serialization, scoring) are IDENTICAL across all three reasoning methods. Only the reasoning step (Phase 4's method choice) varies. Never let CoT/ToT/GoT see different evidence or serialization — it invalidates the whole experiment.

**Build order** (do not skip ahead): ground-truth parser → evidence graph + reduction + serializer (with the hard gate: 100% of ground-truth events must survive reduction) → scorer (LOCK before writing any reasoning method) → CoT → ToT → GoT → full grid → aggregation ablation.

## Current status (update this section as work progresses)

- Repo cloned on the remote GPU node at `~/sajeev/graph-of-thoughts` (asaicomputemaster, 2× RTX 6000 Ada 48GB, driver 580.142/CUDA 13.0, Python 3.12.3, both GPUs idle as of last check).
- Planning docs (`idea.md`, `phases.md`, `workflow.md`, `proposal_document.md`) committed and pushed — this is Phase 0.
- In progress: `.venv` + `requirements.txt` (vllm, graph_of_thoughts, pandas, xlrd, openpyxl, numpy, scipy, matplotlib, tqdm) being installed; `PKU-ASAL/Simulated-Data` being cloned with `git lfs pull` in parallel.
- Not yet done: verifying vLLM serves all 3 models (Llama 3.1 8B, Qwen 2.5 7B, Phi-4), unzipping the dataset, `src/` code (nothing written yet — `parse_ground_truth.py` is next once setup is confirmed working).

## Working conventions established so far

- Inference backend: **vLLM** (not Ollama) — chosen for continuous-batching throughput given the many-call ToT/GoT grid (~225 cells).
- No training/fine-tuning anywhere — inference-only comparison is the whole point.
- The reduction/candidate-selection step must stay structurally simple (event-type + external-IP rules only) — it must NOT do content-based malicious/benign judgment, or the experiment quietly reintroduces a trained-detector-like component and undermines the "LLM does the investigative work" claim.
- Repo layout follows `workflow.md`'s recommended structure: `src/{parse_ground_truth,build_graph,serialize,scorer,run_grid}.py`, `src/methods/{cot,tot,got}.py`, `data/` (gitignored), `results/`.
