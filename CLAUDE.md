# Project Context — Read This First

This file is auto-loaded by Claude Code whenever a session starts in this repo (any machine, any SSH connection). It exists so a fresh session has full context without needing the original conversation transcript, which is stored locally on whichever machine started it and does not sync across machines.

## What this project is

Full detail lives in `idea.md`, `phases.md`, `workflow.md`, and `proposal_document.md` in this repo — **read all four before doing anything else.** One-line summary: we compare Chain-of-Thought, Tree-of-Thoughts, and Graph-of-Thoughts reasoning (via `spcl/graph-of-thoughts`, over local LLMs) on reconstructing multi-stage cyber-attack kill chains from the NODLINK Simulated-Data dataset, holding everything except reasoning topology fixed. Hypothesis: GoT's Aggregate operation gives higher stage-level recall than CoT/ToT. See `proposal_document.md` Section 6 for the formal hypothesis and Section 8 for the pipeline diagram.

**The one invariant that must never break** (from `workflow.md`): Phases 1–4 and 6 (ground truth, evidence graph, reduction, serialization, scoring) are IDENTICAL across all three reasoning methods. Only the reasoning step (Phase 4's method choice) varies. Never let CoT/ToT/GoT see different evidence or serialization — it invalidates the whole experiment.

**Build order** (do not skip ahead): ground-truth parser → evidence graph + reduction + serializer (with the hard gate: 100% of ground-truth events must survive reduction) → scorer (LOCK before writing any reasoning method) → CoT → ToT → GoT → full grid → aggregation ablation.

## Current status (update this section as work progresses)

**Phase 0 is DONE.**

- Repo cloned on the remote GPU node at `~/sajeev/graph-of-thoughts` (asaicomputemaster, 2× RTX 6000 Ada 48GB, both GPUs idle as of last check).
- Planning docs (`idea.md`, `phases.md`, `workflow.md`, `proposal_document.md`, `CLAUDE.md`) committed and pushed.
- `.venv` created with `uv venv` (Python 3.10.20) + `uv pip install -r requirements.txt` (217 packages, includes vllm 0.26.0, graph-of-thoughts 0.0.2, torch 2.11.0+cu130). `requirements.txt` itself still needs to be `git add`ed/committed — it was untracked as of last check.
- `PKU-ASAL/Simulated-Data` cloned to `~/sajeev/Simulated-Data` with `git lfs pull` (git-lfs was not preinstalled and there's no sudo on this box — installed the v3.7.1 binary release straight into `~/.local/bin`, no root needed).
- WS12 unzipped: `SimulatedWS12/hw20.zip` → `anomaly.json` (1,087,311 JSONL events) + `benign.json` (2.4GB). Ubuntu/W10 zips are pulled but **not yet unzipped**.
- WS12 attack understood by hand (6 annotation files → command text maps to xls stages: recon+backdoor download → Initialize access, reg/netsh/sticky-keys → Persistence, WinBrute → Credential theft, nbtscan → Discovery, PAExec → Lateral movement).
- vLLM hello-world confirmed working for **all 3 models** (Qwen 2.5 7B, Llama 3.1 8B, Phi-4) via `LLM(...).chat(...)` on GPU 0. Llama 3.1 8B required a Hugging Face access request (gated) — approved, token stored at `~/.cache/huggingface/token`.
- Not yet started: unzipping Ubuntu/W10, all `src/` code (`parse_ground_truth.py` is next — Phase 1).

## Important discovery: WS12 ground-truth ID scheme mismatch

`attack_annotation/A1.txt`...`A6.txt` are numbered sequentially by time within the host, but `attack_analysis.xls`'s ID column uses a **different, non-sequential global ID space** (observed for WS12: `A1`, a blank-ID continuation row, `A7`, `A24`, `A25`, `A32`) — apparently IDs shared across the full attack description, not per-host. **Do not join annotation files to xls rows by ID string equality** — it silently produces wrong stage labels. Join by matching `pCommand` text between the two instead (confirmed working for all 6 WS12 nodes). Check whether Ubuntu/APT29/Sidewinder/FIN6 have the same mismatch before writing `parse_ground_truth.py`'s join logic.

## Working conventions established so far

- Inference backend: **vLLM** (not Ollama) — chosen for continuous-batching throughput given the many-call ToT/GoT grid (~225 cells).
- No training/fine-tuning anywhere — inference-only comparison is the whole point.
- The reduction/candidate-selection step must stay structurally simple (event-type + external-IP rules only) — it must NOT do content-based malicious/benign judgment, or the experiment quietly reintroduces a trained-detector-like component and undermines the "LLM does the investigative work" claim.
- Repo layout follows `workflow.md`'s recommended structure: `src/{parse_ground_truth,build_graph,serialize,scorer,run_grid}.py`, `src/methods/{cot,tot,got}.py`, `data/` (gitignored), `results/`.
- **Home dir (`/dist_home`) is GlusterFS (network storage), not local disk.** vLLM/Triton's JIT compile cache breaks on it (`errno 61: No data available` writing temp files during autotuning). Always set `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` to somewhere on local disk (e.g. `/var/tmp/<user>_cache/...`, confirmed `ext4`) before running vLLM. `/tmp` is also local disk.
- When invoking `.venv/bin/python` directly (without `source .venv/bin/activate`), prepend `.venv/bin` to `PATH` — some deps (e.g. `ninja`, needed by FlashInfer's JIT kernel build) are only found via `PATH`, not via the interpreter path alone.
- No system `pip`/`conda` on this machine — use `uv` (`~/.local/bin/uv`) for all Python env/package management.
- HF token for gated models (Llama 3.1) lives at `~/.cache/huggingface/token`.
