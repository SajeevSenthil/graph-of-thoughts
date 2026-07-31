# idea.md — Project Overview

## Title
**Graph-of-Thoughts for Multi-Stage Attack Investigation: Comparing Chain, Tree, and Graph Reasoning for Kill-Chain Reconstruction**

## One-sentence summary
We test whether the *structure* of an LLM's reasoning — a straight chain, a branching tree, or a merging graph — changes how well it can reconstruct a multi-stage cyber attack from raw system logs, holding everything else fixed so the reasoning topology is the only variable.

## The problem
When a machine is attacked, the attack leaves a few malicious actions (a payload download, a network scan, a jump to another host) scattered among ~1 million benign log events. An analyst must find those few events and connect them into one ordered, stage-labeled **kill chain** (initial access → execution → discovery → lateral movement → ...).

## The gap (state of the art)
State-of-the-art systems like **OCR-APT** (CCS 2025, F1 ≈ 0.96 on this dataset) use a **trained GNN detector** to find suspicious subgraphs, then let the LLM only *summarize* what the detector already found. The LLM is a writer, not an investigator. **Nobody has asked whether the LLM's own reasoning — and specifically the *shape* of that reasoning — can do the investigative work, or whether reasoning topology matters at all.**

## The hypothesis (falsifiable)
> Graph-of-Thoughts reasoning, through its **aggregation** operation (merging conclusions reached separately for different parts of the log), reconstructs multi-stage attacks with **higher stage-level recall** than Chain- or Tree-of-Thoughts — because aggregation can connect attack steps that look benign in isolation but are clearly malicious once linked.

We will either see this recall gap or we won't. Both outcomes are publishable.

## Why this framing (reasoning-first, cyber as testbed)
The focus is **reasoning**; cyber attack investigation is the ideal **testbed** because the task has exactly the property that should expose topology differences: scattered clues that only mean something once connected. We do **not** compete with OCR-APT on absolute F1 (they have a trained detector; we use prompts). Our baseline is Chain-of-Thought, and our result is how much the graph's aggregation adds over it. This is a question OCR-APT never asked and their 0.96 cannot answer — so on our question, we define the state of the art.

## Core design principle (the thing that makes it valid)
**Controlled, inference-only comparison.** Detection step, candidate evidence, input format, and model weights are ALL held fixed. Only the reasoning topology (CoT / ToT / GoT) changes. No training, no fine-tuning. Therefore any measured difference is attributable to reasoning topology alone.

## What we build vs. what we borrow
- **We build:** a simple process-graph constructor (parse PID/ParentID), a reduction/serialization step, CoT/ToT/GoT implementations (via the `spcl/graph-of-thoughts` framework), and a deterministic scorer.
- **We do NOT reuse:** OCR-APT's trained GNN (that would break the "no training" design) or ExCyTIn's SQL environment. OCR-APT is a comparison baseline; ExCyTIn is motivation and a possible second dataset.

## Two graphs — never conflate them
1. **Evidence graph** — built from logs by our constructor (nodes = processes/files/IPs, edges = "spawned"/"wrote"/"connected"). Shared by all methods. Serialized to TEXT before going to the LLM.
2. **Thought graph** — built by the LLM's reasoning (nodes = hypotheses, edges = "derived from"). This is what CoT/ToT/GoT differ on. "Graph-of-Thoughts" refers to THIS graph.

## Dataset
**NODLINK Simulated-Data** — https://github.com/PKU-ASAL/Simulated-Data
Five multi-stage attacks across Windows Server 2012, Ubuntu, and Windows 10 (APT29, Sidewinder, FIN6). Each host: a `benign.json` (background) and `anomaly.json` (~1M events, attack hidden inside). Ground truth: `attack_annotation/` files (exact malicious commands) + `attack_analysis.xls` (kill-chain stage labels).

## Evaluation (three layers)
1. **Event-level:** precision / recall / F1 on flagged malicious events (comparable to OCR-APT 0.96, NODLINK 0.248).
2. **Stage-level:** partial-credit recall of kill-chain stages (where aggregation should win).
3. **Efficiency:** LLM calls, tokens, wall-clock latency per investigation (the accuracy-cost tradeoff).

## Models
Llama 3.1 8B, Qwen 2.5 7B, Phi-4 — open-weight, run locally, no fine-tuning.

## Hardware
Single RTX 6000 Ada (48 GB VRAM). Holds any of these models at full/near-full precision. Inference is effectively free, so the full evaluation grid + multiple seeds is affordable.

## Honest scope boundaries
- **In:** inference-only CoT/ToT/GoT comparison on 5 simulated attacks; event + stage + efficiency metrics.
- **Out:** training/fine-tuning; building a novel detector; real-time detection; generalization claims beyond the 5 scenarios.
- **Known limitation:** ground-truth sets are small (6-37 events/scenario), so per-scenario numbers are noisy → report across all scenarios + multiple seeds; stage-recall is the more stable signal.
