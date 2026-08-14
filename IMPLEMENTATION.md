# Implementation Log

Running record of what has actually been built, phase by phase, updated every time a phase lands. Planning documents (`idea.md`, `phases.md`, `workflow.md`, `proposal_document.md`) describe the intended design; this file records what was actually implemented, what was verified against real data, and every place reality forced a deviation from the plan. `CLAUDE.md` stays as the short cross-session pointer -- this file is the detailed version.

---

## Phase 0 -- Environment & Data Setup ✅

**Goal:** working local inference environment, dataset on disk, one attack understood by hand.

**What was done:**
- `.venv` created with `uv venv` + `uv pip install -r requirements.txt` on the GPU node (`asaicomputemaster`, 2× RTX 6000 Ada 48GB).
- `PKU-ASAL/Simulated-Data` cloned via `git lfs pull` (git-lfs wasn't preinstalled and there's no sudo on this machine -- installed the v3.7.1 binary release directly into `~/.local/bin`).
- All 5 scenarios unzipped: `SimulatedWS12`, `realAPTlinux/hw17` (Ubuntu), `realAPTWin10/win10` (shared by APT29/Sidewinder/FIN6 -- see Phase 2 notes on why).
- vLLM hello-world confirmed working for all 3 models (Llama 3.1 8B, Qwen 2.5 7B, Phi-4) via `LLM(...).chat(...)` on GPU 0.
- WS12's 6-step attack read and understood by hand from `attack_annotation/*.txt`.

**Real problems hit and fixed** (see `CLAUDE.md` "Working conventions" for the full list): vLLM/Triton's compile cache breaks on the home directory's GlusterFS mount (`errno 61`) -- must point `VLLM_CACHE_ROOT`/`TRITON_CACHE_DIR` at local disk (`/var/tmp/...`); `ninja` (needed by FlashInfer's JIT build) is only found via `PATH`, not via the interpreter path alone, when not using `source .venv/bin/activate`.

**Input:** none (setup phase). **Output:** working `.venv`, dataset on disk under `~/sajeev/Simulated-Data/`, no code yet.

---

## Phase 1 -- Ground-Truth Parser ✅

**Goal:** load the ground truth (attacker's real commands + kill-chain stage labels) into a clean structure, before touching the logs.

**Input:** `attack_annotation/A*.txt` (one file per attack step: `occurTime`, `hostIp`, `pCommand`, optional `pFilePath`) + `attack_analysis.xls` (stage label per command) for each of the 5 scenarios.

**Output:** `data/ground_truth/{scenario}.json` (+ combined `all_scenarios.json`) -- a list of `{id, stage, match_substring, full_command, occurTime_start, occurTime_end, hostIp}` per scenario, sorted chronologically.

**Code:** `src/parse_ground_truth.py`, `src/visualize_ground_truth.py`

### Why this couldn't be automated blindly

Checked all 5 scenarios by hand before writing the parser and found **three different relationships** between `attack_annotation` filenames and `attack_analysis.xls`'s own ID column:

| Scenario | Pattern |
|---|---|
| WS12 | xls IDs (`A1`, a blank-ID row, `A7`, `A24`, `A25`, `A32`) are a **totally different, non-sequential ID space** than the annotation filenames (`A1`-`A6`). ID-string matching silently produces wrong stage labels. |
| Ubuntu | xls IDs *do* align 1:1 with annotation filenames, but the **rows are out of numeric order** in the sheet (the `A4` row sits physically after the `A6`/`A7` row). |
| APT29 / Sidewinder / FIN6 | xls IDs align 1:1 *and* are in order -- the clean case. Even here, one node's command text differs slightly between the two sources (APT29 `A34`: xls has a placeholder URL, the annotation file has the real one). |

Given three different failure modes, every `(stage, match_substring)` pair was resolved **by hand**, once, by reading both sources side by side for all 89 nodes -- not by trusting one automated join rule across all five scenarios. The resolved table lives in `parse_ground_truth.py`'s `CURATED` dict.

### Self-validation

The parser asserts every `match_substring` is a literal substring of its own node's real command at parse time -- a curation typo fails loudly instead of silently corrupting Phase 3 scoring later. This caught a real bug during development: WS12/A4's annotation *file* uses literal double backslashes in a registry path (`Execution Options\\sethc.exe`) that the real raw log does not (confirmed directly against `anomaly.json` while building Phase 2 -- see below). The validation has a normalization fallback for this exact case, with a printed `[note]` so it stays visible rather than silently passing.

### Known weak nodes

Two nodes have `match_substring: null` -- their real attacker command is a bare `whoami`, genuinely not distinctive (Ubuntu/A4, Sidewinder/A12). Documented rather than forced into a fragile heuristic.

### Results

**89 ground-truth nodes** across 5 scenarios: WS12 6, Ubuntu 8, Sidewinder 15, FIN6 23, APT29 37.

```
WS12          6 nodes -> data/ground_truth/WS12.json
Ubuntu        8 nodes -> data/ground_truth/Ubuntu.json   [weak/non-distinctive: A4]
APT29        37 nodes -> data/ground_truth/APT29.json
Sidewinder   15 nodes -> data/ground_truth/Sidewinder.json   [weak/non-distinctive: A12]
FIN6         23 nodes -> data/ground_truth/FIN6.json
```

Visible output: `data/ground_truth/summary.png` -- small-multiples bar chart of node count per scenario per kill-chain stage.

---

## Phase 2 -- Evidence Graph Constructor + Reduction Filter + Hard Gate + Serializer ✅

**Goal:** turn each scenario's raw host log (~1-4M events) into a small, structurally-filtered candidate graph that provably retains every ground-truth attack event, then serialize it into the plain text every reasoning method (Phase 4) will read identically.

**Code:** `src/build_graph.py`, `src/serialize.py`, `src/visualize_graph.py`

### Step 1: Evidence Graph Constructor + Reduction Filter (`build_graph.py`)

**Input:** a scenario's `anomaly.json` (raw host log, JSONL, streamed line-by-line, never loaded whole into memory -- WS12 alone is 1,087,311 lines; Ubuntu is 3,832,342).

**Output:** `data/evidence_graph/{scenario}.json` -- `{nodes, edges, stats, hard_gate}`. A node is one process (from a Process/Start-equivalent event); edges are `spawned` (parent→child), `connected_to` (process→external IP), `wrote` (process→suspicious file path). Construction and reduction happen in one streaming pass: a node is only ever created from a process-start event, and `connected_to`/`wrote` edges only ever exist when the structural keep-rule already matched -- so there's no separate "shrink it afterward" step to reconcile.

**Critical, non-negotiable rule:** this module never reads the raw log's own `is_warn` field. Spot-checking WS12's `anomaly.json` shows `is_warn` is already `"True"` on the malicious `cmd.exe`/`certutil.exe` processes and `"False"` on benign ones -- it is the dataset's own ground-truth label sitting inside the evidence. If it ever leaked into a node's attributes or the serialized text, the LLM would be handed the answer directly and the entire experiment would be invalid.

#### Real bugs found and fixed while building this (in the order they were caught)

1. **Two different raw-log schemas.** WS12/APT29/Sidewinder/FIN6 use a Windows ETW-style schema (`EventName`, top-level `PID`, `ImageFileName`, `CommandLine`). Ubuntu uses a completely different **sysdig/Falco-style syscall trace** (`evt.type`, `proc.cmdline`, `proc.pname`, `fd.name`) with no numeric PID on most event types -- process ownership there is resolved by matching `proc.cmdline` text to the most recent `execve` that produced it, not by numeric PID lookup. `_construct_windows` and `_construct_linux` are separate adapters sharing one `_finalize` reduction step.

2. **PID reuse silently overwrote nodes.** WS12 has 3,139 `Process/Start` events but only 593 distinct PIDs -- the OS reuses PIDs constantly over a long capture. Keying nodes by raw PID let a later process silently overwrite an earlier one; if the earlier one had been part of the attack, it would have vanished with no error. Fixed by keying nodes on `(PID, MSec)` and resolving PID→current-owner via a temporal `pid_to_current_node` map updated while streaming forward. (The raw log's own `UniqueProcessKey` field looked like a safer unique key at first, but turned out to be a recycled kernel-object address, not actually unique either -- one value repeated 802 times. Verified directly before trusting it.)

3. **`is_external_ip`/suspicious-write matching produced 35,372 edges for 593 nodes** -- traced to one `postgres.exe` process alone generating 23,010 raw `TcpIp/Send` events for only 64 distinct targets (one packet = one log line for a long-lived connection). Fixed by deduplicating to one edge per `(process, target)` pair.

4. **Even after PID-reuse and edge-dedup fixes, WS12 had 3,139 legitimate process-start nodes** -- ~10x the ~285-593 estimate in the planning docs. Traced to Chrome alone spawning dozens of `--type=renderer`/`--type=gpu-process`/`--type=utility` helper subprocesses. Collapsing exact-duplicate `(image, command_line)` repeats (a helper respawning with the *identical* command) got this to 360; collapsing near-duplicates that differ only in a volatile numeric parameter (`--renderer-client-id=6` vs `=5`, `--launch-time-ticks=...`) via a grouping-key normalization (`_normalize_for_grouping`, collapses runs of 3+ digits) got it to **234**. This normalization is used only for deciding which raw nodes collapse together -- the actual stored/matched `command_line` is never altered, and the hard gate was re-verified to still pass after adopting it (confirming no two distinct ground-truth commands in the same scenario accidentally collapsed together).

5. **WS12/A4's ground-truth match_substring didn't match the real raw log**, even though it matched the annotation *file* -- see Phase 1 above. The raw log is the authority (that's what the LLM actually sees); the annotation file's text isn't always byte-identical to it.

### Hard gate

For each scenario, every ground-truth `match_substring` is checked against the reduced candidate set's command lines. **All 5 scenarios pass**, with two categories of documented, non-silent exception:

- **Unscoreable** (from Phase 1): Ubuntu/A4, Sidewinder/A12 -- `match_substring: null`, can't be checked at all.
- **Explained exception**: Ubuntu/A8 -- `hostIp` for this node is `192.168.0.244`, not `192.168.0.155` like every other Ubuntu node. It's evidence from a **second host** reached via the A6/A7 lateral-movement step, running `JuicyPotato_x64.exe` (a Windows privilege-escalation tool -- meaningless on the Linux box whose log we actually have). Confirmed via `grep -c JuicyPotato` on the raw log = 0. We only have a log for the originating Linux host; this is a genuine dataset scope boundary, not a bug, and is tracked in `build_graph.py`'s `EXPECTED_GATE_EXCEPTIONS` so it's a visible, explained skip rather than a silent one.

### Results (current, after all fixes above)

```
WS12                         1,087,311 events -> 3139 process starts (2905 repeats collapsed) ->  234 nodes /  255 edges   hard gate: PASS
Ubuntu                       3,832,342 events -> 2901 process starts (2515 repeats collapsed) ->  386 nodes /  478 edges   hard gate: PASS  [explained exception: A8] [unscoreable: A4]
APT29                        1,878,778 events -> 2365 process starts (1741 repeats collapsed) ->  624 nodes /  933 edges   hard gate: PASS
Sidewinder   (shared W10 log)                                                                ->  624 nodes /  933 edges   hard gate: PASS  [unscoreable: A12]
FIN6         (shared W10 log)                                                                ->  624 nodes /  933 edges   hard gate: PASS
```

Note: APT29/Sidewinder/FIN6 share **one** W10 host log (three attacks run against the same captured machine at different times, confirmed by inspecting the W10 zip -- it contains a single `anomaly.json`, not three). `build_graph.py` constructs that shared graph once and caches it rather than re-streaming ~650MB three times.

Full runtime for all 5 scenarios (streaming ~8.7M raw events total): **~24 seconds** on this machine.

### Step 2: Serializer (`serialize.py`)

**Input:** `data/evidence_graph/{scenario}.json`. **Output:** `data/serialized/{scenario}.txt` -- indented plain text preserving parent→child causality through indentation (one process tree per root, roots ordered chronologically), identical in format across every scenario and every reasoning method that will consume it in Phase 4.

Command-line text is truncated for **display only** at 600 characters (a `[truncated, N chars total]` marker is appended) to control token cost from a handful of very verbose but uninteresting commands. This is safe: the hard gate always checks the full, untruncated `command_line` stored in `evidence_graph/*.json`, never this display copy, and 600 chars leaves comfortable margin above the furthest any real ground-truth `match_substring` ends within its own command (472, Ubuntu/A7 -- checked across all 89 nodes before picking the threshold).

**Honest finding -- token budget is larger than planned.** `phases.md` targets ~1.5K-8K tokens per scenario. Real counts (tiktoken `cl100k_base`, a real tokenizer, not a `len()/4` guess):

```
WS12          46,276 chars -> 16,197 tokens   [OVER the ~8K target]
Ubuntu        48,945 chars -> 19,411 tokens   [OVER the ~8K target]
APT29        162,532 chars -> 58,396 tokens   [OVER the ~8K target]
Sidewinder   162,532 chars -> 58,396 tokens   [OVER the ~8K target]
FIN6         162,532 chars -> 58,396 tokens   [OVER the ~8K target]
```

This is a real, useful finding, not a bug being hidden: the ~8K estimate assumed ~50-300 candidate nodes, but strictly honoring "keep every Process/Start node" (phases.md's own rule, never violated here) on a real host running everyday applications (Chrome alone, on WS12) legitimately produces more than that even after every structural reduction described above. All three chosen models (Llama 3.1 8B, Qwen 2.5 7B, Phi-4) support context windows well beyond 58K tokens, so this does not block Phase 4 -- it just recalibrates the original estimate. Recorded here so Phase 4/5's cost accounting starts from real numbers instead of the planning-doc estimate.

### Step 3: Visualization (`visualize_graph.py`)

**Input:** `data/evidence_graph/{scenario}.json`. **Output:** `data/evidence_graph/{scenario}_graph.png`.

Renders using `networkx` + `matplotlib` with a hand-written recursive tree layout (each leaf gets its own x-slot in DFS order, parents centered above their children) -- no dependency on system graphviz. Process / external-IP / file-write nodes are colored distinctly using the same reference palette as the project's other figures (`data/ground_truth/summary.png`, `kill_chain_pipeline.png`).

The full per-scenario graph (234-624 nodes) is illegible as a static image, so the **picture** (not the underlying data) is filtered to the branches that actually touch an external IP or a suspicious write, plus the process ancestry needed to reach them -- `data/evidence_graph/{scenario}.json` always has the complete, uncapped graph; only the rendered figure is trimmed, and every trim is stated in the image's own title (e.g. "82 of 624 total process nodes shown, earliest 6 of 47 branches"). Two rounds of real layout bugs were hit and fixed here: an initial same-depth-row layout produced a smeared, crossing-line mess because unrelated processes' IP/file leaves shared one visual row with no relation to their actual parent (fixed by switching to a proper per-parent tree layout); and one process alone (WS12) connecting to 60+ distinct external IPs single-handedly blew the figure out to 15,000+ pixels wide (fixed by capping displayed leaves-per-process to 4, with a "+N more" summary node, and capping root branches shown to the earliest 6-15 depending on scenario scale).

---

## What's next -- Phase 3

Per the build order in `phases.md`/`workflow.md`: the **scorer** (`src/scorer.py`) is next, and must be written and unit-tested *before* any reasoning method (CoT/ToT/GoT) is implemented, so there is never a temptation to tune model output to flatter the metric. It consumes `data/ground_truth/{scenario}.json` and a model's structured JSON output, producing event-level P/R/F1, stage-level recall, and hallucination rate.
