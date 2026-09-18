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

## Phase 3 -- Scorer (`src/scorer.py`, `tests/test_scorer.py`)

**Goal:** a deterministic function that grades any reasoning method's output against ground truth, locked with unit tests *before* any reasoning method (CoT/ToT/GoT) is written -- per `phases.md`, so there is never a temptation, conscious or not, to tune a method's output to flatter this metric.

**Input:** a model's structured output `{malicious_events: [{command, stage}], narrative}` + `data/ground_truth/{scenario}.json` + `data/serialized/{scenario}.txt` (for hallucination-checking -- what the model actually saw). **Output:** `{event: {precision, recall, f1, tp, fp, fn}, stage: {precision, recall, f1, tp, fp, fn, recall_partial}, hallucination_rate, cost}`. Ground truth is only ever read here, at grading time -- never shown to a model.

**Event-level** follows `phases.md` literally: a flagged event is TP if its command contains any ground-truth `match_substring`, else FP; a ground-truth node is FN if no flagged event's command contains its substring. Deliberately *not* a strict one-to-one bipartite match -- precision's TP (flagged events that matched something real) and recall's TP (ground-truth nodes that got matched) are counted from two different perspectives, matching what happens when one substring legitimately matches multiple flagged commands, or two ground-truth nodes share a substring (documented in Phase 1 -- e.g. the same scheduled-task command reused three times in APT29).

**Stage-level** adds one requirement on top of event-level: the flagged event's `stage` must also match the ground-truth node's true stage. Reports the strict precision/recall/f1 (workflow.md's contract) *and* a separate `recall_partial` -- phases.md's own name for this metric, "stage-level partial-credit recall" -- which gives a node 1.0 credit if found with the correct stage, 0.5 if found but mislabeled, 0.0 if never found at all. Both are kept, never blended, so "missed entirely" stays distinguishable from "found but mislabeled."

**Hallucination rate:** extract IOCs (IPs via regex, filenames via a `word.ext`-shaped regex) from the model's narrative + flagged commands, and check what fraction never appear anywhere in the serialized evidence text the model actually saw.

**A real bug caught by the test itself, on the first run:** the filename regex `[\w\-]+\.[a-zA-Z0-9]{2,5}` also matched purely-numeric fragments of an IP address -- e.g. `10.0.0.99` spuriously yielded a fake "filename" `0.99`, since a numeric `.99` satisfies "word, dot, 2-5 alnum chars" just as well as a real extension does. Fixed by requiring the extension to start with a letter (`[a-zA-Z][a-zA-Z0-9]{1,4}`), which every real extension in this project already does (`.exe`, `.txt`, `.zip`, ...) and no IP octet ever does. Caught immediately because the known-bad unit test asserted the *exact* missing-IOC set, not just "hallucination_rate > 0" -- a looser assertion would have passed with the bug still present, silently over-counting hallucinations by one on every score.

**Unit tests** (`tests/test_scorer.py`) run against WS12's real 6-node ground truth, not a synthetic stand-in:
- **Known-good:** all 6 events correctly flagged with correct stages, evidence text built as a superset of the flagged commands by construction (zero hallucination by design) -> asserts a clean P=R=F1=1.0 sweep on both event- and stage-level, `recall_partial = 1.0`, `hallucination_rate = 0.0`.
- **Known-bad:** one node (`nbtscan`) never flagged at all; one node (`sethc.exe`) flagged with the *correct* command but the *wrong* stage; one fully fabricated event matching no ground-truth substring at all; one fabricated IP deliberately absent from the evidence text. Hand-verified expected values: event `TP=5, FP=1, FN=1`; stage `TP=4, FP=2, FN=2, recall_partial=4.5/6`; exactly one hallucinated IOC (`missing == {"10.0.0.99"}`, asserted as an exact set, which is what caught the regex bug above).

`.venv/bin/python tests/test_scorer.py` -- both pass. Also spot-checked `score_scenario()` (the convenience wrapper that loads ground truth + serialized evidence from disk directly) against all 5 real scenarios with an empty output, confirming every file path resolves correctly end-to-end before Phase 4 needs it.

**scorer.py is now LOCKED.** Any future change to its scoring logic must be justified independently of what it does to any specific reasoning method's numbers.

---

## Phase 4 -- Reasoning methods (`src/llm_client.py`, `src/methods/`)

**Goal:** implement CoT, ToT, and GoT as the only variable, all consuming `data/serialized/{scenario}.txt` and producing `{malicious_events, narrative, _cost}` in exactly the shape `scorer.py` already expects.

### `llm_client.py` -- vLLM wrapper implementing the framework's `AbstractLanguageModel`

Bakes in the Phase 0 environment fixes (`VLLM_CACHE_ROOT`/`TRITON_CACHE_DIR` on local disk, `.venv/bin` on `PATH`) so they hold regardless of how the module is invoked, applied *before* `import vllm` since vLLM reads some of these at import time. Accumulates `prompt_tokens`/`completion_tokens`/`wall_clock_s`/`llm_calls` across every call on the instance, matching what the framework's own `Controller.output_graph()` expects to read.

**A real upstream bug hit immediately:** `graph_of_thoughts==0.0.2`'s `controller/__init__.py` unconditionally imports its `ChatGPT` class, which references `openai.error.OpenAIError` -- an attribute that only existed in `openai<1.0`. Our installed `openai` SDK is newer and has no `.error` submodule, so *any* import from `graph_of_thoughts.controller` failed, even just for `AbstractLanguageModel`, which has nothing to do with OpenAI. Fixed by stubbing a fake `openai.error` module (`OpenAIError = Exception`) before the `graph_of_thoughts` import, rather than downgrading `openai` or patching site-packages -- `ChatGPT` is never instantiated, so the stub only needs to satisfy the class-body-level `@backoff.on_exception(..., openai.error.OpenAIError, ...)` decorator evaluated at import time.

### `methods/prompter.py`, `methods/parser.py` -- shared across all three topologies

One `KillChainPrompter`/`KillChainParser` pair, used identically by CoT, ToT, and GoT -- this is the concrete mechanism behind "reasoning topology is the only variable": the wording of every prompt and the JSON-extraction logic never differs by condition, only which operations are wired together and how many times each prompt is used. `parser.py`'s `extract_json` strips markdown code fences, takes the largest `{...}` block, and validates shape before accepting it; a failed parse increments `lm.parse_retries` (a tracked metric, not a silent failure) and falls back to an empty reconstruction rather than crashing the run.

### `methods/cot.py` -- one `Generate(1,1)` operation via the framework's own `Controller`. Simplest topology, used as-is.

### `methods/tot.py` -- `Generate(1,k)` -> `Score` (scored via a `scoring_function` callable that makes its own small LLM call asking "how well-supported is this reconstruction, 0.0-1.0") -> `KeepBestN(1)`. Uses `graph_of_thoughts.operations.Score`'s documented `scoring_function` bypass (confirmed by reading `Score._execute`: when set, it's called directly on each thought's state, skipping `prompter.score_prompt`/`parser.parse_score_answer` entirely) rather than templating a second prompt shape for what's naturally a single free-form question.

### `methods/got.py` -- segment -> Generate per segment -> Aggregate -> Refine

`split_into_segments()` splits the serialized evidence into a **fixed segment count** (default 4), not a fixed size -- grouping contiguous root-level process trees (a root = a line with no leading indentation in the serializer's output; WS12 has 82 such root blocks). A fixed count, not size, keeps GoT's call count -- and therefore its cost relative to CoT/ToT -- comparable across scenarios ranging from WS12's 82 root blocks to APT29's several hundred.

**Framework-integration limitation, documented rather than hidden:** `graph_of_thoughts`'s `Controller` runs one shared `problem_parameters` dict for an entire graph, which works cleanly for CoT and ToT (one `evidence_text` for the whole run) but not for GoT's fan-out, where each of N independent `Generate` calls needs a genuinely *different* input (its own segment). Each segment's Generate step therefore runs as its own tiny `Controller` (sharing the same `lm` instance, so cost still accumulates correctly across all of them), and the Aggregate/Refine steps replicate `graph_of_thoughts.operations.operations.Aggregate`/`Improve`'s own `_execute` logic directly -- identical `prompter`/`parser` calls, same state-merging -- rather than fighting the single-shared-state assumption with awkward workarounds. Every prompt template still comes from the one shared `KillChainPrompter`.

### End-to-end validation on WS12 (Qwen 2.5 7B) -- and what it found

All three methods ran to completion and produced valid, parseable JSON on the first real run (`parse_ok: True` for all three) -- satisfying Phase 4's literal done-when bar for this model. But "runs without crashing" and "produces sensible output" are different bars, and looking at the actual output surfaced two real, substantively different findings:

**1. A genuine scorer robustness bug (fixed).** CoT correctly identified the ground-truth PAExec lateral-movement event, but wrote its UNC path with a single backslash (`PAExec.exe \192.168.0.244`) where the raw log -- and therefore the ground-truth `match_substring` -- uses two (`PAExec.exe \\192.168.0.244`). A cosmetic reformatting difference silently zeroed out an otherwise-correct match. This is the exact same backslash-escaping quirk already documented in Phase 1's `parse_ground_truth.py` self-check, just showing up on the model-output side instead of the annotation-file side. Fixed by adding `_normalize_backslashes()` + `_contains()` to `scorer.py`, collapsing repeated backslashes on *both* sides of every substring comparison before checking containment -- applied symmetrically to every method and model, not tuned to flatter this one run. **Re-ran the locked unit tests after the change (`tests/test_scorer.py`) -- both still pass, no regression.** Re-scoring the already-saved raw outputs (no LLM calls needed) recovered the match: CoT/ToT event TP went 2 -> 3, stage TP 0 -> 1.

**2. A genuine stage-taxonomy disagreement (not a bug -- left as-is).** The model labeled the `sangforcat.exe` download as `"Execution"` where our ground truth says `"Initial Access"`, and the `sethc.exe` sticky-keys backdoor as `"Defense Evasion"` where ground truth says `"Persistence"` -- both defensible alternative readings of ambiguous MITRE-adjacent techniques, not obviously wrong. This is exactly the phenomenon `recall_partial` (Phase 3) was built to capture separately from strict stage match: `recall_partial = 0.333` (two of six nodes found-but-mislabeled = 2 x 0.5 / 6) vs. strict `stage_recall = 0.167` (one of six found-and-correctly-labeled) -- the two metrics correctly disagree in exactly the way they're supposed to. Nothing here was changed; this is real signal, not noise.

**3. GoT reconstructed a *different*, still-plausible attack story (qualitative finding, kept for Phase 6).** On this single run/seed, GoT's aggregated output matched **zero** of the six ground-truth events -- but not because it failed. It instead built a coherent narrative around a *different* apparent intrusion also present in WS12's log: a phpStudy-backdoor command chain hitting a different C2 IP (`111.229.170.12`, vs. the ground truth's `124.223.85.207`), local user-account creation (`net user nbb ... /add`), and Windows-Defender-adjacent tooling (`MRT.exe`, `wermgr.exe`). Whether this is a second real attack thread NODLINK's authors left unannotated, or the model chasing a red herring, isn't something one seed on one model can answer -- which is exactly why the grid runs multiple seeds. Not treated as a bug and nothing was adjusted to steer GoT toward the ground-truth narrative; saved verbatim as a candidate example for the Phase 6 qualitative case study either way.

**Follow-up: finding #3 above was investigated rather than left alone, since a complete miss on the exact metric the hypothesis depends on deserved a look before trusting it.** Checked which segment(s) actually contained the six ground-truth events by re-running `split_into_segments` standalone: all six landed in one segment that also happened to hold ~78% of the entire evidence text (36K of 46K characters), while the other three segments were nearly empty (2-4.5K characters each) -- a real bug, not a hypothesis-relevant result. The segmenter split by root-block *count* (82 blocks / 4), but block size varies enormously (a few noisy processes like Chrome and Postgres carry dozens of external-connection edges each), so equal count did not mean equal size. Confirmed the oversized segment was itself the problem, not just a symptom, by running it through Generate alone: it flagged **zero** events -- worse than CoT's already-partial recall on the *full* evidence, which it had almost as much of.

Fixed `split_into_segments` to accumulate blocks by character count until each segment reaches its target share of the total (`src/methods/got.py`). WS12 now splits into 3 segments of 20K/14K/11K characters instead of the old 36K/2K/4.5K/3.5K. Re-ran the full WS12/Qwen test after the fix: **GoT's event recall went from 0/6 to 1/6**, and its narrative shifted from being entirely about an unrelated activity cluster to being clearly anchored in the real attack (agent.exe download, persistence, privilege escalation) even though it still blends in a second, still-unresolved thread (see below). This confirms the size imbalance was genuinely suppressing GoT's signal, not a cosmetic issue -- and that the fix, while not sufficient on its own to make GoT competitive with CoT/ToT on this single run, was necessary.

**A pattern visible across all three methods, on every run, worth flagging rather than treating as three independent hallucinations:** CoT, ToT, and the pre- and post-fix GoT runs all independently surfaced the *same* second command chain -- a phpStudy-backdoor sequence hitting a different C2 IP (`111.229.170.12`, vs. the ground truth's `124.223.85.207`), downloading `servert.exe`/`ywm.exe`, plus local user-account creation (`net user nbb ... /add`) and Windows-Defender-adjacent tooling (`MRT.exe`, `wermgr.exe`). Three independent runs converging on the same unannotated activity is much more consistent with "this is real activity in the log that our 6-node ground truth doesn't cover" than with "the model is making things up" -- worth checking against WS12's own attack description text in `attack_analysis.xls` before the full grid runs, since if it's a real second attack thread NODLINK's authors left unannotated, it's depressing precision on every method roughly equally (not a topology effect), but still worth knowing about before over-interpreting precision numbers.

### Validating Llama 3.1 8B and Phi-4 -- a shared-GPU lesson and a model-lineup change

This is a shared GPU machine (confirmed via `nvidia-smi --query-compute-apps`: other users' processes already resident on both GPUs). vLLM's default `gpu_memory_utilization=0.9` is a fraction of *total* device memory, not free memory, so it would try to grab ~43GB on a 48GB card regardless of what else is running there. Capped it to per-model values instead of one flat number (`MODEL_GPU_MEMORY_UTILIZATION` in `llm_client.py`), since a single value either wastes headroom or doesn't fit -- model checkpoint sizes here range from Phi-3.5-mini's ~7.6GB to Llama's ~15GB. Also added per-model `max_model_len` caps (`MODEL_MAX_MODEL_LEN`): both a floor (Llama's own default is 131072, demanding 16GB of KV cache alone, more than the trimmed memory budget) and, as it turned out, a hard ceiling that couldn't be worked around at all.

**Phi-4 was dropped from the model lineup.** It loaded fine, but failed at the *query* step, not model load: `This model's maximum context length is 16384 tokens. However, ... your prompt contains at least 16385 input tokens`. WS12 -- our *smallest* scenario -- already exceeds Phi-4's hard context ceiling by one token under its own tokenizer. The larger scenarios (APT29/Sidewinder/FIN6, ~58K tokens) would fail by a much wider margin, and CoT/ToT's design (the full evidence in one call) has no way around this -- it isn't a settings problem, it's the model's architecture. Presented this as a decision point rather than picking unilaterally, since it changes a model the proposal names explicitly. **Replaced Phi-4 with Phi-3.5-mini-instruct** (3.8B params, 128K native context -- real margin over even the largest scenario). Updated every planning/proposal doc that named Phi-4 (`idea.md`, `phases.md`, `workflow.md`, `proposal_document.md`, `README.md`) to keep them consistent with what's actually implemented.

**Llama 3.1 8B validated successfully** on WS12 -- CoT and ToT both produced clean, parseable output (13 and 7 flagged events respectively). **GoT's final Refine step failed to parse** on this run (`parse_ok: False` after 2 retries) -- a real example of the parser's fallback logic working as designed rather than a crash: `parse_improve_answer` returns the prior (pre-Refine) state unchanged when the model's response doesn't parse, rather than discarding good data. That preserved pre-Refine state turned out to be the **best result across every model/method combination tested so far**: event recall 5/7 (~71%), stage `recall_partial` 0.42. Worth flagging plainly: this is one run, one seed, one scenario -- encouraging, not conclusive, and exactly the kind of number multiple seeds exist to confirm or correct.

### Phi-3.5-mini-instruct -- four fix attempts, still blocked

Phi-4's replacement loaded and ran within its context window (unlike Phi-4), but failed to produce parseable output on every method, every attempt. Diagnosed rather than guessed at each step -- `common.py`'s `build_output()` was first fixed to actually preserve the raw model text on a parse failure (it was silently discarding a field the parser had already captured), specifically so this could be investigated instead of re-run blind.

**Attempt 1 -- prompted JSON only (the original design):** failed with `completion_tokens` pegged exactly at `max_tokens`. Raw text showed why: the model copied the full multi-line tree entry -- including its embedded literal `"` -- straight into a `"command"` string value, which is invalid JSON regardless of length.

**Attempt 2 -- vLLM structured/guided decoding** (`StructuredOutputsParams(json=schema)`, applied to every model, not a Phi-3.5-mini-only patch): forces valid JSON *syntax* grammar-token-by-token. This was still worth adding on its own merits (a real, symmetric robustness improvement), but didn't fix this failure -- the grammar closed the string at the embedded quote as designed, then the model, confused by its own broken structure, degenerated into emitting nothing but whitespace/tab tokens for the rest of the budget.

**Attempt 3 -- explicit prompt instruction + worked example** (`prompter.py`'s `TASK_PREAMBLE`, shared by all three methods): told the model exactly what to extract vs. not extract, with a concrete before/after example. No effect -- the model still opened with the same raw tree-line copy pattern.

**Attempt 4 -- `disable_any_whitespace=True`** (targets exactly the whitespace-padding trap): still failed the same way -- the padding persisted, suggesting the flag isn't taking effect for this model/backend combination as expected, or the degenerate state it reaches isn't the one this flag guards against.

**A parallel finding worth keeping separate:** at temperature=0 (greedy), the *same* model produced a *different* failure -- it correctly extracted one clean single-line command this time, then repeated that exact JSON object verbatim until the token budget ran out (a textbook repetition trap). Added `repetition_penalty=1.15` (again, applied to every model symmetrically) to address it -- confirmed it didn't fix the temperature=0.7 whitespace-padding failure, haven't yet re-tested whether it fixes the temperature=0 repetition failure in isolation.

**Resolution: model lineup finalized at 2, not 3.** Brought back to the user as a decision rather than a fifth unilateral fix attempt; chose to proceed with Qwen 2.5 7B and Llama 3.1 8B only rather than try a third replacement model. Every proposal/planning doc (`idea.md`, `phases.md`, `workflow.md`, `proposal_document.md`, `README.md`) and the relevant `src/` files updated to match -- Phi-3.5-mini-instruct's `MODEL_IDS`/`MODEL_GPU_MEMORY_UTILIZATION`/`MODEL_MAX_MODEL_LEN` entries removed, `run_grid.py`/`analyze_grid.py`'s model lists trimmed. The `disable_any_whitespace`/`repetition_penalty`/structured-decoding additions were kept (they're symmetric robustness improvements, not Phi-3.5-mini-specific, and don't hurt Qwen/Llama).

**Status: Phase 4 is DONE. Both models in the final lineup (Qwen 2.5 7B, Llama 3.1 8B) fully validated end-to-end on WS12 -- all three methods, valid parseable output.**

### Phase 5 smoke test on Sidewinder found a second, model-architecture context limit

Before committing to the full unattended grid, ran one more de-risking smoke test: Qwen 2.5 7B + CoT + seed 0 on Sidewinder (`data/serialized/Sidewinder.txt`, 162,532 chars, ~58K tokens -- APT29/Sidewinder/FIN6 share one large W10 log, see Phase 2). This exercised a code path flagged as unverified in an earlier comment: Qwen's `max_model_len` had been raised from vLLM's detected default of 32768 to 65536, to match Llama's cap, on the assumption ("Qwen2.5-7B-Instruct's own config supports much longer context via rope scaling") that this was safe. It was not.

vLLM refused to even load the model:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, User-specified max_model_len (65536) is greater than the derived max_model_len
  (max_position_embeddings=32768.0 or model_max_length=None in model's config.json). To allow
  overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
  VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme caution. If the model uses relative
  position encoding (RoPE), positions exceeding derived_max_model_len lead to nan. If the model
  uses absolute position encoding, positions exceeding derived_max_model_len will cause a CUDA
  array out-of-bounds error.
```

This is a hard load-time failure, not a runtime degeneracy like Phi-3.5-mini's -- and critically, the suggested override is not a safe workaround: exceeding a RoPE model's trained position range risks silently producing NaN output, which would be far worse than a loud crash (a confounding factor indistinguishable from "the model reasoned badly" without deliberately checking for it). Qwen 2.5 7B-Instruct's real, config.json-derived native ceiling is 32768 tokens -- comfortably above WS12 (~16K) and Ubuntu (~19K), but well under APT29/Sidewinder/FIN6's ~58K. CoT and ToT both require the full evidence in a single call, so there is no safe way to fit those three scenarios for Qwen without either the NaN risk above or truncating evidence in a way that would confound "worse reasoning" with "never saw the evidence" -- exactly the failure mode this project's fixed-evidence-pipeline principle exists to prevent.

Brought back to the user as a decision point (same pattern as the Phi-4/Phi-3.5-mini swap) rather than silently capping `max_model_len` back down and calling it fixed, since it changes the grid's actual scope. **Decision (2026-08-24): Qwen runs WS12 + Ubuntu only; Llama runs all 5 scenarios** (native 131072 context, safely covers even the largest scenario at the 65536 cap -- not independently re-verified on Sidewinder yet, but has 2x the real margin Qwen would have needed). Fixed `llm_client.py`'s `MODEL_MAX_MODEL_LEN["qwen2.5-7b"]` back to 32768 (the real safe value, not a truncation workaround) and added `run_grid.py`'s `MODEL_SCENARIO_ALLOWLIST` to enforce the split in code, with skip messages printed per run rather than silently omitted rows. The resulting grid is intentionally asymmetric: full 2-model comparison is only valid on WS12/Ubuntu; the 5-scenario comparison is Llama-only.

---

### Phase 5 first full run (2026-08-24): 63 runs, 5 scenarios, 3 seeds

WS12+Ubuntu (both models, 36 runs) then Sidewinder/FIN6/APT29 (Llama only per the allowlist, 27 runs) -- all via `src/run_grid.py`, summarized with `src/analyze_grid.py` into `results/summary_*.csv` and `results/plots/*.png`.

**Headline pattern, holds across all 3 seeds:** models find roughly a quarter to half of real malicious events (`event_f1` 0.15-0.44 on WS12/Ubuntu) but almost never assign the *exact* correct kill-chain stage (`stage_recall` mostly 0.00-0.12; `stage_recall_partial`, 0.5 credit for right-event-wrong-stage, sits 2-4x higher). This is a stage-vocabulary/labeling gap, not a detection gap. Qwen's CoT and ToT hit an exact stage_recall of 0.000 across all 6 WS12+Ubuntu runs each (not just low -- a hard, repeated zero); GoT breaks that pattern for Qwen (0.079-0.111), consistent with segmentation changing its output style.

**Found while investigating why Sidewinder/FIN6/APT29 score so much worse than WS12/Ubuntu (event_f1 collapses to 0.04-0.23, stage_recall to 0.00-0.07):** `data/serialized/Sidewinder.txt`, `FIN6.txt`, and `APT29.txt` are byte-identical (confirmed via md5sum -- same hash, same 165,345-byte size). Traced upstream: `data/evidence_graph/{Sidewinder,FIN6,APT29}.json` all have exactly 624 nodes and 933 edges (file sizes differ by <1%, likely just the `scenario`/`hard_gate` metadata fields, not node/edge content) -- so the post-reduction evidence these three scenarios present to every reasoning method is the same, only the ground-truth annotation (which subset of that shared evidence counts as "malicious" for each of the 3 different attack narratives) differs. This is the concrete consequence of the already-documented "APT29/Sidewinder/FIN6 share one W10 log" data characteristic (Phase 2, `IMPLEMENTATION.md`) -- not a newly-introduced bug in `serialize.py`/`build_graph.py`, but its first real impact on inference results: models are being asked to pick 3 different "correct" attack narratives out of literally the same reduced evidence, then graded against 3 different, mutually-exclusive answer keys. Read the low Sidewinder/FIN6/APT29 scores with that in mind -- they conflate "the model reasoned badly" with "the task's evidence doesn't actually disambiguate which of 3 narratives is being asked for," which is a harder and different failure mode than WS12/Ubuntu's numbers reflect. Not fixed or altered here (Phase 2 is locked, and this reflects the real dataset, not a processing bug) -- flagged for the write-up as a genuine interpretive caveat on the 5-scenario comparison.

Full comparison tables: `results/summary_by_scenario_method_model.csv` (63 rows -> 21 aggregated cells) and `results/summary_by_method_model.csv`.

### Significance testing and the interpretability follow-up (2026-08-24)

Ran paired Wilcoxon signed-rank tests (paired by scenario+seed) on every method-vs-method comparison, every metric, every slice of the grid. **Not one comparison clears p<0.05** -- the closest is CoT vs. ToT on Llama/WS12+Ubuntu event_f1 (p=0.062, n=6), and even that reverses direction once the other 3 scenarios are added (p=0.177). Effect sizes (0.03-0.18) are consistently smaller than the standard deviation of the paired differences (0.09-0.32). Conclusion: at this sample size, method-to-method differences are statistically indistinguishable from seed-to-seed sampling noise.

More seeds would sharpen these estimates but would not address the actual bottleneck, per a direct read of `results/raw/*.json` against `data/ground_truth/*.json` (`src/analyze_stage_confusion.py`, checking every ground-truth node with a `match_substring` against every flagged event across the whole 63-run grid, independent of topology):

- **Detection is the primary failure, not labeling.** Only 138 of 900 checkable ground-truth events (15.3%) were ever flagged by any method/model/seed at all. 84.7% are missed outright, before stage assignment is even relevant.
- **Of the events that are found, exact stage accuracy is ~24%** (33/138) -- barely above the ~10% baseline for picking one of 10 MITRE stages at random.
- **Two distinct, non-random causes, from `results/stage_confusion_matrix.csv`:**
  1. Some "wrong" labels are defensible alternate readings of genuinely dual-purpose techniques -- e.g. the `sethc.exe` Image File Execution Options hijack (ground truth: Persistence) predicted as Privilege Escalation 17x (MITRE T1546.008 legitimately spans both); PowerShell download-and-execute payloads (ground truth: Execution) predicted as Command and Control 13x (a fileless downloader is simultaneously both). The ground truth's single-label taxonomy is stricter than some techniques' actual ambiguity.
  2. **Each model has a stable label prior that's miscalibrated to this dataset -- not noise.** WS12+Ubuntu's ground truth is 72% "Initial Access" or "Lateral Movement" (5/11 + 3/11 checkable nodes). Across every event Qwen ever flagged (`results/stage_prediction_distribution.csv`), "Initial Access" is 3.9% of its output and "Lateral Movement" is 1.1% -- it essentially never reaches for the two labels this dataset needs most, biasing instead toward Persistence/Command-and-Control/Execution (22.8%/18.9%/18.3%). This is the precise, quantified explanation for Qwen's flat 0.000 stage_recall on CoT/ToT (`summary_by_method_model.csv`) -- a systematic prior, not sampling variance. Llama shows a different but equally systematic skew (22% Command and Control, 15.7% Defense Evasion, only 2.7% Initial Access).

**Implication for the project's actual conclusion**: at 7-8B model scale, the bottleneck for this task is event detection recall and MITRE-stage label calibration -- both properties of the base model, not of how many reasoning steps wrap around it. Reasoning topology (CoT/ToT/GoT) shows no statistically defensible effect on top of that ceiling with the current data. This is a legitimate, interesting finding in its own right, not a failed experiment -- but it reframes the project's likely conclusion away from "topology X wins" toward "topology has limited effect at this model scale; the base model's stage-vocabulary calibration is the dominant bottleneck." Artifacts: `results/stage_confusion_matrix.csv`, `results/stage_prediction_distribution.csv`, `results/plots/stage_confusion_heatmap_{model}.png`.

---

## What's next -- Phase 5

`src/run_grid.py` (resumable, one OS process per model, `MODEL_SCENARIO_ALLOWLIST`-aware) and `src/analyze_grid.py` (summary CSVs at three levels of aggregation, per-scenario topology x model bars, the cross-scenario comparison, a scenario x method heatmap, and an accuracy-vs-cost Pareto plot) are running/ready against the grid: Qwen on WS12+Ubuntu, Llama on all 5 scenarios, both across all 3 methods and seeds 0/1/2. First launch (2026-08-24): WS12 + Ubuntu, both models, 3 seeds (36 runs) -- a fully symmetric slice to answer the user's "try another scenario and 3 seeds" request without yet committing to the multi-hour full run across the 3 large Llama-only scenarios. Once that completes and results look sane, extend to the remaining 3 scenarios (Llama only) via `--scenarios Sidewinder FIN6 APT29 --seeds 0 1 2` (no `--model` needed; the allowlist will naturally skip Qwen for those).
