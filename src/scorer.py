"""
Phase 3 -- Deterministic scorer. LOCKED before any reasoning method (CoT/ToT/
GoT) is written, per phases.md, so there is never a temptation -- conscious
or not -- to tune a reasoning method's output to flatter this metric.

INPUT:  a reasoning method's structured output --
            {"malicious_events": [{"command": str, "stage": str}, ...],
             "narrative": str,
             "_cost": {"llm_calls": int, "prompt_tokens": int,
                        "completion_tokens": int, "wall_clock_s": float}}  (optional)
        + a scenario's ground_truth -- data/ground_truth/{scenario}.json
        + that scenario's serialized evidence text -- data/serialized/{scenario}.txt
          (only used for hallucination-checking: is an IOC the model
          mentions actually present in what it was shown?)

OUTPUT: a metrics dict --
            {"event": {precision, recall, f1, tp, fp, fn},
             "stage": {precision, recall, f1, recall_partial},
             "hallucination_rate": float,
             "cost": <passed through unchanged, or None>}

Ground truth is NEVER shown to any model -- it is read here, at grading
time only, exactly like an answer key a teacher keeps and never hands out
during the exam.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "ground_truth"
SERIALIZED_DIR = REPO_ROOT / "data" / "serialized"

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Extension must start with a letter -- otherwise this also matches numeric
# fragments of an IP address (e.g. "10.0.0.99" spuriously yielding a fake
# "filename" "0.99", caught by tests/test_scorer.py's known-bad case).
_FILENAME_RE = re.compile(r"\b[\w\-]+\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _extract_iocs(text: str) -> set[str]:
    return set(_IP_RE.findall(text)) | set(_FILENAME_RE.findall(text))


def score_event_level(flagged_events: list[dict], ground_truth: list[dict]) -> dict:
    """
    Per phases.md: for each flagged event, TP if its command contains a
    ground-truth match_substring, else FP. FN for any ground-truth node no
    flagged event matched. Nodes with match_substring=None (documented
    non-distinctive commands, e.g. a bare "whoami") are excluded entirely --
    they cannot be scored by substring matching, by construction.

    Note the asymmetry this implies and why it's intentional: TP for
    precision's numerator (flagged events that matched *something* real) and
    TP for recall's numerator (ground-truth nodes that got matched by
    *something*) are counted from two different perspectives, matching
    phases.md's literal per-flagged-event / per-ground-truth-node definition
    rather than forcing a strict one-to-one bipartite match. This matters
    when one substring legitimately matches multiple flagged commands, or
    when two ground-truth nodes share a match_substring (documented in
    parse_ground_truth.py -- e.g. the same scheduled-task command reused
    three times in APT29).
    """
    checkable_gt = [n for n in ground_truth if n["match_substring"] is not None]
    gt_substrings = [n["match_substring"] for n in checkable_gt]

    tp_events = 0
    fp_events = 0
    for event in flagged_events:
        cmd = event.get("command", "")
        if any(ms in cmd for ms in gt_substrings):
            tp_events += 1
        else:
            fp_events += 1

    flagged_commands = [e.get("command", "") for e in flagged_events]
    found_gt = sum(1 for n in checkable_gt if any(n["match_substring"] in cmd for cmd in flagged_commands))
    fn_nodes = len(checkable_gt) - found_gt

    metrics = _prf1(tp=tp_events, fp=fp_events, fn=fn_nodes)
    metrics["tp"] = tp_events
    metrics["fp"] = fp_events
    metrics["fn"] = fn_nodes
    return metrics


def score_stage_level(flagged_events: list[dict], ground_truth: list[dict]) -> dict:
    """
    Same TP/FP/FN structure as event-level, but a match only counts if the
    flagged event's `stage` also matches the ground-truth node's true stage
    (case-insensitive). Reports both the strict precision/recall/f1 (workflow.md's
    output contract) AND a partial-credit recall (phases.md's own name for
    this metric: "stage-level partial-credit recall") that gives half credit
    to a node whose event was found but mislabeled with the wrong stage,
    rather than zero -- both are kept, not blended, so a reader can see the
    difference between "missed entirely" and "found but mislabeled."
    """
    checkable_gt = [n for n in ground_truth if n["match_substring"] is not None]

    tp_events = 0
    fp_events = 0
    for event in flagged_events:
        cmd = event.get("command", "")
        event_stage = (event.get("stage") or "").strip().lower()
        matched_correctly = any(
            n["match_substring"] in cmd and n["stage"].strip().lower() == event_stage for n in checkable_gt
        )
        if matched_correctly:
            tp_events += 1
        else:
            fp_events += 1

    partial_credits = []
    found_and_correct = 0
    for n in checkable_gt:
        matches = [e for e in flagged_events if n["match_substring"] in e.get("command", "")]
        if not matches:
            partial_credits.append(0.0)
            continue
        correct_stage = any((e.get("stage") or "").strip().lower() == n["stage"].strip().lower() for e in matches)
        if correct_stage:
            partial_credits.append(1.0)
            found_and_correct += 1
        else:
            partial_credits.append(0.5)  # found the event, but mislabeled the stage

    fn_nodes = len(checkable_gt) - found_and_correct
    metrics = _prf1(tp=tp_events, fp=fp_events, fn=fn_nodes)
    metrics["tp"] = tp_events
    metrics["fp"] = fp_events
    metrics["fn"] = fn_nodes
    metrics["recall_partial"] = sum(partial_credits) / len(checkable_gt) if checkable_gt else 0.0
    return metrics


def score_hallucination(model_output: dict, evidence_text: str) -> float:
    """Fraction of IOCs (IPs, filenames) the model mentions that never appear
    anywhere in the evidence text it was actually shown."""
    text = model_output.get("narrative", "") + " " + " ".join(
        e.get("command", "") for e in model_output.get("malicious_events", [])
    )
    iocs = _extract_iocs(text)
    if not iocs:
        return 0.0
    hallucinated = sum(1 for ioc in iocs if ioc not in evidence_text)
    return hallucinated / len(iocs)


def score(model_output: dict, ground_truth: list[dict], evidence_text: str) -> dict:
    flagged = model_output.get("malicious_events", [])
    return {
        "event": score_event_level(flagged, ground_truth),
        "stage": score_stage_level(flagged, ground_truth),
        "hallucination_rate": score_hallucination(model_output, evidence_text),
        "cost": model_output.get("_cost"),
    }


def score_scenario(scenario: str, model_output: dict) -> dict:
    """Convenience wrapper for real use (Phase 5): loads ground truth and
    the serialized evidence for `scenario` from disk automatically."""
    ground_truth = json.loads((GROUND_TRUTH_DIR / f"{scenario}.json").read_text())
    evidence_text = (SERIALIZED_DIR / f"{scenario}.txt").read_text()
    return score(model_output, ground_truth, evidence_text)


if __name__ == "__main__":
    import sys

    for scenario_path in sorted(GROUND_TRUTH_DIR.glob("*.json")):
        if scenario_path.stem == "all_scenarios":
            continue
        print(scenario_path.stem, "ground truth loaded OK, scorer ready.")
    print("\nRun `python -m src.test_scorer` (or see tests/test_scorer.py) for the locked unit tests.", file=sys.stderr)
