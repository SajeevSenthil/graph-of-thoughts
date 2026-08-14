"""
Phase 3 -- locked unit tests for src/scorer.py, per phases.md's "done when":
feed a hand-written fake output and get correct precision/recall/F1 +
stage-recall back, with a known-good and a known-bad example.

Uses WS12's real ground truth (data/ground_truth/WS12.json, 6 nodes, all
checkable -- no None match_substrings on this scenario) so the test is
grading against the actual answer key, not a synthetic stand-in. Flagged
commands are kept short ("ran: <real match_substring> ...") rather than the
full realistic multi-clause commands, purely so the expected IOC set stays
easy to hand-verify -- the scoring logic itself only ever checks substring
containment, so this doesn't weaken what's being tested.

Run: .venv/bin/python tests/test_scorer.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scorer import score, _extract_iocs  # noqa: E402

GROUND_TRUTH = json.loads((REPO_ROOT / "data" / "ground_truth" / "WS12.json").read_text())
assert len(GROUND_TRUTH) == 6 and all(n["match_substring"] is not None for n in GROUND_TRUTH), (
    "This test assumes WS12's real ground truth shape (6 nodes, all checkable). "
    "If parse_ground_truth.py's WS12 table changed, update this test's fixtures too."
)


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Known-good example: every event correctly flagged with the correct stage,
# nothing hallucinated. Expect a clean sweep across every metric.
# ---------------------------------------------------------------------------

GOOD_COMMANDS = [
    ("ran: ping www.baidu.com -c 1", "Initial Access"),
    ("ran: sangforcat.exe download", "Initial Access"),
    ("ran: WinBrute.exe crack", "Credential Access"),
    ("ran: Execution Options\\sethc.exe hijack", "Persistence"),
    ("ran: nbtscan.exe scan", "Discovery"),
    ("ran: PAExec.exe \\\\192.168.0.244 move", "Lateral Movement"),
]
GOOD_OUTPUT = {
    "malicious_events": [{"command": c, "stage": s} for c, s in GOOD_COMMANDS],
    "narrative": "Full kill chain reconstructed across all six stages.",
}
GOOD_EVIDENCE = " ".join(c for c, _ in GOOD_COMMANDS)  # superset by construction -> zero hallucination

good = score(GOOD_OUTPUT, GROUND_TRUTH, GOOD_EVIDENCE)

assert good["event"]["tp"] == 6 and good["event"]["fp"] == 0 and good["event"]["fn"] == 0
assert _approx(good["event"]["precision"], 1.0)
assert _approx(good["event"]["recall"], 1.0)
assert _approx(good["event"]["f1"], 1.0)

assert _approx(good["stage"]["precision"], 1.0)
assert _approx(good["stage"]["recall"], 1.0)
assert _approx(good["stage"]["f1"], 1.0)
assert _approx(good["stage"]["recall_partial"], 1.0)

assert good["hallucination_rate"] == 0.0

print("PASS: known-good example scores a clean sweep (P=R=F1=1.0, hallucination=0.0)")

# ---------------------------------------------------------------------------
# Known-bad example:
#   - A5 (nbtscan) never flagged at all                      -> event FN, stage FN
#   - A4 (sethc.exe) flagged with the WRONG stage             -> event TP, stage FP + partial credit 0.5
#   - one fabricated event matching no ground-truth substring -> event FP, stage FP
#   - the fabricated event's IP (10.0.0.99) never appears in the evidence
#     text it was supposedly reasoning over                   -> exactly one hallucinated IOC
# ---------------------------------------------------------------------------

BAD_COMMANDS = [
    ("ran: ping www.baidu.com -c 1", "Initial Access"),
    ("ran: sangforcat.exe download", "Initial Access"),
    ("ran: WinBrute.exe crack", "Credential Access"),
    ("ran: Execution Options\\sethc.exe hijack", "Discovery"),  # wrong stage (real: Persistence)
    ("ran: PAExec.exe \\\\192.168.0.244 move", "Lateral Movement"),
    ("net use \\\\10.0.0.99 /user:hacker", "Lateral Movement"),  # fabricated -- matches nothing
]
BAD_OUTPUT = {
    "malicious_events": [{"command": c, "stage": s} for c, s in BAD_COMMANDS],
    "narrative": "Attacker pivoted to 10.0.0.99 using stolen credentials.",
}
# Evidence only covers the 5 legitimate commands -- the fabricated node's IP is deliberately absent.
BAD_EVIDENCE = " ".join(c for c, _ in BAD_COMMANDS[:5])

bad = score(BAD_OUTPUT, GROUND_TRUTH, BAD_EVIDENCE)

# Event-level: 5 of 6 flagged commands are real (A4's command text is still
# correct even though its stage label is wrong -- event-level never looks at
# stage), 1 fabricated -> TP=5, FP=1. A5 was never flagged -> FN=1.
assert bad["event"]["tp"] == 5 and bad["event"]["fp"] == 1 and bad["event"]["fn"] == 1, bad["event"]
assert _approx(bad["event"]["precision"], 5 / 6)
assert _approx(bad["event"]["recall"], 5 / 6)

# Stage-level: A1,A2,A3,A6 correct (TP=4); A4 (wrong stage) and the
# fabricated event are both FP (TP=4, FP=2); A4 and A5 both end up FN from
# the recall side (found_and_correct=4, so FN = 6-4 = 2).
assert bad["stage"]["tp"] == 4 and bad["stage"]["fp"] == 2 and bad["stage"]["fn"] == 2, bad["stage"]
assert _approx(bad["stage"]["precision"], 4 / 6)
assert _approx(bad["stage"]["recall"], 4 / 6)
# Partial credit: A1,A2,A3,A6=1.0 each, A4=0.5 (found, wrong stage), A5=0.0 (never found) -> 4.5/6
assert _approx(bad["stage"]["recall_partial"], 4.5 / 6), bad["stage"]["recall_partial"]

# Hallucination: exactly one IOC (10.0.0.99) is missing from the evidence text.
text = BAD_OUTPUT["narrative"] + " " + " ".join(e["command"] for e in BAD_OUTPUT["malicious_events"])
iocs = _extract_iocs(text)
missing = {i for i in iocs if i not in BAD_EVIDENCE}
assert missing == {"10.0.0.99"}, missing
assert _approx(bad["hallucination_rate"], 1 / len(iocs)), bad["hallucination_rate"]
assert bad["hallucination_rate"] > 0.0

print("PASS: known-bad example correctly separates 'missed', 'found but mislabeled', 'fabricated', and 'hallucinated'")
print("\nAll Phase 3 scorer tests passed. Locking scorer.py -- do not modify to flatter a reasoning method's output.")
