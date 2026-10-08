"""
Triage pre-pass (variant v4): ask the model, one process at a time (shown
with its parent chain for context), whether that process looks malicious.
Flagged processes are kept together with their ancestors, so the kept text
is still a readable tree; everything else is dropped before CoT/ToT/GoT run.

Per-process rather than per-subtree: a first version judged whole root
subtrees, but busy benign trees (WS12's Chrome tree alone is 44% of the
evidence) were flagged as a unit, so 69-99% of the text survived triage.

Triage runs once per (scenario, model, seed) and its filtered evidence is
reused by all three topologies, so within a variant the topologies still see
byte-identical input -- triage is a shared upstream step, like the evidence
pipeline, not part of any one reasoning method.
"""

import json
import re

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "suspicious": {"type": "boolean"},
    },
    "required": ["reason", "suspicious"],
}

TRIAGE_PROMPT = (
    "You are a security analyst triaging a host's activity log, one process at a time. Below is one "
    "process with any external network connections or suspicious file writes it made, followed by "
    "the chain of parent processes that led to it. Decide whether THIS process is likely part of a "
    "malicious attack, as opposed to normal user, system, or administrative activity.\n\n"
    'Respond with ONLY a JSON object: {{"reason": "<one sentence>", "suspicious": true or false}}\n\n'
    "Process:\n{process}\n\nParent chain (nearest parent first):\n{parents}"
)

_PROCESS_LINE_RE = re.compile(r"^( *)(└─ )?\S.*? \(PID [^)]*\): ")
_ANCESTOR_CHARS = 300

TRIAGE_MAX_TOKENS = 256


def split_process_nodes(evidence_text: str) -> list[dict]:
    """One entry per process line, owning the lines after it up to the next
    process line (its connected-to/wrote lines, plus continuation lines of a
    command that itself contains newlines). Depth comes from indentation."""
    nodes = []
    for line in evidence_text.split("\n"):
        m = _PROCESS_LINE_RE.match(line)
        if m:
            depth = len(m.group(1)) // 2
            nodes.append({"depth": depth, "lines": [line]})
        elif nodes:
            nodes[-1]["lines"].append(line)
    parent_stack = []
    for i, node in enumerate(nodes):
        del parent_stack[node["depth"]:]
        node["ancestors"] = list(parent_stack)
        parent_stack.append(i)
    return nodes


def split_root_blocks(evidence_text: str) -> list[str]:
    """One block per root process: its line plus every indented line beneath it."""
    blocks, current = [], []
    for line in evidence_text.split("\n"):
        if line and not line[0].isspace():
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _is_suspicious(text: str) -> bool:
    # An unparseable verdict keeps the block: dropping evidence on a formatting
    # failure would turn a parse error into a silent detection miss.
    match = _JSON_OBJECT_RE.search(text)
    try:
        verdict = json.loads(match.group(0))["suspicious"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict = None
    return verdict if isinstance(verdict, bool) else True


def _triage_prompt(nodes: list[dict], i: int) -> str:
    parents = [nodes[a]["lines"][0].strip()[:_ANCESTOR_CHARS] for a in reversed(nodes[i]["ancestors"])]
    return TRIAGE_PROMPT.format(
        process="\n".join(l.strip() for l in nodes[i]["lines"]), parents="\n".join(parents) or "(none -- root process)"
    )


def run_triage(evidence_text: str, lm) -> dict:
    nodes = split_process_nodes(evidence_text)
    before = lm.cost_snapshot()
    responses = lm.query_batch(
        [_triage_prompt(nodes, i) for i in range(len(nodes))], json_schema=TRIAGE_SCHEMA, max_tokens=TRIAGE_MAX_TOKENS
    )
    after = lm.cost_snapshot()
    flagged = [i for i, r in enumerate(responses) if _is_suspicious(r)]
    keep = set(flagged)
    for i in flagged:
        keep.update(nodes[i]["ancestors"])
    kept_text = "\n".join(line for i in sorted(keep) for line in nodes[i]["lines"])
    return {
        "evidence_text": kept_text,
        "n_blocks": len(nodes),
        "n_flagged": len(flagged),
        "n_kept_with_ancestors": len(keep),
        "flagged_indices": flagged,
        "chars_before": len(evidence_text),
        "chars_after": len(kept_text),
        "cost": {k: (round(after[k] - before[k], 3) if isinstance(after[k], float) else after[k] - before[k]) for k in after},
    }
