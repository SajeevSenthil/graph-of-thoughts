"""
Phase 2 -- Serializer.

INPUT:  data/evidence_graph/{scenario}.json (Phase 2's evidence-graph output:
        candidate nodes + spawned/connected_to/wrote edges).
OUTPUT: data/serialized/{scenario}.txt -- indented plain text preserving
        parent -> child causality through indentation, one process tree per
        root, roots ordered chronologically. This exact text is the ONLY
        thing any reasoning method (CoT/ToT/GoT, Phase 4) will ever see --
        it must stay byte-for-byte identical across all three methods.

Token counts are reported using tiktoken's cl100k_base encoding (a real
tokenizer, not a len()/4 guess) so they can be checked against the ~1.5K-8K
token budget from phases.md.
"""

import json
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "data" / "evidence_graph"
OUTPUT_DIR = REPO_ROOT / "data" / "serialized"

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Display-only truncation: some benign processes (Chrome's renderer/gpu/
# utility subprocesses, confirmed the dominant token cost on WS12) carry
# 700-900+ character command lines (base64 GPU preference blobs, mojo
# handles) that add token cost with zero investigative value. Truncating the
# *displayed* text is safe -- the hard gate always checks the full,
# untruncated command_line stored in data/evidence_graph/*.json, never this
# truncated version. 600 chars leaves comfortable margin: the furthest any
# real ground-truth match_substring ends within its own command, checked
# across all 89 nodes, is 472 (Ubuntu/A7).
_MAX_COMMAND_CHARS = 600


def _display_command(command_line: str) -> str:
    if len(command_line) <= _MAX_COMMAND_CHARS:
        return command_line
    return command_line[:_MAX_COMMAND_CHARS] + f"...[truncated, {len(command_line)} chars total]"


def _msec_key(node):
    try:
        return float(node["msec"])
    except (TypeError, ValueError):
        return 0.0


def serialize_graph(nodes: dict, edges: list) -> str:
    children = {}  # parent_key -> [child_key, ...]
    attachments = {}  # node_key -> [edge, ...]  (connected_to / wrote)

    for e in edges:
        if e["type"] == "spawned":
            children.setdefault(e["from_node"], []).append(e["to_node"])
        else:
            attachments.setdefault(e["from_node"], []).append(e)

    for key in children:
        children[key].sort(key=lambda k: _msec_key(nodes[k]))
    for key in attachments:
        attachments[key].sort(key=lambda e: float(e["msec"]) if e["msec"] is not None else 0.0)

    child_keys = {c for kids in children.values() for c in kids}
    roots = sorted((k for k in nodes if k not in child_keys), key=lambda k: _msec_key(nodes[k]))

    lines = []

    def render(key, depth, visited):
        if key in visited:  # defensive: guard against any accidental cycle
            return
        visited = visited | {key}
        node = nodes[key]
        prefix = "  " * depth + ("└─ " if depth > 0 else "")
        repeat = f"  [x{node['repeat_count']}]" if node.get("repeat_count", 1) > 1 else ""
        lines.append(f'{prefix}{node["image"]} (PID {node["pid"]}): "{_display_command(node["command_line"])}"{repeat}')

        for edge in attachments.get(key, []):
            verb = "connected to" if edge["type"] == "connected_to" else "wrote"
            lines.append("  " * (depth + 1) + f'    {verb} → {edge["target"]}')

        for child_key in children.get(key, []):
            render(child_key, depth + 1, visited)

    for root_key in roots:
        render(root_key, 0, frozenset())

    return "\n".join(lines)


def run_scenario(scenario: str) -> dict:
    graph = json.loads((EVIDENCE_DIR / f"{scenario}.json").read_text())
    text = serialize_graph(graph["nodes"], graph["edges"])
    token_count = len(_ENCODING.encode(text))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{scenario}.txt"
    out_path.write_text(text, encoding="utf-8")

    return {"scenario": scenario, "chars": len(text), "tokens": token_count, "out_path": out_path}


def main():
    for scenario in ["WS12", "Ubuntu", "APT29", "Sidewinder", "FIN6"]:
        result = run_scenario(scenario)
        budget_flag = "" if result["tokens"] <= 8000 else "  [OVER the ~8K token target]"
        print(
            f"{scenario:12s} {result['chars']:>7,} chars -> {result['tokens']:>6,} tokens (cl100k_base)"
            f"   -> {result['out_path'].relative_to(REPO_ROOT)}{budget_flag}"
        )


if __name__ == "__main__":
    main()
