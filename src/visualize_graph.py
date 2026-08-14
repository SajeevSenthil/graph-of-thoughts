"""
Phase 2 -- visible-output artifact: renders the constructed evidence graph.

INPUT:  data/evidence_graph/{scenario}.json
OUTPUT: data/evidence_graph/{scenario}_graph.png

Uses networkx for the graph data structure and a manual layered layout (BFS
depth from each root -> y position; spread within a depth -> x position) so
this never depends on system graphviz being installed. Process nodes, IP
targets, and file targets are colored distinctly using the project's
existing reference palette (same tokens as data/ground_truth/summary.png
and kill_chain_pipeline.png, for visual consistency across the paper's
figures).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "data" / "evidence_graph"

PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
PROCESS_COLOR = "#3f7cac"   # blue  (matches CoT accent in kill_chain_pipeline.png)
IP_COLOR = "#a6373a"        # crimson (matches GoT accent -- "danger/network")
FILE_COLOR = "#b9791f"      # amber (matches ToT accent -- "artifact/write")
EDGE_COLOR = "#c3c2b7"


def build_networkx_graph(nodes: dict, edges: list, max_leaves_per_process: int = 4) -> nx.DiGraph:
    """
    max_leaves_per_process caps how many connected_to/wrote leaves get drawn
    per process, collapsing the rest into one "+N more" summary leaf --
    purely a figure-legibility choice (confirmed necessary: one process in
    WS12 alone touches 60+ distinct external IPs, which single-handedly blew
    the figure out to 15,000+ pixels wide). The full, uncapped edge list is
    always in data/evidence_graph/{scenario}.json.
    """
    g = nx.DiGraph()
    for key, node in nodes.items():
        label = f"{node['image']}\nPID {node['pid']}"
        g.add_node(key, kind="process", label=label)

    leaf_edges = {}  # from_node -> [edge, ...]
    for e in edges:
        if e["type"] == "spawned":
            g.add_edge(e["from_node"], e["to_node"], kind="spawned")
        else:
            leaf_edges.setdefault(e["from_node"], []).append(e)

    for from_node, e_list in leaf_edges.items():
        shown, overflow = e_list[:max_leaves_per_process], e_list[max_leaves_per_process:]
        for e in shown:
            target_id = f"{e['type']}::{e['from_node']}::{e['target']}"
            g.add_node(target_id, kind="ip" if e["type"] == "connected_to" else "file", label=e["target"])
            g.add_edge(from_node, target_id, kind=e["type"])
        if overflow:
            kind = "ip" if overflow[0]["type"] == "connected_to" else "file"
            summary_id = f"overflow::{from_node}"
            g.add_node(summary_id, kind=kind, label=f"+{len(overflow)} more")
            g.add_edge(from_node, summary_id, kind=overflow[0]["type"])

    return g


def _tree_layout(g: nx.DiGraph):
    """
    Classic recursive tree layout: each leaf gets its own x-slot in
    left-to-right DFS order, and each parent is centered above its own
    children -- not a shared same-depth row across unrelated parents (which
    is what produced a smeared, crossing-line mess when first tried: IP/file
    leaves from many different processes all landing in one wide row with no
    relation to which process they actually belonged to). No graphviz
    dependency.
    """
    process_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "process"]
    roots = [n for n in process_nodes if g.in_degree(n) == 0]

    pos = {}
    next_x = [0]

    def assign(node, depth, visited):
        if node in visited:
            return next_x[0]
        visited = visited | {node}
        children = list(g.successors(node))
        if not children:
            x = next_x[0]
            next_x[0] += 1.4
            pos[node] = (x, -depth)
            return x
        xs = [assign(c, depth + 1, visited) for c in children]
        x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    max_depth = 0
    for root in roots:
        assign(root, 0, frozenset())
    for n in g.nodes:
        if n not in pos:
            assign(n, 0, frozenset())
    if pos:
        max_depth = int(max(-y for _x, y in pos.values()))
    leaf_count = next_x[0]
    return pos, max_depth, leaf_count


def _focus_subgraph(nodes: dict, edges: list, max_roots: int = 6) -> tuple[dict, list, int]:
    """
    Rendering choice ONLY -- the full graph stays intact in
    data/evidence_graph/{scenario}.json and data/serialized/{scenario}.txt;
    this never touches either. A 200+ node graph (WS12's own scale) is
    illegible as a figure, so the *picture* shows the branches that actually
    touch an external IP or a suspicious file write, plus the process
    ancestry needed to see how they were reached -- everything else in the
    real data (which the LLM does still receive in full) is omitted from
    this image only.

    The W10 scenarios (APT29/Sidewinder/FIN6 share one log covering three
    combined attacks) still have 40+ such branches even after that filter,
    which is too many root trees to lay out side by side -- max_roots caps
    to the earliest N chronologically, since the figure's job is showing
    what the graph *looks like*, not serving as the complete data (that's
    what the JSON is for).
    """
    has_signal = {e["from_node"] for e in edges if e["type"] in ("connected_to", "wrote")}
    parent_of = {e["to_node"]: e["from_node"] for e in edges if e["type"] == "spawned"}

    keep = set(has_signal)
    for key in list(has_signal):
        cur = parent_of.get(key)
        while cur and cur not in keep:
            keep.add(cur)
            cur = parent_of.get(cur)

    roots = sorted(
        (n for n in keep if parent_of.get(n) not in keep),
        key=lambda n: float(nodes[n]["msec"]) if nodes[n]["msec"] is not None else 0.0,
    )
    total_roots = len(roots)
    kept_roots = set(roots[:max_roots])

    if len(kept_roots) < total_roots:
        # Re-derive `keep` as only the descendants of the kept roots.
        children_of = {}
        for e in edges:
            if e["type"] == "spawned":
                children_of.setdefault(e["from_node"], []).append(e["to_node"])
        keep = set()
        stack = list(kept_roots)
        while stack:
            n = stack.pop()
            if n in keep:
                continue
            keep.add(n)
            stack.extend(children_of.get(n, []))

    focused_nodes = {k: v for k, v in nodes.items() if k in keep}
    focused_edges = [
        e for e in edges
        if (e["type"] == "spawned" and e["from_node"] in keep and e["to_node"] in keep)
        or (e["type"] in ("connected_to", "wrote") and e["from_node"] in keep)
    ]
    return focused_nodes, focused_edges, total_roots


def render(scenario: str):
    graph_data = json.loads((EVIDENCE_DIR / f"{scenario}.json").read_text())
    all_nodes, all_edges = graph_data["nodes"], graph_data["edges"]
    nodes, edges, total_roots = _focus_subgraph(all_nodes, all_edges)
    shown_roots = sum(
        1 for n in nodes
        if not any(e["type"] == "spawned" and e["to_node"] == n for e in edges)
    )

    g = build_networkx_graph(nodes, edges)
    pos, max_depth, leaf_count = _tree_layout(g)

    fig_w = min(40, max(10, leaf_count * 0.85))
    fig_h = max(6, (max_depth + 1) * 1.7 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=PAGE)
    ax.set_facecolor(SURFACE)

    colors = {"process": PROCESS_COLOR, "ip": IP_COLOR, "file": FILE_COLOR}
    for kind, color in colors.items():
        node_list = [n for n, d in g.nodes(data=True) if d["kind"] == kind]
        if not node_list:
            continue
        nx.draw_networkx_nodes(
            g, pos, nodelist=node_list, node_color=color, node_size=260 if kind == "process" else 140,
            ax=ax, linewidths=0,
        )

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=EDGE_COLOR, arrows=True, arrowsize=8, width=0.8, alpha=0.85)

    # Process labels: horizontal, drawn via networkx (they have more room --
    # process nodes are the sparser layer). Leaf (ip/file) labels: drawn
    # manually at a rotation, since siblings under the same parent can sit
    # only 1 x-unit apart and horizontal text collides at that spacing.
    process_labels = {n: d["label"] for n, d in g.nodes(data=True) if d["kind"] == "process"}
    nx.draw_networkx_labels(g, pos, labels=process_labels, font_size=6.5, ax=ax, font_color=INK_PRIMARY)

    for n, d in g.nodes(data=True):
        if d["kind"] == "process":
            continue
        x, y = pos[n]
        ax.text(
            x, y - 0.05, d["label"], rotation=55, rotation_mode="anchor",
            ha="right", va="top", fontsize=6, color=INK_PRIMARY,
        )

    root_note = f", earliest {shown_roots} of {total_roots} branches" if shown_roots < total_roots else ""
    ax.set_title(
        f"{scenario} evidence graph -- branches with network/file activity "
        f"({len(nodes)} of {len(all_nodes)} total process nodes shown{root_note}; full graph in {scenario}.json)",
        fontsize=11, color=INK_PRIMARY, fontweight="bold", loc="left",
    )
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROCESS_COLOR, markersize=8, label="process"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=IP_COLOR, markersize=7, label="external IP"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FILE_COLOR, markersize=7, label="file write"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8, labelcolor=INK_MUTED)
    ax.axis("off")

    out_path = EVIDENCE_DIR / f"{scenario}_graph.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"{scenario}: saved {out_path.relative_to(REPO_ROOT)}")
    return out_path


if __name__ == "__main__":
    scenarios = sys.argv[1:] or ["WS12"]
    for s in scenarios:
        render(s)
