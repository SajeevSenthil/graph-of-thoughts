"""
Review-deck figures built purely from existing data (results/raw/*.json,
data/evidence_graph/*.json, data/ground_truth/*.json) -- no new inference.
Outputs to results/review_figures/.
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.scorer import _contains  # noqa: E402
from src.methods.prompter import STAGES  # noqa: E402

RAW = REPO_ROOT / "results" / "raw"
GT = REPO_ROOT / "data" / "ground_truth"
EV = REPO_ROOT / "data" / "evidence_graph"
OUT = REPO_ROOT / "results" / "review_figures"
OUT.mkdir(parents=True, exist_ok=True)

PAGE, INK, MUTED, GRID = "#f9f9f7", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, CRIMSON, AMBER, TEAL, GREEN = "#3f7cac", "#a6373a", "#b9791f", "#2c766e", "#3c8a55"
MODEL_COLOR = {"qwen2.5-7b": "#2a78d6", "llama3.1-8b": "#eb6834"}


def load_runs():
    for f in sorted(RAW.glob("*.json")):
        scenario, method, model, seed = f.stem.split("__")
        yield scenario, method, model, seed, json.loads(f.read_text())["output"]


def funnel():
    total = detected = correct = 0
    for scenario, _m, _mo, _s, out in load_runs():
        flagged = out.get("malicious_events", [])
        for n in json.loads((GT / f"{scenario}.json").read_text()):
            if n["match_substring"] is None:
                continue
            total += 1
            hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
            if hits:
                detected += 1
                if any((e.get("stage") or "").strip().lower() == n["stage"].strip().lower() for e in hits):
                    correct += 1
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=PAGE)
    labels = ["Real attack events\n(across all 63 runs)", "Detected at all", "Detected AND\nexact stage correct"]
    vals = [total, detected, correct]
    colors = [MUTED, AMBER, CRIMSON]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6, zorder=3)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + total * 0.01, bar.get_y() + bar.get_height() / 2, f"{v}  ({v / total:.1%})",
                va="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlim(0, total * 1.28)
    ax.set_facecolor(PAGE)
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.set_title("Where the reconstruction fails: detection, then labeling", fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "funnel.png", dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("funnel", total, detected, correct)


def pred_vs_true():
    small = {"WS12", "Ubuntu"}
    true = collections.Counter()
    for s in small:
        for n in json.loads((GT / f"{s}.json").read_text()):
            if n["match_substring"] is not None:
                true[n["stage"].strip().lower()] += 1
    pred = {m: collections.Counter() for m in MODEL_COLOR}
    for scenario, _m, model, _s, out in load_runs():
        if scenario in small:
            for e in out.get("malicious_events", []):
                pred[model][(e.get("stage") or "").strip().lower()] += 1
    stages = [s.lower() for s in STAGES]
    tt = sum(true.values())
    series = {"Ground truth": [true[s] / tt for s in stages]}
    for m, c in pred.items():
        t = sum(c.values())
        series[m] = [c[s] / t for s in stages]
    colors = {"Ground truth": INK, **MODEL_COLOR}
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=PAGE)
    w = 0.27
    for i, (name, vals) in enumerate(series.items()):
        ax.bar([x + (i - 1) * w for x in range(len(stages))], vals, width=w, label=name, color=colors[name], zorder=3)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.title() for s in stages], rotation=25, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.set_facecolor(PAGE)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Stage labels each model outputs vs. what the ground truth contains (WS12 + Ubuntu)",
                 fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "stage_pred_vs_true.png", dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("pred_vs_true done")


def attack_chain():
    g = json.loads((EV / "WS12.json").read_text())
    nodes, edges = g["nodes"], g["edges"]
    root = next(k for k, n in nodes.items() if n["pid"] == "1996" and "sangforcat" in n["command_line"])
    kids, leaves = collections.defaultdict(list), collections.defaultdict(list)
    for e in edges:
        (kids if e["type"] == "spawned" else leaves)[e["from_node"]].append(e)
    G = nx.DiGraph()
    kind, label = {}, {}

    def add(key, depth):
        n = nodes[key]
        G.add_node(key)
        kind[key] = "process"
        label[key] = f"{n['image']}\nPID {n['pid']}"
        for e in leaves[key][:3]:
            leaf = (key, e["target"])
            G.add_edge(key, leaf)
            kind[leaf] = "ip" if e["type"] == "connected_to" else "file"
            t = e["target"].replace("\\", "/").split("/")[-1] if e["type"] == "wrote" else e["target"]
            label[leaf] = t[:26]
        if depth < 2:
            for e in kids[key]:
                G.add_edge(key, e["to_node"])
                add(e["to_node"], depth + 1)

    add(root, 0)
    levels = {root: 0}
    for u, v in nx.bfs_edges(G, root):
        levels[v] = levels[u] + 1
    by_level = collections.defaultdict(list)
    for n, d in levels.items():
        by_level[d].append(n)
    pos = {}
    for d, ns in by_level.items():
        for i, n in enumerate(ns):
            pos[n] = ((i - (len(ns) - 1) / 2) * 2.4, -d * 1.6)
    col = {"process": BLUE, "ip": CRIMSON, "file": AMBER}
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=PAGE)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c3c2b7", arrows=True, arrowsize=12, node_size=900)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[kind[n]] for n in G.nodes], node_size=900)
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.32, label[n], ha="center", va="top", fontsize=9, color=INK)
    for k, c in (("process", "Process"), ("ip", "External IP contacted"), ("file", "Suspicious file written")):
        ax.scatter([], [], c=col[k], s=80, label=c)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("WS12 evidence graph -- the real attack chain (ground-truth event A2: Initial Access)",
                 fontsize=13, fontweight="bold", loc="left")
    ax.axis("off")
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(OUT / "attack_chain_graph.png", dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("attack_chain done", len(G.nodes))


def demo_panel(run_file="WS12__CoT__llama3.1-8b__seed1.json"):
    d = json.loads((RAW / run_file).read_text())
    out, score = d["output"], d["score"]
    flagged = out.get("malicious_events", [])
    rows = []
    for n in json.loads((GT / "WS12.json").read_text()):
        hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
        if not hits:
            verdict, pred, color = "MISSED", "-", CRIMSON
        elif any((e.get("stage") or "").strip().lower() == n["stage"].lower() for e in hits):
            verdict, pred, color = "FOUND, stage correct", n["stage"], GREEN
        else:
            verdict, pred, color = "FOUND, wrong stage", hits[0].get("stage", "?"), AMBER
        rows.append((n["id"], n["match_substring"][:34], n["stage"], pred, verdict, color))
    fig, ax = plt.subplots(figsize=(13, 3.9), facecolor=PAGE)
    ax.axis("off")
    cols = ["ID", "Ground-truth command", "True stage", "Model's stage", "Outcome"]
    xs = [0.01, 0.07, 0.42, 0.60, 0.77]
    for x, c in zip(xs, cols):
        ax.text(x, 0.93, c, fontsize=11, fontweight="bold", transform=ax.transAxes, color=INK)
    for i, (rid, cmd, ts, ps, verdict, color) in enumerate(rows):
        y = 0.80 - i * 0.13
        for x, t in zip(xs, [rid, cmd, ts, ps]):
            ax.text(x, y, t, fontsize=10.5, transform=ax.transAxes, color=INK, family="monospace" if x == xs[1] else None)
        ax.text(xs[4], y, verdict, fontsize=10.5, fontweight="bold", transform=ax.transAxes, color=color)
    s, e = score["stage"], score["event"]
    ax.set_title(f"One real run: WS12 / CoT / Llama 3.1 8B / seed 1  --  "
                 f"event F1 {e['f1']:.2f}, exact-stage recall {s['recall']:.2f}, {out['_cost']['llm_calls']} LLM call, "
                 f"{out['_cost']['total_tokens']:,} tokens", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "demo_panel.png", dpi=200, facecolor=PAGE)
    plt.close(fig)
    print("demo_panel done", run_file, "parse_ok:", out.get("_parse_ok"))


if __name__ == "__main__":
    funnel()
    pred_vs_true()
    attack_chain()
    demo_panel()
