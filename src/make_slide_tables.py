"""
Builds results/review_figures/slide_tables.xlsx (one sheet per review-deck
table, computed from results/grid_results.csv where numeric) and a cropped
evidence-graph image for slides. Read-only over existing results.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from PIL import Image
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.scorer import _contains  # noqa: E402

OUT = REPO / "results" / "review_figures"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(REPO / "results" / "grid_results.csv")
df["parse_ok"] = df["parse_ok"].astype(bool)
small = df[df.scenario.isin(["WS12", "Ubuntu"])]

sheets = {}

sheets["S3 LitReview"] = pd.DataFrame([
    ["Reasoning topologies", "CoT (Wei 2022); ToT (Yao 2023); GoT (Besta 2024)", "Validated only on clean, synthetic tasks"],
    ["Attack reconstruction", "NODLINK (NDSS 2024); OCR-APT (CCS 2025)", "A trained detector does the work; the LLM only summarizes"],
    ["LLM security agents", "ExCyTIn-Bench (ICML 2026)", "Reasoning is confounded with SQL/tool-use skill"],
], columns=["Theme", "Key works", "Limitation"])

sheets["S5 Status"] = pd.DataFrame([
    ["Ground-truth parser", "89 hand-verified events, 5 scenarios", "Done (locked)"],
    ["Evidence graph + reduction + hard gate", "Gate passes on all 5 scenarios", "Done"],
    ["Serializer", "Identical text for all methods", "Done"],
    ["Scorer", "Event + stage metrics; unit-tested", "Done (locked)"],
    ["CoT / ToT / GoT on vLLM", "Shared prompt and parser", "Done"],
    ["Experiment runner + analysis", "63 runs saved individually", "Done"],
], columns=["Module", "Evidence", "Status"])

ds = []
for s, gt_n, tok in [("WS12", None, "~16K"), ("Ubuntu", None, "~19K"), ("APT29", None, "~58K"), ("Sidewinder", None, "~58K"), ("FIN6", None, "~58K")]:
    g = json.loads((REPO / "data" / "evidence_graph" / f"{s}.json").read_text())
    st = g["stats"]
    gt = json.loads((REPO / "data" / "ground_truth" / f"{s}.json").read_text())
    ds.append([s, f"{st['total_events']:,}", st["candidate_nodes"], st["candidate_edges"], tok,
               len(gt), sum(1 for n in gt if n["match_substring"])])
sheets["S6 Dataset"] = pd.DataFrame(ds, columns=["Scenario", "Raw log events", "Graph nodes", "Graph edges", "Evidence tokens", "GT events", "GT checkable"])

sheets["S6 RunMath"] = pd.DataFrame([
    ["Full symmetric grid", "5 scenarios x 3 methods x 2 models x 3 seeds", 90],
    ["Llama 3.1 8B", "5 scenarios x 3 methods x 3 seeds", 45],
    ["Qwen 2.5 7B (context limit)", "2 scenarios x 3 methods x 3 seeds", 18],
    ["Executed total", "45 + 18", 63],
    ["Not run (Qwen, 3 large scenarios)", "3 scenarios x 3 methods x 3 seeds", 27],
    ["Check", "63 + 27 = 90", 90],
], columns=["Item", "Calculation", "Runs"])

rows = []
for method in ["CoT", "ToT", "GoT"]:
    for model in ["qwen2.5-7b", "llama3.1-8b"]:
        s = small[(small.method == method) & (small.model == model)]
        rows.append([method, model, len(s), round(s.event_precision.mean(), 2), round(s.event_recall.mean(), 2),
                     round(s.event_f1.mean(), 2), round(s.stage_recall.mean(), 2), round(s.stage_recall_partial.mean(), 2),
                     round(s.hallucination_rate.mean(), 2), int(s.total_tokens.mean()), round(s.wall_clock_s.mean())])
sheets["S7 Results"] = pd.DataFrame(rows, columns=["Method", "Model", "Runs", "Event precision", "Event recall", "Event F1",
                                                   "Stage recall (exact)", "Stage recall (partial)", "Hallucination rate",
                                                   "Tokens/run", "Seconds/run"])

sig = []
for model in ["qwen2.5-7b", "llama3.1-8b"]:
    sub = small[small.model == model]
    for metric in ["event_f1", "stage_recall"]:
        piv = sub.pivot_table(index=["scenario", "seed"], columns="method", values=metric)
        for a, b in combinations(["CoT", "ToT", "GoT"], 2):
            d = (piv[a] - piv[b])
            try:
                p = stats.wilcoxon(piv[a], piv[b]).pvalue
            except ValueError:
                p = float("nan")
            sig.append([model, metric, f"{a} vs {b}", len(d), round(d.mean(), 3), round(p, 3), "No" if not p < 0.05 else "Yes"])
sheets["S7 Significance"] = pd.DataFrame(sig, columns=["Model", "Metric", "Comparison", "Pairs (n)", "Mean difference", "Wilcoxon p", "Significant (p<0.05)?"])

sens = []
for method in ["CoT", "ToT", "GoT"]:
    s = small[(small.method == method) & (small.model == "llama3.1-8b")]
    ok = s[s.parse_ok]
    sens.append([method, len(s), int((~s.parse_ok).sum()), round(s.event_f1.mean(), 2), round(ok.event_f1.mean(), 2) if len(ok) else None])
sheets["S7 ParseSensitivity"] = pd.DataFrame(sens, columns=["Method (Llama, WS12+Ubuntu)", "Runs", "Parse failures", "Mean event F1 (all runs)", "Mean event F1 (parse OK only)"])

pf = df[~df.parse_ok].groupby(["method", "model"]).size().reset_index(name="parse_failures")
pf["of_runs"] = pf.apply(lambda r: int(((df.method == r.method) & (df.model == r.model)).sum()), axis=1)
sheets["S10 ParseFailures"] = pf

total = detected = correct = 0
for _, r in df.iterrows():
    out = json.loads((REPO / r.raw_path).read_text())["output"]
    flagged = out.get("malicious_events", [])
    for n in json.loads((REPO / "data" / "ground_truth" / f"{r.scenario}.json").read_text()):
        if n["match_substring"] is None:
            continue
        total += 1
        hits = [e for e in flagged if _contains(n["match_substring"], e.get("command", ""))]
        if hits:
            detected += 1
            if any((e.get("stage") or "").strip().lower() == n["stage"].strip().lower() for e in hits):
                correct += 1
sheets["S8 Funnel"] = pd.DataFrame([
    ["Real attack events checked (all 63 runs)", total, "100%"],
    ["Detected at all", detected, f"{detected / total:.1%}"],
    ["Detected AND exact stage correct", correct, f"{correct / total:.1%}"],
], columns=["Step", "Count", "Share of all"])

with pd.ExcelWriter(OUT / "slide_tables.xlsx", engine="openpyxl") as w:
    for name, sdf in sheets.items():
        sdf.to_excel(w, sheet_name=name, index=False)
        ws = w.sheets[name]
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(len(str(c.value)) for c in col if c.value is not None) + 3)
print("wrote slide_tables.xlsx:", list(sheets))
print(sheets["S7 ParseSensitivity"].to_string(index=False))
print(sheets["S10 ParseFailures"].to_string(index=False))

im = Image.open(REPO / "data" / "evidence_graph" / "WS12_graph.png")
w_, h_ = im.size
im.crop((int(w_ * 0.345), 0, int(w_ * 0.745), h_)).save(OUT / "evidence_graph_attack_cluster.png")
print("cropped", im.size)
