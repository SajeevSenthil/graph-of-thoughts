"""Regenerates every figure and generated table of the thesis from results/ and data/.
Run from the repo root:  .venv/bin/python thesis/scripts/make_figures.py"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

R, D, OUT, TB = "results/", "data/", "thesis/figures/", "thesis/tables/"
M = ["CoT", "ToT", "GoT"]
C = {"CoT": "#4C72B0", "ToT": "#DD8452", "GoT": "#55A868"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(name, dpi=300):
    plt.savefig(OUT + name + ".pdf", bbox_inches="tight")
    plt.savefig(OUT + name + ".png", dpi=dpi, bbox_inches="tight")
    plt.close()


def load(path):
    return pd.read_csv(R + path)


base = load("grid_results.csv")
v = {k: load(f"variants/{f}/grid_results.csv") for k, f in
     [("v2", "v2_maxtok4096"), ("v3", "v3_stagedefs"), ("v4", "v4_triage"), ("v5", "v5_got_links"),
      ("scale", "scale_v4"), ("scale_v3", "scale_v3"), ("primus", "primus_v4")]}
v["v1"] = base

# ---- copy figures that already exist from earlier phases ----
shutil.copy(D + "evidence_graph/WS12_graph.png", OUT + "fig_evidence_ws12.png")
shutil.copy(D + "evidence_graph/Ubuntu_graph.png", OUT + "fig_evidence_ubuntu.png")
shutil.copy(R + "plots/stage_confusion_heatmap_llama3.1-8b.png", OUT + "fig_confusion_llama8b.png")
shutil.copy(R + "plots/stage_confusion_heatmap_qwen2.5-7b.png", OUT + "fig_confusion_qwen7b.png")
shutil.copy(R + "plots/pareto_stage_recall_vs_cost.png", OUT + "fig_pareto.png")

# ---- variant progress (pooled) ----
rows = [(n, d.event_precision.mean(), d.event_recall.mean(), d.stage_recall.mean())
        for n, d in [("v1\nbaseline", v["v1"]), ("v2\n4096 tokens", v["v2"]), ("v3\nstage defs", v["v3"]), ("v4\ntriage", v["v4"])]]
r = pd.DataFrame(rows, columns=["variant", "Precision", "Event recall", "Stage recall"]).set_index("variant")
ax = r.plot.bar(figsize=(6.4, 3.2), rot=0, color=["#8172B2", "#55A868", "#C44E52"])
ax.set_ylabel("Mean score"); ax.legend(frameon=False, fontsize=8)
save("fig_variants")

# ---- baseline by method ----
g = base.groupby("method")[["event_f1", "stage_recall", "event_recall"]].mean().loc[M]
g.columns = ["Event F1", "Stage recall", "Event recall"]
ax = g.plot.bar(figsize=(5.6, 3.1), rot=0); ax.set_ylabel("Mean score"); ax.legend(frameon=False, fontsize=8)
save("fig_baseline_methods")

# ---- model x topology comparison (v4 setup for every model) ----
l4 = v["v4"]
sets = [("Qwen 2.5 7B", l4[l4.model == "qwen2.5-7b"]), ("Llama 3.1 8B", l4[l4.model == "llama3.1-8b"]),
        ("Llama-Primus 8B", v["primus"]),
        ("Qwen 2.5 32B", v["scale"][v["scale"].model == "qwen2.5-32b-awq"]),
        ("Llama 3.1 70B", v["scale"][v["scale"].model == "llama3.1-70b-awq"])]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
for ax, (col, ttl) in zip(axs, [("event_f1", "Event F1"), ("event_recall", "Event recall"), ("stage_recall", "Exact stage recall")]):
    w = 0.26
    for i, m in enumerate(M):
        vals = [d[d.method == m][col].mean() for _, d in sets]
        ax.bar(np.arange(len(sets)) + (i - 1) * w, vals, w, label=m, color=C[m])
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s for s, _ in sets], rotation=25, ha="right", fontsize=8)
    ax.set_title(ttl, fontsize=10)
axs[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); save("fig_model_topology")
tab = pd.DataFrame([{"model": s, "method": m, "runs": int((d.method == m).sum()),
                     **{c: round(d[d.method == m][c].mean(), 3) for c in ["event_precision", "event_recall", "event_f1", "stage_recall", "hallucination_rate"]}}
                    for s, d in sets for m in M])
tab.to_csv(TB + "model_topology_v4.csv", index=False)
with open(TB + "model_topology_v4.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nModel & Method & Runs & Precision & Recall & Event F$_1$ & Stage recall \\\\\n\\midrule\n")
    for i, row in tab.iterrows():
        name = row.model if row.method == "CoT" else ""
        f.write(f"{name} & {row.method} & {row.runs} & {row.event_precision:.3f} & {row.event_recall:.3f} & {row.event_f1:.3f} & {row.stage_recall:.3f} \\\\\n")
        if row.method == "GoT" and i < len(tab) - 1:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- scenario x method heatmap (baseline, event recall and stage recall) ----
fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.9))
scen = ["WS12", "Ubuntu", "Sidewinder", "FIN6", "APT29"]
for ax, (col, ttl) in zip(axs, [("event_recall", "Event recall"), ("stage_recall", "Exact stage recall")]):
    h = base.pivot_table(index="scenario", columns="method", values=col).loc[scen, M]
    im = ax.imshow(h.values, cmap="Blues", vmin=0, vmax=max(0.6, h.values.max()))
    ax.set_xticks(range(3)); ax.set_xticklabels(M); ax.set_yticks(range(5)); ax.set_yticklabels(scen); ax.set_title(ttl, fontsize=10)
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{h.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout(); save("fig_scenario_heatmap")

# ---- input-length effect: 8B vs 70B, no triage, v3 prompt ----
a = v["v3"][v["v3"].model == "llama3.1-8b"].copy(); b = v["scale_v3"][v["scale_v3"].model == "llama3.1-70b-awq"].copy()
for d in (a, b):
    d["len"] = np.where(d.scenario.isin(["WS12", "Ubuntu"]), "16-19K tokens\n(WS12, Ubuntu)", "58K tokens\n(Windows 10 x3)")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3))
rows = []
for ax, (col, ttl) in zip(axs, [("event_recall", "Event recall"), ("event_precision", "Event precision")]):
    labs = sorted(a["len"].unique())
    for k, (nm, d, c) in enumerate([("Llama 3.1 8B", a, "#8172B2"), ("Llama 3.1 70B (AWQ)", b, "#C44E52")]):
        vals = [d[d["len"] == l][col].mean() for l in labs]
        ax.bar(np.arange(2) + (k - .5) * .35, vals, .35, label=nm, color=c)
        for l, val in zip(labs, vals): rows.append((nm, l.replace("\n", " "), col, round(val, 3)))
    ax.set_xticks(range(2)); ax.set_xticklabels(labs); ax.set_title(ttl, fontsize=10)
axs[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); save("fig_length_effect")
pd.DataFrame(rows, columns=["model", "input", "metric", "value"]).to_csv(TB + "length_effect.csv", index=False)

# ---- Primus parse success and table ----
P = v["primus"]
ps = P.groupby("method").parse_ok.mean().reindex(M)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.bar(M, ps.values, color=[C[m] for m in M]); ax.set_ylim(0, 1); ax.set_ylabel("Share of runs with parseable answer")
for i, x in enumerate(ps.values): ax.text(i, x + .02, f"{x:.2f}", ha="center")
save("fig_primus_parse")
t = P.groupby("method")[["event_precision", "event_recall", "event_f1", "stage_recall", "hallucination_rate"]].mean().reindex(M)
with open(TB + "primus_table.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrr@{}}\\toprule\nMethod & Precision & Recall & Event F$_1$ & Stage recall & Halluc.\\\\\\midrule\n")
    for m, row in t.iterrows():
        f.write(f"{m} & " + " & ".join(f"{x:.3f}" for x in row) + "\\\\\n")
    f.write(f"\\midrule\nAll ({len(P)} runs) & " + " & ".join(f"{x:.3f}" for x in P[t.columns].mean()) + "\\\\\n\\bottomrule\\end{tabular}\n")

# ---- hypothesis tests: paired Wilcoxon between topologies for each experiment ----
def ptest(d, col, a_, b_):
    pv = d.pivot_table(index=["scenario", "model", "seed"], columns="method", values=col)
    x, y = pv[a_], pv[b_]
    if np.allclose(x, y): return 1.0, 0.0, len(pv)
    return wilcoxon(x, y).pvalue, (x - y).mean(), len(pv)

exps = [("Baseline (v1)", v["v1"]), ("v3 stage definitions", v["v3"]), ("v4 triage", v["v4"]),
        ("Scale: Qwen 32B, Llama 70B", v["scale"]), ("Llama-Primus 8B", v["primus"])]
rows = []
for name, d in exps:
    for col, lab in [("stage_recall", "Stage recall"), ("event_recall", "Event recall")]:
        row = [name, lab]
        for a_, b_ in [("GoT", "CoT"), ("GoT", "ToT"), ("ToT", "CoT")]:
            p, dm, n = ptest(d, col, a_, b_)
            row += [dm, p]
        rows.append(row + [n])
T = pd.DataFrame(rows, columns=["exp", "metric", "d_gc", "p_gc", "d_gt", "p_gt", "d_tc", "p_tc", "n"])
T.round(4).to_csv(TB + "topology_tests.csv", index=False)
with open(TB + "topology_tests.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{GoT vs CoT} & \\multicolumn{2}{c}{GoT vs ToT} & \\multicolumn{2}{c}{ToT vs CoT} & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\nExperiment & Metric & $\\Delta$ & $p$ & $\\Delta$ & $p$ & $\\Delta$ & $p$ & $n$ \\\\\n\\midrule\n")
    for i, r_ in T.iterrows():
        nm = r_.exp if r_.metric == "Stage recall" else ""
        f.write(f"{nm} & {r_.metric} & {r_.d_gc:+.3f} & {r_.p_gc:.3f} & {r_.d_gt:+.3f} & {r_.p_gt:.3f} & {r_.d_tc:+.3f} & {r_.p_tc:.3f} & {int(r_.n)} \\\\\n")
        if r_.metric == "Event recall" and i < len(T) - 1:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- variant tests (v_prev vs v_next) on matched Llama/Qwen cells ----
def vtest(d1, d2, col):
    k = ["scenario", "model", "method", "seed"]
    m = d1.merge(d2, on=k, suffixes=("_a", "_b"))
    return m[col + "_a"].mean(), m[col + "_b"].mean(), wilcoxon(m[col + "_a"], m[col + "_b"]).pvalue, len(m)
rows = []
for lab, (a_, b_) in [("v1 to v2", ("v1", "v2")), ("v2 to v3", ("v2", "v3")), ("v3 to v4", ("v3", "v4"))]:
    for col, cl in [("stage_recall", "Stage recall"), ("event_precision", "Precision"), ("event_recall", "Event recall"), ("event_f1", "Event F1")]:
        x, y, p, n = vtest(v[a_], v[b_], col)
        rows.append((lab, cl, x, y, p, n))
pd.DataFrame(rows, columns=["step", "metric", "before", "after", "p", "n"]).round(4).to_csv(TB + "variant_tests.csv", index=False)
with open(TB + "variant_tests.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nStep & Metric & Before & After & $p$ & $n$ \\\\\n\\midrule\n")
    prev = None
    for s_, c_, x, y, p, n in rows:
        f.write(f"{s_ if s_ != prev else ''} & {c_} & {x:.3f} & {y:.3f} & {p:.4f} & {n} \\\\\n"); prev = s_
    f.write("\\bottomrule\n\\end{tabular}\n")
print("figures and tables written")
