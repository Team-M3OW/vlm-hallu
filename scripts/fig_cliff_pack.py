"""Figures for the size-cliff appendix. Every panel from measured, cached data.

fig_cliff_curve    accuracy vs tokens-on-target (log): uniform steps, oracle flat
fig_cliff_budget   accuracy vs target side fraction at two budgets: the step does not move
fig_lens           per-layer logit-lens decodability: uniform flat, oracle steps at L22
fig_footprint      distribution of target side in merged-token units (0.25 / 1.0 lines)
fig_exchange       oracle crop at B0 vs the best uniform rung per model (the 26x gap)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
F = f"{D}/paper/figs"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BAD, GOOD, HL, GREY = "#c0392b", "#1a6b54", "#2c5aa0", "#95a5a6"
rng = np.random.default_rng(242)


def save(fig, n):
    fig.savefig(f"{F}/{n}.png"); fig.savefig(f"{F}/{n}.pdf"); plt.close(fig)


def load(f): return [json.loads(l) for l in open(f)]


q3 = load(f"{D}/data/phase78_w_sweep.jsonl")
q2 = load(f"{D}/data/phase97m_merged_qwen2vl.jsonl")
m2 = {r["question_id_full"]: r for r in load(f"{D}/data/phase74_Qwen2_VL_7B_Instruct.jsonl")}
m3 = {r["question_id_full"]: r for r in load(f"{D}/data/phase30c_attn_maps_all.jsonl")}
for rows, maps in ((q3, m3), (q2, m2)):
    for r in rows:
        if r.get("gt_area_frac") is None:
            x0, y0, x1, y1 = maps[r["question_id_full"]]["gt_box_frac"]
            r["gt_area_frac"] = (x1 - x0) * (y1 - y0)
        r["grid"] = maps[r["question_id_full"]]["grid"]


def ok(r, arm):
    p = r["probs"].get(arm)
    return None if p is None else float(int(np.argmax(p)) == r["label"])


def curve(rows, arm, xs, edges):
    ys, lo, hi, cs = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xs >= a) & (xs < b)
        v = np.array([ok(r, arm) for r, k in zip(rows, m) if k], float)
        if len(v) < 5: cs.append(np.sqrt(a * b)); ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        bt = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
        cs.append(np.sqrt(a * b)); ys.append(v.mean() * 100)
        lo.append(np.percentile(bt, 2.5) * 100); hi.append(np.percentile(bt, 97.5) * 100)
    return np.array(cs), np.array(ys), np.array(lo), np.array(hi)


# ---------------- fig_cliff_curve
edges = np.logspace(np.log10(0.02), np.log10(8.0), 9)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, (name, rows) in zip(axes, (("Qwen3-VL-2B", q3), ("Qwen2-VL-7B", q2))):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    xs = np.array([r["gt_area_frac"] * 300 for r in sr])
    for arm, c, lab in (("uniform@300", BAD, "uniform @300"), ("oracle@0.25", GOOD, "oracle crop")):
        c_, y_, l_, h_ = curve(sr, arm, xs, edges)
        ax.plot(c_, y_, "-o", ms=2.6, lw=1.4, color=c, label=lab)
        ax.fill_between(c_, l_, h_, color=c, alpha=0.18, lw=0)
    ax.axvspan(0.15, 0.25, color=GREY, alpha=0.25, lw=0)
    ax.axhline(25, ls="--", lw=0.9, color=GREY)
    ax.text(0.021, 27, "chance", fontsize=6, color=GREY)
    ax.set_xscale("log"); ax.set_xlim(0.02, 8); ax.set_ylim(0, 105)
    ax.set_xlabel("tokens on target (merged, $B_0{=}300$)"); ax.set_title(name, fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
axes[0].set_ylabel("accuracy (%)")
fig.text(0.5, -0.16, "shaded band: the 0.15--0.25 merged-token cliff; intervals are item bootstraps",
         ha="center", fontsize=6.2, color="#444444")
save(fig, "fig_cliff_curve")

# ---------------- fig_cliff_budget
edges = np.logspace(np.log10(0.3), np.log10(30.0), 9)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, (name, rows) in zip(axes, (("Qwen3-VL-2B", q3), ("Qwen2-VL-7B", q2))):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    for arm, B, c, lab in (("uniform@300", 300.0, BAD, "uniform @300"), ("uniform@600", 588.0, HL, "uniform @588")):
        xs = np.array([np.sqrt(r["gt_area_frac"]) * 100 for r in sr])
        c_, y_, l_, h_ = curve(sr, arm, xs, edges)
        ax.plot(c_, y_, "-o", ms=2.6, lw=1.4, color=c, label=lab)
        ax.fill_between(c_, l_, h_, color=c, alpha=0.15, lw=0)
    ax.axvline(2.9, ls=":", lw=1.1, color="k")
    ax.text(3.05, 5, "cliff ($\\approx$3\\% side)", fontsize=6, color="k")
    ax.set_xscale("log"); ax.set_xlim(0.3, 30); ax.set_ylim(0, 105)
    ax.set_xlabel("target side (\\% of image, log)"); ax.set_title(name, fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
axes[0].set_ylabel("accuracy (%)")
fig.text(0.5, -0.16, "the step sits at the same relative target size at both budgets; only the token threshold moves",
         ha="center", fontsize=6.2, color="#444444")
save(fig, "fig_cliff_budget")

# ---------------- fig_lens
lens = []
rows60 = load(f"{D}/data/phase60_logit_lens.jsonl")
for arm in ("uniform", "oracle"):
    acc = []
    for l in range(28):
        acc.append(np.mean([int(np.argmax(r[arm][l])) == r["label"] for r in rows60]) * 100)
    lens.append((arm, np.array(acc)))
rows84 = load(f"{D}/data/phase84_lens_qwen2vl.jsonl")
lens2 = []
for arm, lab in (("uniform@300", "uniform"), ("oracle@0.15", "oracle")):
    acc = []
    for l in range(28):
        acc.append(np.mean([int(np.argmax(r["traj"][arm][l])) == r["label"] for r in rows84]) * 100)
    lens2.append((lab, np.array(acc)))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, (name, data) in zip(axes, (("Qwen3-VL-2B", lens), ("Qwen2-VL-7B", lens2))):
    for arm, y in data:
        c = GOOD if arm == "oracle" else BAD
        ax.plot(np.arange(28), y, "-o", ms=2.2, lw=1.4, color=c, label=f"{arm} encoding")
    ax.axhline(25, ls="--", lw=0.9, color=GREY)
    ax.axvline(22, ls=":", lw=1.0, color="k")
    ax.text(22.4, 30, "L22", fontsize=6, color="k")
    ax.set_ylim(0, 105); ax.set_xlabel("layer"); ax.set_title(name, fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
axes[0].set_ylabel("lens argmax accuracy (%)")
fig.text(0.5, -0.16, "uniform: no layer above 52.9\\%; oracle: the answer appears abruptly at L22",
         ha="center", fontsize=6.2, color="#444444")
save(fig, "fig_lens")

# ---------------- fig_footprint
fig, ax = plt.subplots(figsize=(4.4, 2.4))
bins = np.linspace(0, 1.2, 25)
for name, rows, c in (("Qwen3-VL-2B", q3, HL), ("Qwen2-VL-7B", q2, BAD)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    sides = np.array([np.sqrt(r["gt_area_frac"] * r["grid"][0] * r["grid"][1]) for r in sr])
    ax.hist(sides, bins=bins, histtype="step", lw=1.5, color=c, label=name, density=True)
ax.axvline(0.25, ls=":", lw=1.2, color="k"); ax.text(0.26, ax.get_ylim()[1] * 0.9, "cliff (0.25)", fontsize=6)
ax.axvline(1.0, ls="--", lw=1.2, color="k"); ax.text(0.9, ax.get_ylim()[1] * 0.7, "one token", fontsize=6, ha="right")
ax.set_xlabel("target side (merged tokens)"); ax.set_ylabel("density")
ax.legend(fontsize=6.2, frameon=False)
fig.text(0.5, -0.18, "every single-region target is smaller than one merged token; 62\\% sit below the cliff",
         ha="center", fontsize=6.2, color="#444444")
save(fig, "fig_footprint")

# ---------------- fig_exchange (Phase 28 numbers)
models = ["Qwen2-VL-7B", "LLaVA-OV-7B", "LLaVA-NeXT-7B", "Qwen3-VL-2B"]
oracle = [93.2, 86.9, 80.6, 97.4]; oracle_tok = [300, 1323, 1176, 300]
ceil = [75.4, 69.6, 38.2, np.nan]; ceil_tok = [7776, 5157, 1464, 0]
ratio = ["$\\geq$25.9$\\times$", "$\\geq$3.9$\\times$", "no axis", "$\\geq$26$\\times$"]
x = np.arange(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.bar(x - w / 2, oracle, w, color=GOOD, ec="k", lw=0.4, label="oracle crop @$B_0$")
ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in ceil], w, color=BAD, ec="k", lw=0.4, label="best uniform rung")
for i in range(len(models)):
    ax.text(i - w / 2, oracle[i] + 1.5, f"{oracle_tok[i]} tok", ha="center", fontsize=5.8)
    if not np.isnan(ceil[i]): ax.text(i + w / 2, ceil[i] + 1.5, f"{ceil_tok[i]} tok", ha="center", fontsize=5.8)
    ax.text(i, 104, ratio[i], ha="center", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels(["Qwen2", "LLaVA-OV", "LLaVA-NeXT", "Qwen3"], fontsize=6.8)
ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 112)
ax.legend(fontsize=6.2, frameon=False, loc="lower left")
save(fig, "fig_exchange")

print("figs written to", F)
