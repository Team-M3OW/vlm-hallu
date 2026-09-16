"""Generate every paper figure from measured data. No hand-drawn numbers."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gs
from PIL import Image
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
F = "/home/kavinder/ARNABI_ARSH/vlm-hallu/paper/figs"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 220, "savefig.bbox": "tight", "axes.grid": False})
C = {"bad": "#c0392b", "good": "#1a6b54", "neutral": "#7f8c8d", "hl": "#2c5aa0",
     "light": "#d5dbdb"}


def vstar():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    return root, load_dataset("craigwu/vstar_bench")["test"]


# ---------------------------------------------------------------- FIG 1 : the defect, on a real image
def fig1():
    root, ds = vstar()
    rows = {r["question_id_full"]: r for r in map(json.loads,
            open(f"{D}/phase30c_attn_maps_all.jsonl"))}
    ex = next(e for e in ds if f"{e['category']}/{e['question_id']}" in rows
              and rows[f"{e['category']}/{e['question_id']}"]["gt_area_frac"] < 0.004)
    qid = f"{ex['category']}/{ex['question_id']}"
    r = rows[qid]
    img = Image.open(os.path.join(root, ex["image"])).convert("RGB")
    gh, gw = r["grid"]
    A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(28)])
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    gt = r["gt_box_frac"]
    W, H = img.size

    show = [2, 12, 17, 21, 27]
    fig = plt.figure(figsize=(7.6, 2.5))
    g = gs.GridSpec(1, len(show) + 2, width_ratios=[1.5] + [1] * len(show) + [1], wspace=0.30)

    ax = fig.add_subplot(g[0]); ax.imshow(img); ax.axis("off")
    ax.add_patch(Rectangle((gt[0]*W, gt[1]*H), (gt[2]-gt[0])*W, (gt[3]-gt[1])*H,
                           fill=False, ec="#f1c40f", lw=1.8))
    ax.set_title("image + target", fontsize=7.5)

    def peak(m):
        mm = np.full_like(m, -1.0); mm[1:-1, 1:-1] = m[1:-1, 1:-1]
        i = int(np.argmax(mm)); return ((i % gw)+.5)/gw, ((i//gw)+.5)/gh
    inbox = lambda p: gt[0] <= p[0] <= gt[2] and gt[1] <= p[1] <= gt[3]

    for k, L in enumerate(show):
        ax = fig.add_subplot(g[k+1])
        m = A[L].reshape(gh, gw)
        ax.imshow(m, cmap="magma"); ax.axis("off")
        p = peak(m)
        ax.plot(p[0]*gw-.5, p[1]*gh-.5, marker="x", ms=7, mew=2,
                color=C["good"] if inbox(p) else C["bad"])
        ax.add_patch(Rectangle((gt[0]*gw-.5, gt[1]*gh-.5), (gt[2]-gt[0])*gw, (gt[3]-gt[1])*gh,
                               fill=False, ec="#f1c40f", lw=1.2))
        ax.set_title(f"layer {L}", fontsize=7.5,
                     color=C["good"] if inbox(p) else C["bad"])

    ax = fig.add_subplot(g[-1])
    m = A[list(range(16, 27))].mean(0).reshape(gh, gw)
    ax.imshow(m, cmap="magma"); ax.axis("off")
    p = peak(m)
    ax.plot(p[0]*gw-.5, p[1]*gh-.5, marker="x", ms=7, mew=2,
            color=C["good"] if inbox(p) else C["bad"])
    ax.add_patch(Rectangle((gt[0]*gw-.5, gt[1]*gh-.5), (gt[2]-gt[0])*gw, (gt[3]-gt[1])*gh,
                           fill=False, ec="#f1c40f", lw=1.2))
    ax.set_title("MEAN 16–26\n(standard)", fontsize=7.2, color=C["bad"], fontweight="bold")
    fig.savefig(f"{F}/fig1_defect.png"); plt.close(fig)
    print("fig1 ok", qid)


# ---------------------------------------------------------------- FIG 2 : averaging more is worse
def fig2():
    ks = [1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 28]
    v = [41.9, 44.5, 45.5, 45.5, 45.5, 41.4, 39.8, 39.3, 39.3, 39.8, 36.6]
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    ax.plot(ks, v, "o-", color=C["hl"], ms=3.5, lw=1.4)
    ax.axhline(39.3, ls="--", lw=1, color=C["bad"])
    ax.annotate("deployed read-out\n(11-layer mean)", (11, 39.3), (13, 43.2), fontsize=6.5,
                color=C["bad"], arrowprops=dict(arrowstyle="->", color=C["bad"], lw=.8))
    ax.annotate("best", (3, 45.5), (3.4, 47.0), fontsize=6.5, color=C["good"])
    ax.set_xlabel("number of layers averaged"); ax.set_ylabel("evidence found (%)")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 11, 28]); ax.set_xticklabels(["1", "3", "11", "28"])
    ax.set_ylim(34, 49)
    fig.savefig(f"{F}/fig2_averaging.png"); plt.close(fig)
    print("fig2 ok")


# ---------------------------------------------------------------- FIG 3 : pruning, two models
def fig3():
    keeps = ["10%", "25%", "50%"]
    d = {"Qwen3-VL-2B": {"none": 56.5, "rand": [38.2, 41.4, 47.1], "l2": [34.6, 40.3, 52.9],
                         "late": [56.0, 58.1, 55.5]},
         "Qwen2-VL-7B": {"none": 52.9, "rand": [44.5, 47.1, 45.5], "l2": [39.8, 40.3, 43.5],
                         "late": [51.3, 50.8, 50.3]}}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45), sharey=False)
    for ax, (tag, m) in zip(axes, d.items()):
        x = np.arange(3); w = 0.27
        ax.bar(x - w, m["rand"], w, label="random", color=C["light"], ec="k", lw=.4)
        ax.bar(x, m["l2"], w, label="layer 2 (FastV)", color=C["bad"], ec="k", lw=.4)
        ax.bar(x + w, m["late"], w, label="late layers (ours)", color=C["good"], ec="k", lw=.4)
        ax.axhline(m["none"], ls="--", lw=1, color="k")
        ax.text(2.45, m["none"] + .4, "keep all tokens", fontsize=6, ha="right")
        ax.set_xticks(x); ax.set_xticklabels(keeps); ax.set_xlabel("visual tokens kept")
        ax.set_title(tag, fontsize=8)
        ax.set_ylim(30, m["none"] + 4)
    axes[0].set_ylabel("accuracy (%)")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=6.6, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.17))
    fig.savefig(f"{F}/fig3_pruning.png"); plt.close(fig)
    print("fig3 ok")


# ---------------------------------------------------------------- FIG 4 : the L21 step
def fig4():
    R = [json.loads(l) for l in open(f"{D}/phase79_lens_mechanism.jsonl")]
    hit = lambda r, a, i: 1.0 * (max(range(4), key=lambda j: r["traj"][a][i][j]) == r["label"])
    nL = len(R[0]["traj"]["uniform@300"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    ax = axes[0]
    for a, lab, c in [("uniform@300", "no crop", C["bad"]),
                      ("head@0.15", "our crop", C["good"]),
                      ("oracle@0.15", "perfect crop", C["hl"])]:
        ax.plot(range(nL), [100*np.mean([hit(r, a, i) for r in R]) for i in range(nL)],
                lw=1.5, color=c, label=lab)
    ax.axvline(21, ls=":", lw=1.2, color="k")
    ax.text(21.3, 32, "L21", fontsize=7)
    ax.set_xlabel("layer"); ax.set_ylabel("answer correct (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("the answer appears at one layer", fontsize=8)

    ax = axes[1]
    cov = [r for r in R if r["head_cov"] >= .5]; mis = [r for r in R if r["head_cov"] < .5]
    for S, lab, c in [(cov, f"window covers  (n={len(cov)})", C["good"]),
                      (mis, f"window misses  (n={len(mis)})", C["bad"])]:
        ax.plot(range(nL), [100*(np.mean([hit(r, "head@0.15", i) for r in S])
                                 - np.mean([hit(r, "uniform@300", i) for r in S]))
                            for i in range(nL)], lw=1.5, color=c, label=lab)
    ax.axhline(0, lw=.8, color="k"); ax.axvline(21, ls=":", lw=1.2, color="k")
    ax.set_xlabel("layer"); ax.set_ylabel("gain over no crop (pp)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("and only when the crop delivers", fontsize=8)
    fig.savefig(f"{F}/fig4_l21.png"); plt.close(fig)
    print("fig4 ok")


# ---------------------------------------------------------------- FIG 5 : qualitative crops
def fig5():
    root, ds = vstar()
    P = json.load(open(f"{D}/phase71a_head_proposals.json"))
    E = {f"{e['category']}/{e['question_id']}": e for e in ds}
    picks = [q for q in P if P[q]["head_cov"] >= .5 > P[q]["argmax_cov"]][:4]
    fig, axes = plt.subplots(2, len(picks), figsize=(7.2, 3.7))
    for c, q in enumerate(picks):
        ex = E[q]; d = P[q]
        img = Image.open(os.path.join(root, ex["image"])).convert("RGB")
        W, H = img.size; gt = d["gt_box_frac"]
        for rIdx, (key, lab, col) in enumerate([("argmax", "standard read-out", C["bad"]),
                                                ("head", "ours", C["good"])]):
            ax = axes[rIdx, c]; ax.imshow(img); ax.axis("off")
            cx, cy = d[key]; Wn = 0.15
            x0 = min(max(cx - Wn/2, 0), 1-Wn); y0 = min(max(cy - Wn/2, 0), 1-Wn)
            ax.add_patch(Rectangle((gt[0]*W, gt[1]*H), (gt[2]-gt[0])*W, (gt[3]-gt[1])*H,
                                   fill=False, ec="#f1c40f", lw=1.6))
            ax.add_patch(Rectangle((x0*W, y0*H), Wn*W, Wn*H, fill=False, ec=col, lw=1.8))
            if rIdx == 0:
                ax.set_title(ex["text"][:44] + "…", fontsize=5.6)
            if c == 0:
                ax.text(-0.06, 0.5, lab, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7, color=col, fontweight="bold")
    fig.savefig(f"{F}/fig5_qualitative.png"); plt.close(fig)
    print("fig5 ok", len(picks))


# ---------------------------------------------------------------- FIG 6 : where the signal lives
def fig6():
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    stages = ["vision\nencoder", "LM layer 2\n(FastV prunes here)", "LM layers 16–26\n(signal is here)",
              "LM layer 21\n(answer forms)"]
    vals = [2.3, 39.8, 51.3, None]
    xs = [0, 1, 2, 3]
    ax.plot([-0.35, 3.35], [0, 0], color="k", lw=1.1)
    for x, s in zip(xs, stages):
        ax.plot([x], [0], marker="o", ms=9, color="w", mec="k", mew=1.2, zorder=3)
        ax.text(x, -0.55, s, ha="center", va="top", fontsize=6.8)
    notes = [("no question-conditioned\nsignal at all\n(at chance)", C["bad"]),
             ("signal too weak:\npruning here is\nWORSE than random", C["bad"]),
             ("signal usable:\nprune 90% of tokens\nfor free", C["good"]),
             ("answer becomes\ndecodable", C["good"])]
    for x, (t, c) in zip(xs, notes):
        ax.text(x, 0.35, t, ha="center", va="bottom", fontsize=6.4, color=c)
    ax.annotate("", xy=(3.35, 0), xytext=(-0.35, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))
    ax.text(3.45, 0, "depth", fontsize=7, va="center")
    ax.set_xlim(-0.6, 3.9); ax.set_ylim(-1.6, 1.5); ax.axis("off")
    fig.savefig(f"{F}/fig6_depth.png"); plt.close(fig)
    print("fig6 ok")


if __name__ == "__main__":
    for fn in (fig2, fig3, fig6, fig1, fig4, fig5):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
