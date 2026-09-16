"""Depth attention maps and result figures. Every panel from measured data."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gs
from PIL import Image
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
F = "/home/kavinder/ARNABI_ARSH/vlm-hallu/paper/figs"
CR = f"{F}/crops"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BAD, GOOD, HL, GREY = "#c0392b", "#1a6b54", "#2c5aa0", "#95a5a6"


def save(fig, n):
    fig.savefig(f"{F}/{n}.png"); fig.savefig(f"{F}/{n}.pdf"); plt.close(fig)


# ------------------------------------------------- FIG: ALL 28 LAYERS, overlaid on real images
def fig_depth_grid():
    R = json.load(open(f"{D}/phase85_figure_inference.json"))[:3]
    nL = 28
    byq = {r["question_id_full"]: r for r in
           (json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl"))}
    fig = plt.figure(figsize=(7.4, 2.05 * len(R)))
    outer = gs.GridSpec(len(R), 2, width_ratios=[1, 7.4], wspace=0.04, hspace=0.34)
    for ri, r in enumerate(R):
        src = byq[r["qid"]]
        gh, gw = src["grid"]
        A = np.stack([np.asarray(src["attn"][f"L{i}"], float) for i in range(nL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        img = Image.open(f"{CR}/{r['qid'].replace('/','_')}_full.png").convert("RGB")
        gt = src["gt_box_frac"]
        W, H = img.size
        ax = fig.add_subplot(outer[ri, 0])
        ax.imshow(img); ax.axis("off")
        ax.add_patch(Rectangle((gt[0]*W, gt[1]*H), (gt[2]-gt[0])*W, (gt[3]-gt[1])*H,
                               fill=False, ec="#f1c40f", lw=1.3))
        q = r["question"].split("(A)")[0].strip()
        ax.set_title("\n".join([q[:22], q[22:44]]).strip(), fontsize=5.0, loc="left",
                     linespacing=1.1)
        inner = gs.GridSpecFromSubplotSpec(2, 14, subplot_spec=outer[ri, 1],
                                           hspace=0.42, wspace=0.06)
        # the paper's criterion everywhere: does the W=0.15 window at this cell COVER
        # the evidence? Marking "cell centre inside the box" instead is much stricter
        # and gave 0/28 on every item, which would have made the caption false.
        inbox = lambda p: P70.coverage(p[0], p[1], gt) >= P70.COV_HIT
        for L in range(nL):
            ax = fig.add_subplot(inner[L // 14, L % 14])
            m = A[L].reshape(gh, gw)
            mm = np.full_like(m, -1.0)
            if gh > 2 and gw > 2: mm[1:-1, 1:-1] = m[1:-1, 1:-1]
            else: mm = m.copy()
            j = int(np.argmax(mm))
            p = (((j % gw)+.5)/gw, ((j//gw)+.5)/gh)
            ax.imshow(m, cmap="inferno", aspect="auto"); ax.axis("off")
            ax.add_patch(Rectangle((gt[0]*gw-.5, gt[1]*gh-.5), (gt[2]-gt[0])*gw,
                                   (gt[3]-gt[1])*gh, fill=False, ec="#f1c40f", lw=.8))
            ax.plot(p[0]*gw-.5, p[1]*gh-.5, "x", ms=4.0, mew=1.2,
                    color=GOOD if inbox(p) else BAD)
            ax.set_title(f"{L}", fontsize=5.0, pad=1.0,
                         color=GOOD if inbox(p) else BAD,
                         fontweight="bold" if 16 <= L <= 26 else "normal")
    save(fig, "fig_depth_grid")
    print("fig_depth_grid ok (28 layers x 3 items, 2x14)")


# ------------------------------------------------- FIG: per-layer gt_pct, all four models
def fig_gtpct_curves():
    spec = [("phase30c_attn_maps_all.jsonl", "Qwen3-VL-2B", 28, HL),
            ("phase74_Qwen2_VL_7B_Instruct.jsonl", "Qwen2-VL-7B", 28, BAD),
            ("phase82_onevision.jsonl", "LLaVA-OneVision-7B", 28, GOOD),
            ("phase82_llavanext.jsonl", "LLaVA-NeXT-7B", 32, "#8e44ad")]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for path, tag, nL, c in spec:
        rows = [json.loads(l) for l in open(f"{D}/{path}")]
        g = [[] for _ in range(nL)]
        for r in rows:
            gh, gw = r["grid"]
            A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(nL)])
            A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
            yy, xx = np.mgrid[0:gh, 0:gw]
            fy, fx = (yy+.5)/gh, (xx+.5)/gw
            cov = np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                            for i in range(gh*gw)])
            rm = np.zeros((gh, gw), bool)
            if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
            else: rm[:] = True
            rm = rm.flatten(); tgt = int(np.argmax(cov))
            for L in range(nL):
                v = np.where(rm, A[L], -1e9)
                g[L].append((v > v[tgt]).sum()/max(rm.sum(), 1))
        m = [float(np.mean(x)) for x in g]
        ax.plot(np.arange(nL)/(nL-1), m, lw=1.4, color=c, label=tag)
    ax.axhline(0.5, ls="--", lw=1, color="k")
    ax.text(0.015, 0.505, "chance", fontsize=6.5, va="bottom")
    ax.set_xlabel("relative depth (0 = first layer, 1 = last)")
    ax.set_ylabel("gt_pct  (lower = better)")
    ax.invert_yaxis()
    ax.legend(fontsize=6.4, frameon=False, ncol=2)
    save(fig, "fig_gtpct_curves")
    print("fig_gtpct_curves ok (4 models)")


# ------------------------------------------------- FIG: coverage dose-response + the cliff
def fig_coverage_cliff():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    cov = ["0%\n(missed)", "partial", "100%\n(full)"]
    delta = [-15.6, 11.0, 37.1]
    lo = [-26.0, 2.0, 24.2]; hi = [-5.2, 20.0, 50.0]
    x = np.arange(3)
    ax.bar(x, delta, 0.55, color=[BAD, GREY, GOOD], ec="k", lw=.4)
    ax.errorbar(x, delta, yerr=[np.array(delta)-np.array(lo), np.array(hi)-np.array(delta)],
                fmt="none", ecolor="k", lw=.9, capsize=2.5)
    ax.axhline(0, lw=.9, color="k")
    ax.set_xticks(x); ax.set_xticklabels(cov)
    ax.set_ylabel("Δ vs no crop (pp)")
    ax.set_title("coverage decides the sign", fontsize=8)
    ax.set_xlabel("evidence covered by the window")

    ax = axes[1]
    for tag, below, above, c in [("Qwen3-VL-2B", 32.4, 77.3, HL), ("Qwen2-VL-7B", 36.6, 63.6, BAD)]:
        ax.plot([0.12, 0.25, 0.25, 0.6], [below, below, above, above], lw=1.6, color=c, label=tag)
    ax.plot([0.12, 0.6], [95.8, 100.0], lw=1.6, ls="--", color=GOOD, label="oracle crop (both)")
    ax.axvline(0.25, ls=":", lw=1.1, color="k")
    ax.axhline(25, ls="--", lw=.9, color=GREY)
    ax.text(0.13, 27, "chance", fontsize=6.3, color=GREY)
    ax.set_xscale("log"); ax.set_xticks([0.15, 0.25, 0.5]); ax.set_xticklabels(["0.15", "0.25", "0.5"])
    ax.set_xlabel("tokens on target"); ax.set_ylabel("accuracy (%)")
    ax.set_title("the encoding cliff", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="center right")
    save(fig, "fig_coverage_cliff")
    print("fig_coverage_cliff ok")


# ------------------------------------------------- FIG: sink enrichment
def fig_sink():
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    models = ["Qwen3-VL-2B", "Qwen2-VL-7B", "LLaVA-OneVision-7B", "LLaVA-NeXT-7B"]
    trail = [3.4, 4.5, 2.15, 2.15]
    err = [[0.4, 0.4, 0.15, 0.15], [0.4, 0.3, 0.15, 0.15]]
    kind = ["implicit\n(last column)", "implicit\n(last column)",
            "explicit\n(image_newline)", "explicit\n(image_newline)"]
    x = np.arange(4)
    ax.bar(x, trail, 0.5, yerr=err, color=[HL, HL, GOOD, GOOD], ec="k", lw=.4,
           error_kw=dict(lw=.9, capsize=2.5))
    ax.axhline(1.0, ls="--", lw=1, color="k")
    ax.text(3.45, 1.08, "fair share", fontsize=6.3, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{k}" for m, k in zip(models, kind)], fontsize=6.0)
    ax.set_ylabel("enrichment (×)")
    ax.set_title("the sink sits on the row boundary — wherever that boundary is marked",
                 fontsize=7.6)
    save(fig, "fig_sink")
    print("fig_sink ok")


if __name__ == "__main__":
    for fn in (fig_sink, fig_coverage_cliff, fig_gtpct_curves, fig_depth_grid):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
