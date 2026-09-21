"""Two-checkpoint version of fig_layer_accuracy: end-task accuracy of cropping at each layer's arg-max (phase 199),
Qwen3-VL-2B and Qwen2-VL-7B, against the same reference arms."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; PURPLE = "#8250df"
NL = 28
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), sharey=True)
for ax, w, name in zip(axes, ["qwen3", "qwen2"], ["Qwen3-VL-2B", "Qwen2-VL-7B"]):
    sw = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase199_layersweep_{w}.jsonl")}
    arm = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase184_allarms_{w}.jsonl")}
    ids = sorted(set(sw) & set(arm)); n = len(ids)
    acc = 100 * np.array([[int(np.argmax(sw[q]["probs"][f"L{l}"]) == sw[q]["label"]) for l in range(NL)] for q in ids]).mean(0)
    se = 100 * np.array([np.std([int(np.argmax(sw[q]["probs"][f"L{l}"]) == sw[q]["label"]) for q in ids]) / np.sqrt(n) for l in range(NL)])
    ref = {k: 100 * np.mean([int(np.argmax(arm[q]["probs"][k]) == arm[q]["label"]) for q in ids if k in arm[q]["probs"]])
           for k in ("uniform@600", "vicrop_block", "ridge", "laser", "vicrop_L14")}
    ax.axvspan(-0.5, 15.5, color="#f6f8fa", zorder=0)
    ax.errorbar(range(NL), acc, yerr=se, fmt="o-", color=BLUE, ms=4.5, lw=1.8, capsize=2.5, zorder=3,
                label="crop at layer $\\ell$'s arg-max")
    ax.axvline(15.5, color=GREEN, ls="--", lw=2, zorder=2)
    for k, lbl, col, ls, xt in [("ridge", "TWR (all layers)", GREEN, "-", 0.2),
                                ("laser", "LASER (per-sample layer)", PURPLE, ":", 0.2),
                                ("uniform@600", "no crop, equal compute", GREY, "--", 8.6),
                                ("vicrop_L14", "published fixed layer (L14)", RED, "-.", 0.2)]:
        ax.axhline(ref[k], color=col, ls=ls, lw=1.6, zorder=1)
        ax.text(xt, ref[k] + 0.6, lbl, fontsize=8, color=col, ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.8))
    b = int(np.argmax(acc))
    ax.annotate(f"best single layer: L{b} = {acc[b]:.1f}%", (b, acc[b]), textcoords="offset points",
                xytext=(24, -20), fontsize=8.6, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.text(7.5, acc.max() - 1.5, "transport window", fontsize=8.4, color=GREEN, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    ax.set_xlabel("read-out layer $\\ell$")
    ax.set_title(f"{name}, $n$={n}", fontsize=10.5)
    ax.set_xlim(-0.8, 27.8); ax.grid(alpha=.22, zorder=0)
    inside = acc[:16]; after = acc[16:]
    print(f"{w}: spread {acc.min():.1f}-{acc.max():.1f}; best L{b}; window max {inside.max():.1f} (vs bar {ref['uniform@600']:.1f}); "
          f"after max {after.max():.1f}; block {ref['vicrop_block']:.1f}; LASER {ref['laser']:.1f}; TWR {ref['ridge']:.1f}")
axes[0].set_ylabel("V$^*$Bench accuracy (%)")
axes[0].legend(fontsize=8.4, loc="lower right")
plt.tight_layout()
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_layer_accuracy2.{e}", dpi=170, bbox_inches="tight")
print("wrote fig_layer_accuracy2")
