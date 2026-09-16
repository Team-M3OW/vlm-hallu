"""
Phase 92: rebuild the re-ranking head on ALL 28 layers instead of the arbitrary L16-26 block.

WHY
---
The head reads L16-26, a band fixed before we knew anything about depth structure. Phase 91 showed
that on per-layer maps alone, giving the head ALL layers beats that block by +5.8pp (50.8% vs 45.0%)
-- and, importantly, that SELECTING good layers is worse than using all of them (top-3 by measured
quality: 44.0%). Poor layers are not noise to exclude; they are negative evidence about where NOT to
look, and filtering them discards that.

This rebuilds the full 65-feature head with L0-27 as input and freezes out-of-fold proposals to disk,
so the end-task GPU script does no fitting and cannot leak.

GATE, fixed before the run
    coverage > 52.9% + 3pp  -> the band was a binding constraint; proceed to the end task
    coverage ~ 52.9%        -> the neighbourhood and geometry features already supplied what the
                               extra layers add; skip to per-item window sizing instead

WATCH: 28 layers x 2 feature types is ~34 more features on 191 items. If coverage rises but the
end-task gain does not follow, that is overfitting, and the two numbers diverging is the signal.
"""
import json
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT = f"{D}/phase92_allayer_proposals.json"
NL = 28


def build(layers):
    rows = [json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl")]
    X, Y, G, DEP = [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        n = gh * gw
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        sel = A[layers]
        M = sel.reshape(len(layers), gh, gw)
        dep = A[list(range(16, 27))].mean(0).reshape(gh, gw)      # deployed map kept as a feature
        R = (np.argsort(np.argsort(-sel, axis=1), axis=1) / max(n - 1, 1)).reshape(len(layers), gh, gw)
        pad = np.pad(dep, 1, mode="edge")
        nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        X.append(np.concatenate([
            M.reshape(len(layers), -1).T, R.reshape(len(layers), -1).T,
            nb.reshape(-1, 1), dep.reshape(-1, 1),
            fx.reshape(-1, 1), fy.reshape(-1, 1),
            np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2).reshape(-1, 1),
            np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)).reshape(-1, 1),
            (xx == gw - 1).astype(float).reshape(-1, 1),
            (yy == gh - 1).astype(float).reshape(-1, 1),
            (xx == 0).astype(float).reshape(-1, 1)], axis=1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                           for i in range(n)]))
        G.append(np.full(n, gi)); DEP.append(dep.flatten())
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), DEP, rows


def evaluate(P, G, Y, DEP, rows, tag):
    hit, inc = [], []
    out = {}
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        m = G == gi
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        rmf = rm.flatten()
        y = Y[m]
        j = int(np.argmax(np.where(rmf, P[m], -1e9)))
        jd = int(np.argmax(np.where(rmf, DEP[gi], -1e9)))
        cell = lambda i: (((i % gw) + .5) / gw, ((i // gw) + .5) / gh)
        hit.append(float(y[j] >= P70.COV_HIT)); inc.append(float(y[jd] >= P70.COV_HIT))
        out[r["question_id_full"]] = {"head": cell(j), "argmax": cell(jd),
                                      "head_cov": float(y[j]), "argmax_cov": float(y[jd]),
                                      "gt_box_frac": r["gt_box_frac"], "category": r["category"]}
    print(f"  {tag:28} coverage {100*st.mean(hit):5.1f}%   (deployed argmax {100*st.mean(inc):.1f}%)")
    return st.mean(hit), out


def main():
    print("rebuilding the head on all 28 layers\n")
    Xa, Y, G, DEP, rows = build(list(range(NL)))
    print(f"  {len(rows)} items, {Xa.shape[1]} features/cell (was 65)\n")
    Pa = P70.oof(Xa, Y, G)
    print()
    ca, out = evaluate(Pa, G, Y, DEP, rows, "ALL 28 layers")

    Xb, Yb, Gb, DEPb, _ = build(list(range(16, 27)))
    Pb = P70.oof(Xb, Yb, Gb)
    print()
    cb, _ = evaluate(Pb, Gb, Yb, DEPb, rows, "deployed L16-26 (control)")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n  delta {100*(ca-cb):+.1f}pp   wrote {OUT}")
    print("\n" + "=" * 66)
    if ca > cb + 0.03:
        print("=> GATE PASSED. The band was a binding constraint; proceed to the end task.")
    else:
        print("=> GATE NOT PASSED. The neighbourhood/geometry features already supply what the")
        print("   extra layers add. Skip to per-item window sizing.")


if __name__ == "__main__":
    main()
