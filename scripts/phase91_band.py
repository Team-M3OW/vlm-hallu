"""
Phase 91: does the head's INPUT BAND matter? (the lever phase 88 exposed)

The head reads L16-26 -- a band fixed before we knew anything about depth structure. Phase 88 then
measured, per layer, that attention quality peaks at L17 (top-1 41.9%) / L19 (gt_pct 0.324) and is
much worse elsewhere INSIDE that same band: L18 22.0%, L22 21.5%, L26 24.1%. So the head is being fed
roughly as many poor layers as good ones, plus layers past L21 where quality is already declining.

ARMS (all out-of-fold, grouped by item, identical harness)
    deployed        L16-26, as shipped
    topk_oof        the k best layers by gt_pct, SELECTED ON TRAINING FOLDS ONLY, k in {3,5,8}
    peak_band       L16-21 (up to answer formation, not past it)
    all_layers      L0-L27 -- let the head choose
    qweighted       every layer, each map pre-scaled by its training-fold quality

⚠ Layer selection MUST be out-of-fold. Phase 74: an in-sample "best layer" read 41.9% and collapsed
to 36.1% out-of-fold, BELOW the block mean it appeared to beat.

READING, fixed before the run
    > 52.9% + 3pp  -> the band was a binding constraint; re-run the end task
    ~ 52.9%        -> the band is not the constraint; the head already extracts what depth offers,
                      and the remaining gap to the 88.5% ceiling is about something else
"""
import json
import numpy as np
import statistics as st
from sklearn.model_selection import GroupKFold
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
NL = 28


def build_all():
    rows = [json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl")]
    A, COV, RING, G = [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        n = gh * gw
        a = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        a = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        cov = np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                        for i in range(n)])
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        A.append(a); COV.append(cov); RING.append(m.flatten()); G.append(gi)
    return rows, A, COV, RING


def gtpct_on(idx, A, COV, RING):
    """mean gt_pct per layer over a subset of items -- used for TRAINING-FOLD layer selection"""
    out = []
    for L in range(NL):
        g = []
        for i in idx:
            rm, cv = RING[i], COV[i]
            tgt = int(np.argmax(cv))
            v = np.where(rm, A[i][L], -1e9)
            g.append((v > v[tgt]).sum() / max(rm.sum(), 1))
        out.append(st.mean(g))
    return out


def run(name, layer_fn, rows, A, COV, RING, weighted=False):
    """layer_fn(train_idx) -> list of layers (and optional weights) chosen on TRAINING folds"""
    n = len(rows)
    feats = None
    hits = np.zeros(n)
    G = np.arange(n)
    for tr, te in GroupKFold(5).split(G, G, G):
        sel = layer_fn(tr)
        w = None
        if weighted:
            q = gtpct_on(tr, A, COV, RING)
            w = np.array([max(0.0, 0.5 - q[L]) for L in sel])
            w = w / max(w.sum(), 1e-9)
        X, Y, GG = [], [], []
        for i in range(n):
            a = A[i][sel]
            if w is not None:
                a = a * w[:, None]
            X.append(a.T); Y.append(COV[i]); GG.append(np.full(len(COV[i]), i))
        X, Y, GG = np.vstack(X), np.concatenate(Y), np.concatenate(GG)
        trm = np.isin(GG, tr)
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(91)
        yt = Y[trm]
        pos = np.where(trm)[0][yt > 0]
        negp = np.where(trm)[0][yt <= 0]
        neg = rng.choice(negp, size=min(len(negp), 30 * len(tr)), replace=False)
        sub = np.concatenate([pos, neg])
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                          random_state=0).fit(X[sub], Y[sub])
        for i in te:
            s = m.predict(X[GG == i])
            j = int(np.argmax(np.where(RING[i], s, -1e9)))
            hits[i] = float(COV[i][j] >= P70.COV_HIT)
    return st.mean(hits)


def main():
    rows, A, COV, RING = build_all()
    print(f"{len(rows)} items, {NL} layers\n")
    DEP = list(range(16, 27))
    arms = {
        "deployed L16-26": (lambda tr: DEP, False),
        "peak band L16-21": (lambda tr: list(range(16, 22)), False),
        "all layers L0-27": (lambda tr: list(range(NL)), False),
        "top-3 by gt_pct (OOF)": (lambda tr: sorted(np.argsort(gtpct_on(tr, A, COV, RING))[:3]), False),
        "top-5 by gt_pct (OOF)": (lambda tr: sorted(np.argsort(gtpct_on(tr, A, COV, RING))[:5]), False),
        "top-8 by gt_pct (OOF)": (lambda tr: sorted(np.argsort(gtpct_on(tr, A, COV, RING))[:8]), False),
        "all layers, quality-weighted": (lambda tr: list(range(NL)), True),
    }
    base = None
    for nm, (fn, w) in arms.items():
        v = run(nm, fn, rows, A, COV, RING, weighted=w)
        if base is None:
            base = v
        print(f"  {nm:32}{100*v:6.1f}%   {100*(v-base):+6.1f}pp vs deployed")
    print("\n  (deployed band with the full 65-feature head = 52.9%; these use the per-layer maps only)")


if __name__ == "__main__":
    main()
