"""
Phase 73: (1) distil the head into a LINEAR layer-reweighting, and (4) say what the layers disagree
about. Both from attention maps already on disk. No GPU.

WHY
---
SS14C's head is a 65-feature gradient-boosted tree. It works (39.3% -> 52.9% top-1 coverage) and
Phase 73's screen showed it genuinely COMBINES depths rather than picking a better one: the best
single layer is L17 at 41.9% and the deployed block-16-26 mean is 39.3%.

But a boosted tree over engineered features is not a "plug-and-play architecture". If a single
LEARNED WEIGHTED SUM of the 28 layer maps recovers most of the gain, the method collapses to

        replace   mean(L16..L26)   with   sum_i w_i * L_i

-- 28 numbers, a one-line change in any VLM's attention read-out. That is a different class of
contribution, so it is worth knowing before writing the method section.

PART 4 asks what the weights are buying. For each layer we measure how well it ranks the GT cell
(gt_pct: 0 = GT ranked first, 0.5 = chance), how much mass it puts on the serialization sink, and
how layers correlate with one another. If early layers track low-level salience and late layers
track queried semantics, the weights should show it.

All predictions OUT-OF-FOLD, grouped by item. A linear arm that is fitted and scored on the same
items would be meaningless.
"""
import json
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
NL = 28


def load():
    return [json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl")]


def maps(r):
    A = np.stack([np.asarray(r["attn"][f"L{i}"], dtype=float) for i in range(NL)])
    return A / np.maximum(A.sum(1, keepdims=True), 1e-12)


def ring(gh, gw):
    m = np.zeros((gh, gw), dtype=bool)
    if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
    else: m[:] = True
    return m.flatten()


def cov_vec(r, gh, gw):
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = (yy + .5) / gh, (xx + .5) / gw
    return np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                     for i in range(gh * gw)])


def top1(rows, score_fn):
    h = []
    for r in rows:
        gh, gw = r["grid"]
        s = np.where(ring(gh, gw), score_fn(r), -1e9)
        h.append(float(cov_vec(r, gh, gw)[int(np.argmax(s))] >= P70.COV_HIT))
    return st.mean(h)


def main():
    rows = load()
    cache = {r["question_id_full"]: (maps(r), *r["grid"]) for r in rows}
    print(f"{len(rows)} items, {NL} layers\n")

    # ---------------- PART 1: linear distillation ----------------
    print("=" * 70)
    print("PART 1 -- can a LINEAR reweighting of the 28 layer maps replace the head?")
    print("=" * 70)
    X, Y, G = [], [], []
    for gi, r in enumerate(rows):
        A, gh, gw = cache[r["question_id_full"]]
        X.append(A.T)                       # cells x 28
        Y.append(cov_vec(r, gh, gw))
        G.append(np.full(gh * gw, gi))
    X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    W = np.zeros(NL)
    P = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(X, Y, G):
        m = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        P[te] = m.predict(X[te])
        W += m.coef_ / 5
    off = {r["question_id_full"]: i for i, r in enumerate(rows)}
    lin = top1(rows, lambda r: P[G == off[r["question_id_full"]]])

    blockmean = top1(rows, lambda r: cache[r["question_id_full"]][0][P70.BLOCK].mean(0))
    bestL = max(range(NL), key=lambda L: top1(rows, lambda r, L=L: cache[r["question_id_full"]][0][L]))
    bestv = top1(rows, lambda r: cache[r["question_id_full"]][0][bestL])
    print(f"  deployed block-16-26 mean        {100*blockmean:5.1f}%")
    print(f"  best single layer (L{bestL})           {100*bestv:5.1f}%")
    print(f"  LINEAR learned reweighting (OOF) {100*lin:5.1f}%")
    print(f"  full head, 65 features (SS14C)    52.9%")
    gain_head, gain_lin = 52.9 - 100*blockmean, 100*lin - 100*blockmean
    print(f"\n  linear recovers {100*gain_lin/max(gain_head,1e-9):.0f}% of the head's gain "
          f"({gain_lin:+.1f}pp of {gain_head:+.1f}pp)")
    print(f"\n  learned weights (normalised, + = evidence, - = distractor):")
    Wn = W / np.abs(W).max()
    for i in range(0, NL, 1):
        if abs(Wn[i]) > 0.15 or i in (0, 27):
            bar = "#" * int(abs(Wn[i]) * 24)
            print(f"    L{i:<2d} {Wn[i]:+6.2f} {'-' if Wn[i]<0 else '+'}{bar}")

    # ---------------- PART 4: what do the layers encode? ----------------
    print("\n" + "=" * 70)
    print("PART 4 -- what are the layers disagreeing ABOUT?")
    print("=" * 70)
    print(f"  {'layer':>6}{'gt_pct':>9}{'top1':>8}{'sink mass':>11}   (gt_pct 0.5 = chance)")
    gtp, sink = [], []
    for L in range(NL):
        g_, s_ = [], []
        for r in rows:
            A, gh, gw = cache[r["question_id_full"]]
            rm = ring(gh, gw)
            c = cov_vec(r, gh, gw)
            tgt = int(np.argmax(c))
            v = np.where(rm, A[L], -1e9)
            g_.append((v > v[tgt]).sum() / max(rm.sum(), 1))
            M = A[L].reshape(gh, gw)
            s_.append(M[:, -1].sum() / max(M.sum(), 1e-12) / (1.0 / gw))
        gtp.append(st.mean(g_)); sink.append(st.mean(s_))
    for L in range(NL):
        t = top1(rows, lambda r, L=L: cache[r["question_id_full"]][0][L])
        mark = "  <-- best" if L == bestL else ""
        if L % 2 == 0 or L == bestL:
            print(f"    L{L:<3d}{gtp[L]:9.3f}{100*t:7.1f}%{sink[L]:10.2f}x{mark}")
    print(f"\n  sink enrichment: early layers {st.mean(sink[:8]):.2f}x, "
          f"mid {st.mean(sink[8:18]):.2f}x, late {st.mean(sink[18:]):.2f}x")
    print(f"  gt ranking:      early {st.mean(gtp[:8]):.3f}, "
          f"mid {st.mean(gtp[8:18]):.3f}, late {st.mean(gtp[18:]):.3f}  (0.5 = chance)")

    C = np.corrcoef(np.vstack([np.concatenate([cache[r["question_id_full"]][0][L]
                                               for r in rows]) for L in range(NL)]))
    print(f"\n  layer-map correlation: adjacent {st.mean([C[i,i+1] for i in range(NL-1)]):.3f}, "
          f"early-vs-late {float(np.mean(C[:8,18:])):.3f}, within-block(16-26) "
          f"{st.mean([C[i,j] for i in P70.BLOCK for j in P70.BLOCK if i<j]):.3f}")
    json.dump({"linear_top1": lin, "blockmean": blockmean, "best_layer": int(bestL),
               "best_layer_top1": bestv, "weights": W.tolist(), "gt_pct": gtp, "sink": sink},
              open(f"{D}/phase73_layer_structure.json", "w"), indent=1)
    print(f"\nwrote {D}/phase73_layer_structure.json")


if __name__ == "__main__":
    main()
