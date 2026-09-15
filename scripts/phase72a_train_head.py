"""
Phase 72a: train the re-ranking head on ALL of V*Bench and freeze it for transfer.

No folds here, and that is correct: the test set is a DIFFERENT BENCHMARK (HR-Bench 4k), so there
is no item overlap to leak through. Phase 70/71 used GroupKFold because they tested on V*Bench
itself. Using all 191 items here is the honest choice -- it is the strongest head we can build from
the training benchmark, which is exactly what a transfer claim should carry.

WHAT MUST NOT HAPPEN: no HR-Bench data of any kind enters this file, and phase72b does no fitting.
HR-Bench ships no boxes, so it could not be fitted on coverage even if we wanted to.

FEATURE PORTABILITY, checked rather than assumed. Every feature is grid-agnostic: per-layer
attention normalised to sum 1, within-layer rank divided by (n-1), 3x3 neighbourhood mean, and
position expressed as fractions. Both benchmarks are fitted to B0=300, so cell counts are
comparable (~295). The saved spec records the training grid stats so phase72b can assert it is not
extrapolating.
"""
import json
import numpy as np
import joblib
import phase70_rerank_head as P70

OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase72a_head.joblib"


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, Y, G, DEP, rows, NGEO = P70.build()
    print(f"training on ALL {len(rows)} V*Bench items, {X.shape[0]} cells, "
          f"{X.shape[1]} features", flush=True)

    rng = np.random.default_rng(72)
    pos = np.where(Y > 0)[0]
    negpool = np.where(Y <= 0)[0]
    neg = rng.choice(negpool, size=min(len(negpool), 30 * len(rows)), replace=False)
    sub = np.concatenate([pos, neg])
    m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                      random_state=0).fit(X[sub], Y[sub])

    ns = [r["n_img_tokens"] for r in rows]
    spec = {"n_features": int(X.shape[1]), "n_layers": 28, "block": P70.BLOCK, "W": P70.W,
            "train_cells_min": int(min(ns)), "train_cells_max": int(max(ns)),
            "train_items": len(rows), "train_rows_used": int(len(sub))}
    joblib.dump({"model": m, "spec": spec}, OUT)

    # sanity: in-sample top-1 coverage must exceed the incumbent (it is not a transfer number)
    P = m.predict(X)
    hit, inc = [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        sel = G == gi
        rm = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        rmf = rm.flatten()
        y = Y[sel]
        hit.append(y[int(np.argmax(np.where(rmf, P[sel], -1e9)))] >= P70.COV_HIT)
        inc.append(y[int(np.argmax(np.where(rmf, DEP[gi], -1e9)))] >= P70.COV_HIT)
    print(f"in-sample coverage {100*np.mean(hit):.1f}% vs incumbent {100*np.mean(inc):.1f}% "
          f"(in-sample, NOT the transfer claim)")
    print(f"cells/item in training: {spec['train_cells_min']}-{spec['train_cells_max']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
