"""
Phase 71a: freeze the re-ranking head's OUT-OF-FOLD proposal for every V*Bench item.

Separated from the GPU run on purpose. The head is trained here, on CPU, from attention maps that
already exist; phase71b only reads the resulting (cx, cy) per item and crops there. That way the
GPU script contains no fitting and cannot leak.

LEAKAGE DISCIPLINE
------------------
GroupKFold(5) x 3 seeds, grouped by item. An item's proposal is produced ONLY by folds that never
saw that item. Training rows are negative-subsampled; prediction covers all cells. This is the same
`oof` used in Phase 70, imported rather than reimplemented so the two cannot drift.

WHAT THE HEAD IS SUPERVISED BY, STATED PLAINLY
----------------------------------------------
Per-cell coverage labels derived from V*Bench's ground-truth boxes. That is lightweight supervised
training on a public benchmark's own annotations -- not a new dataset, and not self-supervision.
Training and evaluating on V*Bench with out-of-fold folds is legitimate; CROSS-BENCHMARK transfer
would be a stronger claim and is not made here.
"""
import json
import numpy as np
import phase70_rerank_head as P70

OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"


def main():
    X, Y, G, DEP, rows, NGEO = P70.build()
    print(f"{len(rows)} items, {X.shape[1]} features/cell", flush=True)
    P = P70.oof(X, Y, G)
    print(flush=True)

    out = {}
    agree = 0
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        m = G == gi
        rm = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2:
            rm[1:-1, 1:-1] = True
        else:
            rm[:] = True
        rmf = rm.flatten()
        s = np.where(rmf, P[m], -1e9)
        d = np.where(rmf, DEP[gi], -1e9)
        j, jd = int(np.argmax(s)), int(np.argmax(d))
        agree += (j == jd)
        cell = lambda i: (((i % gw) + .5) / gw, ((i // gw) + .5) / gh)
        out[r["question_id_full"]] = {
            "head": cell(j), "argmax": cell(jd),
            "head_cov": float(Y[m][j]), "argmax_cov": float(Y[m][jd]),
            "gt_box_frac": r["gt_box_frac"], "category": r["category"]}
    json.dump(out, open(OUT, "w"), indent=1)
    hc = np.mean([v["head_cov"] >= P70.COV_HIT for v in out.values()])
    ac = np.mean([v["argmax_cov"] >= P70.COV_HIT for v in out.values()])
    print(f"head covers {100*hc:.1f}%   argmax covers {100*ac:.1f}%   "
          f"proposals identical on {100*agree/len(rows):.1f}% of items")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
