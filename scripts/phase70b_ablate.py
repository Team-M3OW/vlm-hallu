"""
Phase 70b: WHY does the re-ranking head work? Feature-group ablation.

Phase 70: deployed argmax 39.3% -> learned head 52.9% (+13.6pp), attention-only 50.8%, with the
geometry-only (3.7%) and shuffled (2.6%) controls both at chance. So the map carries information the
argmax discards. This file asks WHICH information, against two mechanistic predictions the paper
already makes:

  (P1) the block-16-26 MEAN destroys the depth profile.  If so, a head given ONLY the deployed map
       (its value + 3x3 neighbourhood + geometry) should gain little, and adding the 28-layer
       profile should be what moves it.
  (P2) the serialization sink corrupts the MAX.  If so, removing the last-column / last-row
       indicators should cost the head measurably -- it can no longer discount sink cells.

Both predictions are falsifiable here. Same harness, same folds, same controls as Phase 70.
"""
import json
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def main():
    X, Y, G, DEP, rows, NGEO = P70.build()
    nL = len([k for k in rows[0]["attn"] if k.startswith("L")])
    ring = []
    for r in rows:
        gh, gw = r["grid"]
        m = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        ring.append(m.flatten())
    # column layout from build(): [depth(nL)][rank(nL)][nb, dep][fx, fy, dcentre, dborder,
    #                              lastcol, lastrow, firstcol]
    depth = list(range(0, nL))
    rank = list(range(nL, 2 * nL))
    nb_dep = [2 * nL, 2 * nL + 1]
    geo_all = list(range(2 * nL + 2, X.shape[1]))
    geo_pos = geo_all[:4]          # fx, fy, dcentre, dborder
    sink = geo_all[4:]             # lastcol, lastrow, firstcol

    inc = []
    for gi in range(len(rows)):
        m = G == gi
        y, d = Y[m], np.where(ring[gi], DEP[gi], -1e9)
        inc.append(1.0 * (y[int(np.argmax(d))] >= P70.COV_HIT))
    inc = st.mean(inc)
    print(f"incumbent (deployed argmax)  {100*inc:.1f}%\n")

    arms = {
        "deployed map only (value + 3x3 + position)": nb_dep + geo_pos,
        "  + sink indicators":                        nb_dep + geo_all,
        "  + DEPTH PROFILE (28 layers)":              depth + nb_dep + geo_all,
        "  + within-layer ranks  [= full head]":      depth + rank + nb_dep + geo_all,
        "full head MINUS sink indicators":            depth + rank + nb_dep + geo_pos,
        "depth profile alone (no geometry at all)":   depth + rank + nb_dep,
    }
    res = {"incumbent": inc, "arms": {}}
    for nm, cols in arms.items():
        print(f"  {nm} ", end="", flush=True)
        p = P70.oof(X[:, cols], Y, G)
        h, c, _, _ = P70.evaluate(p, G, Y, DEP, ring)
        res["arms"][nm] = h
        print(f"\n  {nm:44} {100*h:5.1f}%  {100*(h-inc):+5.1f}pp")
    json.dump(res, open(f"{D}/phase70b_ablate.json", "w"), indent=1)
    a = res["arms"]
    print("\n" + "=" * 74)
    d_depth = a["  + DEPTH PROFILE (28 layers)"] - a["  + sink indicators"]
    d_sink = a["  + within-layer ranks  [= full head]"] - a["full head MINUS sink indicators"]
    print(f"P1  depth profile is worth {100*d_depth:+.1f}pp  "
          f"({'CONFIRMED' if d_depth > 0.04 else 'NOT SUPPORTED'})")
    print(f"P2  sink indicators are worth {100*d_sink:+.1f}pp  "
          f"({'CONFIRMED' if d_sink > 0.02 else 'NOT SUPPORTED'})")


if __name__ == "__main__":
    main()
