"""
Phase 76 analyzer: does the VISION TOWER localise, and does the SS14F depth structure appear there
too? Same code path as phase73 so the LM and vision results cannot drift apart.

GATE, fixed before the data existed
    vision top-1 coverage clearly above chance AND a learned combination beats the best single
    vision layer  -> single-pass foveation is buildable; the signal exists before the LM runs.
    at or near chance                                     -> the direction is dead at step one.

Chance here is the rate a randomly chosen ring-masked cell covers the box (~3%, SS14C).
"""
import json
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def main():
    rows = [json.loads(l) for l in open(f"{D}/phase76_vision_tower.jsonl")]
    nL = len([k for k in rows[0]["attn"] if k.startswith("L")])
    print(f"{len(rows)} items, {nL} vision layers\n")
    C, COV, RING = {}, {}, {}
    for r in rows:
        gh, gw = r["grid"]
        A = np.stack([np.asarray(r["attn"][f"L{i}"], dtype=float) for i in range(nL)])
        C[r["question_id_full"]] = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        COV[r["question_id_full"]] = np.array(
            [P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
             for i in range(gh * gw)])
        m = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING[r["question_id_full"]] = m.flatten()

    def top1(fn):
        h = []
        for r in rows:
            q = r["question_id_full"]
            s = np.where(RING[q], fn(C[q]), -1e9)
            h.append(float(COV[q][int(np.argmax(s))] >= P70.COV_HIT))
        return st.mean(h)

    rng = np.random.default_rng(76)
    # 200 draws per item: ONE draw per item gave 0.0% and made the gate threshold meaningless
    chance = st.mean([float(COV[r["question_id_full"]][j] >= P70.COV_HIT)
                      for r in rows
                      for j in rng.choice(np.where(RING[r["question_id_full"]])[0], 200)])
    print(f"  random ring-masked cell (chance)     {100*chance:5.1f}%")
    solo = [(top1(lambda A, L=L: A[L]), L) for L in range(nL)]
    for v, L in sorted(solo, reverse=True)[:5]:
        print(f"  vision layer L{L:<2d}                     {100*v:5.1f}%")
    bv, bL = max(solo)
    print(f"  >> best vision layer: L{bL} at {100*bv:.1f}%")
    print(f"  mean of ALL {nL} vision layers            {100*top1(lambda A: A.mean(0)):5.1f}%")

    X, Y, G = [], [], []
    for gi, r in enumerate(rows):
        q = r["question_id_full"]
        X.append(C[q].T); Y.append(COV[q]); G.append(np.full(len(COV[q]), gi))
    X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    P = np.zeros(len(Y)); W = np.zeros(nL)
    for tr, te in GroupKFold(5).split(X, Y, G):
        m = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        P[te] = m.predict(X[te]); W += m.coef_ / 5
    off = {r["question_id_full"]: i for i, r in enumerate(rows)}
    h = []
    for r in rows:
        q = r["question_id_full"]
        s = np.where(RING[q], P[G == off[q]], -1e9)
        h.append(float(COV[q][int(np.argmax(s))] >= P70.COV_HIT))
    lin = st.mean(h)
    print(f"  LEARNED signed combination (OOF)     {100*lin:5.1f}%")
    print(f"\n  for reference, the LM read-out (SS14C): argmax 39.3%, head 52.9%")
    print(f"  weights: {int((W>0).sum())} positive / {int((W<0).sum())} negative; "
          f"last layer weight {W[-1]:+.3f}")

    print("\n" + "=" * 70)
    if max(lin, bv) < 3 * max(chance, 0.01):
        print("GATE FAILED: vision-tower attention is at or below the random-cell rate. The")
        print("  'where to look' signal does NOT exist before the LM runs -- it is CONSTRUCTED by")
        print("  the language model from the question. Single-pass foveation steered by vision")
        print("  saliency is dead at step one.")
    elif lin > max(bv, 2.5 * chance) and lin > 0.20:
        print("GATE PASSED: the vision tower localises BEFORE the LM runs, and depth structure")
        print("  appears there too. Single-pass foveation is buildable.")
    elif bv > 2.5 * chance:
        print("PARTIAL: vision attention is above chance but a learned combination does not beat")
        print("  the best single layer. Foveation is possible; the depth story is LM-specific.")
    else:
        print("GATE FAILED: vision-tower attention is at/near chance. Single-pass foveation")
        print("  cannot be steered by it. The direction is dead at step one -- report and stop.")
    json.dump({"chance": chance, "best_layer": int(bL), "best_layer_top1": bv,
               "linear_oof": lin, "weights": W.tolist()},
              open(f"{D}/phase76_vision_tower_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
