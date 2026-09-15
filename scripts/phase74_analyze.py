"""
Phase 74 analyzer: does the READ-OUT DEFECT replicate on other architectures?

SS14F established on Qwen3-VL-2B: the deployed block-mean (39.3%) is barely beaten by the best
single layer (41.9%), a learned SIGNED combination reaches 45.5% linear / 52.9% full, the weights
take both signs, and the FINAL layer is anti-correlated with the target (gt_pct 0.529 > 0.500
chance). The claim is that a mean can only add, so it cancels layers that point at distractors.

PRE-REGISTERED REPLICATION CRITERIA (a claim about VLM read-outs, not about one checkpoint)
    R1  best single layer is only marginally above the block mean   -> not "the wrong layers"
    R2  a learned LINEAR signed combination beats BOTH              -> contrast is the mechanism
    R3  the learned weights take both signs
    R4  the FINAL layer's gt_pct is at or worse than 0.500 chance   -> the sharpest prediction

R4 is the one that would be very hard to get by accident, so it carries the most weight. R1-R3
failing would mean SS2 must be rewritten as "on this checkpoint".

The block is rescaled to each model's depth (Qwen3-VL used L16-26 of 28 = 0.57-0.93 of the stack).
"""
import json
import sys
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def analyse(path, tag):
    rows = [json.loads(l) for l in open(path)]
    if len(rows) < 50:
        print(f"{tag}: only {len(rows)} items -- skipping"); return None
    nL = len([k for k in rows[0]["attn"] if k.startswith("L")])
    BLOCK = list(range(int(0.57 * nL), int(0.93 * nL) + 1))
    print(f"\n{'='*72}\n{tag}  --  {len(rows)} items, {nL} layers, block L{BLOCK[0]}-{BLOCK[-1]}\n{'='*72}")

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

    def gt_pct(L):
        g = []
        for r in rows:
            q = r["question_id_full"]
            rm, c = RING[q], COV[q]
            tgt = int(np.argmax(c))
            v = np.where(rm, C[q][L], -1e9)
            g.append((v > v[tgt]).sum() / max(rm.sum(), 1))
        return st.mean(g)

    bm = top1(lambda A: A[BLOCK].mean(0))
    solo = [(top1(lambda A, L=L: A[L]), L) for L in range(nL)]
    bv, bL = max(solo)
    allm = top1(lambda A: A.mean(0))

    X, Y, G = [], [], []
    for gi, r in enumerate(rows):
        q = r["question_id_full"]
        X.append(C[q].T); Y.append(COV[q]); G.append(np.full(len(COV[q]), gi))
    X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    P = np.zeros(len(Y)); W = np.zeros(nL)
    for tr, te in GroupKFold(5).split(X, Y, G):
        m = Ridge(alpha=1.0).fit(X[tr], Y[tr]); P[te] = m.predict(X[te]); W += m.coef_ / 5
    off = {r["question_id_full"]: i for i, r in enumerate(rows)}
    lin = st.mean([float(COV[r["question_id_full"]][int(np.argmax(np.where(
        RING[r["question_id_full"]], P[G == off[r["question_id_full"]]], -1e9)))] >= P70.COV_HIT)
        for r in rows])

    print(f"  deployed block mean            {100*bm:5.1f}%")
    print(f"  best single layer (L{bL})        {100*bv:5.1f}%")
    print(f"  LEARNED linear signed (OOF)    {100*lin:5.1f}%")
    print(f"  mean of ALL {nL} layers            {100*allm:5.1f}%")
    fin = gt_pct(nL - 1)
    print(f"  FINAL layer gt_pct             {fin:.3f}   (0.500 = chance)")
    print(f"  weights: {int((W>0).sum())} pos / {int((W<0).sum())} neg, last {W[-1]:+.3f}")

    r1 = bv - bm < 0.06
    r2 = lin > max(bm, bv)
    r3 = (W > 0).sum() > 0 and (W < 0).sum() > 0
    r4 = fin >= 0.48
    for nm, ok, why in [("R1 mean is not just wrong layers", r1, f"best-single − block = {100*(bv-bm):+.1f}pp"),
                        ("R2 signed combination beats both", r2, f"linear {100*lin:.1f}% vs {100*max(bm,bv):.1f}%"),
                        ("R3 weights take both signs", r3, f"{int((W>0).sum())}/{int((W<0).sum())}"),
                        ("R4 final layer at/below chance", r4, f"gt_pct {fin:.3f}")]:
        print(f"    {'PASS' if ok else 'FAIL'}  {nm:36} {why}")
    return {"tag": tag, "block_mean": bm, "best_single": bv, "linear": lin,
            "final_gt_pct": fin, "R": [bool(r1), bool(r2), bool(r3), bool(r4)], "weights": W.tolist()}


if __name__ == "__main__":
    out = []
    for path, tag in [(f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", "Qwen2-VL-7B"),
                      (f"{D}/phase74_llava_onevision_qwen2_7b_ov_hf.jsonl", "LLaVA-OneVision-7B")]:
        try:
            r = analyse(path, tag)
            if r: out.append(r)
        except FileNotFoundError:
            print(f"\n{tag}: not extracted yet")
    if out:
        json.dump(out, open(f"{D}/phase74_replication.json", "w"), indent=1)
        print(f"\n{'='*72}")
        print("REPLICATION SUMMARY (Qwen3-VL-2B: 39.3 / 41.9 / 45.5, final gt_pct 0.529)")
        for r in out:
            print(f"  {r['tag']:22} {100*r['block_mean']:5.1f} / {100*r['best_single']:5.1f} / "
                  f"{100*r['linear']:5.1f}   final gt_pct {r['final_gt_pct']:.3f}   "
                  f"{sum(r['R'])}/4 criteria")
