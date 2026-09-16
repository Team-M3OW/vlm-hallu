"""
Phase 88: are "where localisation lives" and "when the answer forms" the SAME depth?

THE OBJECTION THIS ANSWERS
--------------------------
"The central 'wrong depth' contribution is partly two entangled findings. The paper conflates
(a) where localization signal lives in a per-layer attention map and (b) when an answer becomes
linearly decodable from the residual stream. These correlate empirically but are conceptually
distinct mechanisms (attention quality vs. computation/answer-formation timing)."

This is correct and it is testable. We have both quantities per layer on the same items:

    (a) ATTENTION QUALITY at layer L -- gt_pct of that layer's map alone (phase30c)
    (b) ANSWER DECODABILITY at layer L -- logit-lens accuracy at that layer (phase79)

If they peak at the same depth and track each other, one "wrong depth" story is defensible. If they
are dissociated -- e.g. attention is already good well before the answer forms -- then they are two
findings and must be reported as two.

PRE-REGISTERED READING
    peaks within ~2 layers AND high rank correlation  -> one mechanism, current framing stands
    peaks separated, or low/negative correlation      -> TWO mechanisms. The paper must separate
                                                         "read attention late" (a claim about maps)
                                                         from "the answer forms late" (a claim about
                                                         the residual stream), and the pruning result
                                                         rests only on (a).

A dissociation would not weaken either result -- it would make the pruning claim INDEPENDENT of the
answer-formation claim, so the pruning finding no longer relies on the lens at all.
"""
import json
import numpy as np
import statistics as st
from scipy.stats import spearmanr
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
NL = 28


def main():
    maps = {r["question_id_full"]: r for r in
            (json.loads(l) for l in open(f"{D}/phase30c_attn_maps_all.jsonl"))}
    lens = [json.loads(l) for l in open(f"{D}/phase79_lens_mechanism.jsonl")]
    lq = {r["question_id_full"]: r for r in lens}
    shared = [q for q in maps if q in lq]
    print(f"n = {len(shared)} items with both measurements\n")

    # (a) attention quality per layer: gt_pct of that layer alone, and top-1 coverage
    gtp, cov1 = [], []
    for L in range(NL):
        g, c = [], []
        for q in shared:
            r = maps[q]
            gh, gw = r["grid"]
            A = np.asarray(r["attn"][f"L{L}"], dtype=float)
            A = A / max(A.sum(), 1e-12)
            rm = np.zeros((gh, gw), bool)
            if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
            else: rm[:] = True
            rm = rm.flatten()
            yy, xx = np.mgrid[0:gh, 0:gw]
            fy, fx = (yy+.5)/gh, (xx+.5)/gw
            cv = np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                           for i in range(gh*gw)])
            tgt = int(np.argmax(cv))
            v = np.where(rm, A, -1e9)
            g.append((v > v[tgt]).sum()/max(rm.sum(), 1))
            c.append(float(cv[int(np.argmax(v))] >= P70.COV_HIT))
        gtp.append(st.mean(g)); cov1.append(st.mean(c))

    # (b) answer decodability per layer, uncropped arm
    hit = lambda r, a, i: 1.0*(max(range(4), key=lambda j: r["traj"][a][i][j]) == r["label"])
    dec_u = [st.mean([hit(lq[q], "uniform@300", i) for q in shared]) for i in range(NL)]
    dec_o = [st.mean([hit(lq[q], "oracle@0.15", i) for q in shared]) for i in range(NL)]

    print(f"  {'L':>3}{'attn gt_pct':>13}{'attn top-1':>12}{'decode(unif)':>14}{'decode(oracle)':>16}")
    for L in range(NL):
        mark = "  <-- deployed block" if L == 16 else ("  <-- answer step" if L == 21 else "")
        if L % 2 == 0 or L in (17, 21):
            print(f"  {L:3d}{gtp[L]:12.3f}{100*cov1[L]:11.1f}%{100*dec_u[L]:13.1f}%"
                  f"{100*dec_o[L]:15.1f}%{mark}")

    best_attn = int(np.argmin(gtp))
    best_cov = int(np.argmax(cov1))
    # answer-formation layer = biggest single-layer jump in the ORACLE arm (where evidence exists)
    jumps = [(dec_o[i]-dec_o[i-1], i) for i in range(1, NL)]
    form = max(jumps)[1]
    print("\n" + "="*70)
    print(f"  attention quality peaks at   L{best_attn} (gt_pct {gtp[best_attn]:.3f}) / "
          f"L{best_cov} (top-1 {100*cov1[best_cov]:.1f}%)")
    print(f"  answer forms at              L{form} (largest oracle jump {100*max(jumps)[0]:+.1f}pp)")
    print(f"  separation                   {abs(form-best_cov)} layers")
    rho_c = spearmanr(cov1, dec_o).statistic
    rho_g = spearmanr([-g for g in gtp], dec_o).statistic
    print(f"\n  rank corr across layers: attn-top1 vs decodability  rho = {rho_c:+.3f}")
    print(f"                           attn-gt_pct vs decodability rho = {rho_g:+.3f}")
    # is attention already good BEFORE the answer forms?
    pre = st.mean(cov1[max(0, form-6):form])
    post = st.mean(cov1[form:])
    print(f"\n  attention top-1 in the 6 layers BEFORE the answer forms: {100*pre:.1f}%")
    print(f"  attention top-1 from the answer layer onward:            {100*post:.1f}%")
    print("\n" + "="*70)
    if abs(form-best_cov) <= 2 and rho_c > 0.5:
        print("=> ONE MECHANISM. Attention quality and answer formation peak together; the")
        print("   current single 'wrong depth' framing is defensible.")
    else:
        print("=> TWO MECHANISMS, DISSOCIATED. Attention quality and answer formation are not the")
        print("   same depth. The paper must separate 'read attention late' (a claim about maps,")
        print("   which the pruning result rests on) from 'the answer forms late' (a claim about")
        print("   the residual stream). The pruning finding then does NOT depend on the lens.")
    json.dump({"gt_pct": gtp, "cov1": cov1, "decode_uniform": dec_u, "decode_oracle": dec_o,
               "best_attn_layer": best_attn, "best_cov_layer": best_cov, "form_layer": form,
               "rho_cov_decode": float(rho_c)},
              open(f"{D}/phase88_two_mechanisms.json", "w"), indent=1)


if __name__ == "__main__":
    main()
