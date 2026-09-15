"""
Phase 66: THE FOUNDING QUESTION, with the logit lens. The model localises -- why is it still wrong?

THE DISSOCIATION
----------------
The ring-masked attention map ranks the GT-centre cell at gt_pct 0.034 -- the top 3.4% of cells,
against an exact chance of 0.500 -- yet uniform@300 answers only 56.5% correctly. The model looks in
roughly the right place and still gets it wrong. That is the observation this project started from,
and until now we could only characterise it from the outside.

TWO EXPLANATIONS, WHICH THE LENS CAN SEPARATE
---------------------------------------------
  PROPAGATION FAILURE  the correct answer IS decodable at some layer on these items, and is lost
                       before the read-out. Then the deficit is internal routing and a decoding
                       intervention could in principle recover it.
  EMPTY LOCALISATION   the correct answer is decodable at NO layer. The attention points at the
                       right place, but the tokens there do not encode what the question asks --
                       because at B0=300 the target occupies less than one merged token. Then
                       "localises well" never implied "has the answer", and no read-out can help.

DECISIVE CONTRAST: take items that are LOCALISED (the W=0.15 window at the attention peak covers the
GT) but ANSWERED WRONG, and ask whether the correct option is ever the argmax at any layer. Then ask
whether the ORACLE CROP -- same question, same budget, target actually resolved -- fixes them. If
the oracle fixes what no layer could decode, the missing ingredient is resolution, not routing.

Intermediate layers come from the Phase 60 lens (correct); the FINAL layer comes from Phase 64's
model logits, because the lens double-normalised it (§12D).
"""
import json
import random
import statistics as st

P60 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"
P64 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase64_component_ablation.jsonl"
P58 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase58_multicrop.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    a = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P60))}
    b = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P64))}
    c = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P58))}
    k = sorted(set(a) & set(b) & set(c))
    nL = len(a[k[0]]["uniform"])
    am = lambda p: max(range(4), key=lambda i: p[i])

    def layers(q, cond):
        """correct per-layer curve: lens for intermediate, model logits for the final."""
        return a[q][cond][:nL - 1] + [b[q]["probs"][cond]["baseline"]]

    print(f"n = {len(k)} items;  localisation = coverage of the GT box by the W=0.15 window")
    print("at the attention peak (0 = the window misses the target entirely).\n")

    cells = {}
    for q in k:
        loc = c[q]["cand_cov"][0] > 0.001
        cor = am(layers(q, "uniform")[-1]) == a[q]["label"]
        cells.setdefault((loc, cor), []).append(q)

    print("=== THE DISSOCIATION, as a 2x2 ===")
    print(f"  {'':<22}{'answered RIGHT':>16}{'answered WRONG':>16}")
    for loc, nm in ((True, "LOCALISED"), (False, "window missed")):
        r_ = len(cells.get((loc, True), []))
        w_ = len(cells.get((loc, False), []))
        print(f"  {nm:<22}{r_:>16}{w_:>16}")
    LW = cells.get((True, False), [])
    print(f"\n  the cell of interest -- LOCALISED but WRONG -- has n = {len(LW)}")

    print("\n=== IS THE CORRECT ANSWER DECODABLE AT *ANY* LAYER? ===")
    print(f"  {'cell':<26}{'n':>5}{'ever argmax at some layer':>28}{'chance if random':>19}")
    for (loc, cor), qs in sorted(cells.items(), key=lambda x: (-x[0][0], x[0][1])):
        if len(qs) < 8:
            continue
        ever = st.mean(1.0 * any(am(layers(q, "uniform")[li]) == a[q]["label"]
                                 for li in range(nL)) for q in qs)
        nm = ("LOCALISED" if loc else "missed") + (" & right" if cor else " & WRONG")
        print(f"  {nm:<26}{len(qs):>5}{100*ever:>27.1f}%{'~68%':>19}")
    print("  (a random-walk baseline: with 28 layers and 4 options, the correct option is argmax")
    print("   at some layer ~1-(3/4)^28 -> ~100% if layers were independent; they are not, so the")
    print("   honest reference is the RIGHT-answered cells in the same table.)")

    print("\n=== FOR 'LOCALISED BUT WRONG': WHERE DOES THE CORRECT ANSWER LIVE? ===")
    if len(LW) >= 8:
        print(f"  {'layer':<8}{'P(correct is argmax)':>22}{'P(correct in top-2)':>22}")
        for li in list(range(16, nL, 2)) + [nL - 1]:
            p1 = st.mean(1.0 * (am(layers(q, "uniform")[li]) == a[q]["label"]) for q in LW)
            p2 = st.mean(1.0 * (sorted(range(4), key=lambda i: -layers(q, "uniform")[li][i])[:2]
                                .count(a[q]["label"]) > 0) for q in LW)
            tag = "  <- final" if li == nL - 1 else ""
            print(f"  L{li:<7}{100*p1:>21.1f}%{100*p2:>21.1f}%{tag}")

        print("\n=== DOES THE ORACLE CROP FIX THEM? (same question, same budget, target resolved) ===")
        orc = [1.0 * (am(layers(q, "oracle")[-1]) == a[q]["label"]) for q in LW]
        lo, hi = boot(orc)
        print(f"  oracle-crop accuracy on the LOCALISED-BUT-WRONG cell: "
              f"{100*st.mean(orc):.1f}%  CI[{100*lo:.1f},{100*hi:.1f}]  (n={len(LW)})")
        ever_u = st.mean(1.0 * any(am(layers(q, "uniform")[li]) == a[q]["label"]
                                   for li in range(nL)) for q in LW)
        print(f"  ... while under uniform the correct option is argmax at SOME layer on only "
              f"{100*ever_u:.1f}%")
        tgt = st.median([a[q]["tokens_on_target"] for q in LW])
        print(f"  median tokens on target for this cell: {tgt:.2f}  "
              f"({'SUB-TOKEN' if tgt < 1 else 'resolvable'})")

    print("\n" + "=" * 74)
    if len(LW) >= 8:
        orc_acc = st.mean(1.0 * (am(layers(q, "oracle")[-1]) == a[q]["label"]) for q in LW)
        ever_u = st.mean(1.0 * any(am(layers(q, "uniform")[li]) == a[q]["label"]
                                   for li in range(nL)) for q in LW)
        if orc_acc > ever_u + 0.15:
            print("  => EMPTY LOCALISATION. On items the model localises but answers wrong, the")
            print("     correct answer is decodable at no layer -- yet resolving the same region")
            print("     fixes them. Attention points at the right place; the tokens there do not")
            print("     carry what the question asks. 'Localises well' never implied 'has the")
            print("     answer', and no read-out intervention can close that gap.")
        else:
            print("  => PROPAGATION FAILURE is plausible: the answer is decodable internally on a")
            print("     comparable fraction of these items, so routing -- not resolution -- may be")
            print("     the deficit. A decoding intervention is worth testing.")


if __name__ == "__main__":
    main()
