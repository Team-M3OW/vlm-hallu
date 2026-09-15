"""Phase 16 analysis, applying the verdict rules fixed in the phase16 docstring BEFORE results:

  SUCCESS  steer_obj beats rand_obj on DISCRIMINATION (recovery on denials - FP on true absences),
           CI excluding zero.
  NULL     steer_obj ~= rand_obj at every alpha. Report as such; do not tune alpha until they split.
  BIAS     both arms raise P(yes) on denials AND on true negatives -> a bias shift, not perception.
"""
import json
import random
import statistics as st

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
ARMS = ["steer_obj", "rand_obj", "steer_all_img", "steer_last", "rand_last"]
ALPHAS = [0.25, 0.5, 1.0, 2.0]


def dd(pa, na, pb, nb_, n=4000, seed=13):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pi = [rng.randrange(len(pa)) for _ in range(len(pa))]
        ni = [rng.randrange(len(na)) for _ in range(len(na))]
        da = sum(pa[i] for i in pi)/len(pa) - sum(na[i] for i in ni)/len(na)
        db = sum(pb[i] for i in pi)/len(pb) - sum(nb_[i] for i in ni)/len(nb_)
        out.append(da - db)
    out.sort()
    return out[int(.025*n)], out[int(.975*n)]


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/phase16_steering_results.jsonl")]
    seen, u = set(), []
    for r in rows:
        if r["uid"] not in seen:
            seen.add(r["uid"]); u.append(r)
    rows = u
    P = [r for r in rows if r["group"] == "denial"]
    N = [r for r in rows if r["group"] == "negative"]
    print(f"RePOPE-clean confident denials n={len(P)}   clean negatives n={len(N)}")
    print(f"baseline P(yes): denials median {st.median([r['baseline'] for r in P]):.2e}  "
          f"recovery {100*sum(1 for r in P if r['baseline']>0.5)/len(P):.1f}%   |   "
          f"negatives FP {100*sum(1 for r in N if r['baseline']>0.5)/len(N):.1f}%")

    print("\n" + "="*88)
    print("RECOVERY on confident denials / FP on true absences / DISCRIMINATION")
    print("="*88)
    print(f"  {'arm':<16}" + "".join(f"{'a='+str(a):>17}" for a in ALPHAS))
    cache = {}
    for arm in ARMS:
        cells = []
        for a in ALPHAS:
            k = f"{arm}@{a}"
            hp = [1 if r["arms"][k] > 0.5 else 0 for r in P]
            hn = [1 if r["arms"][k] > 0.5 else 0 for r in N]
            cache[k] = (hp, hn)
            rec, fp = sum(hp)/len(hp), sum(hn)/len(hn)
            cells.append(f"{100*rec:5.1f}/{100*fp:4.1f}/{100*(rec-fp):+5.1f}")
        print(f"  {arm:<16}" + "".join(f"{c:>17}" for c in cells))
    print("  (cells are recovery% / false-positive% / discrimination-pp)")

    print("\n" + "="*88)
    print("THE DECIDING CONTRAST: steer_obj vs NORM-MATCHED RANDOM direction, same positions")
    print("="*88)
    any_win = False
    for a in ALPHAS:
        sp, sn = cache[f"steer_obj@{a}"]
        rp, rn = cache[f"rand_obj@{a}"]
        lo, hi = dd(sp, sn, rp, rn)
        win = lo > 0
        any_win |= win
        print(f"  alpha={a:<5} diff-in-discrimination CI [{100*lo:+6.1f}, {100*hi:+6.1f}] pp"
              f"{'   <-- SEPARATES' if win else '   (spans 0)'}")

    print("\n" + "="*88)
    print("VERDICT")
    print("="*88)
    if any_win:
        print("  SUCCESS at >=1 alpha: the derived direction beats a norm-matched random direction")
        print("  on discrimination. The L16 category signal is causally usable.")
    else:
        print("  NULL. steer_obj is indistinguishable from a norm-matched RANDOM direction at every")
        print("  alpha. Per the pre-registration this is reported as a null -- and, as in Phase 6e,")
        print("  it is a statement about the INTERVENTION, not proof the mechanism is absent:")
        print("  Phase 15 measured the L16 category signal at AUROC ~0.71, which is real but faint,")
        print("  and adding a single mean-difference direction may simply be too blunt to exploit it.")
    for a in ALPHAS:
        sp, sn = cache[f"steer_obj@{a}"]
        if sum(sp)/len(sp) > 0.1 and sum(sn)/len(sn) > 0.1:
            print(f"  BIAS-SHIFT WARNING at alpha={a}: raises P(yes) on denials AND true absences.")


if __name__ == "__main__":
    main()
