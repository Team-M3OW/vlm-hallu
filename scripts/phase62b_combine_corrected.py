"""
Phase 62b -- RE-RUN of Phase 62 with the CORRECT final layer.

Phase 62 asked whether allocation (multi-crop) and a layer read-out COMPOSE. Its read-out half rested
on the same double-normalisation bug as Phase 61, so the composition claim was withdrawn. This redoes
it with the final layer taken from the model's own logits.

    intermediate layers  <- Phase 62 lens (correct; never touched the final layer)
    final layer          <- Phase 58 `probs` (model.logits), same items, same deterministic decoding

TWO JOIN CHECKS before anything is reported:
  1. Phase 62's intermediate layers must agree with Phase 60's on the uniform arm (both are lens runs
     on the same items, and intermediate layers were never corrupted).
  2. Phase 62's own final layer must DISAGREE with Phase 58's on a meaningful fraction -- that
     disagreement IS the bug, and its absence would mean the join is pointing at the wrong thing.

The question re-asked: with a correct baseline, is there anything for allocation to compose WITH?
"""
import json
import random
import statistics as st

P58 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase58_multicrop.jsonl"
P60 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"
P62 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase62_combine.jsonl"
MAP = {"uniform": "uniform@300", "top1": "top1@0.15", "multi4": "multi4"}


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    c = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P62))}
    f = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P58))}
    g0 = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P60))}
    k = sorted(set(c) & set(f) & set(g0))
    nL = len(c[k[0]]["uniform"])
    am = lambda p: max(range(4), key=lambda i: p[i])

    inter = sum(1 for q in k if am(c[q]["uniform"][20]) == am(g0[q]["uniform"][20])) / len(k)
    finl = sum(1 for q in k if am(c[q]["uniform"][-1]) == am(f[q]["probs"]["uniform@300"])) / len(k)
    print(f"n = {len(k)}")
    print(f"  join check 1 -- intermediate layers agree across lens runs: {100*inter:.1f}%  "
          f"{'OK' if inter > 0.95 else 'FAIL'}")
    print(f"  join check 2 -- phase62's own final vs model.logits:        {100*finl:.1f}%  "
          f"{'(the bug, as expected)' if finl < 0.95 else 'FAIL -- expected disagreement'}")
    assert inter > 0.95, "lens runs disagree on intermediate layers; join unsafe"
    assert finl < 0.95, "no final-layer disagreement; the join is not pointing at the bug"

    def lay(q, arm):
        return c[q][arm][:nL - 1] + [f[q]["probs"][MAP[arm]]]

    sub = [q for q in k if c[q]["tokens_on_target"] < 1.0]
    ARMS = ["uniform", "top1", "multi4"]

    for nm, G in (("ALL ITEMS", k), ("SUB-TOKEN", sub)):
        n = len(G)
        idx = list(range(n))
        random.Random(11).shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        cell, picked = {}, {}
        for arm in ARMS:
            fin = [1.0 * (am(lay(G[i], arm)[-1]) == c[G[i]]["label"]) for i in range(n)]
            oof = [0.0] * n
            ps = []
            for fd in range(5):
                te = set(folds[fd])
                tr = [i for i in idx if i not in te]
                best = max(range(nL), key=lambda li: st.mean(
                    1.0 * (am(lay(G[i], arm)[li]) == c[G[i]]["label"]) for i in tr))
                ps.append(best)
                for i in te:
                    oof[i] = 1.0 * (am(lay(G[i], arm)[best]) == c[G[i]]["label"])
            cell[(arm, "final")] = fin
            cell[(arm, "cv")] = oof
            picked[arm] = ps
        print(f"\n=== {nm} (n={n}) ===")
        print(f"  {'arm':<10}{'final layer':>14}{'CV layer':>11}{'read-out gain':>16}   layers")
        for arm in ARMS:
            d = [x - y for x, y in zip(cell[(arm, "cv")], cell[(arm, "final")])]
            lo, hi = boot(d)
            print(f"  {arm:<10}{100*st.mean(cell[(arm,'final')]):>13.1f}%"
                  f"{100*st.mean(cell[(arm,'cv')]):>10.1f}%{100*st.mean(d):>+13.1f}"
                  f"   [{100*lo:+.1f},{100*hi:+.1f}]  {picked[arm]}")
        base = cell[("uniform", "final")]
        print(f"\n  {'effect vs uniform/final':<34}{'delta':>9}{'95% CI':>18}")
        eff = {}
        for lbl, key in (("allocation alone (multi4, final)", ("multi4", "final")),
                         ("read-out alone (uniform, CV layer)", ("uniform", "cv")),
                         ("BOTH (multi4, CV layer)", ("multi4", "cv")),
                         ("top1 alone (final)", ("top1", "final"))):
            d = [x - y for x, y in zip(cell[key], base)]
            lo, hi = boot(d)
            eff[lbl] = st.mean(d)
            print(f"  {lbl:<34}{100*st.mean(d):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
                  f"{'  SIG' if lo > 0 else ''}")
        A = eff["allocation alone (multi4, final)"]
        R = eff["read-out alone (uniform, CV layer)"]
        B = eff["BOTH (multi4, CV layer)"]
        print(f"\n  additive prediction {100*(A+R):+.1f}pp   observed {100*B:+.1f}pp"
              f"   interaction {100*(B-A-R):+.1f}pp")
        if abs(R) < 0.01:
            print("  => there is NOTHING to compose with: the read-out term is zero, so the")
            print("     combination is just allocation. Phase 62's composition claim is void, and")
            print("     the allocation effect it measured stands on its own.")


if __name__ == "__main__":
    main()
