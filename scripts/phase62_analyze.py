"""
Phase 62 analyzer: do multi-crop (allocation) and the L24 read-out compose?

They act on different failures -- one supplies information the representation lacks, the other stops
information being discarded late -- so their gains should add. Measured here as a 2x2: {uniform,
top1, multi4} x {final layer, L24}, on the same items, with the read-out layer chosen out of fold.

The interaction term is the question: if (multi4, L24) - (uniform, final) equals the sum of the two
main effects, they compose; if it is smaller, they are competing for the same headroom.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase62_combine.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 60:
        print(f"only {len(rows)} rows; wait")
        return
    nL = len(rows[0]["uniform"])
    hit = lambda p, y: 1.0 * (max(range(4), key=lambda i: p[i]) == y)
    ARMS = ["uniform", "top1", "multi4"]
    for nm, g in (("ALL ITEMS", rows),
                  ("SUB-TOKEN", [r for r in rows if r["tokens_on_target"] < 1.0])):
        n = len(g)
        if n < 40:
            continue
        idx = list(range(n))
        random.Random(11).shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        # read-out layer chosen out of fold, per arm
        cell, picked = {}, {}
        for a in ARMS:
            oof = [0.0] * n
            ps = []
            for f in range(5):
                te = set(folds[f])
                tr = [idx_ for idx_ in idx if idx_ not in te]
                best = max(range(nL), key=lambda li: st.mean(
                    hit(g[i][a][li], g[i]["label"]) for i in tr))
                ps.append(best)
                for i in te:
                    oof[i] = hit(g[i][a][best], g[i]["label"])
            cell[(a, "cvL")] = oof
            picked[a] = ps
            cell[(a, "final")] = [hit(r[a][-1], r["label"]) for r in g]
        print(f"\n=== {nm} (n={n}) ===")
        print(f"  {'':<10}{'final layer':>14}{'CV layer':>12}{'read-out gain':>16}   layers picked")
        for a in ARMS:
            d = [x - y for x, y in zip(cell[(a, "cvL")], cell[(a, "final")])]
            lo, hi = boot(d)
            print(f"  {a:<10}{100*st.mean(cell[(a,'final')]):>13.1f}%"
                  f"{100*st.mean(cell[(a,'cvL')]):>11.1f}%{100*st.mean(d):>+13.1f}"
                  f"   [{100*lo:+.1f},{100*hi:+.1f}]  {picked[a]}")
        base = cell[("uniform", "final")]
        print(f"\n  {'effect':<34}{'delta vs uniform/final':>24}{'95% CI':>18}")
        eff = {}
        for lbl, key in (("allocation alone (multi4, final)", ("multi4", "final")),
                         ("read-out alone (uniform, L_cv)", ("uniform", "cvL")),
                         ("BOTH (multi4, L_cv)", ("multi4", "cvL")),
                         ("top1 alone (final)", ("top1", "final")),
                         ("top1 + read-out", ("top1", "cvL"))):
            d = [x - y for x, y in zip(cell[key], base)]
            lo, hi = boot(d)
            eff[lbl] = st.mean(d)
            print(f"  {lbl:<34}{100*st.mean(d):>+23.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")
        a_ = eff["allocation alone (multi4, final)"]
        r_ = eff["read-out alone (uniform, L_cv)"]
        b_ = eff["BOTH (multi4, L_cv)"]
        print(f"\n  additive prediction {100*(a_+r_):+.1f}pp   observed {100*b_:+.1f}pp"
              f"   interaction {100*(b_-a_-r_):+.1f}pp")
        if b_ >= a_ + r_ - 0.01:
            print("  => they COMPOSE (observed >= sum of main effects).")
        elif b_ > max(a_, r_):
            print("  => PARTIAL composition: better than either alone, less than their sum;")
            print("     they overlap on some of the same items.")
        else:
            print("  => they do NOT compose: the combination is no better than the best single one.")


if __name__ == "__main__":
    main()
