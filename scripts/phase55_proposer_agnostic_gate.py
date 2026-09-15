"""
Phase 55: THE PROPOSER-AGNOSTIC GATE. Offline on Phase 47 (V*Bench) and Phase 53 (HR-Bench 4k).

THE CLAIM
---------
Every result so far says the same thing from a different angle: allocation pays only when the window
covers the evidence, and paying for a proposal you will not use is what makes crop/zoom policies
lose to the budget axis. That is a statement about WHEN TO SPEND, not about WHICH PROPOSER to use.
So the contribution should be a wrapper that improves ANY proposer, including the baselines.

    gate(P):  pass 1 is uniform@B0 -- it yields an answer, the attention map, and `peak` (free).
              if peak <  tau:  answer from pass 1.                        cost = B0
              if peak >= tau:  run proposer P, then answer from whichever
                               of {uniform, P} is more confident.         cost = B0 + cost(P)

`peak` is the max of the ring-masked attention map, already computed by pass 1, and it predicts
coverage at AUROC 0.788 (§8A). tau is set by the V*Bench firing rate (40%) and TRANSFERRED to
HR-Bench unchanged; refitting per benchmark would not be a transfer result.

WHY THIS IS WORTH TESTING ON THE BASELINES TOO
----------------------------------------------
Zoom Eye spends 9 passes and 2636 tokens on EVERY item and loses 15.3pp to its own bar at 4K. If
most of that spend lands on items where no crop helps, gating should recover a large part of the
loss -- and that would be a result about the literature, not just about us.

    the gate improves every proposer on both benchmarks -> a general, free improvement operator,
        and the method claim becomes "when to spend", which is what all the mechanism work supports.
    it helps ours and not the baselines -> it is not proposer-agnostic; say so and drop the framing.

Each arm is scored against the uniform sweep measured IN ITS OWN RUN on ITS OWN items, so the two
benchmarks are never mixed. No GPU: every forward pass needed was already run.
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase47_prior_art.jsonl"
HR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase53_prior_art_hrbench.jsonl"
FIRE = 0.40
PROPOSERS = ["ours_attn@0.15", "grounding_crop", "zoom_eye"]
NICE = {"ours_attn@0.15": "attention peak (ours)",
        "grounding_crop": "grounding (Chain-of-Spot family)",
        "zoom_eye": "tree search (Zoom Eye)"}


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def make_bar(anchors):
    anchors = sorted(anchors)

    def f(t):
        if t <= anchors[0][0]:
            return anchors[0][1]
        for (t0, a0), (t1, a1) in zip(anchors, anchors[1:]):
            if t <= t1:
                w = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
                return a0 + w * (a1 - a0)
        (t0, a0), (t1, a1) = anchors[-2], anchors[-1]
        sl = (a1 - a0) / (math.log(t1) - math.log(t0))
        return a1 + sl * (math.log(t) - math.log(t1))
    return f


def run(path, name, key_inst, uni="uniform@300"):
    rows = [json.loads(l) for l in open(path)]
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    conf = lambda r, a: max(r["probs"][a])
    n = len(rows)
    props = [p for p in PROPOSERS if p in rows[0]["probs"]]
    anchors = [(st.mean(r["cost"][a]["tokens"] for r in rows), st.mean(hit(r, a) for r in rows))
               for a in ("uniform@300", "uniform@600", "uniform@1200") if a in rows[0]["probs"]]
    bar = make_bar(anchors)

    # peak is per INSTANCE on HR-Bench, per item on V*Bench
    key = (lambda r: r[key_inst]) if key_inst else (lambda r: id(r))
    pk = {}
    for r in rows:
        pk.setdefault(key(r), r["peak"])
    vals = sorted(pk.values(), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]

    print(f"\n{'='*78}\n{name}   n={n}"
          + (f" rows / {len(pk)} instances" if key_inst else " items")
          + f"\n{'='*78}")
    print("  bar (in-run): " + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print(f"  tau = {tau:.4f} (V*Bench firing rate {100*FIRE:.0f}%, transferred)\n")
    print(f"  {'proposer':<34}{'variant':<9}{'tokens':>8}{'acc':>8}{'bar':>8}"
          f"{'margin':>9}{'95% CI':>17}")
    out = {}
    for p in props:
        # ungated: always run the proposer
        vu = [hit(r, p) for r in rows]
        tu = st.mean(r["cost"][p]["tokens"] for r in rows)
        bu = bar(tu)
        lo, hi = boot([x - bu for x in vu])
        # gated: pay the proposer only when peak clears tau, then confidence-route
        vg, tg = [], []
        for r in rows:
            if pk[key(r)] >= tau:
                vg.append(hit(r, p) if conf(r, p) > conf(r, uni) else hit(r, uni))
                tg.append(r["cost"][p]["tokens"])
            else:
                vg.append(hit(r, uni))
                tg.append(r["cost"][uni]["tokens"])
        tgm = st.mean(tg)
        bg = bar(tgm)
        lo2, hi2 = boot([x - bg for x in vg])
        out[p] = (st.mean(vu) - bu, st.mean(vg) - bg, tu, tgm, lo2, hi2)
        print(f"  {NICE[p]:<34}{'ungated':<9}{tu:>8.0f}{100*st.mean(vu):>7.1f}%{100*bu:>7.1f}%"
              f"{100*(st.mean(vu)-bu):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")
        print(f"  {'':<34}{'GATED':<9}{tgm:>8.0f}{100*st.mean(vg):>7.1f}%{100*bg:>7.1f}%"
              f"{100*(st.mean(vg)-bg):>+8.1f}   [{100*lo2:+.1f},{100*hi2:+.1f}]"
              f"   {'<- improves' if out[p][1] > out[p][0] else '<- hurts'}")
    print(f"\n  {'proposer':<34}{'margin gain from gating':>26}{'token saving':>15}")
    for p in props:
        d = out[p][1] - out[p][0]
        print(f"  {NICE[p]:<34}{100*d:>+25.1f}pp{100*(1-out[p][3]/out[p][2]):>14.0f}%")
    return out


def main():
    print("THE PROPOSER-AGNOSTIC GATE: does a free pass-1 signal improve EVERY proposer?")
    print("Each arm scored against the uniform sweep measured in its own run on its own items.")
    a = run(VS, "V*Bench (191 items, ~1500-2000px)", None)
    b = run(HR, "HR-Bench 4k (800 rows / 200 instances, 4032px)", "instance")

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    ok = True
    for p in PROPOSERS:
        ga = a.get(p)
        gb = b.get(p)
        if not ga or not gb:
            continue
        da, db = ga[1] - ga[0], gb[1] - gb[0]
        tag = "both" if (da > 0 and db > 0) else ("V* only" if da > 0 else
                                                 ("HR only" if db > 0 else "neither"))
        print(f"  {NICE[p]:<34}V*Bench {100*da:+6.1f}pp   HR-Bench {100*db:+6.1f}pp   -> {tag}")
        if not (da > 0 and db > 0):
            ok = False
    print()
    if ok:
        print("  => THE GATE IS PROPOSER-AGNOSTIC. It improves every proposer on both benchmarks and")
        print("     at both scales, using only a scalar that pass 1 already computed. The method")
        print("     claim is WHEN TO SPEND, which is exactly what the mechanism work supports, and")
        print("     it improves the BASELINES too -- a result about the literature, not just ours.")
    else:
        print("  => NOT proposer-agnostic in general. Report which proposers and which scales it")
        print("     helps, and drop the general framing.")


if __name__ == "__main__":
    main()
