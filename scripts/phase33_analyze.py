"""
Phase 33 analyzer: did the V*Bench gate TRANSFER to HR-Bench 4k?

Nothing is fitted here. W=0.15 and both gate rules come from V*Bench; this file only scores them.
If any number below required retuning on HR-Bench it would not be a transfer result.

PRE-REGISTERED VERDICT (fixed before the data existed)
------------------------------------------------------
  gate > uniform2x  AND  gate > uniform      -> TRANSFERS. The §6.2 threat is answered: the rule is
                                                not the V*Bench category annotation in disguise.
  gate > uniform  BUT  gate <= uniform2x     -> PARTIAL. Helps, but does not earn the second pass on
                                                this benchmark. Report as a bounded transfer.
  gate <= uniform                            -> DOES NOT TRANSFER. Report it. The likely mechanism is
                                                already identified: HR-Bench uses spatial
                                                prepositions REFERENTIALLY ("the number above the
                                                entrance") where cropping should help, so the
                                                keyword rule skips items it should have taken.

Two metrics, both reported:
  per-row       all 800 rows independently
  circular      HR-Bench's intended CircularEval: an instance scores only if ALL 4 option
                permutations are answered correctly. Strictly harder and the benchmark's own metric.

NO ORACLE ARM: HR-Bench ships no boxes, so no oracle-capture fraction is computable and none is
quoted.
"""
import json
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33_hrbench_transfer.jsonl"
W_MAIN, W_ALT = 0.15, 0.25
REL = ("left", "right", "above", "below", "under", "over", "behind",
       "front", "side", "between", "beneath", "underneath", "top of", "bottom of")


def is_relational(q):
    ql = str(q).lower()
    return any(k in ql for k in REL)


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 50:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    inst = len({r["instance"] for r in rows})
    print(f"n = {n} rows over {inst} instances")
    print("  categories:", dict((c, sum(1 for r in rows if r["category"] == c))
                                for c in sorted({r["category"] for r in rows})))
    gate_fires = sum(1 for r in rows if is_relational(r["question"])) / n
    agree = max(
        sum(1 for r in rows if is_relational(r["question"]) == (r["category"] == c)) / n
        for c in ("cross", "single"))
    print(f"  keyword gate fires on {100*gate_fires:.1f}% of rows; "
          f"concordance with `category` {100*agree:.1f}%  (V*Bench was 99.0%)")

    pred = lambda r, a: max(range(4), key=lambda i: r["probs"][a][i])
    conf = lambda r, a: max(r["probs"][a])
    hit = lambda r, a: 1.0 * (pred(r, a) == r["label"])

    print("\n=== BUDGET GATE ===")
    for a in sorted(rows[0]["realized_tokens"]):
        med = st.median([r["realized_tokens"][a] for r in rows])
        tgt = 600 if a == "uniform2x" else 300
        off = abs(med - tgt) / tgt
        print(f"  {a:12} {med:6.0f} tok (target {tgt})  {100*off:4.1f}% off  "
              f"{'OK' if off < .10 else '!! VOID'}")

    def gate(r, W, kind):
        if kind == "conf":
            return hit(r, f"attn@{W}") if conf(r, f"attn@{W}") > conf(r, "uniform") \
                else hit(r, "uniform")
        if kind == "text":
            return hit(r, "uniform") if is_relational(r["question"]) else hit(r, f"attn@{W}")
        return hit(r, "uniform") if is_relational(r["question"]) else gate(r, W, "conf")

    arms = {
        "uniform@300": [hit(r, "uniform") for r in rows],
        "uniform@600 (compute-matched)": [hit(r, "uniform2x") for r in rows],
        f"always attn@{W_MAIN}": [hit(r, f"attn@{W_MAIN}") for r in rows],
        f"always attn@{W_ALT} (secondary)": [hit(r, f"attn@{W_ALT}") for r in rows],
        f"conf_route@{W_MAIN}": [gate(r, W_MAIN, "conf") for r in rows],
        f"text_gate@{W_MAIN}": [gate(r, W_MAIN, "text") for r in rows],
        f"TEXT+CONF@{W_MAIN} (the gate)": [gate(r, W_MAIN, "both") for r in rows],
    }
    u = arms["uniform@300"]
    u2 = arms["uniform@600 (compute-matched)"]

    print("\n=== PER-ROW ACCURACY (n={}) ===".format(n))
    print(f"  {'arm':34}{'acc':>7}{'vs uniform':>13}{'vs uniform2x':>15}")
    for k, v in arms.items():
        lo, hi = boot(v, u)
        print(f"  {k:34}{100*st.mean(v):6.1f}%{100*(st.mean(v)-st.mean(u)):+12.1f}pp"
              f"{100*(st.mean(v)-st.mean(u2)):+14.1f}pp   CI[{100*lo:+.1f},{100*hi:+.1f}]")

    # ---- CircularEval: instance counts only if all 4 permutations are right
    print("\n=== CIRCULAR EVAL (instance correct only if ALL cycles correct) ===")
    byinst = defaultdict(list)
    for i, r in enumerate(rows):
        byinst[r["instance"]].append(i)
    full = [k for k, v in byinst.items() if len(v) >= 4]
    print(f"  instances with all 4 cycles present: {len(full)}/{len(byinst)}")
    if full:
        print(f"  {'arm':34}{'circ acc':>10}{'vs uniform':>13}")
        cu = [min(arms['uniform@300'][i] for i in byinst[k]) for k in full]
        for k, v in arms.items():
            c = [min(v[i] for i in byinst[kk]) for kk in full]
            print(f"  {k:34}{100*st.mean(c):9.1f}%{100*(st.mean(c)-st.mean(cu)):+12.1f}pp")

    print("\n=== by category (per-row) ===")
    for cat in sorted({r["category"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        g = arms[f"TEXT+CONF@{W_MAIN} (the gate)"]
        print(f"  {cat:10} n={len(idx):4d}  gate {100*st.mean([g[i] for i in idx]):5.1f}%"
              f"  uniform {100*st.mean([u[i] for i in idx]):5.1f}%"
              f"  uniform2x {100*st.mean([u2[i] for i in idx]):5.1f}%")

    print("\n=== DIAGNOSTIC: does the keyword rule skip items it should have taken? ===")
    fired = [i for i, r in enumerate(rows) if is_relational(r["question"])]
    notf = [i for i, r in enumerate(rows) if not is_relational(r["question"])]
    a = arms[f"always attn@{W_MAIN}"]
    for nm, idx in [("gate FIRED (skipped alloc)", fired), ("gate did NOT fire", notf)]:
        if not idx:
            continue
        d = 100 * (st.mean([a[i] for i in idx]) - st.mean([u[i] for i in idx]))
        print(f"  {nm:28} n={len(idx):4d}  allocation would have scored {d:+.1f}pp vs uniform")
    print("  (a POSITIVE number on the FIRED rows means the rule skipped items allocation helps --")
    print("   the referential-preposition failure mode predicted before running)")

    print("\n" + "=" * 68)
    print("VERDICT vs pre-registered rule")
    print("=" * 68)
    g = arms[f"TEXT+CONF@{W_MAIN} (the gate)"]
    ga, ua, u2a = 100 * st.mean(g), 100 * st.mean(u), 100 * st.mean(u2)
    lo, hi = boot(g, u2)
    print(f"  gate {ga:.1f}%   uniform {ua:.1f}%   uniform2x {u2a:.1f}%")
    print(f"  gate - uniform2x = {ga-u2a:+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]")
    if ga > u2a and ga > ua:
        print("\n  => TRANSFERS. The rule is not the V*Bench annotation in disguise.")
    elif ga > ua:
        print("\n  => PARTIAL TRANSFER. Helps over baseline but does not earn the second pass here.")
    else:
        print("\n  => DOES NOT TRANSFER. Report as a negative; check the diagnostic above for the")
        print("     referential-preposition mechanism.")


if __name__ == "__main__":
    main()
