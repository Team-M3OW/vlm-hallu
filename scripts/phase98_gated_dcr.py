"""
Phase 98: DCR HAS NEVER BEEN GATED.

Phase 31 established that a FIXED always-crop policy cannot pay for its second forward pass.
Phase 32 fixed that for the OLD argmax proposer by answering from whichever pass is more confident,
reaching 68.6% and clearing the compute bar. Phase 45 then showed `peak` predicts coverage for free.

Every DCR arm ever run (71, 80, 90) is ALWAYS-CROP and answers from the crop unconditionally. The
better proposer has never been combined with the gate that made the worse one pay. That is the
untried lever, and it is free: every number below comes off disk.

Why it should help more now, not less: SS6E showed the two-pass confidence comparison is an
unsupervised COVERAGE detector, and DCR raises coverage 39.3 -> 52.9%. The detector gets a
better-populated positive class. And half of all windows still miss (SS6D), costing -15.6pp each --
a gate is exactly what removes that loss.

PRE-REGISTERED, before looking at any output:

  PRIMARY    conf(head@0.25): per item answer from whichever of {uniform@300, head@0.25} is more
             confident.  Cost = 2 passes.  Bar = uniform@600.  No threshold, so no selection bias.
             DECISION: CI clear of zero -> the gated method beats equal compute. Else it does not.

  SECONDARY  gate on a FREE pass-1 signal (peak / peak_over_median / top1_frac / entropy), pay for
             pass 2 only when it fires. Threshold chosen OUT-OF-FOLD, GroupKFold by item.
             Cost = 300*(1+f).  BAR = RANDOM ROUTING AT THE SAME RATE f -- not uniform@300.
             Phase 45 found random routing already earns +3.3pp from the concavity of the uniform
             curve, so anything scored against uniform@300 is inflated.

  CONTROL    always-crop head@0.25 vs uniform@600 = the +7.7pp [-0.6,+16.0] we are trying to beat.
  CEILING    conf(oracle@0.25).

Budget gate: tokens MEASURED from realized_tokens; >10% drift voids the contrast.
"""
import json, numpy as np
from collections import defaultdict

W = "0.25"
SW = "data/phase78_w_sweep.jsonl"      # Qwen3-VL-2B, n=191, all W arms + both uniform rungs
GT = "data/phase32_conditional.jsonl"  # same items; carries the free pass-1 attn features

rows = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(SW)}
feats = {}
for l in open(GT):
    r = json.loads(l)
    if "attn_feats" in r:
        feats[r["question_id_full"]] = r["attn_feats"]
ids = [q for q in rows if q in feats]
print(f"n = {len(ids)} items joined (sweep {len(rows)}, feats {len(feats)})")

P = {q: rows[q]["probs"] for q in ids}
T = {q: rows[q]["realized_tokens"] for q in ids}
y = np.array([rows[q]["label"] for q in ids])
cat = np.array([rows[q]["category"] for q in ids])

def correct(arm):
    return np.array([int(np.argmax(P[q][arm]) == rows[q]["label"]) for q in ids], float)
def conf(arm):
    return np.array([max(P[q][arm]) for q in ids])
def toks(arm):
    return np.array([T[q][arm] for q in ids], float)

rng = np.random.default_rng(98)
def ci(d, B=4000):
    n = len(d); idx = rng.integers(0, n, (B, n))
    b = d[idx].mean(1)
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

u300, u600 = correct("uniform@300"), correct("uniform@600")
crop = correct(f"head@{W}")
orac = correct(f"oracle@{W}")
c300, ccrop, corac = conf("uniform@300"), conf(f"head@{W}"), conf(f"oracle@{W}")

# --- PRIMARY: two-pass confidence rule -------------------------------------
pick = (ccrop >= c300)
confhead = np.where(pick, crop, u300)
confor   = np.where(corac >= c300, orac, u300)

tok_method = toks("uniform@300") + toks(f"head@{W}")
tok_bar    = toks("uniform@600")
drift = abs(tok_method.mean() - tok_bar.mean()) / tok_bar.mean()
print(f"\nbudget: method {tok_method.mean():.0f} tok vs bar {tok_bar.mean():.0f} tok "
      f"-> drift {100*drift:.1f}%  {'OK' if drift <= .10 else 'VOID'}")

print(f"\n{'arm':>26} {'acc':>7}")
for nm, a in [("uniform@300 (pass 1)", u300), ("uniform@600 (THE BAR)", u600),
              (f"head@{W} always-crop", crop), (f"conf(head@{W})  PRIMARY", confhead),
              (f"conf(oracle@{W}) ceiling", confor), (f"oracle@{W}", orac)]:
    print(f"{nm:>26} {a.mean()*100:6.1f}%")
print(f"\n  crop fires on {pick.mean()*100:.1f}% of items (confidence favoured the crop)")

print(f"\n{'contrast':>34} {'delta':>8}  95% CI")
for nm, d in [("always-crop - bar  (the CONTROL)", crop - u600),
              ("conf(head) - bar   (PRIMARY)", confhead - u600),
              ("conf(head) - always-crop", confhead - crop),
              ("conf(head) - pass 1", confhead - u300),
              ("conf(oracle) - bar (CEILING)", confor - u600)]:
    m, lo, hi = ci(d)
    print(f"{nm:>34} {m:+7.1f}pp  [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo > 0 else ''}")

# per-category, pre-registered split (single-region is where the mechanism says it lives)
print(f"\n{'category':>22} {'n':>4} {'bar':>7} {'conf':>7} {'delta':>8}")
for c in sorted(set(cat)):
    m_ = cat == c
    d = (confhead - u600)[m_]
    print(f"{c:>22} {m_.sum():4d} {u600[m_].mean()*100:6.1f}% {confhead[m_].mean()*100:6.1f}% {d.mean()*100:+7.1f}pp")

# --- SECONDARY: free pass-1 gate, threshold chosen OUT OF FOLD -------------
print("\n--- free pass-1 gate, OOF threshold, bar = random routing at the same rate ---")
K = 5
order = np.arange(len(ids)); rng2 = np.random.default_rng(7); rng2.shuffle(order)
fold = np.zeros(len(ids), int)
for i, o in enumerate(order):
    fold[o] = i % K

for fname in ["peak", "peak_over_median", "top1_frac", "entropy_norm"]:
    s = np.array([feats[q][fname] for q in ids], float)
    gated = np.empty(len(ids)); fired = np.zeros(len(ids), bool)
    for k in range(K):
        tr, te = fold != k, fold == k
        best, bthr = -1, None
        for thr in np.quantile(s[tr], np.linspace(0.0, 0.95, 20)):
            f_ = s[tr] >= thr
            a = np.where(f_, crop[tr], u300[tr]).mean()
            if a > best:
                best, bthr = a, thr
        f_te = s[te] >= bthr
        fired[te] = f_te
        gated[te] = np.where(f_te, crop[te], u300[te])
    f = fired.mean()
    # matched-cost control: route the same FRACTION at random to uniform@600
    reps = 200; ctrl = np.zeros(len(ids))
    for _ in range(reps):
        rr = rng.random(len(ids)) < f
        ctrl += np.where(rr, u600, u300)
    ctrl /= reps
    m, lo, hi = ci(gated - ctrl)
    m2, lo2, hi2 = ci(gated - u300)
    print(f"  {fname:>17}  fires {f*100:4.1f}%  acc {gated.mean()*100:5.1f}%  "
          f"vs random-routing {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'  CLEARS' if lo>0 else ''}"
          f"   (vs pass 1 {m2:+5.1f})")
