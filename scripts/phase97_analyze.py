"""Phase 97 analysis: does the held-out W=0.25 from Qwen3-VL transfer to Qwen2-VL?

PRE-REGISTERED in the phase 97 header, before the run:
  PRIMARY  head@0.25 - uniform@600  (equal compute; W transferred, NOT selected here)
  W=0.35 is context only and is NOT eligible to be reported as the method's window.
"""
import json, numpy as np
rows = [json.loads(l) for l in open("data/phase97_w_transfer_qwen2vl.jsonl")]
n = len(rows)
WS = ["0.15", "0.25", "0.35"]
def arm(a): return np.array([int(np.argmax(r["probs"][a]) == r["label"]) for r in rows], float)
def tok(a): return np.array([r["realized_tokens"][a] for r in rows], float)
cat = np.array([r["category"] for r in rows])
rng = np.random.default_rng(97)
def ci(d, B=4000):
    idx = rng.integers(0, n, (B, n)); b = d[idx].mean(1)
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

u300, u600 = arm("uniform@300"), arm("uniform@600")
print(f"Qwen2-VL-7B, V*Bench, n = {n}")
tm, tb = tok("uniform@300") + tok("head@0.25"), tok("uniform@600")
d = abs(tm.mean()-tb.mean())/tb.mean()
print(f"budget: method {tm.mean():.0f} vs bar {tb.mean():.0f} -> drift {100*d:.1f}% {'OK' if d<=.10 else 'VOID'}\n")
print(f"{'arm':>22} " + " ".join(f"{w:>8}" for w in WS))
for k in ["argmax", "head", "rand", "oracle"]:
    print(f"{k:>22} " + " ".join(f"{arm(f'{k}@{w}').mean()*100:7.1f}%" for w in WS))
print(f"\n  uniform@300 {u300.mean()*100:.1f}%    uniform@600 (THE BAR) {u600.mean()*100:.1f}%")

print(f"\n{'contrast':>38} {'delta':>8}  95% CI")
for nm, dd in [("head@0.25 - uniform@600   (PRIMARY)", arm("head@0.25") - u600),
               ("head@0.15 - uniform@600  (phase 80b)", arm("head@0.15") - u600),
               ("head@0.25 - head@0.15  (does W transfer)", arm("head@0.25") - arm("head@0.15")),
               ("head@0.25 - argmax@0.25", arm("head@0.25") - arm("argmax@0.25")),
               ("head@0.25 - rand@0.25", arm("head@0.25") - arm("rand@0.25")),
               ("head@0.25 - uniform@300 (vs vanilla)", arm("head@0.25") - u300),
               ("head@0.35 - uniform@600  (context only)", arm("head@0.35") - u600)]:
    m, lo, hi = ci(dd)
    print(f"{nm:>38} {m:+7.1f}pp  [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''}")
print(f"\n{'category':>22} {'n':>4} {'bar':>7} {'head@.25':>9} {'delta':>8}")
for c in sorted(set(cat)):
    m_ = cat == c
    print(f"{c:>22} {m_.sum():4d} {u600[m_].mean()*100:6.1f}% {arm('head@0.25')[m_].mean()*100:8.1f}% "
          f"{(arm('head@0.25')-u600)[m_].mean()*100:+7.1f}pp")
