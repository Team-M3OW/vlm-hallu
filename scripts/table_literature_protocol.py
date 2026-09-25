"""
The literature-protocol table: a FAIR comparison with ViCrop Table 2.

Their rules, replicated:
  * no-crop  = the model on the full image at its budget (for LLaVA the anyres ladder minimum;
               for Qwen the same token count as the crop arm);
  * crop     = the attention/ridge-placed crop re-encoded at the SAME token count;
  * the localise pass is NOT charged (crop@N vs no-crop@N, exactly as published);
  * metrics: accuracy for V* (MCQ), normalised any-of-N match for TextVQA/DocVQA/GQA open items
    (the harness's approximation of VQA accuracy / ANLS).

Sources: phase228/229 (Qwen, equal 300), phase225 (LLaVA-OneVision, equal realised tokens).
ViCrop's published numbers are quoted for reference only.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
BENCH = [("vstar", "V*"), ("textvqa", "TextVQA"), ("docvqa", "DocVQA"), ("gqa", "GQA")]


def rows_of(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def acc(rows, a):
    v = [B.item_correct(r, a) for r in rows if (a in r.get("probs", {}) or a in r.get("preds", {}))]
    return 100 * float(np.mean(v)) if v else None


def tok(rows, a):
    v = [r["tokens"][a] for r in rows if a in r.get("tokens", {})]
    return int(np.mean(v)) if v else 0


def delta(rows, a, b, Bn=10000):
    d = [B.item_correct(r, a) - B.item_correct(r, b) for r in rows
         if (a in r.get("probs", {}) or a in r.get("preds", {})) and (b in r.get("probs", {}) or b in r.get("preds", {}))]
    if len(d) < 8: return None
    d = np.array(d); m = d[rng.integers(0, len(d), (Bn, len(d)))].mean(1)
    lo, hi = np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
    return d.mean() * 100, lo, hi


print("=" * 112)
print("PANEL A -- OUR MODELS UNDER THE LITERATURE PROTOCOL (equal answer tokens, no localise charge)")
print("=" * 112)
print(f"{'model':16s}{'benchmark':10s}{'n':>5s}{'no-crop':>9s}{'crop(W=.25)':>12s}{'crop(W=.5)':>11s}{'delta best':>20s}{'tokens':>16s}")

# Qwen 2B / 7B, phase 228/229 (300 vs 300)
for mk, name in [("qwen3_2b", "Qwen3-VL-2B"), ("qwen2_7b", "Qwen2-VL-7B")]:
    for bk, bn in BENCH:
        rows = rows_of(f"{D}/data/phase229_{mk}_{bk}.jsonl") or rows_of(f"{D}/data/phase228_{mk}_{bk}.jsonl")
        if not rows:
            print(f"{name:16s}{bn:10s}{'--':>5s}{'running':>9s}"); continue
        base = "uniform@300"
        c25, c50 = acc(rows, "crop25@300"), acc(rows, "crop50@300")
        best = "crop25@300" if (c25 or 0) >= (c50 or 0) else "crop50@300"
        d = delta(rows, best, base)
        dtxt = f"{d[0]:+5.1f} [{d[1]:+5.1f},{d[2]:+5.1f}]" if d else "n/a"
        print(f"{name:16s}{bn:10s}{len(rows):5d}{acc(rows,base):9.1f}{c25 if c25 is not None else float('nan'):12.1f}"
              f"{c50 if c50 is not None else float('nan'):11.1f}{dtxt:>20s}"
              f"{str(tok(rows,'uniform@300'))+' vs '+str(tok(rows,'crop25@300')):>16s}")

# LLaVA-OneVision, phase 225 (its ladder minimum; equal realised tokens both arms)
for bk, bn in BENCH:
    rows = rows_of(f"{D}/data/phase225_llava_ov_{bk}.jsonl")
    if not rows: continue
    base = "uniform@lo"
    c_block, c_dwa = acc(rows, "block"), acc(rows, "dwa_t")
    if c_block is None: continue
    best = "dwa_t" if (c_dwa or 0) >= (c_block or 0) else "block"
    d = delta(rows, best, base)
    dtxt = f"{d[0]:+5.1f} [{d[1]:+5.1f},{d[2]:+5.1f}]" if d else "n/a"
    print(f"{'LLaVA-OV-7B':16s}{bn:10s}{len(rows):5d}{acc(rows,base):9.1f}{c_block:12.1f}{c_dwa:11.1f}{dtxt:>20s}"
          f"{str(tok(rows,'uniform@lo'))+' vs '+str(tok(rows,'block')):>16s}")

print()
print("=" * 112)
print("PANEL B -- VICROP'S PUBLISHED TABLE 2 (reference; fixed-resolution baselines, no localise charge)")
print("=" * 112)
print(f"{'model':16s}{'TextVQA':>10s}{'V*':>10s}{'POPE':>10s}{'DocVQA':>10s}{'AOKVQA':>10s}{'GQA':>10s}{'VQAv2':>10s}")
print(f"{'LLaVA-1.5 no-crop':16s}{47.80:10.2f}{42.41:10.2f}{85.27:10.2f}{15.97:10.2f}{59.01:10.2f}{60.48:10.2f}{75.57:10.2f}")
print(f"{'LLaVA-1.5 rel-att':16s}{55.17:10.2f}{62.30:10.2f}{87.25:10.2f}{19.63:10.2f}{60.66:10.2f}{60.97:10.2f}{76.51:10.2f}")
print(f"{'InstructBLIP no-crop':16s}{33.48:10.2f}{35.60:10.2f}{84.89:10.2f}{9.20:10.2f}{60.06:10.2f}{49.41:10.2f}{76.25:10.2f}")
print(f"{'InstructBLIP rel-att':16s}{45.44:10.2f}{42.41:10.2f}{86.64:10.2f}{9.95:10.2f}{61.28:10.2f}{49.75:10.2f}{76.84:10.2f}")
