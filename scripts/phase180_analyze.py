"""
Combined analyzer: phase 178 (glocal) + phase 180 (DPR-Fovea, global-heavy splits + oracle sweep).

Merge rule: the two runs are pooled ONLY if their shared anchors -- uniform@600 and the DPR crop --
agree within the repo's 1-2.5pp noise floor. Otherwise the merge is void and each is reported alone.
Split naming is global/local out of a 300-token answer pass, so 83/17 == 236/64 (the 64-merged-token
floor, bug #21), 67/33 == 200/100, 50/50 == 150/150.
"""
import json, sys
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
rng = np.random.default_rng(180)
SPLIT_SRC = {  # requested ratio -> (file tag, arm name, oracle arm name)
    "83/17": ("180", "fovea_83_17", "oracle_fovea_83_17"),
    "75/25": ("180", "fovea_75_25", "oracle_fovea_75_25"),
    "67/33": ("178", "glocal_200_100", None),
    "50/50": ("178", "glocal_150_150", "oracle_glocal_150_150"),
    "33/67": ("178", "glocal_100_200", None),
}


def ci(d, B=8000):
    n = len(d)
    if n == 0: return (float("nan"),)*3
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


def load(which):
    out = {}
    for tag, path in [("178", f"{D}/phase178_glocal_{which}.jsonl"),
                      ("180", f"{D}/phase180_dprfovea_{which}_full.jsonl")]:
        try:
            out[tag] = {r["question_id_full"]: r for r in
                        (json.loads(l) for l in open(path))}
        except FileNotFoundError:
            out[tag] = {}
    return out


gtt, cat = {}, {}
for l in open(f"{D}/phase30c_attn_maps_all.jsonl"):
    r = json.loads(l); gtt[r["question_id_full"]] = r["gt_tokens"]; cat[r["question_id_full"]] = r["category"]
BINS = [(0, 0.15, "sub-token <0.15"), (0.15, 1, "0.15-1 tok"), (1, 4, "1-4 tok"), (4, 1e9, ">4 tok")]

for which, mdl in [("qwen3", "Qwen3-VL-2B"), ("qwen2", "Qwen2-VL-7B")]:
    S = load(which)
    ids = sorted(set(S["178"]) & set(S["180"])) if S["180"] else sorted(S["178"])
    if not ids:
        print(f"\n{mdl}: no data yet"); continue
    acc = lambda tag, arm: np.array(
        [int(np.argmax(S[tag][i]["probs"][arm]) == S[tag][i]["label"]) for i in ids], float)
    print(f"\n{'='*92}\n{mdl}   n={len(ids)} (items present in both runs)" if S["180"]
          else f"\n{'='*92}\n{mdl}   n={len(ids)} (phase 178 only)")

    # ---- merge anchors
    if S["180"]:
        for a178, a180, nm in [("uniform@600", "uniform@600", "bar"), ("head@0.25", "dpr_crop", "DPR crop")]:
            x, y = acc("178", a178), acc("180", a180)
            d = 100*(y-x).mean()
            print(f"  anchor {nm:9} 178 {100*x.mean():5.1f}%  180 {100*y.mean():5.1f}%  "
                  f"diff {d:+5.1f}pp  {'OK (pooled)' if abs(d) <= 2.5 else 'VOID — do not pool'}")
    bar, dpr = acc("178", "uniform@600"), acc("178", "head@0.25")
    print(f"\n  {'global/local':>12} {'src':>4} {'acc':>7} | {'Δ vs DPR crop':>24} {'Δ vs uniform@600':>24}")
    rows = {}
    for ratio, (tag, arm, _) in SPLIT_SRC.items():
        if tag not in S or not S[tag] or arm not in S[tag][ids[0]]["probs"]: continue
        a = acc(tag, arm); rows[ratio] = a
        d1, d2 = ci(a-dpr), ci(a-bar)
        print(f"  {ratio:>12} {tag:>4} {100*a.mean():6.1f}% | {d1[0]:+6.1f} [{d1[1]:+6.1f},{d1[2]:+6.1f}]"
              f"{'✔' if d1[1] > 0 else ' '}  {d2[0]:+6.1f} [{d2[1]:+6.1f},{d2[2]:+6.1f}]{'✔' if d2[1] > 0 else ' '}")
    print(f"  {'DPR crop':>12} {'178':>4} {100*dpr.mean():6.1f}% | {'—':>24} "
          f"{ci(dpr-bar)[0]:+6.1f} [{ci(dpr-bar)[1]:+6.1f},{ci(dpr-bar)[2]:+6.1f}]")
    print(f"  {'uniform@600':>12} {'178':>4} {100*bar.mean():6.1f}% |")

    best = max(rows, key=lambda k: rows[k].mean()) if rows else None
    # ---- breakdowns on the best split + the 50/50 primary
    for ratio in dict.fromkeys([r for r in (best, "50/50") if r in rows]):
        a = rows[ratio]
        print(f"\n  --- breakdown, global/local {ratio} ---")
        c = np.array([cat[i] for i in ids]); t = np.array([gtt[i] for i in ids], float)
        strata = [("single-object", c == "direct_attributes"), ("relational", c == "relative_position")]
        strata += [(nm, (t >= lo) & (t < hi)) for lo, hi, nm in BINS]
        for nm, m in strata:
            if m.sum() < 8: continue
            d1, d2 = ci((a-dpr)[m]), ci((a-bar)[m])
            print(f"    {nm:16} n={m.sum():3d}  fovea {100*a[m].mean():5.1f}%  dpr {100*dpr[m].mean():5.1f}%  "
                  f"bar {100*bar[m].mean():5.1f}% | vs DPR {d1[0]:+5.1f} [{d1[1]:+5.1f},{d1[2]:+5.1f}]"
                  f"{'✔' if d1[1] > 0 else ''} | vs bar {d2[0]:+5.1f} [{d2[1]:+5.1f},{d2[2]:+5.1f}]{'✔' if d2[1] > 0 else ''}")

    # ---- oracle sweep: is localisation or representation the bottleneck?
    print(f"\n  --- ORACLE-FOVEA (GT-centred ROI, same budgets) ---")
    ocrop = acc("178", "oracle@0.25")
    print(f"    {'oracle crop':>22} {100*ocrop.mean():5.1f}%   (DPR crop {100*dpr.mean():5.1f}%, "
          f"localisation gap {100*(ocrop-dpr).mean():+5.1f}pp)")
    for ratio, (tag, arm, oarm) in SPLIT_SRC.items():
        if oarm is None or tag not in S or not S[tag] or oarm not in S[tag][ids[0]]["probs"]: continue
        o = acc(tag, oarm)
        gap = ci(o - rows[ratio]) if ratio in rows else (float('nan'),)*3
        print(f"    {('oracle '+ratio):>22} {100*o.mean():5.1f}%   localisation gap vs its own fovea "
              f"{gap[0]:+5.1f} [{gap[1]:+5.1f},{gap[2]:+5.1f}]{'✔' if gap[1] > 0 else ''}")

    # ---- tokens and latency
    if S["180"]:
        r0 = S["180"][ids[0]]
        tk = {k: np.mean([S["180"][i]["realized_tokens"][k] for i in ids]) for k in r0["realized_tokens"]}
        lt = {k: np.mean([S["180"][i]["latency_s"][k] for i in ids]) for k in r0["latency_s"]}
        print(f"\n  --- tokens (answer pass; +300 localise for every crop/fovea arm) and latency ---")
        for k in tk:
            tot = tk[k] + (0 if k.startswith("uniform") else 300)
            print(f"    {k:22} {tk[k]:6.1f} tok | total {tot:6.1f} | drift vs 600 "
                  f"{100*(tot-600)/600:+5.1f}% | {1000*lt[k]:6.0f} ms")
