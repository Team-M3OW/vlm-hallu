"""
Phase 140 (Track 3): THE NO-TRAINING COUNTERPART -- compose every label-free fix that works.

Three label-free corrections each beat the deployed block-mean argmax: norm weighting alpha*||v||
(+4.7), max over layers (+7.3), S_v head selection (+11.5). They attack the serialisation sink by
different routes. Nobody has composed them. Also untried: ReAttn's entropy rescaling, phase 30d's
leave-one-out positional background, VEA's isolated-patch denoising, and a layer weighting by phase
95's label-free question-divergence signal.

EVERYTHING HERE IS A FIXED-CONSTANT RULE. No out-of-fold selection is used anywhere: block =
L16-26 (Qwen3) / L15-26 (Qwen2) as deployed; head fraction q=0.10; VEA lambda=10; background grid
8x8 as in phase 30d; divergence gate = layers with divergence >= 0.5 * max. Where a rule has no
selectable parameter it says so; the ONLY per-item information used is the model's own attention.

||v|| per token per layer is not stored separately, but nw_layers/raw_layers = normalise(A*vn)/
normalise(A) is proportional to vn up to a per-layer constant, which renormalisation removes. So
norm weighting can be applied to head-selected maps offline without a GPU.

PRE-REGISTERED
    primary contrast  composite - normw_max (56.5 on Qwen3): the strongest existing fixed rule
    secondary         composite - deployed argmax
    a component is 'additive' if its marginal gain clears zero on BOTH Qwen models
    LLaVA: only raw-map components can be tested (no norm-weighting, no per-head maps stored);
           reported as such, and any Qwen-only composite is labelled Qwen-only
Metric: top-1 coverage at W=0.25, ring-masked, hit = >=50% of the GT box inside the window.
Noise floor ~1pp (SS14Z). Reference: learned head 63.4 / 54.5.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
P70.W = 0.25; COV = P70.COV_HIT
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
Q, LAM, BG_R, BG_C, DIV_GATE = 0.10, 10.0, 8, 8, 0.5
DIV = json.load(open(f"{D}/phase95_question_depth.json"))

def ring(gh, gw):
    m = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
    else: m[:] = True
    return m.ravel()

def covvec(gh, gw, gt):
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
    return np.array([P70.coverage(float(fx[i]), float(fy[i]), gt) for i in range(gh*gw)])

def nrm(A):  # per-layer sum-normalise, rows = layers
    return A / np.maximum(A.sum(-1, keepdims=True), 1e-12)

def entropy_w(A, direction):
    p = nrm(A); H = -(p*np.log(p+1e-12)).sum(-1); Hn = H/np.log(A.shape[-1])
    w = Hn if direction == "high" else (1-Hn)
    return w/max(w.sum(), 1e-12)

def denoise(m, gh, gw, lam=LAM):
    g = m.reshape(gh, gw).copy(); p = np.pad(g, 1, mode="edge"); out = g.copy()
    for i in range(gh):
        for j in range(gw):
            nb = p[i:i+3, j:j+3].copy(); nb[1, 1] = -np.inf
            nbv = p[i:i+3, j:j+3].ravel(); nbv = np.delete(nbv, 4)
            if g[i, j] > lam*nb.max(): out[i, j] = nbv.mean()
    return out.ravel()

def bg_coords(gh, gw):
    yy, xx = np.mgrid[0:gh, 0:gw]
    r = np.minimum(BG_R-1, (BG_R*(yy+.5)/gh).astype(int)); c = np.minimum(BG_C-1, (BG_C*(xx+.5)/gw).astype(int))
    return (r*BG_C+c).ravel()

def load_qwen(which):
    b0, b1 = (16, 27) if which == "qwen3" else (15, 27)
    meta = json.load(open(f"{D}/phase108b_perhead_{which}_meta.json"))
    z = np.load(f"{D}/phase108b_perhead_{which}.npz")
    loc = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/phase104b_locators_{which}.jsonl")}
    div = np.array(DIV["Qwen3-VL-2B" if which == "qwen3" else "Qwen2-VL-7B"])
    items = []
    for i, mt in enumerate(meta):
        q = mt["question_id_full"]
        if q not in loc: continue
        r = loc[q]; gh, gw = mt["grid"]; n = mt["n_img"]
        if r["grid"] != mt["grid"]: continue
        raw = nrm(np.asarray(r["raw_layers"], float)); nw = nrm(np.asarray(r["nw_layers"], float))
        vn = nw/np.maximum(raw, 1e-12); vn = vn/np.maximum(vn.mean(-1, keepdims=True), 1e-12)
        hm = z[f"m{i}"].astype(np.float32); sv = z[f"sv{i}"].astype(np.float32)   # (NL,H,n), (NL,H)
        NL, H = sv.shape; k = max(1, int(round(Q*H)))
        idx = np.argsort(-sv, axis=1)[:, :k]
        hs = nrm(np.stack([hm[l, idx[l]].mean(0) for l in range(NL)]))
        hs_nw = nrm(hs*vn)
        items.append(dict(q=q, gh=gh, gw=gw, n=n, cov=covvec(gh, gw, mt["gt_box_frac"]), rm=ring(gh, gw),
                          raw=raw, nw=nw, hs=hs, hs_nw=hs_nw, cat=mt["category"]))
    return items, (b0, b1), div

def load_llava(tag):
    rows = [json.loads(l) for l in open(f"{D}/phase82_{tag}.jsonl")]
    NL = len([k for k in rows[0]["attn"] if k.startswith("L")]); b0, b1 = int(.57*NL), int(.93*NL)+1
    items = []
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh*gw != n: continue
        raw = nrm(np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)]))
        items.append(dict(q=r["question_id_full"], gh=gh, gw=gw, n=n, cov=covvec(gh, gw, r["gt_box_frac"]),
                          rm=ring(gh, gw), raw=raw, cat=r["category"]))
    return items, (b0, b1), None

def lo_background(items, key, fn):
    """Leave-one-out positional background of the (already aggregated) map fn(item), on an 8x8 grid."""
    per = []
    for it in items:
        m = fn(it); m = m/max(m.sum(), 1e-12); bc = bg_coords(it["gh"], it["gw"])
        acc = np.zeros(BG_R*BG_C); cnt = np.zeros(BG_R*BG_C)
        np.add.at(acc, bc, m); np.add.at(cnt, bc, 1)
        per.append((acc, cnt))
    tot = sum(a for a, _ in per); totc = sum(c for _, c in per)
    out = []
    for it, (a, c) in zip(items, per):
        bg = (tot-a)/np.maximum(totc-c, 1); bc = bg_coords(it["gh"], it["gw"])
        out.append(fn(it)/np.maximum(bg[bc], 1e-9))
    return out

def run(items, blk, div, has_nw):
    b0, b1 = blk
    arms = {}
    arms["deployed: raw mean block"] = [it["raw"][b0:b1].mean(0) for it in items]
    arms["raw max block"] = [it["raw"][b0:b1].max(0) for it in items]
    arms["raw + denoise"] = [denoise(it["raw"][b0:b1].mean(0), it["gh"], it["gw"]) for it in items]
    arms["raw + bgLOO"] = lo_background(items, "raw", lambda it: it["raw"][b0:b1].mean(0))
    arms["raw max + bgLOO"] = lo_background(items, "raw", lambda it: it["raw"][b0:b1].max(0))
    for d in ["high", "low"]:
        arms[f"raw entropy-w({d}) mean all"] = [(entropy_w(it["raw"], d)[:, None]*it["raw"]).sum(0) for it in items]
    if div is not None:
        w = div/div.sum(); gate = div >= DIV_GATE*div.max()
        arms["raw div-weighted mean all"] = [(w[:, None]*it["raw"]).sum(0) for it in items]
        arms["raw div-gated max"] = [it["raw"][gate].max(0) for it in items]
    if has_nw:
        arms["nw mean block"] = [it["nw"][b0:b1].mean(0) for it in items]
        arms["nw max block  (ref 56.5)"] = [it["nw"][b0:b1].max(0) for it in items]
        arms["heads mean block"] = [it["hs"][b0:b1].mean(0) for it in items]
        arms["heads max block  (ref 55.0)"] = [it["hs"][b0:b1].max(0) for it in items]
        arms["COMPOSITE heads+nw max block"] = [it["hs_nw"][b0:b1].max(0) for it in items]
        arms["COMPOSITE + denoise"] = [denoise(it["hs_nw"][b0:b1].max(0), it["gh"], it["gw"]) for it in items]
        arms["COMPOSITE + bgLOO"] = lo_background(items, "hs_nw", lambda it: it["hs_nw"][b0:b1].max(0))
        arms["COMPOSITE + bgLOO + denoise"] = [denoise(m, it["gh"], it["gw"]) for m, it in zip(arms["COMPOSITE + bgLOO"], items)]
        for d in ["high", "low"]:
            arms[f"nw entropy-w({d}) mean all"] = [(entropy_w(it["nw"], d)[:, None]*it["nw"]).sum(0) for it in items]
        if div is not None:
            w = div/div.sum(); gate = div >= DIV_GATE*div.max()
            arms["nw div-weighted mean all"] = [(w[:, None]*it["nw"]).sum(0) for it in items]
            arms["COMPOSITE div-gated max"] = [it["hs_nw"][gate].max(0) for it in items]
            arms["COMPOSITE div-gated max + bgLOO"] = lo_background(items, "x", lambda it: it["hs_nw"][gate].max(0))
    hits = {}
    for k, maps in arms.items():
        hits[k] = np.array([float(it["cov"][int(np.argmax(np.where(it["rm"], m, -1e9)))] >= COV) for it, m in zip(items, maps)])
    return hits

def report(name, hits, ref_keys):
    n = len(next(iter(hits.values()))); rng = np.random.default_rng(140)
    def ci(d):
        b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
        return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    base = hits["deployed: raw mean block"]
    ref = hits.get(ref_keys[0]) if ref_keys[0] in hits else None
    print(f"\n=== {name}  n={n} ===")
    print(f"  {'rule':>34} {'cov':>6}   vs deployed          " + (f"vs {ref_keys[1]}" if ref is not None else ""))
    for k, h in hits.items():
        m, lo, hi = ci(h-base); s = f"  {k:>34} {h.mean()*100:5.1f}%   {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 else ' '}"
        if ref is not None:
            m2, lo2, hi2 = ci(h-ref); s += f"   {m2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]{'*' if lo2 > 0 else ' '}"
        print(s)
    return hits

out = {}
for which, label in [("qwen3", "Qwen3-VL-2B"), ("qwen2", "Qwen2-VL-7B")]:
    items, blk, div = load_qwen(which)
    out[label] = report(label, run(items, blk, div, True), ("nw max block  (ref 56.5)", "nw-max"))
for tag, label in [("llavanext", "LLaVA-NeXT-7B"), ("onevision", "LLaVA-OneVision-7B")]:
    items, blk, div = load_llava(tag)
    out[label] = report(label, run(items, blk, None, False), ("raw max block", "raw-max"))
json.dump({k: {a: v.tolist() for a, v in h.items()} for k, h in out.items()}, open(f"{D}/phase140_ladder_hits.json", "w"))
print("\n* = CI clear of zero.  reference: learned head 63.4 (Q3) / 54.5 (Q2); LLaVA head 28.3 / 37.2 at W=0.25")
