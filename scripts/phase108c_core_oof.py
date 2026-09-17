"""
Phase 108c: CoRe's contrastive head criterion, selected OUT-OF-FOLD -- the fair test.

Phase 108's CoRe arms scored heads using each item's own GT box, so its 60.7% is an in-sample upper
bound, not a result. CoRe's actual protocol selects heads on a held-out set and transfers them; this
reproduces that -- the head set for each fold is chosen by averaging the contrastive score over the
TRAINING items only, then applied unchanged to the held-out items.

    S_core(h,l) = softmax over {mean attention on evidence cells, mean attention on the rest}
                  at temperature t, i.e. how much this head PREFERS evidence to non-evidence.
    S_v(h,l)    = share of this head's mass on visual tokens -- Lu et al. / QR-style, ABSOLUTE,
                  and label-free.

The comparison CoRe makes in text, made here in vision, with one asymmetry stated up front: S_v needs
no labels at all, while S_core needs boxes on the training fold. A contrastive win must therefore be
large enough to justify that cost, and a contrastive LOSS is decisive against it.

Runs entirely offline from phase 108b's per-head dump; no GPU.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
B0, B1 = (16, 27) if WHICH == "qwen3" else (15, 27)
P70.W, COV, K, T = 0.25, 0.5, 5, 0.05

z = np.load(f"{D}/phase108b_perhead_{WHICH}.npz")
meta = json.load(open(f"{D}/phase108b_perhead_{WHICH}_meta.json"))
N = len(meta)
M, SV, SC, COVv, RING = [], [], [], [], []
for i, mt in enumerate(meta):
    v = z[f"m{i}"].astype(np.float32)                      # (NL, H, n_img)
    gh, gw = mt["grid"]; n = mt["n_img"]; gt = mt["gt_box_frac"]
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
    inb = ((fx >= gt[0]) & (fx <= gt[2]) & (fy >= gt[1]) & (fy <= gt[3])).astype(np.float32)
    if inb.sum() == 0:
        d2 = (fx-(gt[0]+gt[2])/2)**2 + (fy-(gt[1]+gt[3])/2)**2
        inb = np.zeros(n, np.float32); inb[int(d2.argmin())] = 1.0
    pos = (v*inb).sum(-1)/max(inb.sum(), 1); neg = (v*(1-inb)).sum(-1)/max((1-inb).sum(), 1)
    e = np.exp(np.stack([pos, neg])/T - (np.stack([pos, neg])/T).max(0, keepdims=True))
    SC.append(e[0]/e.sum(0))                               # (NL, H)
    SV.append(z[f"sv{i}"].astype(np.float32))
    M.append(v)
    COVv.append(np.array([P70.coverage(float(fx[j]), float(fy[j]), gt) for j in range(n)]))
    rm = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
    else: rm[:] = True
    RING.append(rm.ravel())
H = M[0].shape[1]
print(f"{WHICH}: {N} items, {M[0].shape[0]} layers, {H} heads")

fold = np.zeros(N, int)
order = np.arange(N); np.random.default_rng(7).shuffle(order)
for i, o in enumerate(order): fold[o] = i % K

def run(score_list, q, agg, oof):
    """score_list: per-item (NL,H) head scores. q: top fraction. oof: select on training folds only."""
    hit = np.zeros(N)
    if oof:
        for k in range(K):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            S = np.mean([score_list[i] for i in tr], 0)                     # (NL,H) fold-honest
            kk = max(1, int(round(q*H)))
            idx = np.argsort(-S, axis=1)[:, :kk]
            for i in te: hit[i] = one(i, idx, agg)
    else:
        for i in range(N):
            kk = max(1, int(round(q*H)))
            idx = np.argsort(-score_list[i], axis=1)[:, :kk]
            hit[i] = one(i, idx, agg)
    return hit

def one(i, idx, agg):
    v = M[i]; NL = v.shape[0]
    sel = np.stack([v[l, idx[l]].mean(0) for l in range(NL)])
    m = agg(sel)
    return float(COVv[i][int(np.argmax(np.where(RING[i], m, -1e9)))] >= COV)

MEAN = lambda A: A[B0:B1].mean(0)
MAXX = lambda A: A[B0:B1].max(0)
allh = np.array([one(i, np.tile(np.arange(H), (M[i].shape[0], 1)), MEAN) for i in range(N)])
rng = np.random.default_rng(1083)
def ci(d):
    b = np.array([d[rng.integers(0, N, N)].mean() for _ in range(6000)])
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

print(f"\n  all heads, mean over block: {allh.mean()*100:.1f}%  (the incumbent)\n")
print(f"  {'rule':>28} {'cov':>7}   vs incumbent")
for tag, sl, q, oof in [("S_v top10 (label-free)", SV, .10, False),
                        ("S_v top25 (label-free)", SV, .25, False),
                        ("CoRe top10 IN-SAMPLE (leaky)", SC, .10, False),
                        ("CoRe top25 IN-SAMPLE (leaky)", SC, .25, False),
                        ("CoRe top10 OUT-OF-FOLD", SC, .10, True),
                        ("CoRe top25 OUT-OF-FOLD", SC, .25, True)]:
    for agg, an in [(MEAN, "mean"), (MAXX, "max")]:
        h = run(sl, q, agg, oof)
        m, lo, hi = ci(h - allh)
        print(f"  {tag+' + '+an:>28} {h.mean()*100:6.1f}%   {m:+6.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''}")
print("\n  reference: learned head 63.4% | normw x max 56.5% | deployed argmax 46.1%")
