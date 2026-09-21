"""
Phase 184a: ARE THE RIGHT RETRIEVAL HEADS EXAMPLE-DEPENDENT? (CPU only, no GPU, no intervention)

VRH (arXiv 2608.27417) claims visual retrieval heads are sparse, causal, UNIVERSAL and "stably
detectable" -- as few as 5-10 examples recover nearly the same top-ranked set as 200. That is a claim
about a FIXED head set. This tests the opposite: per (image, question, DPR region), score heads

    S_h = G_h * V_h * R_h
    G_h  global VRH quality        (fold-honest referent-region mass, TRAINING items only)
    V_h  visual sensitivity HERE   (phase 108's per-item S_v: share of this head's mass on visual tokens)
    R_h  agreement with DPR's R    (this head's attention mass inside the W=0.25 window)

and takes H* = TopK(S_h) per example.

TWO QUESTIONS, both answerable without touching the model:
  Q1 STABILITY.  How much does H* differ across examples, and from the global set? If per-example sets
     are ~the global set, the dynamic idea is empty and VRH's universality claim stands as written.
  Q2 UTILITY.    Does the per-example set make a BETTER VERIFIER than the fixed set? §20H's verifier
     (fixed heads) reads AUROC 0.762 / 0.755 for "does the window cover the evidence"; a dynamic set
     that beats it is a concrete improvement on a signal that already converts (§20I gate).

⚠ CIRCULARITY GUARD. R_h uses the window, and Q2's verifier also scores the window. Selecting heads by
R_h and then scoring R with them is circular, so Q2 reports BOTH S_h = G*V*R and the non-circular
S_h = G*V. Only the latter is eligible to be called a better verifier.
PRE-REGISTERED: Q1 is descriptive. Q2 adopts the dynamic verifier only if G*V beats the fixed set with
a paired CI clear of zero on BOTH models.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W = 0.25
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
SVF = {"qwen3": f"{D}/phase108_heads_qwen3.jsonl", "qwen2": f"{D}/phase108_heads_qwen2.jsonl"}
rng = np.random.default_rng(184)


def auroc(s, p):
    p = np.asarray(p, bool)
    if p.all() or not p.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum() - n1*(n1+1)/2) / (n1*n0)


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    SV = {}
    with open(SVF[which]) as f:
        for l in f:
            r = json.loads(l); SV[r["question_id_full"]] = np.asarray(r["Sv"], np.float32)
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in SV]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    R = np.zeros((N, NL, H), np.float32)     # mass in the DPR window
    Gref = np.zeros((N, NL, H), np.float32)  # mass on the referent (for the global G_h)
    V = np.zeros((N, NL, H), np.float32)
    cov = np.zeros(N, bool)
    for i, e in enumerate(idx):
        q = e["question_id_full"]; p = props[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        cx, cy = p["head"]
        x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
        win = (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
        g = p["gt_box_frac"]
        ref = (fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3])
        A = imap(e)
        R[i] = A[:, :, win].sum(-1); Gref[i] = A[:, :, ref].sum(-1) if ref.any() else 0.0
        V[i] = SV[q]; cov[i] = p["head_cov"] >= 0.5

    k = max(1, int(round(0.25*H)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))

    def topk_mask(score2d):
        m = np.zeros((NL, H), bool)
        for l in range(NL): m[l, np.argsort(-score2d[l])[:k]] = True
        return m

    fixed = np.zeros((N, NL, H), bool)
    dyn_gvr = np.zeros((N, NL, H), bool)
    dyn_gv = np.zeros((N, NL, H), bool)
    for tr, te in folds:
        G_ = Gref[tr].mean(0)
        fm = topk_mask(G_)
        for i in te:
            fixed[i] = fm
            dyn_gvr[i] = topk_mask(G_ * V[i] * R[i])
            dyn_gv[i] = topk_mask(G_ * V[i])

    # ---- Q1 stability
    def jac(a, b): return (a & b).sum() / max((a | b).sum(), 1)
    ov_fixed = np.array([jac(dyn_gv[i], fixed[i]) for i in range(N)])
    ov_pair = np.array([jac(dyn_gv[i], dyn_gv[j]) for i, j in
                        rng.integers(0, N, (400, 2)) if i != j])
    ov_gvr = np.array([jac(dyn_gvr[i], fixed[i]) for i in range(N)])
    print(f"\n{'='*86}\n{which}  n={N}  {NL} layers x {H} heads, top-{k} per layer")
    print(f"  Q1 STABILITY (Jaccard overlap of selected head sets)")
    print(f"     dynamic G*V   vs the fixed global set : {ov_fixed.mean():.3f}  "
          f"(sd {ov_fixed.std():.3f})")
    print(f"     dynamic G*V*R vs the fixed global set : {ov_gvr.mean():.3f}")
    print(f"     dynamic G*V   between two random items: {ov_pair.mean():.3f}")
    print(f"     -> {'EXAMPLE-DEPENDENT' if ov_fixed.mean() < 0.8 else 'essentially the fixed set'}")

    # ---- Q2 utility as a verifier
    print(f"  Q2 VERIFIER AUROC for 'DPR window covers the evidence' (base rate {100*cov.mean():.1f}%)")
    scores = {}
    for nm, masks in [("fixed heads (§20H)", fixed), ("dynamic G*V (non-circular)", dyn_gv),
                      ("dynamic G*V*R (CIRCULAR)", dyn_gvr)]:
        s = np.zeros(N)
        for i, e in enumerate(idx):
            A = imap(e); m = masks[i]
            sel = (A * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1)
            g_ = sel[GATE[which]].mean(0)
            s[i] = R[i][GATE[which]][m[GATE[which]]].sum() if False else 0
            gh, gw = e["grid"]; n = e["n_cells"]
            yy, xx = np.mgrid[0:gh, 0:gw]
            fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
            p = props[e["question_id_full"]]; cx, cy = p["head"]
            x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
            win = (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
            s[i] = g_[win].sum() / max(g_.sum(), 1e-12)
        scores[nm] = s
        print(f"     {nm:30} AUROC {auroc(s, cov):.3f}")
    base = scores["fixed heads (§20H)"]
    for nm in ["dynamic G*V (non-circular)", "dynamic G*V*R (CIRCULAR)"]:
        d = auroc(scores[nm], cov) - auroc(base, cov)
        b = []
        for _ in range(4000):
            i = rng.integers(0, N, N)
            x = auroc(scores[nm][i], cov[i]) - auroc(base[i], cov[i])
            if not np.isnan(x): b.append(x)
        lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
        print(f"     {nm:30} ΔAUROC vs fixed {d:+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"{'  BETTER' if lo > 0 else ''}")
