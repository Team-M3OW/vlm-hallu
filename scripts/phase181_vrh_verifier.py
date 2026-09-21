"""
Phase 181: VRH AS A REGION VERIFIER, not as another localisation map.

SS20A closed VRH as a localiser input to DPR (1 of 2: helps Qwen2, nothing on Qwen3). This tests the
other function: given a region DPR already proposed, does VRH tell us whether the model is actually
RETRIEVING from it? That is a scoring problem over one region, not a ranking problem over cells, and
it has never been run.

  verifier score  V = attention mass landing inside the DPR window, summed over VRH-SELECTED heads
                      (heads selected fold-honestly by referent-region mass on TRAINING items only)

TARGETS (AUROC, higher = the score separates the classes)
  P1  does the proposed window actually cover the evidence?   <- "is this region real evidence"
  P2  is the DPR crop arm's answer correct?                   <- does verification predict the fix working
  P3  is the ORIGINAL full-image answer correct?              <- the selective-crop detector use

BASELINES, without which an AUROC is meaningless
  all-heads mass in the same window   <- the naive version: does head SELECTION add anything?
  top1_frac of the gated map          <- the SS18E dispersion signal that already routes at +5.8/+5.8
  window coverage by chance           <- shuffled-label control, must land at 0.500

NOTE ON THE CEILING. Phase 77 (SS14H) measured the oracle of answer-changing deletion methods at
+2.6pp [-0.5,+5.8], so a verifier cannot be worth much as an answer-changer. It is tested here as a
DETECTOR, where that bound does not apply.
CPU only; reads phase 170's per-head dump and the existing OOF proposals.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W = 0.25
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
OUTC = {"qwen3": (f"{D}/phase78_w_sweep.jsonl", "head@0.25", "uniform@300"),
        "qwen2": (f"{D}/phase97m_merged_qwen2vl.jsonl", "head@0.25", "uniform@300")}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
rng = np.random.default_rng(181)


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any(): return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1*(n1+1)/2) / (n1*n0))


def auroc_ci(score, pos, B=4000):
    a = auroc(score, pos); n = len(score)
    b = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = auroc(score[i], np.asarray(pos)[i])
        if not np.isnan(v): b.append(v)
    return a, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    path, crop_arm, base_arm = OUTC[which]
    out = {r["question_id_full"]: r for r in (json.loads(l) for l in open(path))}
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in out]
    N = len(idx)

    def imap(e):
        return buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    # per-item: cells inside the DPR window, cells inside the GT box (for fold-honest head scoring)
    inwin, inref, cov_ok, crop_ok, base_ok, disp = [], [], [], [], [], []
    for e in idx:
        q = e["question_id_full"]; p = props[q]; r = out[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        cx, cy = p["head"]
        x0, x1 = min(max(0, cx-W/2), 1-W), min(max(0, cx-W/2), 1-W)+W
        y0, y1 = min(max(0, cy-W/2), 1-W), min(max(0, cy-W/2), 1-W)+W
        inwin.append((fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1))
        g = p["gt_box_frac"]
        inref.append((fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3]))
        cov_ok.append(p["head_cov"] >= 0.5)
        lab = r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        crop_ok.append(int(np.argmax(r["probs"][crop_arm]) == lab))
        base_ok.append(int(np.argmax(r["probs"][base_arm]) == lab))
        A = imap(e); s = A.mean(1)[GATE[which]].max(0)
        disp.append(float(s.max() / max(s.sum(), 1e-12)))
    cov_ok, crop_ok, base_ok = map(np.array, (cov_ok, crop_ok, base_ok))
    disp = np.array(disp)

    # fold-honest VRH head selection, then verifier scores
    k = max(1, int(round(0.25*H)))
    ref_mass = np.zeros((N, NL, H), np.float32)
    for i, e in enumerate(idx):
        A = imap(e)
        ref_mass[i] = A[:, :, inref[i]].sum(-1) if inref[i].any() else 0.0
    V_vrh, V_all = np.zeros(N), np.zeros(N)
    G = np.arange(N)
    for tr, te in GroupKFold(5).split(G.reshape(-1, 1), G, G):
        sc = ref_mass[tr].mean(0)
        m = np.zeros((NL, H), bool)
        for l in range(NL): m[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(idx[i])
            sel = (A * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1)
            V_vrh[i] = sel[GATE[which]].mean(0)[inwin[i]].sum() / max(sel[GATE[which]].mean(0).sum(), 1e-12)
            al = A.mean(1)
            V_all[i] = al[GATE[which]].mean(0)[inwin[i]].sum() / max(al[GATE[which]].mean(0).sum(), 1e-12)

    print(f"\n{'='*86}\n{which}  n={N}   VRH-selected heads: top-25% per layer ({k}/{H}), fold-honest")
    print(f"  base rates: window covers evidence {100*cov_ok.mean():.1f}% | "
          f"crop arm correct {100*crop_ok.mean():.1f}% | full-image correct {100*base_ok.mean():.1f}%")
    shuf = rng.permutation(V_vrh)
    for tname, target in [("P1 window covers evidence", cov_ok),
                          ("P2 DPR crop answer correct", crop_ok),
                          ("P3 full-image answer correct", base_ok)]:
        print(f"  --- {tname}")
        for sname, s in [("VRH mass in window", V_vrh), ("all-head mass in window", V_all),
                         ("map dispersion (top1_frac)", disp), ("shuffled CONTROL", shuf)]:
            a, lo, hi = auroc_ci(s, target)
            print(f"      {sname:28} AUROC {a:.3f} [{lo:.3f},{hi:.3f}]"
                  f"{'  ✔' if lo > 0.5 else ''}")
