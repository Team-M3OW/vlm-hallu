"""
Phase 234 (CPU): is there a deployable per-item policy that does not LOSE on either stratum?

Constraint 7 permits adaptation only from the attention map. This re-evaluates the §18E map-only
skip with per-stratum results and with AVR as the alternative arm (the newer, scene-level policy):

  signal  (per item, label-free, from the pass-1 map):
    disp      top-1 share of the ring-masked block-mean map          (§18B statistic)
    vrh       attention mass inside the DWA cell's W=0.25 window     (§20H verifier statistic)
    ridge_top top-decile mass share of the ridge score map
  threshold (per fold, no free parameter): median of the TRAINING-fold signal, OOF GroupKFold(5)
  action    signal > median -> DWA crop@300 ; else -> uniform@600 (or AVR@900 where available)
  outcome   the recorded per-item correctness of the chosen arm (phase225 grid, identical items)

Reports pooled and per-stratum deltas vs the bar, alongside always-DWA / always-AVR / per-item oracle.
"""
import json, os, numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25
MAPS = {"qwen3_2b": ("qwen3", "phase30c_attn_maps_all.jsonl", (16, 27)),
        "qwen2_7b": ("qwen2", "phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}
SEED = 234
rng = np.random.default_rng(SEED)


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def acc1(r, a):
    if a not in r.get("probs", {}) and a not in r.get("preds", {}):
        return None
    if "kind" in r and r["kind"].startswith("mcq"):
        return 1.0 if int(np.argmax(r["probs"][a])) == int(r["gold"]) else 0.0
    import sys; sys.path.insert(0, f"{D}/scripts"); import benchmarks as B
    p = r["preds"][a]; gs = r["gold"] if isinstance(r["gold"], (list, tuple)) else [r["gold"]]
    return 1.0 if any(B.norm(g) and B.norm(g) == B.norm(p) for g in gs) else 0.0


def boot(d, Bn=8000):
    d = np.asarray(d, float); n = len(d)
    if n < 8: return d.mean() * 100, float("nan"), float("nan"), n
    m = d[rng.integers(0, n, (Bn, n))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100, n


for mk, (wt, fn, BLK) in MAPS.items():
    grid = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/phase225_{mk}_vstar.jsonl")}
    maps = {}
    for line in open(f"{D}/data/{fn}"):
        r = json.loads(line)
        if "attn" in r and "grid" in r: maps[r["question_id_full"]] = r
    fs = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_{wt}.jsonl")}
    qids = [q for q in grid if q in maps and q in fs and "uniform@lo" in grid[q].get("probs", {})]
    sig = {}
    for q in qids:
        gh, gw = maps[q]["grid"]; nc = gh * gw
        A = np.stack([np.asarray(maps[q]["attn"][f"L{l}"], float) for l in range(28)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A[BLK[0]:BLK[1]].mean(0)
        rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
        disp = float(np.max(np.where(rm, M, -1e9))) / max(M.sum(), 1e-12)
        # ridge score + DWA cell from the deployed score grid
        S = np.array(fs[q]["score"], float).ravel()
        if S.size != nc: continue
        cx, cy = fs[q]["ridge_cell"]
        inwin = ((np.abs((np.arange(nc) % gw + .5) / gw - cx) <= W / 2) &
                 (np.abs((np.arange(nc) // gw + .5) / gh - cy) <= W / 2))
        vrh = float(M[inwin].sum()) / max(M.sum(), 1e-12)
        pos = np.maximum(S, 0); top = np.sort(pos)[-max(1, nc // 10):]
        sig[q] = {"disp": disp, "vrh": vrh, "ridge_top": float(top.sum() / max(pos.sum(), 1e-12))}

    strata = np.array([grid[q]["stratum"] for q in qids])
    arms = [a for a in ("uniform@lo", "dwa_t", "avr", "block") if all(acc1(grid[q], a) is not None for q in qids)]
    print(f"\n================ {mk}  n={len(qids)}  arms={arms} ================")
    base = np.array([acc1(grid[q], "uniform@lo") for q in qids])
    for name in ("disp", "vrh", "ridge_top"):
        s = np.array([sig[q][name] for q in qids])
        # OOF median threshold, GroupKFold(5) by item; assignment uses the chosen alt arm
        alt = "avr" if "avr" in arms else "uniform@lo"
        chosen = np.zeros(len(qids), dtype=bool)
        for f in range(5):
            te = np.arange(f, len(qids), 5); tr = np.setdiff1d(np.arange(len(qids)), te)
            thr = np.median(s[tr]); chosen[te] = s[te] > thr
        out = np.array([acc1(grid[q], "dwa_t") if chosen[i] else acc1(grid[q], alt) for i, q in enumerate(qids)])
        d = out - base
        m, lo, hi, n = boot(d)
        dm, dlo, dhi, _ = boot(np.array([acc1(grid[q], "dwa_t") for q in qids]) - base)
        routed = 100 * chosen.mean()
        per = []
        for st in ("direct_attributes", "relative_position"):
            msk = strata == st
            pm, plo, phi, _ = boot(d[msk])
            per.append(f"{'single' if st.startswith('direct') else 'cross'}: {pm:+.1f} [{plo:+.1f},{phi:+.1f}] route {100*chosen[msk].mean():.0f}%")
        oracle = np.array([max(acc1(grid[q], "dwa_t"), acc1(grid[q], alt)) for q in qids]) - base
        om, olo, ohi, _ = boot(oracle)
        # controls: (a) random routing at the same rate, (b) shuffled signal
        rnd = np.array([acc1(grid[q], "dwa_t") if rng.random() < chosen.mean() else acc1(grid[q], alt) for i, q in enumerate(qids)]) - base
        rm, rlo, rhi, _ = boot(rnd)
        sh = s.copy(); rng.shuffle(sh)
        ch2 = sh > np.median(sh)
        shuf = np.array([acc1(grid[q], "dwa_t") if ch2[i] else acc1(grid[q], alt) for i, q in enumerate(qids)]) - base
        sm, slo, shi, _ = boot(shuf)
        av = np.array([acc1(grid[q], alt) for q in qids]) - base
        am, alo, ahi, _ = boot(av)
        print(f"  {name:10s} routed {routed:4.0f}%  policy {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' *' if lo>0 else ''}"
              f"  | DWA {dm:+5.1f}  {alt.upper()} {am:+5.1f}  random-route {rm:+5.1f} [{rlo:+5.1f},{rhi:+5.1f}]  shuffled-signal {sm:+5.1f}")
        print(f"             " + "  ".join(per) + f"   | per-item oracle {om:+5.1f}")
