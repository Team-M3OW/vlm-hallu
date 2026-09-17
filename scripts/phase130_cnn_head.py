"""
Phase 130 (Track 2): a more sophisticated head than the gradient-boosted tree.

PRE-REGISTERED, before any run
------------------------------
The incumbent is 150 shallow trees over 65 hand-built per-cell features. Known (phases 102, 112):
an MLP on the same features ties it, listwise loss hurts, cross-attention over the LAYER axis
collapses, adding max/normw features adds nothing. Capacity is not the limit at n=191.

Untried: convolution over the SPATIAL axes. The tree's only spatial feature is a 3x3 mean of the
deployed map; a conv sees the neighbourhood of every layer's map at once.

ARMS (all out-of-fold, GroupKFold(5) grouped by item, fold assignment identical to phase70.oof:
group permutation with rng 700+s, evaluation on ALL cells, ring mask, top-1 coverage at 0.5)
    deployed      argmax of the block-mean map                                   <- the field's rule
    gbt3          phase-70 GBT, 3 seeds                                          <- INCUMBENT
    gbt9          the same, 9 seeds (ensembling only)
    gbt3_flip     the same, training rows augmented by a horizontal flip of map + box + geometry
    cnn_mse       spatial CNN, channels = per-layer map + per-layer rank + 7 geometry, MSE on coverage
    cnn_pair      the same net, pairwise (RankNet) loss: covering cell must outrank a non-covering one
    ens           rank-average of gbt3 and the better CNN within each item

CNN: Conv(3x3, C->16) ReLU Conv(3x3, 16->16) ReLU Conv(1x1, 16->1). Adam 3e-3, wd 1e-3, <=40 epochs,
early stop on 20% of the TRAINING items (never the test fold), 3 seeds averaged, flip augmentation.

DECISION RULES
    A replacement must beat gbt3 with CI clear of zero on BOTH Qwen models AND keep the tree's
    cross-family record (3 of 4 at W=0.15: it must clear on LLaVA-NeXT and may fail only on
    LLaVA-OneVision, whose layers agree). Anything else -> keep the tree; report ensembling gains
    only if >1pp (the measured noise floor) with CI clear.
Primary W=0.15 (the cross-family bar); W=0.25 reported for the Qwen models as secondary.
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

torch.set_num_threads(4)
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
FILES = {
    "qwen3":     (f"{D}/phase30c_attn_maps_all.jsonl",       list(range(16, 27))),
    "qwen2":     (f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", list(range(15, 27))),
    "llavanext": (f"{D}/phase82_llavanext.jsonl",            None),
    "onevision": (f"{D}/phase82_onevision.jsonl",            None),
}
WHICH = sys.argv[1]
W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
COV_HIT, SEEDS, K = 0.5, 3, 5
OUT = f"{D}/phase130_{WHICH}_W{W}.json"


def coverage(cx, cy, gt, w):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2
    if x0 < 0: x0, x1 = 0.0, w
    if y0 < 0: y0, y1 = 0.0, w
    if x1 > 1: x0, x1 = 1 - w, 1.0
    if y1 > 1: y0, y1 = 1 - w, 1.0
    gx0, gy0, gx1, gy1 = gt
    inter = max(0.0, min(gx1, x1) - max(gx0, x0)) * max(0.0, min(gy1, y1) - max(gy0, y0))
    return inter / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def geom(gh, gw):
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = (yy + .5) / gh, (xx + .5) / gw
    return np.stack([fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                     np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                     (xx == gw - 1).astype(float), (yy == gh - 1).astype(float),
                     (xx == 0).astype(float)])                                  # (7, gh, gw)


def item_tensors(A, gh, gw, block, gt, w):
    """A: (NL, n) sum-normalised. Returns tabular feats (n, 2NL+9), CNN channels (2NL+7, gh, gw),
    coverage (n,), deployed map (n,)."""
    NL, n = A.shape
    M = A.reshape(NL, gh, gw)
    dep = M[block].mean(0)
    R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).reshape(NL, gh, gw)
    pad = np.pad(dep, 1, mode="edge")
    nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
    G7 = geom(gh, gw)
    F = np.concatenate([M.reshape(NL, -1).T, R.reshape(NL, -1).T,
                        nb.reshape(-1, 1), dep.reshape(-1, 1), G7.reshape(7, -1).T], 1)
    C = np.concatenate([M, R, G7], 0).astype(np.float32)
    fx, fy = G7[0].ravel(), G7[1].ravel()
    cov = np.array([coverage(float(fx[i]), float(fy[i]), gt, w) for i in range(n)])
    return F, C, cov, dep.ravel()


def load(which, w):
    path, block = FILES[which]
    rows = [json.loads(l) for l in open(path)]
    items = []
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh * gw != n: continue
        NL = len([k for k in r["attn"] if k.startswith("L")])
        blk = block if block is not None else list(range(int(.57 * NL), int(.93 * NL) + 1))
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        F, C, cov, dep = item_tensors(A, gh, gw, blk, r["gt_box_frac"], w)
        gt = r["gt_box_frac"]
        Af = A.reshape(NL, gh, gw)[:, :, ::-1].reshape(NL, -1)              # horizontal flip
        gtf = [1 - gt[2], gt[1], 1 - gt[0], gt[3]]
        Ff, Cf, covf, _ = item_tensors(np.ascontiguousarray(Af), gh, gw, blk, gtf, w)
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        items.append(dict(F=F, C=C, cov=cov, dep=dep, ring=rm.ravel(), gh=gh, gw=gw,
                          Ff=Ff, Cf=Cf, covf=covf))
    return items


def folds(N, seed):
    """Identical fold assignment to phase70.oof for a given seed."""
    rng = np.random.default_rng(700 + seed)
    gs = np.arange(N)
    perm = {g: i for i, g in enumerate(rng.permutation(gs))}
    Gp = np.array([perm[g] for g in gs])
    dummy = np.zeros((N, 1))
    return list(GroupKFold(K).split(dummy, dummy, Gp)), rng


def top1(items, scores):
    return np.array([float(it["cov"][int(np.argmax(np.where(it["ring"], s, -1e9)))] >= COV_HIT)
                     for it, s in zip(items, scores)])


# ---------------- GBT ----------------
def gbt_oof(items, seeds=3, flip=False, neg_per_item=30):
    N = len(items)
    S = [np.zeros(len(it["cov"])) for it in items]
    for s in range(seeds):
        fl, rng = folds(N, s)
        for tr, te in fl:
            X, Y = [], []
            for i in tr:
                it = items[i]
                for F, cov in ([(it["F"], it["cov"])] + ([(it["Ff"], it["covf"])] if flip else [])):
                    pos = np.where(cov > 0)[0]
                    negpool = np.where(cov <= 0)[0]
                    neg = rng.choice(negpool, size=min(len(negpool), neg_per_item), replace=False)
                    idx = np.concatenate([pos, neg])
                    X.append(F[idx]); Y.append(cov[idx])
            X, Y = np.vstack(X), np.concatenate(Y)
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                              random_state=s).fit(X, Y)
            for i in te:
                S[i] += m.predict(items[i]["F"])
    return [s / seeds for s in S]


# ---------------- CNN ----------------
class Net(nn.Module):
    def __init__(self, cin, ch=16):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(ch, 1, 1))
    def forward(self, x): return self.f(x)[:, 0]


def cnn_oof(items, loss_kind, seeds=3, epochs=40, lr=3e-3, wd=1e-3):
    N = len(items); cin = items[0]["C"].shape[0]
    S = [np.zeros(len(it["cov"])) for it in items]
    T = [torch.tensor(it["C"])[None] for it in items]
    Tf = [torch.tensor(it["Cf"])[None] for it in items]
    Y = [torch.tensor(it["cov"].reshape(it["gh"], it["gw"]), dtype=torch.float32)[None] for it in items]
    Yf = [torch.tensor(it["covf"].reshape(it["gh"], it["gw"]), dtype=torch.float32)[None] for it in items]
    for s in range(seeds):
        fl, rng = folds(N, s)
        for tr, te in fl:
            torch.manual_seed(1000 + s)
            tr = np.array(tr); rng.shuffle(tr)
            nval = max(8, len(tr) // 5); val, trn = tr[:nval], tr[nval:]
            net = Net(cin); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            best, best_state, bad = -1, None, 0
            for ep in range(epochs):
                net.train()
                order = rng.permutation(trn)
                for i in order:
                    for x, y in ((T[i], Y[i]), (Tf[i], Yf[i])):
                        out = net(x)[0].reshape(-1); yy = y[0].reshape(-1)
                        if loss_kind == "mse":
                            loss = ((out - yy) ** 2).mean()
                        else:
                            pos = torch.where(yy >= COV_HIT)[0]; neg = torch.where(yy < COV_HIT)[0]
                            if len(pos) == 0 or len(neg) == 0: continue
                            pi = pos[torch.randint(len(pos), (64,))]
                            ni = neg[torch.randint(len(neg), (64,))]
                            loss = torch.nn.functional.softplus(-(out[pi] - out[ni])).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    vs = [net(T[i])[0].reshape(-1).numpy() for i in val]
                acc = top1([items[i] for i in val], vs).mean()
                if acc > best + 1e-9:
                    best, bad = acc, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8: break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                for i in te:
                    S[i] += net(T[i])[0].reshape(-1).numpy()
    return [s / seeds for s in S]


def rankavg(A, B):
    out = []
    for a, b in zip(A, B):
        ra = np.argsort(np.argsort(a)) / max(len(a) - 1, 1)
        rb = np.argsort(np.argsort(b)) / max(len(b) - 1, 1)
        out.append(ra + rb)
    return out


def main():
    t0 = time.time()
    items = load(WHICH, W)
    N = len(items)
    print(f"{WHICH} W={W}: {N} items, channels {items[0]['C'].shape[0]}, feats {items[0]['F'].shape[1]}", flush=True)
    res = {}
    res["deployed"] = top1(items, [it["dep"] for it in items])
    res["gbt3"] = top1(items, gbt_oof(items, 3));            print(f"  gbt3 {res['gbt3'].mean()*100:.1f}  t={time.time()-t0:.0f}s", flush=True)
    res["gbt9"] = top1(items, gbt_oof(items, 9));            print(f"  gbt9 {res['gbt9'].mean()*100:.1f}", flush=True)
    res["gbt3_flip"] = top1(items, gbt_oof(items, 3, flip=True)); print(f"  gbt3_flip {res['gbt3_flip'].mean()*100:.1f}", flush=True)
    gbt_scores = gbt_oof(items, 3)
    cnn_m = cnn_oof(items, "mse");  res["cnn_mse"] = top1(items, cnn_m);   print(f"  cnn_mse {res['cnn_mse'].mean()*100:.1f}  t={time.time()-t0:.0f}s", flush=True)
    cnn_p = cnn_oof(items, "pair"); res["cnn_pair"] = top1(items, cnn_p); print(f"  cnn_pair {res['cnn_pair'].mean()*100:.1f}  t={time.time()-t0:.0f}s", flush=True)
    best_cnn = cnn_p if res["cnn_pair"].mean() >= res["cnn_mse"].mean() else cnn_m
    res["ens"] = top1(items, rankavg(gbt_scores, best_cnn));  print(f"  ens {res['ens'].mean()*100:.1f}", flush=True)
    rng = np.random.default_rng(130)
    def ci(d):
        b = np.array([d[rng.integers(0, N, N)].mean() for _ in range(6000)])
        return float(d.mean() * 100), float(np.percentile(b, 2.5) * 100), float(np.percentile(b, 97.5) * 100)
    out = {"which": WHICH, "W": W, "n": N, "acc": {k: float(v.mean() * 100) for k, v in res.items()},
           "vs_gbt3": {k: ci(v - res["gbt3"]) for k, v in res.items()},
           "vs_deployed": {k: ci(v - res["deployed"]) for k, v in res.items()}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{'arm':>10} {'cov':>6}   vs gbt3               vs deployed")
    for k in res:
        a, lo, hi = out["vs_gbt3"][k]; a2, lo2, hi2 = out["vs_deployed"][k]
        print(f"{k:>10} {res[k].mean()*100:5.1f}%  {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]   {a2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]")
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


main()
