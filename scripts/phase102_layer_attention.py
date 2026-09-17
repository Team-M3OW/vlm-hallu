"""
Phase 102: is the head's limit CAPACITY, OBJECTIVE, or DATA?

The incumbent is 150 shallow trees over 65 features, regressing per-cell coverage. Two things about
that are open:

  ARCHITECTURE  a GBT learns ONE global function of the 28-layer profile; its splits are shared
                across items. Cross-attention over the layer readouts can weight layers per cell,
                conditioned on that cell's own profile.
  OBJECTIVE     we regress coverage per cell independently, then DEPLOY argmax over cells. The loss
                and the metric are different. A listwise softmax over the cells within an item,
                pushing mass onto covering cells, optimises what we actually use.

2x2 isolates them, with the incumbent alongside. Same folds, same ring mask, same negative handling,
top-1 coverage evaluated on ALL cells exactly as deployed. Both models.

PRIOR, stated before the run: phase 45 found a fitted multi-feature model LOST to a single scalar at
n=191, so capacity has already failed once here. If nothing beats the GBT, the binding constraint is
data and this direction closes with a number instead of an opinion.

Conversion rate for context: 39.3 -> 52.9% coverage bought +8.4pp end-task, about 0.62pp of accuracy
per pp of coverage. Ceiling is 88.5%.
"""
import json, sys, math
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
COV_HIT = 0.5


def build(model):
    """Returns per-cell features split into (layer stack, non-layer tail), plus labels/groups."""
    if model == "qwen3":
        src, NL, BLOCK = f"{D}/phase30c_attn_maps_all.jsonl", 28, list(range(16, 27))
    else:
        src, NL, BLOCK = f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", 28, list(range(15, 27))
    rows = [json.loads(l) for l in open(src)]
    AL, TL, Y, G, DEP, RING = [], [], [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        R = np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)
        M = A.reshape(NL, gh, gw)
        dep = M[BLOCK].mean(0)
        pad = np.pad(dep, 1, mode="edge")
        nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = (yy + .5) / gh, (xx + .5) / gw
        AL.append(np.stack([A, R], -1).transpose(1, 0, 2))          # (cells, NL, 2)
        TL.append(np.stack([nb.ravel(), dep.ravel(), fx.ravel(), fy.ravel(),
                            np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2).ravel(),
                            np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)).ravel(),
                            (xx == gw - 1).astype(float).ravel(), (yy == gh - 1).astype(float).ravel(),
                            (xx == 0).astype(float).ravel()], 1))   # (cells, 9)
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                           for i in range(n)]))
        G.append(np.full(n, gi)); DEP.append(dep.ravel())
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING.append(m.ravel())
    return (np.concatenate(AL), np.concatenate(TL), np.concatenate(Y),
            np.concatenate(G), DEP, RING, rows, NL)


class Net(nn.Module):
    def __init__(self, NL, arch, d=32):
        super().__init__()
        self.arch = arch
        if arch == "attn":
            self.emb = nn.Linear(2, d)
            self.lay = nn.Parameter(torch.randn(NL, d) * .02)
            self.q = nn.Parameter(torch.randn(1, d) * .02)
            self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
            self.d = d
            head_in = d + 9
        else:
            head_in = NL * 2 + 9
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, a, t):                       # a:(B,NL,2)  t:(B,9)
        if self.arch == "attn":
            h = self.emb(a) + self.lay             # (B,NL,d)
            w = (self.k(h) @ self.q.T).squeeze(-1) / math.sqrt(self.d)   # (B,NL)
            w = torch.softmax(w, -1)
            z = (w.unsqueeze(-1) * self.v(h)).sum(1)                     # (B,d)
        else:
            z = a.reshape(a.shape[0], -1)
        return self.head(torch.cat([z, t], -1)).squeeze(-1)


def fit_eval(AL, TL, Y, G, DEP, RING, NL, arch, loss_kind, seed=0, epochs=30):
    P = np.zeros(len(Y))
    gs = np.unique(G)
    rng = np.random.default_rng(700 + seed)
    perm = {g: i for i, g in enumerate(rng.permutation(gs))}
    Gp = np.vectorize(perm.get)(G)
    a_all = torch.tensor(AL, dtype=torch.float32, device=DEV)
    t_all = torch.tensor(TL, dtype=torch.float32, device=DEV)
    y_all = torch.tensor(Y, dtype=torch.float32, device=DEV)
    for tr, te in GroupKFold(5).split(AL, Y, Gp):
        torch.manual_seed(seed)
        net = Net(NL, arch).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        tr_items = np.unique(G[tr])
        idx_by_item = {g: np.where(G == g)[0] for g in tr_items}
        for ep in range(epochs):
            order = rng.permutation(tr_items)
            if loss_kind == "mse":
                for b in range(0, len(order), 16):
                    ii = np.concatenate([idx_by_item[g] for g in order[b:b + 16]])
                    opt.zero_grad()
                    l = ((net(a_all[ii], t_all[ii]) - y_all[ii]) ** 2).mean()
                    l.backward(); opt.step()
            else:                                   # listwise: softmax over cells within an item
                for g in order:
                    ii = idx_by_item[g]
                    yy = y_all[ii]
                    if (yy >= COV_HIT).sum() == 0: continue
                    opt.zero_grad()
                    lp = torch.log_softmax(net(a_all[ii], t_all[ii]), 0)
                    l = -torch.logsumexp(lp[yy >= COV_HIT], 0)
                    l.backward(); opt.step()
        with torch.no_grad():
            P[te] = net(a_all[te], t_all[te]).cpu().numpy()
    return P


def topcov(score, G, Y, DEP, RING):
    hit = []
    for gi in np.unique(G):
        m = G == gi
        s = np.where(RING[gi], score[m], -1e9)
        hit.append(1.0 * (Y[m][int(np.argmax(s))] >= COV_HIT))
    return float(np.mean(hit))


for model in ["qwen3", "qwen2"]:
    AL, TL, Y, G, DEP, RING, rows, NL = build(model)
    print(f"\n=== {model}: {len(rows)} items, {AL.shape[0]} cells, {NL} layers ===", flush=True)
    dep_hit = topcov(np.concatenate(DEP), G, Y, DEP, RING)
    X65 = np.concatenate([AL[:, :, 0], AL[:, :, 1], TL], 1)
    gbt = topcov(P70.oof(X65, Y, G, seeds=1), G, Y, DEP, RING)
    print(f"\n  {'deployed argmax (block-mean)':>34} {dep_hit*100:5.1f}%")
    print(f"  {'GBT, MSE  (INCUMBENT)':>34} {gbt*100:5.1f}%", flush=True)
    for arch in ["mlp", "attn"]:
        for lk in ["mse", "list"]:
            s = fit_eval(AL, TL, Y, G, DEP, RING, NL, arch, lk)
            nm = f"{'LayerAttn' if arch=='attn' else 'MLP'}, {'listwise' if lk=='list' else 'MSE'}"
            print(f"  {nm:>34} {topcov(s,G,Y,DEP,RING)*100:5.1f}%   "
                  f"({'+' if topcov(s,G,Y,DEP,RING)>gbt else ''}{(topcov(s,G,Y,DEP,RING)-gbt)*100:.1f} vs GBT)",
                  flush=True)
