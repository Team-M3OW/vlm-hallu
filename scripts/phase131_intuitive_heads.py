"""
Phase 131: replace the GBT with something INTUITIVE -- at parity or better.

Four rounds (phases 45, 102, 112, 130) show more expressive heads do not beat the tree at n=191: the
limit is boxed training items, not model class. So the question here is different: can a model whose
structure is READABLE match the tree? What the tree learned (SS14C(b)): the depth profile is worth
+7.3pp, sink indicators +0.0 -- it exploits disagreement across layers. Candidates with exactly that
inductive bias:

    A  layer-gated log-linear    s = sum_l w_l*log A_l + sum_l v_l*rank_l + u.geometry   (L1, ~63 params)
                                 "which layers, how much" -- w_l is the weight of layer l
    B  additive GAM              HistGradientBoosting with interaction_cst = one feature per group:
                                 each feature gets ONE shape function, no interactions; plottable
    C  weighted rank fusion      s = sum_l w_l * rank_l   (Borda with learned layer weights)

PRE-REGISTERED
    non-inferiority:  arm >= GBT - 1.5pp on BOTH Qwen models (noise floor 1-2.5pp, track 1) AND the
                      cross-family pattern preserved (LLaVA-NeXT clears vs deployed, OneVision does not)
    superiority:      only with a CI clear of zero on BOTH Qwen models
Folds, negatives, ring mask, COV_HIT identical to phase 70. W in {0.15, 0.25}.
"""
import json, sys, warnings
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
SRC = {"Qwen3-VL": (f"{D}/phase30c_attn_maps_all.jsonl", 28), "Qwen2-VL": (f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", 28),
       "LLaVA-NeXT": (f"{D}/phase82_llavanext.jsonl", 32), "LLaVA-OneVision": (f"{D}/phase82_onevision.jsonl", 28)}

def build(src, NL, W):
    P70.W = W
    b0, b1 = int(.57*NL), int(.93*NL)+1
    rows = [json.loads(l) for l in open(src)]
    X, Y, G, DEP, RING = [], [], [], [], []
    gi = 0
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh*gw != n: continue
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A/np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(NL, gh, gw); dep = M[b0:b1].mean(0)
        R = (np.argsort(np.argsort(-A, axis=1), axis=1)/max(n-1, 1)).reshape(NL, gh, gw)
        pad = np.pad(dep, 1, mode="edge"); nb = sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3))/9.0
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = (yy+.5)/gh, (xx+.5)/gw
        geo = np.concatenate([nb.reshape(-1,1), dep.reshape(-1,1), fx.reshape(-1,1), fy.reshape(-1,1),
                              np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1,1),
                              np.minimum(np.minimum(fx,1-fx), np.minimum(fy,1-fy)).reshape(-1,1),
                              (xx==gw-1).astype(float).reshape(-1,1), (yy==gh-1).astype(float).reshape(-1,1),
                              (xx==0).astype(float).reshape(-1,1)], 1)
        X.append(np.concatenate([M.reshape(NL,-1).T, R.reshape(NL,-1).T, geo], 1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"]) for i in range(n)]))
        G.append(np.full(n, gi)); DEP.append(dep.ravel())
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING.append(m.ravel()); gi += 1
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), DEP, RING, NL

def oof(X, Y, G, make, seeds=3, neg_per_item=30):
    P = np.zeros(len(Y)); gs = np.unique(G)
    for s in range(seeds):
        rng = np.random.default_rng(700+s); perm = {g: i for i, g in enumerate(rng.permutation(gs))}
        Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            ytr = Y[tr]; pos = tr[ytr > 0]; negpool = tr[ytr <= 0]
            k = min(len(negpool), neg_per_item*len(np.unique(G[tr]))); neg = rng.choice(negpool, size=k, replace=False)
            sub = np.concatenate([pos, neg]); m = make(s); m.fit(X[sub], Y[sub]); P[te] += m.predict(X[te])
    return P/seeds

def top1(P, Y, G, RING):
    return np.array([float(Y[G==gi][int(np.argmax(np.where(RING[gi], P[G==gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(G)])

class LogLinear:
    """A: log-attention (NL) + ranks (NL) + geometry (9), standardised, Lasso."""
    def __init__(self, NL, alpha=1e-4): self.NL, self.alpha = NL, alpha
    def _f(self, X):
        A = np.log(X[:, :self.NL] + 1e-9); return np.concatenate([A, X[:, self.NL:]], 1)
    def fit(self, X, y):
        F = self._f(X); self.mu, self.sd = F.mean(0), F.std(0)+1e-9
        self.m = Lasso(alpha=self.alpha, max_iter=5000).fit((F-self.mu)/self.sd, y); return self
    def predict(self, X): return self.m.predict((self._f(X)-self.mu)/self.sd)

class RankFusion:
    def __init__(self, NL): self.NL = NL
    def fit(self, X, y):
        R = X[:, self.NL:2*self.NL]; self.m = Ridge(alpha=1.0).fit(R, y); return self
    def predict(self, X): return self.m.predict(X[:, self.NL:2*self.NL])

def gam(NL, s):
    nf = 2*NL+9
    return HistGradientBoostingRegressor(max_depth=None, max_iter=150, learning_rate=0.10, random_state=s,
                                         interaction_cst=[[i] for i in range(nf)])

rng_ci = np.random.default_rng(131)
def ci(d):
    n = len(d); b = np.array([d[rng_ci.integers(0, n, n)].mean() for _ in range(4000)])
    return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100

for W in [0.15, 0.25]:
    print(f"\n================= W = {W} =================", flush=True)
    print(f"{'model':>16} {'deployed':>9} {'GBT':>7} | {'A loglin':>9} {'vs GBT':>18} | {'B GAM':>7} {'vs GBT':>18} | {'C rank':>7} {'vs GBT':>18}")
    for name, (src, NL) in SRC.items():
        X, Y, G, DEP, RING, NL = build(src, NL, W)
        dep = np.array([float(Y[G==gi][int(np.argmax(np.where(RING[gi], DEP[gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(G)])
        arms = {
            "GBT": lambda s: HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s),
            "A": lambda s: LogLinear(NL),
            "B": lambda s: gam(NL, s),
            "C": lambda s: RankFusion(NL),
        }
        res = {k: top1(oof(X, Y, G, f), Y, G, RING) for k, f in arms.items()}
        line = f"{name:>16} {dep.mean()*100:8.1f}% {res['GBT'].mean()*100:6.1f}% |"
        for k in ["A", "B", "C"]:
            m, lo, hi = ci(res[k]-res["GBT"])
            tag = "≥" if lo > 0 else ("~" if m > -1.5 else "✗")
            line += f" {res[k].mean()*100:6.1f}%  {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{tag} |"
        print(line, flush=True)
