"""
Phase 132: can the readable head close its Qwen deficit with MINIMAL added structure?

Phase 131: the 63-parameter log-linear head is the only head that clears the deployed argmax on all
four architectures, but loses 2-3pp (W=0.15) / 3-7pp (W=0.25) to the tree on Qwen. Two additions
that keep it readable:
    A+  quadratic per layer:   + sum_l q_l * (log A_l)^2          (a per-layer 1-D curve; 28 params)
    A++ A+ plus products of the neighbourhood term with the 6 largest-|w| layers from A (6 params)
    A+nb  A with the 3x3 neighbourhood computed PER LAYER instead of only on the block mean (28 more)
PRE-REGISTERED: same non-inferiority rule as 131 (>= GBT - 1.5pp on both Qwen; family pattern kept).
"""
import sys, warnings, numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
src = open("/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts/phase131_intuitive_heads.py").read()
exec(src[:src.index("for W in [0.15, 0.25]:")])
from sklearn.linear_model import Lasso
warnings.filterwarnings("ignore")

class LogLinPlus:
    def __init__(self, NL, mode, alpha=1e-4): self.NL, self.mode, self.alpha = NL, mode, alpha
    def _f(self, X):
        NL = self.NL; A = np.log(X[:, :NL] + 1e-9); parts = [A, X[:, NL:]]
        if self.mode in ("A+", "A++"): parts.append(A**2)
        if self.mode == "A++":
            nb = X[:, 2*NL:2*NL+1]; parts.append(A[:, self.top] * nb)
        return np.concatenate(parts, 1)
    def fit(self, X, y):
        if self.mode == "A++":
            base = LogLinear(self.NL).fit(X, y); w = base.m.coef_[:self.NL]; self.top = np.argsort(-np.abs(w))[:6]
        F = self._f(X); self.mu, self.sd = F.mean(0), F.std(0)+1e-9
        self.m = Lasso(alpha=self.alpha, max_iter=5000).fit((F-self.mu)/self.sd, y); return self
    def predict(self, X): return self.m.predict((self._f(X)-self.mu)/self.sd)

def build_nb(src_, NL, W):
    """same as build() but with a per-layer 3x3 neighbourhood block appended (NL extra cols)"""
    X, Y, G, DEP, RING, NL = build(src_, NL, W)
    import json
    rows = [json.loads(l) for l in open(src_)]; ex = []
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh*gw != n: continue
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)]); A = A/np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(NL, gh, gw); out = np.zeros((n, NL))
        for l in range(NL):
            pad = np.pad(M[l], 1, mode="edge"); out[:, l] = (sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        ex.append(out)
    return np.hstack([X, np.vstack(ex)]), Y, G, DEP, RING, NL

class LogLinNB(LogLinear):
    def _f(self, X):
        NL = self.NL; A = np.log(X[:, :NL]+1e-9); NB = np.log(X[:, -NL:]+1e-9)
        return np.concatenate([A, X[:, NL:2*NL+9], NB], 1)

for W in [0.15, 0.25]:
    print(f"\n================= W = {W} =================", flush=True)
    print(f"{'model':>16} {'GBT':>7} {'A':>7} | {'A+':>7} {'vs GBT':>18} | {'A++':>7} {'vs GBT':>18} | {'A+nb':>7} {'vs GBT':>18}")
    for name, (s_, NL) in SRC.items():
        X, Y, G, DEP, RING, NL = build(s_, NL, W)
        Xn, _, _, _, _, _ = build_nb(s_, NL, W)
        gbt = top1(oof(X, Y, G, lambda s: HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s)), Y, G, RING)
        a = top1(oof(X, Y, G, lambda s: LogLinear(NL)), Y, G, RING)
        res = {"A+": top1(oof(X, Y, G, lambda s: LogLinPlus(NL, "A+")), Y, G, RING),
               "A++": top1(oof(X, Y, G, lambda s: LogLinPlus(NL, "A++")), Y, G, RING),
               "A+nb": top1(oof(Xn, Y, G, lambda s: LogLinNB(NL)), Y, G, RING)}
        line = f"{name:>16} {gbt.mean()*100:6.1f}% {a.mean()*100:6.1f}% |"
        for k in ["A+", "A++", "A+nb"]:
            m, lo, hi = ci(res[k]-gbt); tag = "≥" if lo > 0 else ("~" if m > -1.5 else "✗")
            line += f" {res[k].mean()*100:6.1f}%  {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{tag} |"
        print(line, flush=True)
