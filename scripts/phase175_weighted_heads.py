"""
Phase 175: SOFT head pooling for the VRH hybrid — two variants of "don't throw the other heads away".

Phase 171 replaced the all-head mean with a HARD subset mean and was 1 of 2 (Qwen2 +5.2, Qwen3 +0.0).
Two softer forms, both keeping every head:

  A  WEIGHTED MEAN (the proposal).  w(l,h) = (1-lam)/H + lam * score(l,h)/sum_h score(l,h)
     lam=0 is the incumbent all-head mean; lam=1 is pure criterion-proportional weighting. Zero extra
     parameters. Swept lam in {0, .25, .5, .75, 1}. NOTE the endpoints are already measured and equal
     on Qwen3 (all-head 62.8, hard vrh_q25 62.8), so this arm is a genuine test of whether the
     interior differs from both ends rather than an expected win.

  B  TWO CHANNELS.  Give the head the all-head mean AND the selected-head mean as separate per-layer
     channels (65 -> 93 features; +28, well inside what phase 92 tolerated at n=191). Motivated by
     SS19: the re-ranker earns its gain by SUBTRACTING layers the block mean adds. A weighted mean can
     only interpolate; two channels let the head learn `selected - all` per layer, WITH SIGN. Arm A is
     the special case where we guess the mixing coefficient; B lets the data pick it, including
     negative.

Criterion is VRH's (referent-region mass from the output token), selected on TRAINING FOLDS ONLY,
inside phase 70's folds (GroupKFold(5) x 3 seeds, rng 700+s). Everything else -- GBT hyperparameters,
negative subsampling, ring mask, coverage@0.25 target -- is the incumbent's.

DECISION RULE (same as phase 171): beat `allheads` by more than the 2.5pp noise floor on BOTH models.
CPU only; reads phase 170's per-head dump.
"""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
from phase171_head_subset import feats_from_map

warnings.filterwarnings("ignore")
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, COV_HIT = 0.25, 0.5


def load(which):
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    boxes = {}
    with open({"qwen3": f"{D}/phase30c_attn_maps_all.jsonl",
               "qwen2": f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"}[which]) as fh:
        for line in fh:
            r = json.loads(line); boxes[r["question_id_full"]] = r["gt_box_frac"]
    idx = [e for e in idx if e["question_id_full"] in boxes]
    P70.W = W
    Ys, Gs, rings, inbox = [], [], [], []
    for i, e in enumerate(idx):
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fyv, fxv = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        y = np.array([P70.coverage(float(fxv[j]), float(fyv[j]), boxes[e["question_id_full"]])
                      for j in range(n)])
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        Ys.append(y); Gs.append(np.full(n, i)); rings.append(rm.ravel()[:n]); inbox.append(y >= COV_HIT)
    return buf, NL, H, idx, Ys, np.concatenate(Ys), np.concatenate(Gs), rings, inbox


def main(which):
    buf, NL, H, idx, Ys, Y, G, rings, inbox = load(which)
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)
    vrh_i = np.zeros((N, NL, H), np.float32)
    for i, e in enumerate(idx):
        A = imap(e)
        vrh_i[i] = A[:, :, inbox[i]].sum(-1) if inbox[i].any() else 0.0
    print(f"{which}: {N} items, {NL} layers, {H} heads")

    def weights(score, lam):
        p = score / np.maximum(score.sum(1, keepdims=True), 1e-12)
        return (1 - lam) / H + lam * p                            # (NL, H), rows sum to 1

    def run(arm, lam=None):
        P = np.zeros(len(Y))
        for s in range(3):
            rng = np.random.default_rng(700 + s)
            perm = {g: j for j, g in enumerate(rng.permutation(np.unique(G)))}
            Gp = np.vectorize(perm.get)(G)
            for tr, te in GroupKFold(5).split(np.zeros((len(Y), 1)), Y, Gp):
                sc = vrh_i[np.unique(G[tr])].mean(0)
                Xs = []
                if arm == "weighted":
                    w = weights(sc, lam)
                    for e in idx:
                        M = (imap(e) * w[:, :, None]).sum(1)
                        Xs.append(feats_from_map(M, *e["grid"])[0])
                else:                                              # two-channel
                    k = max(1, int(round(0.25 * H)))
                    m = np.zeros((NL, H), bool)
                    for l in range(NL): m[l, np.argsort(-sc[l])[:k]] = True
                    for e in idx:
                        A = imap(e)
                        M_all = A.mean(1)
                        M_sel = (A * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1)
                        base = feats_from_map(M_all, *e["grid"])[0]
                        sel = (M_sel / np.maximum(M_sel.sum(1, keepdims=True), 1e-12)).T
                        Xs.append(np.concatenate([base, sel], axis=1))
                X = np.vstack(Xs); del Xs
                ytr = Y[tr]
                pos = tr[ytr > 0]; negpool = tr[ytr <= 0]
                neg = rng.choice(negpool, size=min(len(negpool), 30*len(np.unique(G[tr]))), replace=False)
                sub = np.concatenate([pos, neg])
                mdl = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                                    random_state=s)
                mdl.fit(X[sub], Y[sub]); P[te] += mdl.predict(X[te]); del X
                print(".", end="", flush=True)
        P /= 3
        hit = [1.0*(Ys[i][int(np.argmax(np.where(rings[i], P[G == i], -1e9)))] >= COV_HIT)
               for i in range(N)]
        return float(np.mean(hit))

    res = {}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        res[f"weighted_lam{lam}"] = run("weighted", lam)
        print(f"\n  weighted lam={lam:<4} {100*res[f'weighted_lam{lam}']:5.1f}%")
    res["two_channel"] = run("twochan")
    print(f"\n  two_channel (93 feats) {100*res['two_channel']:5.1f}%")
    json.dump(res, open(f"{D}/phase175_weighted_{which}.json", "w"), indent=1)
    base = res["weighted_lam0.0"]
    hard = {"qwen3": 0.628, "qwen2": 0.555}[which]
    print(f"\n  reference: all-head mean (lam=0) {100*base:.1f}% | hard vrh_q25 (phase171) {100*hard:.1f}%")
    for k, v in res.items():
        if k == "weighted_lam0.0": continue
        print(f"    {k:18} {100*(v-base):+5.1f} vs all-head   "
              f"{'PASSES FLOOR' if v-base > 0.025 else 'below floor'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3")
