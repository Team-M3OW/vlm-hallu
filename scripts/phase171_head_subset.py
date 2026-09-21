"""
Phase 171 (training-free import #1 and #2): cross a HEAD-SELECTION criterion with the re-ranking head.

Every locator in this project averages over attention heads. Two 2026 papers say that is the wrong
default, and they disagree about how to pick the heads:

  VRH    "Retrieval Heads Meet Vision" (arXiv 2608.27417, 2026-08-27, KAIST). Visual retrieval heads
         are causal, sparse (<2.6% of heads) and universal. Their criterion: the head's attention
         MASS ON THE REFERENT REGION, scored from OUTPUT query tokens, averaged across samples.
         Needs a few labelled boxes to select; nothing is trained.
  ProViP "Not All Attention Heads Contribute..." (arXiv 2608.25332, 2026-08-26, HKUST). Their
         criterion is the VARIANCE of a head's attention over visual tokens -- a head that is
         actually looking at something is spiky; one ignoring the image is flat. Fully label-free.

THE GRAFT (fixed before the run)
    The head keeps its 65 features. Only the SOURCE of the 28-layer profile changes: per layer, the
    map is the mean over the SELECTED heads instead of the mean over all of them. Per-head channels
    are NOT appended -- Qwen2 would add 28x28=784 columns at n=191 and phase 92 already flagged 34
    extra features as an overfitting risk. Subset-averaging is new information (the all-heads mean
    cannot be inverted to recover it), so SS14Z's redundancy lesson does not pre-kill it.

FOLD HONESTY -- the trap phase 108 fell into
    Phase 108's first implementation scored heads on each item's own GT box and read 60.7%; fold-honest
    it collapsed. Here the head subset is selected on TRAINING-FOLD ITEMS ONLY, inside the same
    GroupKFold(5) x 3 seeds as phase 70 (rng 700+s, grouped by item) -- for the label-free criterion
    too, since top-K over all items is still selection on the evaluation set.

ARMS (all OOF, ring-masked, top-1 coverage at W=0.25, identical folds)
    allheads          the incumbent, and a REPRODUCTION CHECK: must land within the 1-2.5pp noise
                      floor of the published 63.4 (qwen3) / 54.5 (qwen2), or the extraction differs
    vrh_q25           per-layer top-25% of heads by referent-region mass        <- VRH criterion
    provip_q25        per-layer top-25% of heads by attention variance          <- ProViP criterion
    vrh_top20         global top-20 (layer, head) pairs by VRH score; layers with none selected fall
                      back to the all-heads mean                                <- VRH's own sparsity
    provip_top20      the same, ProViP score
    random_q25        25% of heads per layer, chosen at random   <- CONTROL: any subset may beat the
                      mean simply by being noisier/sparser, and this is what says otherwise

PRE-REGISTERED DECISION RULE
    Coverage is the SCREEN, not the claim. An arm earns a GPU end-task run only if it beats
    `allheads` by more than the 2.5pp noise floor (SS14Z/SS16A) on BOTH Qwen models AND beats
    `random_q25` by the same margin. Otherwise the head-selection axis is recorded as closed with a
    named reason. No end-task numbers are produced here.

CPU only. Reads phase170's per-head dump; streams one item at a time when building features.
"""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
warnings.filterwarnings("ignore")

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, COV_HIT, BLOCK = 0.25, 0.5, list(range(16, 27))
NGEO = 7


def feats_from_map(M, gh, gw):
    """P70's 65-feature block, verbatim, but from a supplied (NL, n) per-layer map."""
    NL = M.shape[0]
    A = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    R = np.argsort(np.argsort(-A, axis=1), axis=1) / max(A.shape[1] - 1, 1)
    M3 = A.reshape(NL, gh, gw)
    R3 = R.reshape(NL, gh, gw)
    dep = M3[BLOCK].mean(0)
    pad = np.pad(dep, 1, mode="edge")
    nb = sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3)) / 9.0
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = (yy + .5) / gh, (xx + .5) / gw
    return np.concatenate([
        M3.reshape(NL, -1).T, R3.reshape(NL, -1).T,
        nb.reshape(-1, 1), dep.reshape(-1, 1),
        fx.reshape(-1, 1), fy.reshape(-1, 1),
        np.sqrt((fx - .5)**2 + (fy - .5)**2).reshape(-1, 1),
        np.minimum(np.minimum(fx, 1-fx), np.minimum(fy, 1-fy)).reshape(-1, 1),
        (xx == gw-1).astype(float).reshape(-1, 1),
        (yy == gh-1).astype(float).reshape(-1, 1),
        (xx == 0).astype(float).reshape(-1, 1),
    ], axis=1), dep.flatten()


def main(which):
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")            # (NL*H, sum n_i) float16
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    src = {"qwen3": f"{D}/phase30c_attn_maps_all.jsonl",
           "qwen2": f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"}[which]
    boxes = {}
    with open(src) as fh:
        for line in fh:
            r = json.loads(line)
            boxes[r["question_id_full"]] = r["gt_box_frac"]
    idx = [e for e in idx if e["question_id_full"] in boxes]
    N = len(idx)
    print(f"{which}: {N} items, {NL} layers, {H} heads, "
          f"cells {min(e['n_cells'] for e in idx)}-{max(e['n_cells'] for e in idx)} (RAGGED)")

    def item_map(e):
        """(NL, H, n) for one item, float32."""
        return buf[:, e["offset"]:e["offset"] + e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    # ---- per-item targets, groups, ring masks (grids differ per item, as in phase 70)
    P70.W = W
    Ys, Gs, rings, inbox = [], [], [], []
    for i, e in enumerate(idx):
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fyv, fxv = ((yy + .5) / gh).ravel()[:n], ((xx + .5) / gw).ravel()[:n]
        y = np.array([P70.coverage(float(fxv[j]), float(fyv[j]), boxes[e["question_id_full"]])
                      for j in range(n)])
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        Ys.append(y); Gs.append(np.full(n, i)); rings.append(rm.ravel()[:n])
        inbox.append(y >= COV_HIT)
    Y, G = np.concatenate(Ys), np.concatenate(Gs)

    # ---- per-item, per-(layer, head) criterion values
    vrh_i = np.zeros((N, NL, H), np.float32); var_i = np.zeros((N, NL, H), np.float32)
    for i, e in enumerate(idx):
        A = item_map(e)
        vrh_i[i] = A[:, :, inbox[i]].sum(-1) if inbox[i].any() else 0.0   # mass on the referent
        var_i[i] = A.var(-1)                                              # spikiness, label-free
    print(f"  criteria computed  (items with no covering cell: {sum(1 for b in inbox if not b.any())})")

    def select(score, mode, rng):
        m = np.zeros((NL, H), bool)
        if mode == "all":
            m[:] = True
        elif mode == "q25":
            k = max(1, int(round(0.25 * H)))
            for l in range(NL): m[l, np.argsort(-score[l])[:k]] = True
        elif mode == "top20":
            m.ravel()[np.argsort(-score.ravel())[:20]] = True
            for l in range(NL):
                if not m[l].any(): m[l] = True        # layer with no selected head -> all-heads mean
        elif mode == "rand25":
            k = max(1, int(round(0.25 * H)))
            for l in range(NL): m[l, rng.choice(H, k, replace=False)] = True
        return m

    def build_X(mask):
        w = np.maximum(mask.sum(1)[:, None], 1)
        out = []
        for e in idx:
            gh, gw = e["grid"]
            M = (item_map(e) * mask[:, :, None]).sum(1) / w        # (NL, n)
            out.append(feats_from_map(M, gh, gw)[0])
            del M
        return np.vstack(out)

    ARMS = [("allheads", None, "all"), ("vrh_q25", "vrh", "q25"), ("provip_q25", "var", "q25"),
            ("vrh_top20", "vrh", "top20"), ("provip_top20", "var", "top20"),
            ("random_q25", None, "rand25")]
    res = {}
    for name, crit, mode in ARMS:
        P = np.zeros(len(Y))
        for s in range(3):
            rng = np.random.default_rng(700 + s)
            perm = {g: j for j, g in enumerate(rng.permutation(np.unique(G)))}
            Gp = np.vectorize(perm.get)(G)
            for tr, te in GroupKFold(5).split(np.zeros((len(Y), 1)), Y, Gp):
                tr_items = np.unique(G[tr])
                sc = {"vrh": vrh_i, "var": var_i}.get(crit)
                sc = sc[tr_items].mean(0) if sc is not None else np.zeros((NL, H))
                X = build_X(select(sc, mode, rng))
                ytr = Y[tr]
                pos = tr[ytr > 0]; negpool = tr[ytr <= 0]
                neg = rng.choice(negpool, size=min(len(negpool), 30 * len(tr_items)), replace=False)
                sub = np.concatenate([pos, neg])
                mdl = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10,
                                                    random_state=s)
                mdl.fit(X[sub], Y[sub])
                P[te] += mdl.predict(X[te])
                del X
                print(".", end="", flush=True)
        P /= 3
        hit = [1.0 * (Ys[i][int(np.argmax(np.where(rings[i], P[G == i], -1e9)))] >= COV_HIT)
               for i in range(N)]
        res[name] = float(np.mean(hit))
        print(f"\n  {name:14} top-1 coverage {100*res[name]:5.1f}%")
    json.dump(res, open(f"{D}/phase171_head_subset_{which}.json", "w"), indent=1)
    base, rnd = res["allheads"], res["random_q25"]
    pub = {"qwen3": 63.4, "qwen2": 54.5}[which]
    print(f"\n  REPRODUCTION CHECK: allheads {100*base:.1f}% vs published {pub}% "
          f"(noise floor 1-2.5pp) -> {'OK' if abs(100*base-pub) <= 2.5 else 'DIVERGENT, read before trusting arms'}")
    print(f"  screen vs allheads ({100*base:.1f}%) and random_q25 ({100*rnd:.1f}%), floor 2.5pp:")
    for k, v in res.items():
        if k in ("allheads", "random_q25"): continue
        ok = (v - base > 0.025) and (v - rnd > 0.025)
        print(f"    {k:14} {100*(v-base):+5.1f} vs allheads   {100*(v-rnd):+5.1f} vs random   "
              f"{'SCREEN PASSED' if ok else 'below floor'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3")
