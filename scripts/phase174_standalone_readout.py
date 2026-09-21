"""
Phase 174: the MISSING BASELINE — the imported head criteria as a TRAINING-FREE READ-OUT, with no
learned head at all.

Phases 171/172 tested three imports as HYBRIDS: DPR kept entirely intact, the import changing only
its input (head subset) or its output (crop + scene). None cleared on both models. That leaves the
other half untested — whether the same criteria help the *deployed argmax*, which is where §14Z says
such corrections usually pay:

    "If you train nothing: take a max across layers and weight by the value norm -- +10.5pp.
     If you train a head on the raw depth profile: both are redundant."

So the hybrid null does not predict the standalone result, in either direction. This measures it.

ARMS (top-1 coverage, ring-masked, no learning anywhere)
    blockmean_all     mean over L16-26, mean over ALL heads      <- the deployed incumbent
    blockmean_vrh     the same, restricted to VRH-selected heads (needs a few boxes to select)
    blockmean_provip  the same, ProViP variance-selected heads   <- fully label-free
    blockmean_rand    random heads, same count                   <- CONTROL
    gatedmax_*        the same four under the label-free gated-max rule (L17-20 / L19-22, §141)
Head selection is still fold-honest (criterion averaged over training-fold items only, GroupKFold(5)),
even though nothing is trained: top-K over all items would be selection on the evaluation set.

Reads phase 170's per-head dump. CPU only, no GPU, no fitting.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BLOCK = list(range(16, 27))
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
COV_HIT = 0.5


def run(which, W):
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    boxes = {}
    with open({"qwen3": f"{D}/phase30c_attn_maps_all.jsonl",
               "qwen2": f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"}[which]) as fh:
        for line in fh:
            r = json.loads(line); boxes[r["question_id_full"]] = r["gt_box_frac"]
    idx = [e for e in idx if e["question_id_full"] in boxes]
    N = len(idx)
    P70.W = W
    Ys, rings, inbox = [], [], []
    for e in idx:
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fyv, fxv = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        y = np.array([P70.coverage(float(fxv[j]), float(fyv[j]), boxes[e["question_id_full"]])
                      for j in range(n)])
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        Ys.append(y); rings.append(rm.ravel()[:n]); inbox.append(y >= COV_HIT)

    def imap(e):
        return buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    vrh_i = np.zeros((N, NL, H), np.float32); var_i = np.zeros((N, NL, H), np.float32)
    for i, e in enumerate(idx):
        A = imap(e)
        vrh_i[i] = A[:, :, inbox[i]].sum(-1) if inbox[i].any() else 0.0
        var_i[i] = A.var(-1)

    k = max(1, int(round(0.25 * H)))
    rng = np.random.default_rng(174)
    G = np.arange(N)
    masks = {c: np.zeros((N, NL, H), bool) for c in ["all", "vrh", "provip", "rand"]}
    for tr, te in GroupKFold(5).split(G.reshape(-1, 1), G, G):
        for c, src in [("vrh", vrh_i), ("provip", var_i)]:
            sc = src[tr].mean(0)
            m = np.zeros((NL, H), bool)
            for l in range(NL): m[l, np.argsort(-sc[l])[:k]] = True
            masks[c][te] = m
        m = np.zeros((NL, H), bool)
        for l in range(NL): m[l, rng.choice(H, k, replace=False)] = True
        masks["rand"][te] = m
    masks["all"][:] = True

    out = {}
    for rule, layers in [("blockmean", BLOCK), ("gatedmax", GATE[which])]:
        for c in ["all", "vrh", "provip", "rand"]:
            hit = []
            for i, e in enumerate(idx):
                A = imap(e)
                mk = masks[c][i]
                sel = (A * mk[:, :, None]).sum(1) / np.maximum(mk.sum(1)[:, None], 1)  # (NL, n)
                sel = sel / np.maximum(sel.sum(1, keepdims=True), 1e-12)
                s = sel[layers].mean(0) if rule == "blockmean" else sel[layers].max(0)
                s = np.where(rings[i], s, -1e9)
                hit.append(1.0 * (Ys[i][int(np.argmax(s))] >= COV_HIT))
            out[f"{rule}_{c}"] = float(np.mean(hit))
    return out


for which in ["qwen3", "qwen2"]:
    for W in [0.15, 0.25]:
        r = run(which, W)
        print(f"\n{which}  W={W}   (top-1 coverage, no learning)")
        for rule in ["blockmean", "gatedmax"]:
            base = r[f"{rule}_all"]; rnd = r[f"{rule}_rand"]
            print(f"  {rule:10} all {100*base:5.1f}%  random {100*rnd:5.1f}%   "
                  f"vrh {100*r[f'{rule}_vrh']:5.1f}% ({100*(r[f'{rule}_vrh']-base):+5.1f})   "
                  f"provip {100*r[f'{rule}_provip']:5.1f}% ({100*(r[f'{rule}_provip']-base):+5.1f})")
