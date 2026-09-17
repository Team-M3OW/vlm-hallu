"""
Phase 107: can the DEPTH PROFILE tell us, BEFORE generating, that the evidence was never encoded?

Existing pre-generation detectors (HALP, EnsemHalDet, VIB-Probe, HARMONY) predict "will this answer
be wrong". We can label something they cannot: whether the evidence was ENCODABLE AT ALL, certified
by the oracle crop. If the same 300 tokens spent on the evidence region answer correctly while the
uniform pass fails, the information was available in principle and the failure is an ENCODING
failure, not a reasoning one.

TARGETS
    err        pass-1 (uniform@300) is wrong                  -- the ordinary hallucination target
    encfail    pass-1 wrong AND oracle crop right             -- OURS: a certified encoding failure
    fixable    oracle right AND pass-1 wrong, among pass-1 wrong items only (the useful triage)

FEATURES -- pre-registered GEOMETRY-FREE. No GT box, no target area, nothing derived from the label.
    per layer (28 each): peak, entropy, top1_frac, top5_frac, peak/median
    cross-layer: max-minus-mean gap, argmax instability (distinct argmax cells, mean shift),
                 consecutive-layer L1 divergence, and where that divergence first rises
    the model's own next-token distribution over A/B/C/D from pass 1 (max prob, entropy)

BASELINES that must be beaten, not just reported
    max-softmax of the pass-1 answer   (the standard confidence baseline)
    predictive entropy of pass 1
    `peak` alone                        (phase 45's free coverage detector, AUROC 0.788 for coverage)

DECISION
    depth profile > max-softmax on BOTH models, AUROC CI clear -> a detector, and a novel target
    otherwise -> record as a negative next to SS14B, which already found free signals cannot predict
                 the BUDGET. Coarsening to binary either rescues it or it does not.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
CFG = {"Qwen3-VL-2B": (f"{D}/phase30c_attn_maps_all.jsonl", f"{D}/phase78_w_sweep.jsonl"),
       "Qwen2-VL-7B": (f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", f"{D}/phase97m_merged_qwen2vl.jsonl")}
NL = 28


def feats(A):
    """A: (NL, n_img), each layer sum-normalised. Returns a geometry-free feature vector."""
    f = []
    srt = np.sort(A, 1)[:, ::-1]
    s = A.sum(1) + 1e-12
    ent = -(A / s[:, None] * np.log(A / s[:, None] + 1e-12)).sum(1)
    f += [A.max(1), ent, srt[:, 0] / s, srt[:, :5].sum(1) / s,
          A.max(1) / (np.median(A, 1) + 1e-12)]
    f = [np.asarray(x, float) for x in f]
    am = A.argmax(1)
    extra = np.array([
        float((A.max(0) - A.mean(0)).mean()),          # how much max-aggregation would change things
        float(len(set(am.tolist()))) / NL,             # argmax instability across depth
        float(np.mean(np.abs(np.diff(A, axis=0)).sum(1))),   # mean consecutive-layer L1 divergence
        float(np.argmax(np.abs(np.diff(A, axis=0)).sum(1) >
                        3 * np.median(np.abs(np.diff(A, axis=0)).sum(1)))) / NL,
        float(np.corrcoef(A[:NL // 2].mean(0), A[NL // 2:].mean(0))[0, 1]),  # early vs late agreement
    ])
    return np.concatenate([np.concatenate(f), extra])


for name, (attn_src, out_src) in CFG.items():
    rows = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(attn_src)}
    outs = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(out_src)}
    ids = sorted(set(rows) & set(outs))
    X, conf, entc, peak = [], [], [], []
    err, enc = [], []
    for q in ids:
        r, o = rows[q], outs[q]
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        X.append(feats(A))
        p = np.asarray(o["probs"]["uniform@300"], float)
        conf.append(p.max()); entc.append(-(p * np.log(p + 1e-12)).sum())
        b0 = 16 if "Qwen3" in name else 15
        peak.append(float(A[b0:27].mean(0).max()))
        lab = o["label"]
        e = int(np.argmax(p) != lab)
        ok = int(np.argmax(np.asarray(o["probs"]["oracle@0.25"], float)) == lab)
        err.append(e); enc.append(int(e and ok))
    X = np.asarray(X); X = np.nan_to_num(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    conf, entc, peak = map(np.asarray, (conf, entc, peak))
    err, enc = np.asarray(err), np.asarray(enc)
    n = len(ids)
    print(f"\n=== {name}  n={n}   pass-1 wrong {err.mean()*100:.1f}%   "
          f"certified encoding failures {enc.mean()*100:.1f}% ===")

    def oof(Z, y, model="gbt"):
        P = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y):
            m = (HistGradientBoostingClassifier(max_depth=3, max_iter=120, random_state=0)
                 if model == "gbt" else LogisticRegression(max_iter=2000, C=0.5))
            m.fit(Z[tr], y[tr]); P[te] = m.predict_proba(Z[te])[:, 1]
        return P

    rng = np.random.default_rng(107)
    def auc_ci(y, s):
        a = roc_auc_score(y, s)
        b = []
        for _ in range(800):
            i = rng.integers(0, n, n)
            if len(set(y[i].tolist())) < 2: continue
            b.append(roc_auc_score(y[i], s[i]))
        return a, np.percentile(b, 2.5), np.percentile(b, 97.5)

    for tname, y in [("err  (pass-1 wrong)", err), ("encfail (certified)", enc)]:
        print(f"  target {tname}  (positives {y.sum()})")
        arms = {"max-softmax (baseline)": -conf, "pred. entropy (baseline)": entc,
                "peak alone (ph 45)": -peak,
                "DEPTH PROFILE, gbt": oof(X, y, "gbt"),
                "DEPTH PROFILE, logistic": oof(X, y, "lr"),
                "depth + confidence, gbt": oof(np.c_[X, conf, entc], y, "gbt")}
        base = roc_auc_score(y, -conf)
        for k, s in arms.items():
            a, lo, hi = auc_ci(y, s)
            print(f"    {k:>26} AUROC {a:.3f} [{lo:.3f},{hi:.3f}]"
                  f"{'   > baseline' if lo > base else ''}")
