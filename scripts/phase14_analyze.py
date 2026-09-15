"""Phase 14 analysis: is presence linearly decodable from internal states?

Verdict rules, fixed before looking:
  * The LOAD-BEARING probe is `obj` (the object's own visual tokens), and it only counts if it
    beats its paired `rand` control on the SAME items -- otherwise any signal is global/scene
    context, not local encoding.
  * `last` (final position) is a CONFOUND CHECK, not a result: it carries the question text and
    co-occurrence priors. `last_blank` is its language-prior floor (black image, same prompt).
  * Everything is held-out (train/test split); an in-sample probe measures nothing.
  * Stratified by tokens_on_object, pre-specified: chance in the sub-token stratum + rising with
    budget = a representational capacity floor with a dose-response curve.
"""
import json
import sys

import numpy as np

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BLOCKS = ["obj", "rand", "last", "last_blank"]


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lab = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    n_pos, n_neg, n = len(pos), len(neg), len(lab)
    rs, i = 0.0, 0
    while i < n:
        j = i
        while j < n and lab[j][0] == lab[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rs += avg * sum(1 for k in range(i, j) if lab[k][1] == 1)
        i = j
    return (rs - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logreg(X, y, epochs=300, lr=0.5, l2=1e-3):
    """Plain logistic regression, standardized features, full-batch gradient descent. Avoids a
    sklearn dependency and is plenty for a linear separability question."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * g
    return lambda Z: ((Z - mu) / sd @ w[:-1] + w[-1])


def main():
    d = np.load(f"{DATA}/phase14_probe_feats.npz")
    F = d["feats"].astype(np.float32)
    layers = d["layers"].tolist()
    meta = [json.loads(l) for l in open(f"{DATA}/phase14_probe_meta.jsonl")]
    y = np.array([1 if m["group"] == "positive" else 0 for m in meta])
    D = F.shape[1] // (len(layers) * len(BLOCKS))
    print(f"n={len(meta)}  positives={int(y.sum())}  layers={layers}  hidden_dim={D}")

    zc = [m for m in meta if m["group"] == "positive" and m["zero_coverage"]]
    pos = [m for m in meta if m["group"] == "positive"]
    den = [m for m in pos if m["p_yes_fp16"] < 0.01]
    print(f"\nZERO-COVERAGE (bbox covers NO merged visual token; nearest-token fallback used):")
    print(f"  all positives      {len(zc)}/{len(pos)} = {100*len(zc)/len(pos):.1f}%")
    if den:
        zd = sum(1 for m in den if m["zero_coverage"])
        print(f"  confident denials  {zd}/{len(den)} = {100*zd/len(den):.1f}%   (n={len(den)})")
    tk = sorted(m["tokens_on_object"] for m in pos)
    tkd = sorted(m["tokens_on_object"] for m in den) if den else []
    print(f"  median tokens_on_object: all positives {tk[len(tk)//2]:.2f}"
          + (f"   confident denials {tkd[len(tkd)//2]:.2f}" if tkd else ""))

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(meta))
    cut = int(.6 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    def block(bi, li):
        off = (li * len(BLOCKS) + bi) * D
        return F[:, off:off + D]

    print("\n" + "=" * 78)
    print("HELD-OUT PROBE AUROC  (train 60% / test 40%)")
    print("=" * 78)
    print(f"  {'layer':<8}" + "".join(f"{b:>13}" for b in BLOCKS))
    scores = {}
    for li, L in enumerate(layers):
        row = []
        for bi, b in enumerate(BLOCKS):
            X = block(bi, li)
            f = fit_logreg(X[tr], y[tr])
            s = f(X[te])
            scores[(L, b)] = s
            row.append(auroc(s[y[te] == 1], s[y[te] == 0]))
        print(f"  L{L:<7}" + "".join(f"{v:13.4f}" for v in row))

    pj = np.array([m["p_yes_fp16"] for m in meta])[te]
    print(f"\n  P(yes) on the SAME held-out items: {auroc(pj[y[te]==1], pj[y[te]==0]):.4f}")

    print("\n" + "=" * 78)
    print("STRATIFIED BY tokens_on_object  (positives binned; SAME negatives reused per bin)")
    print("=" * 78)
    tks = np.array([m["tokens_on_object"] for m in meta])
    best_L = layers[-1]
    edges = [0, 1, 4, 16, 1e9]
    names = ["<1 (sub-token)", "1-4", "4-16", ">16"]
    print(f"  {'stratum':<18}{'n_pos':>6}{'obj':>9}{'rand':>9}{'last':>9}{'P(yes)':>9}")
    for k in range(4):
        m = (y[te] == 1) & (tks[te] >= edges[k]) & (tks[te] < edges[k + 1])
        npos = int(m.sum())
        if npos < 10:
            print(f"  {names[k]:<18}{npos:>6}   (too few)")
            continue
        negm = y[te] == 0
        cells = []
        for b in ("obj", "rand", "last"):
            s = scores[(best_L, b)]
            cells.append(auroc(s[m], s[negm]))
        cells.append(auroc(pj[m], pj[negm]))
        print(f"  {names[k]:<18}{npos:>6}" + "".join(f"{c:9.4f}" for c in cells))

    print(f"\n  (probe columns at layer {best_L})")
    print("\n  VERDICT: `obj` must beat `rand` for local encoding to exist. In the sub-token")
    print("  stratum, obj ~= rand ~= chance would mean nothing is encoded at the object's")
    print("  location -- a capacity floor no activation/attention method can cross.")


if __name__ == "__main__":
    main()
