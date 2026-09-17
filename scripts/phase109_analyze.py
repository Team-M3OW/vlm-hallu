"""
Phase 109 analysis: Orgad et al. (ICLR'25) ported to VLMs -- does WHERE you probe decide whether the
signal exists, and does the encoding cliff bound their claim?

THEIR CLAIM: "truthfulness information is concentrated in the exact answer tokens", and probing the
last position or a pooled mean misses it. THEIR SECOND CLAIM: such detectors are skill-specific and
do not generalise.

OUR PREDICTION, sharper than theirs and testable here: probes should work ABOVE the encoding cliff
(their regime, where the evidence is in the tokens) and COLLAPSE BELOW it -- phase 66 found that on
those items the correct answer is decodable at NO layer.

THE CONTROL THAT DECIDES INTERPRETATION (phase 15's lesson, and phase 24's retraction): `evid` must
be compared against `rand`, a size-matched random region in the same image and the same forward pass.
Without it a probe can separate "annotated object region" from "random rectangle", which is trivial
and says nothing about truthfulness. Phase 24 reported AUROC 1.000 from exactly that artefact.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
z = np.load(f"{D}/phase109_probe_qwen3.npz", allow_pickle=True)
X, ids, tot, POS = z["X"].astype(np.float32), list(z["ids"]), z["tokens_on_target"], list(z["positions"])
outs = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/phase78_w_sweep.jsonl")}
keep = [i for i, q in enumerate(ids) if q in outs]
X, tot, ids = X[keep], tot[keep], [ids[i] for i in keep]
err, enc = [], []
for q in ids:
    o = outs[q]
    p = np.asarray(o["probs"]["uniform@300"], float)
    e = int(np.argmax(p) != o["label"])
    ok = int(np.argmax(np.asarray(o["probs"]["oracle@0.25"], float)) == o["label"])
    err.append(e); enc.append(int(e and ok))
err, enc = np.asarray(err), np.asarray(enc)
n, NP, NL, dim = X.shape
print(f"n={n}  positions={POS}  layers={NL}  dim={dim}")
print(f"pass-1 wrong {err.mean()*100:.1f}%   certified encoding failures {enc.mean()*100:.1f}%")
print(f"tokens_on_target: median {np.median(tot):.2f}, below cliff(<0.25) {100*np.mean(tot<0.25):.0f}%")

rng = np.random.default_rng(109)
def auc_oof(F, y):
    if len(set(y.tolist())) < 2: return np.nan
    P = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(F, y):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=3000, C=0.05).fit((F[tr]-mu)/sd, y[tr])
        P[te] = m.predict_proba((F[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, P)

def ci(y, s, B=800):
    b = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2: continue
        b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

for tname, y in [("err (pass-1 wrong)", err), ("encfail (certified)", enc)]:
    print(f"\n=== target: {tname}  (positives {int(y.sum())}) ===")
    print(f"  {'position':>10}  best layer   AUROC   (layer chosen by the SAME OOF score -- optimistic)")
    best = {}
    for pi, pname in enumerate(POS):
        a = [auc_oof(X[:, pi, l, :], y) for l in range(NL)]
        bl = int(np.nanargmax(a))
        best[pname] = (bl, a[bl])
        print(f"  {pname:>10}  L{bl:<2d}         {a[bl]:.3f}    (L0 {a[0]:.3f} | L14 {a[14]:.3f} | L{NL-1} {a[-1]:.3f})")
    print(f"\n  CONTROL  evid − rand at their own best layers: "
          f"{best['evid'][1]-best['rand'][1]:+.3f}  "
          f"({'evid carries more than region identity' if best['evid'][1]-best['rand'][1] > 0.05 else 'NOT separable from region identity -- do not interpret evid as truthfulness'})")
    print(f"  ORGAD    evid − last: {best['evid'][1]-best['last'][1]:+.3f}  "
          f"({'token selection matters, as they report' if best['evid'][1]-best['last'][1] > 0.05 else 'token selection does NOT reproduce here'})")

print("\n=== THE CLIFF TEST: probe at each position, split by tokens_on_target ===")
lo, hi = tot < 0.25, tot >= 0.25
print(f"  below cliff n={int(lo.sum())}   above n={int(hi.sum())}")
for pi, pname in enumerate(POS):
    if pname not in ("last", "evid"): continue
    row = []
    for m_, tag in [(lo, "below"), (hi, "above")]:
        yy = err[m_]
        a = max([auc_oof(X[m_][:, pi, l, :], yy) for l in range(NL)]) if len(set(yy.tolist())) > 1 else np.nan
        row.append(f"{tag} {a:.3f}")
    print(f"  {pname:>10}  " + "   ".join(row))
