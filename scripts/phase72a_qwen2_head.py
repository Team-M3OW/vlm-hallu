"""
Phase 72a (Qwen2): train the Qwen2-VL-7B re-ranking head on ALL V*Bench items and freeze it, so
phase 72c (Qwen2) can apply it zero-shot to HR-Bench 4k. Mirrors phase72a_train_head.py exactly,
pointed at phase80a's Qwen2 feature builder (phase74 attention, block L15-26 of 28 = the same stack
fraction as Qwen3-VL's L16-26). This is the crossing run for the METHOD: 72c showed the head clears
the equal-compute bar at 4K on Qwen3-VL (+7.5 [+2.2,+12.5] single-region); this puts the second
model on the second benchmark.
"""
import json, sys
import numpy as np, joblib
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
import phase80a_qwen2vl_head as P80

OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase72a_qwen2_head.joblib"
from sklearn.ensemble import HistGradientBoostingRegressor
X, Y, G, DEP, rows = P80.build()
print(f"training on ALL {len(rows)} V*Bench items, {X.shape[0]} cells, {X.shape[1]} features", flush=True)
rng = np.random.default_rng(72)
pos = np.where(Y > 0)[0]; negpool = np.where(Y <= 0)[0]
neg = rng.choice(negpool, size=min(len(negpool), 30 * len(rows)), replace=False)
sub = np.concatenate([pos, neg])
m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=0).fit(X[sub], Y[sub])
ns = [r["n_img_tokens"] for r in rows]
spec = {"n_features": int(X.shape[1]), "n_layers": P80.NL, "block": P80.BLOCK, "W": P70.W,
        "train_cells_min": int(min(ns)), "train_cells_max": int(max(ns)),
        "train_items": len(rows), "train_rows_used": int(len(sub)), "model_id": "Qwen/Qwen2-VL-7B-Instruct"}
joblib.dump({"model": m, "spec": spec}, OUT)
P = m.predict(X); hit, inc = [], []
for gi, r in enumerate(rows):
    gh, gw = r["grid"]; sel = G == gi
    rm = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
    else: rm[:] = True
    y = Y[sel]; rmf = rm.flatten()
    hit.append(y[int(np.argmax(np.where(rmf, P[sel], -1e9)))] >= P70.COV_HIT)
    inc.append(y[int(np.argmax(np.where(rmf, DEP[gi], -1e9)))] >= P70.COV_HIT)
print(f"in-sample coverage {100*np.mean(hit):.1f}% vs incumbent {100*np.mean(inc):.1f}% (in-sample, NOT the transfer claim)")
print(f"block {spec['block'][0]}-{spec['block'][-1]}, wrote {OUT}")
