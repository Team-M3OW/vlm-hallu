"""
Phase 188a: freeze the re-ranking head's PER-CELL out-of-fold scores to disk.

Every pruning phase (75/83/87/93) ranks tokens by raw attention -- layer-K, the block mean, or §14F's
signed linear combination. None has ever used the 65-feature GBT head itself, because only its ARGMAX
cell was ever stored (phase 71a/80a). Pruning needs the whole score map. This produces it once, with
phase 70's machinery and folds, so the GPU pruning run does no fitting and cannot leak.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"

for which in ["qwen3", "qwen2"]:
    P70.W = 0.25
    if which == "qwen3":
        X, Y, G, DEP, rows, _ = P70.build()
    else:
        import phase80a_qwen2vl_head as P80
        X, Y, G, DEP, rows = P80.build()
    P = P70.oof(X, Y, G, seeds=3)
    out = {}
    for gi, r in enumerate(rows):
        out[r["question_id_full"]] = [round(float(v), 6) for v in P[G == gi]]
    json.dump(out, open(f"{D}/phase188a_headscores_{which}.json", "w"))
    print(f"\n{which}: wrote {len(out)} items, "
          f"{np.mean([len(v) for v in out.values()]):.0f} cells each")
