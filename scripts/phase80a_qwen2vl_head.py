"""
Phase 80a: train the re-ranking head on Qwen2-VL-7B and freeze its out-of-fold proposals.

STANDARD ADOPTED 2026-09-16: every claim in the paper must hold on MORE THAN ONE MODEL, and any
claim that does not survive is REJECTED rather than caveated. SS14K applied that to the read-out
defect (survived on 2) and to the signed-contrast mechanism (rejected). This file begins applying it
to the METHOD itself, which is currently single-backbone.

Reuses phase70's feature construction and OOF machinery verbatim, pointed at phase74's Qwen2-VL
attention maps, so the two models' heads cannot drift apart in implementation.

WHAT WOULD REJECT THE METHOD: if the head does not beat the deployed argmax on Qwen2-VL's proposals,
DCR is a Qwen3-VL-2B artifact and the method section comes out of the paper.
"""
import json
import numpy as np
import statistics as st
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
SRC = f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"
OUT = f"{D}/phase80a_qwen2vl_proposals.json"
NL = 28
BLOCK = list(range(int(.57 * NL), int(.93 * NL) + 1))


def build():
    rows = [json.loads(l) for l in open(SRC)]
    X, Y, G, DEP = [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        n = r["n_img_tokens"]
        A = np.stack([np.asarray(r["attn"][f"L{i}"], dtype=float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(NL, gh, gw)
        dep = M[BLOCK].mean(0)
        R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).reshape(NL, gh, gw)
        pad = np.pad(dep, 1, mode="edge")
        nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        X.append(np.concatenate([
            M.reshape(NL, -1).T, R.reshape(NL, -1).T,
            nb.reshape(-1, 1), dep.reshape(-1, 1),
            fx.reshape(-1, 1), fy.reshape(-1, 1),
            np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2).reshape(-1, 1),
            np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)).reshape(-1, 1),
            (xx == gw - 1).astype(float).reshape(-1, 1),
            (yy == gh - 1).astype(float).reshape(-1, 1),
            (xx == 0).astype(float).reshape(-1, 1)], axis=1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                           for i in range(n)]))
        G.append(np.full(n, gi))
        DEP.append(dep.flatten())
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), DEP, rows


def main():
    X, Y, G, DEP, rows = build()
    print(f"Qwen2-VL-7B: {len(rows)} items, {X.shape[1]} features/cell", flush=True)
    P = P70.oof(X, Y, G)
    print(flush=True)
    out, hc, ac = {}, [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]
        m = G == gi
        rm = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        rmf = rm.flatten()
        j = int(np.argmax(np.where(rmf, P[m], -1e9)))
        jd = int(np.argmax(np.where(rmf, DEP[gi], -1e9)))
        cell = lambda i: (((i % gw) + .5) / gw, ((i // gw) + .5) / gh)
        y = Y[m]
        out[r["question_id_full"]] = {
            "head": cell(j), "argmax": cell(jd),
            "head_cov": float(y[j]), "argmax_cov": float(y[jd]),
            "gt_box_frac": r["gt_box_frac"], "category": r["category"]}
        hc.append(float(y[j] >= P70.COV_HIT)); ac.append(float(y[jd] >= P70.COV_HIT))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Qwen2-VL-7B coverage: argmax {100*st.mean(ac):.1f}% -> head {100*st.mean(hc):.1f}% "
          f"({100*(st.mean(hc)-st.mean(ac)):+.1f}pp)")
    print(f"  (Qwen3-VL-2B was 39.3% -> 52.9%, +13.6pp)")
    print(f"  proposals differ on {100*st.mean([tuple(v['head'])!=tuple(v['argmax']) for v in out.values()]):.1f}% of items")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
