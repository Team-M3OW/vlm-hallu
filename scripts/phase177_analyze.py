"""Phase 177 analysis: CLD on the answer. Rules per budget: final layer; CLD valley (K in 6,10); min-entropy layer in last K;
per-layer lens accuracy; DoLa-style (final − premature by max JSD over letters). Paired CIs vs final."""
import json, sys, numpy as np
tag=sys.argv[1]; rows=[json.loads(l) for l in open(f"data/phase177_cld_{tag}.jsonl")]; rng=np.random.default_rng(177)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return f"{d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
sm=lambda x: np.exp(x-x.max())/np.exp(x-x.max()).sum()
for B in ("300","600"):
    Lm=np.array([r["budgets"][B]["letters"] for r in rows]); E=np.array([r["budgets"][B]["entropy"] for r in rows]); y=np.array([r["label"] for r in rows]); NL=Lm.shape[1]
    acc_layer=np.array([(Lm[:,l].argmax(1)==y).mean() for l in range(NL)]); final=(Lm[:,-1].argmax(1)==y).astype(float)
    res={"final layer":final}; PICKS={}
    # PRIMARY = full-vocab entropy valley (faithful CLD); letter-entropy is the variant
    for K in (6,10):
        pick=[]
        for i in range(len(rows)):
            l=NL-1
            while l>NL-K and E[i,l-1]<E[i,l]: l-=1
            pick.append(l)
        pick=np.array(pick); res[f"CLD valley K={K}"]=(Lm[np.arange(len(rows)),pick].argmax(1)==y).astype(float); PICKS[K]=pick
        res[f"min-entropy layer in last {K}"]=(Lm[np.arange(len(rows)),NL-K+E[:,NL-K:].argmin(1)].argmax(1)==y).astype(float)
        # letter-entropy variant
        El=np.array([[-(sm(Lm[i,l])*np.log(sm(Lm[i,l])+1e-12)).sum() for l in range(NL)] for i in range(len(rows))])
        res[f"min LETTER-entropy layer in last {K}"]=(Lm[np.arange(len(rows)),NL-K+El[:,NL-K:].argmin(1)].argmax(1)==y).astype(float)
    # DoLa-style: contrast final letter log-probs with the premature layer of max JSD among layers NL-10..NL-2
    dola=[]
    for i in range(len(rows)):
        pf=sm(Lm[i,-1]); best=(-1,None)
        for l in range(NL-10,NL-1):
            pl=sm(Lm[i,l]); m=(pf+pl)/2; j=0.5*((pf*np.log(pf/m+1e-12)).sum()+(pl*np.log(pl/m+1e-12)).sum())
            if j>best[0]: best=(j,pl)
        dola.append(int(np.argmax(np.log(pf+1e-12)-np.log(best[1]+1e-12)))==y[i])
    res["DoLa contrast (letters)"]=np.array(dola,float)
    print(f"\n{tag} @ {B} tokens  n={len(rows)}   per-layer lens accuracy: "+" ".join(f"L{l}:{acc_layer[l]*100:.0f}" for l in range(max(0,NL-12),NL)))
    for K,pk in PICKS.items(): print(f"   CLD K={K}: median picked layer L{int(np.median(pk))}, picks final layer on {np.mean(pk==NL-1)*100:.0f}% of items")
    print(f"   entropy rises at the final layer on {(E[:,-1]>E[:,-2]).mean()*100:.0f}% of items (CLD's 'alignment tax' signature)")
    ANCH={"qwen3":"data/phase78_w_sweep.jsonl","qwen2":"data/phase97m_merged_qwen2vl.jsonl"}[tag]
    sw={json.loads(l)["question_id_full"]:json.loads(l) for l in open(ANCH)}
    arm="uniform@300" if B=="300" else "uniform@600"
    ok=[(r["question_id_full"] in sw) for r in rows]
    ref=np.array([float(int(np.argmax(sw[r["question_id_full"]]["probs"][arm]))==sw[r["question_id_full"]]["label"]) if r["question_id_full"] in sw else np.nan for r in rows])
    lab=np.array([sw[r["question_id_full"]]["label"] if r["question_id_full"] in sw else -1 for r in rows])
    m=~np.isnan(ref)
    print(f"   ANCHOR vs stored {arm}: ours {final[m].mean()*100:.1f}%  stored {ref[m].mean()*100:.1f}%  per-item agreement {np.mean(final[m]==ref[m])*100:.0f}%  (label match {np.mean(lab[m]==y[m])*100:.0f}%, n={int(m.sum())})")
    for k,v in res.items():
        if v is None: continue
        print(f"   {k:>36} {v.mean()*100:5.1f}%   vs final {ci(v-final)}")
