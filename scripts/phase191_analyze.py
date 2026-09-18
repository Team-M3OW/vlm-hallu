import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(191)
tag="qwen3" if "qwen3" in f else "qwen2"
SW={"qwen3":"data/phase78_w_sweep.jsonl","qwen2":"data/phase97m_merged_qwen2vl.jsonl"}[tag]
sw={json.loads(l)["question_id_full"]:json.loads(l) for l in open(SW)}
u3=np.array([float(int(np.argmax(sw[r["question_id_full"]]["probs"]["uniform@300"]))==sw[r["question_id_full"]]["label"]) if r["question_id_full"] in sw else np.nan for r in rows])
u6ref=np.array([float(int(np.argmax(sw[r["question_id_full"]]["probs"]["uniform@600"]))==sw[r["question_id_full"]]["label"]) if r["question_id_full"] in sw else np.nan for r in rows])
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
tok={k:np.mean([r["realized_tokens"][k] for r in rows]) for k in arms}; txt={k:np.mean([r["text_tokens"][k] for r in rows]) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]} n={n}   image tokens (incl. localiser where separate): "+"  ".join(f"{k} {tok[k]:.0f}" for k in arms)+"\n   text tokens: "+"  ".join(f"{k} {txt[k]:.0f}" for k in arms))
print(f"  {'stratum':>11} "+" ".join(f"{k:>11}" for k in arms))
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    print(f"  {t:>11} "+" ".join(f"{acc[k][m].mean()*100:10.1f}%" for k in arms))
ok=~np.isnan(u3)
d6,l6,h6=ci((acc["uniform@600"]-u6ref)[ok])
print(f"  ANCHOR: our uniform@600 {acc['uniform@600'][ok].mean()*100:.1f}% vs stored {u6ref[ok].mean()*100:.1f}% ({d6:+.1f}, agreement {np.mean(acc['uniform@600'][ok]==u6ref[ok])*100:.0f}%)  |  stored uniform@300 = {u3[ok].mean()*100:.1f}% (300 tokens, HALF compute)")
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    mm=m&ok; d,lo,hi=ci((acc["ridge_ctx"]-u3)[mm])
    print(f"  {t:>11}  ** ctx - uniform@300 (the cheap control) {d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}   [u@300 {u3[mm].mean()*100:.1f}%  ctx {acc['ridge_ctx'][mm].mean()*100:.1f}%]")
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    out=[]
    for nm,a,b in [("P1/P2 ctx-bar",acc["ridge_ctx"],acc["uniform@600"]),("GUARD ctx-ridge300",acc["ridge_ctx"],acc["ridge300"]),("oracle_ctx-oracle300",acc["oracle_ctx"],acc["oracle300"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>11}  "+"   ".join(out))
