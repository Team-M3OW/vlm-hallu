import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(193)
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
tok={k:np.mean([r["realized_tokens"][k] for r in rows]) for k in arms}
tag="qwen3" if "qwen3" in f else "qwen2"; SW={"qwen3":"data/phase78_w_sweep.jsonl","qwen2":"data/phase97m_merged_qwen2vl.jsonl"}[tag]
sw={json.loads(l)["question_id_full"]:json.loads(l) for l in open(SW)}
u3=np.array([float(int(np.argmax(sw[r["question_id_full"]]["probs"]["uniform@300"]))==sw[r["question_id_full"]]["label"]) if r["question_id_full"] in sw else np.nan for r in rows]); ok=~np.isnan(u3)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]} n={n}   total image tokens: "+"  ".join(f"{k} {tok[k]:.0f}" for k in arms)+f"   | stored uniform@300 {u3[ok].mean()*100:.1f}%")
print(f"  {'stratum':>11} "+" ".join(f"{k:>12}" for k in arms))
S=[("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]
for t,m in S: print(f"  {t:>11} "+" ".join(f"{acc[k][m].mean()*100:11.1f}%" for k in arms))
for t,m in S:
    out=[]
    for nm,a,b in [("P1 2x150-1x300",acc["ridge2@150"],acc["ridge1@300"]),("S1 2x150-1x150",acc["ridge2@150"],acc["ridge1@150"]),
                   ("S2 1x150-1x300",acc["ridge1@150"],acc["ridge1@300"]),("3x100-1x300",acc["ridge3@100"],acc["ridge1@300"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>11}  "+"   ".join(out))
print("  token-count curve (single crop): "+"  ".join(f"{k} {acc[k].mean()*100:.1f}%" for k in ("ridge1@100","ridge1@150","ridge1@300")))
