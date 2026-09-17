import json, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu/data"; R=json.load(open(f"{D}/phase153_router.json"))
SETS={("qwen3","V*Bench"):(f"{D}/phase78_w_sweep.jsonl","question_id_full","head@0.25","category","direct_attributes","vstar"),
      ("qwen2","V*Bench"):(f"{D}/phase97m_merged_qwen2vl.jsonl","question_id_full","head@0.25","category","direct_attributes","vstar"),
      ("qwen3","HR-Bench"):(f"{D}/phase72c_hrbench_fullprompt.jsonl","row_id","head@0.15","category","single","hrbench"),
      ("qwen2","HR-Bench"):(f"{D}/phase72c_hrbench_fullprompt_qwen2.jsonl","row_id","head@0.15","category","single","hrbench")}
def ci(d,rng):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"{'model':>6} {'bench':>9} {'router':>26} {'routed%':>8} {'agree':>6} {'pooled':>7} {'bar':>6}   pooled - bar")
for (m,bench),(path,idk,harm,catk,single,rk) in SETS.items():
    rows=[json.loads(l) for l in open(path)]; n=len(rows); rng=np.random.default_rng(153)
    bar=np.array([int(np.argmax(r["probs"]["uniform@600"])==r["label"]) for r in rows],float)
    dpr=np.array([int(np.argmax(r["probs"][harm])==r["label"]) for r in rows],float)
    orc=np.array([r[catk]==single for r in rows])
    p=np.array([R[m][rk][str(r[idk])] for r in rows])
    for name,route in [("R3 model text-only, P(rel)<0.5",p<0.5),("R2 oracle category",orc)]:
        acc=np.where(route,dpr,bar); v,lo,hi=ci(acc-bar,rng)
        print(f"{m:>6} {bench:>9} {name:>26} {route.mean()*100:7.1f}% {100*np.mean(route==orc):5.1f}% {acc.mean()*100:6.1f}% {bar.mean()*100:5.1f}%   {v:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''}")
