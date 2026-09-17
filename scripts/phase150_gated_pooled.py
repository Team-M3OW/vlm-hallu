"""
Phase 150 (reviewer W1): DPR needs a question-type router -- so route, and score the POOLED result.

The reviewer is right that a practical system must decide when to apply DPR. We evaluate the
composition at exactly matched tokens: on routed items spend localise@300 + crop@300 (DPR), on
unrouted items spend uniform@600 (the bar). Pooled accuracy vs uniform@600 everywhere.

ROUTERS (cheapest first; a router's own cost is stated)
    R0  none                : DPR on everything                          (the pooled number in Table 1)
    R1  keyword rule (free) : route UNLESS the question contains a relational cue
                              (left|right|above|below|next to|between|behind|in front|closer|farther|
                               top of|bottom of|beside|near|far from|under|over|side)
    R2  benchmark category  : the ORACLE router (single vs relational labels) -- the ceiling
Outcomes are on disk for both models on V*Bench (phase78 / phase97m) and HR-Bench (72c / 72c-qwen2).
PRE-REGISTERED: R1 pooled - bar with CI clear of zero on both models on both benchmarks -> the
composition is deployable with a free rule. Report R0 and R2 alongside.
"""
import json, re, numpy as np, glob, pyarrow.parquet as pq, os
D="/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
REL=re.compile(r"\b(left|right|above|below|next to|between|behind|in front|closer|farther|further|top of|bottom of|beside|near|nearer|far from|under|over|side of|adjacent|opposite|facing|toward)\b", re.I)
def ci(d, rng):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)])
    return d.mean()*100, np.percentile(b,2.5)*100, np.percentile(b,97.5)*100
# --- V*Bench: need question text -> from the HF dataset
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ["HF_DATASETS_OFFLINE"]="1"
from datasets import load_dataset
vs=load_dataset("craigwu/vstar_bench")["test"]
vtext={f"{e['category']}/{e['question_id']}": e["text"].split("\n")[0] for e in vs}
# --- HR-Bench question text by row index
f=glob.glob(f"{os.environ['HF_HUB_CACHE']}/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet")[0]
t=pq.read_table(f, columns=["index","question","category"]); htext={int(t.column("index")[i].as_py()): t.column("question")[i].as_py() for i in range(t.num_rows)}
SETS={
 ("Qwen3-VL","V*Bench"): (f"{D}/phase78_w_sweep.jsonl","question_id_full","head@0.25","category","direct_attributes",vtext),
 ("Qwen2-VL","V*Bench"): (f"{D}/phase97m_merged_qwen2vl.jsonl","question_id_full","head@0.25","category","direct_attributes",vtext),
 ("Qwen3-VL","HR-Bench"):(f"{D}/phase72c_hrbench_fullprompt.jsonl","row_id","head@0.15","category","single",htext),
 ("Qwen2-VL","HR-Bench"):(f"{D}/phase72c_hrbench_fullprompt_qwen2.jsonl","row_id","head@0.15","category","single",htext),
}
print(f"{'model':>9} {'bench':>9} {'router':>22} {'routed%':>8} {'pooled acc':>11} {'bar':>6}   pooled - bar")
for (model,bench),(path,idk,harm,catk,single,textmap) in SETS.items():
    rows=[json.loads(l) for l in open(path)]; n=len(rows)
    ids=[r[idk] for r in rows]
    bar=np.array([int(np.argmax(r["probs"]["uniform@600"])==r["label"]) for r in rows],float)
    dpr=np.array([int(np.argmax(r["probs"][harm])==r["label"]) for r in rows],float)
    q=[textmap.get(i if not isinstance(i,str) else i, "") for i in ids]
    routers={"R0 always DPR": np.ones(n,bool),
             "R1 keyword rule (free)": np.array([not REL.search(s) for s in q]),
             "R2 oracle category": np.array([r[catk]==single for r in rows])}
    rng=np.random.default_rng(150)
    for name,route in routers.items():
        acc=np.where(route,dpr,bar); m,lo,hi=ci(acc-bar,rng)
        print(f"{model:>9} {bench:>9} {name:>22} {route.mean()*100:7.1f}% {acc.mean()*100:10.1f}% {bar.mean()*100:5.1f}%   {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''}")
    # router quality vs oracle
    r1=routers["R1 keyword rule (free)"]; r2=routers["R2 oracle category"]
    print(f"{'':>9} {'':>9}   keyword-vs-category agreement {100*np.mean(r1==r2):.1f}%  (routes {r1.sum()} / oracle {r2.sum()})")
