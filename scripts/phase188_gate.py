"""Pre-specified launch gate for 188b: ridge coverage@0.25 with layers<=20 on the 400-token maps must be >= the 300-token
all-layer reference (point estimate) on BOTH models. Prints PASS/FAIL and the numbers; exit code 0 = PASS."""
import json,sys,numpy as np, importlib.util
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
spec=importlib.util.spec_from_file_location("ee","scripts/phase188c_earlyexit_ridge.py")
ok=True; out=[]
for tag in ("qwen3","qwen2"):
    sys.argv=["x",tag]; ee=importlib.util.module_from_spec(spec)
    # reuse prep/ridge_hits by executing the module body up to the analysis (it prints its own table; we only need functions)
    src=open("scripts/phase188c_earlyexit_ridge.py").read(); src=src[:src.index("rows={k:[json.loads(l)")]
    ns={}; exec(compile(src,"ee","exec"),ns)
    r300=[json.loads(l) for l in open(ns["SRC"]["300 stored"])]; r400=[json.loads(l) for l in open(f"data/phase188a_loc400_unpruned_{tag}.jsonl")]
    common=sorted(set(r["question_id_full"] for r in r300)&set(r["question_id_full"] for r in r400))
    r300=sorted([r for r in r300 if r["question_id_full"] in common],key=lambda r:r["question_id_full"]); r400=sorted([r for r in r400 if r["question_id_full"] in common],key=lambda r:r["question_id_full"])
    ref=ns["ridge_hits"](*ns["prep"](r300,27)).mean(); new=ns["ridge_hits"](*ns["prep"](r400,20)).mean()
    out.append(f"{tag}: ridge 300/all {ref*100:.1f}  vs  400/L<=20 {new*100:.1f}  ({(new-ref)*100:+.1f})"); ok&=(new>=ref)
print("\n".join(out)); print("GATE","PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
