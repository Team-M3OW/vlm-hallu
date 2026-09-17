"""
Phase 153 (reviewer W1, part 2): a ROUTER that is not tuned on the evaluation set.

Phase 150: a free keyword rule routes V*Bench perfectly and makes DPR clear the bar POOLED on both
models, but HR-Bench's relational questions use different words ("relative position of X compared
to Y", "how many", "where is", "which two"), so the rule routes 84.5% there and pooled is null. Tuning
the rule on HR-Bench would be selection on the evaluation set. Instead: ask the MODEL, text-only, no
image, zero-shot -- one short pass whose visual-token cost is zero:

    "Does answering this question require comparing, counting, or locating two or more objects
     relative to each other? Answer Yes or No.\n\nQuestion: <q>"

Route to DPR iff the answer is No. Pre-registered primary: pooled (route->DPR, else uniform@600) minus
uniform@600 everywhere, CI clear on both models on BOTH benchmarks. Router quality vs the category
labels reported alongside. Cost: one text-only forward pass per item (~60 tokens, no image).
"""
import json, os, sys, glob, numpy as np, torch, pyarrow.parquet as pq
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ["HF_DATASETS_OFFLINE"]="1"
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
MODELS={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}
PROMPT=("Does answering this question require comparing, counting, or locating two or more objects "
        "relative to each other? Answer Yes or No.\n\nQuestion: {q}")
from datasets import load_dataset
vs=load_dataset("craigwu/vstar_bench")["test"]; vq={f"{e['category']}/{e['question_id']}": e["text"].split("\n")[0] for e in vs}
f=glob.glob(f"{os.environ['HF_HUB_CACHE']}/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet")[0]
t=pq.read_table(f,columns=["index","question"]); hq={int(t.column("index")[i].as_py()): t.column("question")[i].as_py() for i in range(t.num_rows)}
out={}
for tag,mid in MODELS.items():
    model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0}); model.eval()
    pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
    yes=sorted({tok(s,add_special_tokens=False)["input_ids"][-1] for s in ["Yes"," Yes","yes"," yes"]})
    no=sorted({tok(s,add_special_tokens=False)["input_ids"][-1] for s in ["No"," No","no"," no"]})
    def p_relational(q):
        m=[{"role":"user","content":[{"type":"text","text":PROMPT.format(q=q)}]}]
        inp=tok(pr.apply_chat_template(m,tokenize=False,add_generation_prompt=True),return_tensors="pt").to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[yes],0),torch.logsumexp(lg[no],0)]),0); return float(p[0])
    out[tag]={"vstar":{k:p_relational(q) for k,q in vq.items()}, "hrbench":{str(k):p_relational(q) for k,q in hq.items()}}
    print(tag,"done",flush=True); del model; torch.cuda.empty_cache()
json.dump(out,open(f"{D}/phase153_router.json","w")); print("Done")
