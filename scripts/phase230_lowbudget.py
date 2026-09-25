"""Phase 230: where does uniform resolution overtake the best crop? Low-budget bar points.
Qwen3-VL-2B, uniform@{120,150,200,300} tokens on docvqa and textvqa, same 300 items as the
CROP_B curve. Compared against crop50@600 (68.9 TextVQA / 56.2 DocVQA) and crop25@300."""
import json, os, sys, time, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import benchmarks as B
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; BK=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 300
OUT=f"{D}/data/phase230_qwen3_2b_{BK}.jsonl"; Image.MAX_IMAGE_PIXELS=None
model=AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct"); tok=pr.tokenizer; itid=model.config.image_token_id
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0]==itid).sum())
def fit(img,target,refine=6,tol=0.10):
    W_,H_=img.size; sc=(target/max(ntok(build(img,"x")),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(32,int(W_*sc)),max(32,int(H_*sc))),Image.BICUBIC); r=ntok(build(cur,"x"))
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/max(r,1))**0.5
    return best
def answer(img,q,kind,target):
    src=fit(img,target)[0]; inp=build(src,q).to(model.device)
    p,t=B.score_item(model,pr,tok,inp,kind); nt=ntok(inp); del inp; torch.cuda.empty_cache(); return p,t,nt
items=B.load(BK,N or None); print(f"{BK}: {len(items)} items",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
n=0; t0=time.time()
with open(OUT,"a") as f:
    for qid,img,q,gold,stratum,kind in items:
        if qid in done: continue
        img=img.convert("RGB")
        rec={"qid":qid,"gold":gold,"stratum":stratum,"kind":kind,"probs":{},"preds":{},"tokens":{}}
        for E in (120,150,200,300):
            p,t,nt=answer(img,q,kind,E)
            if p is not None: rec["probs"][f"uniform@{E}"]=p
            if t is not None: rec["preds"][f"uniform@{E}"]=t
            rec["tokens"][f"uniform@{E}"]=nt
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
