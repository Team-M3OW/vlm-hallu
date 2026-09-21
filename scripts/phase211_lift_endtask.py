"""
Phase 211: the UNSUPERVISED lift read-out, at the END TASK.

§61-R2 showed that dividing each cell by its value on a typical item recovers 71-86% of DWA's COVERAGE with no
labels. Coverage over-reports (§93b), so this settles whether it survives on answers. If it does, DWA's ~50 boxed
examples buy little and the paper should say so.

Arms, all answering on a W=0.25 crop at 300 tokens, localiser at 300 (the same pipeline as phase 184/199):
  block      arg max of the deployed block-mean map                    (the incumbent)
  lift       arg max of blockmean / item_mean                          (UNSUPERVISED, no labels anywhere)
  lift_rm    same, with the outer-ring mask, to separate the two corrections
The item-mean is computed LEAVE-ONE-OUT within each grid shape, so no item contributes to its own normaliser,
and no label is used at any point.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
MAPS,BLK={"qwen3":("phase30c_attn_maps_all.jsonl",(16,27)),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",(15,27))}[WHICH]
OUT=f"{D}/data/phase211_lift_{WHICH}.jsonl"; W=0.25; Image.MAX_IMAGE_PIXELS=None
rows=[json.loads(l) for l in open(f"{D}/data/{MAPS}")]
rows=[r for r in rows if "attn" in r and "grid" in r]
by={}
for r in rows: by.setdefault(tuple(r["grid"]),[]).append(r)
dep_of={}; 
for g,rs in by.items():
    M=np.stack([np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(28)]) for r in rs])
    M=M/np.maximum(M.sum(2,keepdims=True),1e-12)
    dep=M[:,BLK[0]:BLK[1]].mean(1)
    tot=dep.sum(0); k=len(rs)
    for i,r in enumerate(rs):
        loo=(tot-dep[i])/max(k-1,1)                      # leave-one-out item mean: no self-contribution
        dep_of[r["question_id_full"]]=(dep[i],loo,g)
print(f"maps: {len(dep_of)} items over {len(by)} grid shapes",flush=True)
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
def fit(img,target=300,refine=6,tol=0.08):
    W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/r)**0.5
    return best[0]
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
def answer(im,t):
    inp=build(fit(im),t).to(model.device)
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
lut={f"{e['category']}/{e['question_id']}":e for e in ds}
done=set()
if os.path.exists(OUT): done={json.loads(l)["question_id_full"] for l in open(OUT)}
t0,n=time.time(),0
with open(OUT,"a") as fout:
    for qid,ex in lut.items():
        if qid in done or qid not in dep_of: continue
        ip=os.path.join(root,ex["image"])
        if not os.path.exists(ip): continue
        dep,loo,(gh,gw)=dep_of[qid]
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel()
        lift=dep/np.maximum(loo,1e-12)
        cells={"block":int(np.argmax(np.where(rm,dep,-1e9))),
               "lift":int(np.argmax(lift)),
               "lift_rm":int(np.argmax(np.where(rm,lift,-1e9)))}
        img=Image.open(ip).convert("RGB")
        lab="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
        rec={"question_id_full":qid,"category":ex["category"],"label":lab,"grid":[gh,gw],"cells":{},"probs":{}}
        for k,j in cells.items():
            cx,cy=float((j%gw+.5)/gw),float((j//gw+.5)/gh)
            rec["cells"][k]=[cx,cy]; rec["probs"][k]=answer(crop(img,cx,cy),ex["text"])
        fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
