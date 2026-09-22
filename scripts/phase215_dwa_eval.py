"""
Phase 215: fit DWA out-of-fold on a family's own maps (phase 214) and evaluate at the END TASK.
Arms: uniform bar | block-mean crop (the incumbent) | DWA crop.
Aborts if the block-mean arg-max does not cover ground-truth boxes above chance, which is the guard on
the base-tile-prefix grid assumption for anyres families.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupKFold
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None; W=0.25
MODELS={"llava_ov":("llava-hf/llava-onevision-qwen2-7b-ov-hf",384),
        "llava_next":("llava-hf/llava-v1.6-vicuna-7b-hf",336),
        "gemma3_4b":("google/gemma-3-4b-it",None)}
MK=sys.argv[1]; mid,base_px=MODELS[MK]
rows=[json.loads(l) for l in open(f"{D}/data/phase214_dwa_{MK}.jsonl")]
print(f"{MK}: {len(rows)} items with maps",flush=True)
gh,gw=rows[0]["grid"]; NL=rows[0]["nl"]; ncell=gh*gw
BLK=(round(0.57*NL),NL-1)
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
rm=np.zeros((gh,gw),bool)
if gh>2 and gw>2: rm[1:-1,1:-1]=True
else: rm[:]=True
rm=rm.ravel()
X=[];Y=[];G=[];blockcell=[];covgrid=[]
for i,r in enumerate(rows):
    a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
    a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
    LA=np.log(a+1e-12).T
    R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(ncell-1,1)).T
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    X.append(np.c_[LA,R,geo])
    c=np.array([cov(float(fx[j]),float(fy[j]),r["gt"]) for j in range(ncell)])
    Y.append(c); covgrid.append(c); G.append(np.full(ncell,i))
    blockcell.append(int(np.argmax(np.where(rm,dep.ravel(),-1e9))))
X=np.vstack(X); Y=np.concatenate(Y); G=np.concatenate(G); covgrid=np.array(covgrid)
bm=np.array([covgrid[i,blockcell[i]] for i in range(len(rows))])
chance=covgrid.mean()
print(f"  GUARD block-mean coverage {bm.mean():.3f} vs chance {chance:.3f}  "
      f"{'PASS' if bm.mean()>2*chance else 'FAIL -> grid assumption wrong, aborting'}",flush=True)
assert bm.mean()>2*chance, "base-tile grid assumption rejected"
P=np.zeros(len(Y))
for s in (700,701,702):
    rng=np.random.default_rng(s); gs=np.unique(G); perm={g:i for i,g in enumerate(rng.permutation(gs))}
    Gp=np.vectorize(perm.get)(G)
    for tr,te in GroupKFold(5).split(X,Y,Gp):
        mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
        A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
        w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
P=(P/3).reshape(len(rows),ncell)
ridgecell=[int(np.argmax(np.where(rm,P[i],-1e9))) for i in range(len(rows))]
rc=np.array([covgrid[i,ridgecell[i]] for i in range(len(rows))])
print(f"  coverage: block-mean {bm.mean():.3f}  DWA {rc.mean():.3f}",flush=True)
# ---- end task ----
model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def ans(im,t):
    inp=pr(images=im,text=chat(t),return_tensors="pt").to(model.device)
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
def crop(img,j):
    cx,cy=(j%gw+.5)/gw,(j//gw+.5)/gh
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset")
ex={f"{e['category']}/{e['question_id']}":e for e in load_dataset("craigwu/vstar_bench")["test"]}
OUT=f"{D}/data/phase215_dwaeval_{MK}.jsonl"
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
t0=0
with open(OUT,"a") as f:
    for i,r in enumerate(rows):
        if r["qid"] in done: continue
        e=ex[r["qid"]]; img=Image.open(os.path.join(root,e["image"])).convert("RGB")
        base=img.resize((base_px,base_px),Image.BICUBIC) if base_px else img
        rec={"qid":r["qid"],"category":r["category"],"label":r["label"],
             "probs":{"bar":ans(base,e["text"]),
                      "block":ans(crop(img,blockcell[i]),e["text"]),
                      "dwa":ans(crop(img,ridgecell[i]),e["text"])},
             "cov":{"block":float(bm[i]),"dwa":float(rc[i])}}
        f.write(json.dumps(rec)+"\n"); f.flush(); t0+=1
        if t0%25==0: print(f"  [{t0}]",flush=True)
print(f"Done -> {OUT}",flush=True)
