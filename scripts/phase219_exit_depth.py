"""
Phase 219: what does exiting the localiser early actually cost, AT THE END TASK, by depth?

WHY. The funded two-crop design (§37) is positive on BOTH strata on Qwen3 (+18.3 single, +2.6 cross) and
fails only on Qwen2, where the whole deficit is the L20 early exit used to pay for it (-6.6 cross, -3.7
pooled). The exit was adopted on COVERAGE evidence (§29: -0.5/+1.6 at L20) and then cost accuracy. §93b
says coverage over-reports, so the exit depth was never chosen on the quantity that matters.

WHAT. Refit DWA out-of-fold using ONLY layers <= L, crop at its arg-max, answer. Sweep L. Report the
end-task cost against the full-depth read-out, and the token-layers each depth frees.
Budget: localiser = 300*(L+1); the bar is 600*28 = 16,800.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupKFold
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)  # the env token is invalid; use the stored login
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None; W=0.25
WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
MAPS,BLK0={"qwen3":("phase30c_attn_maps_all.jsonl",16),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",15)}[WHICH]
OUT=f"{D}/data/phase219_exit_{WHICH}.jsonl"; NL=28; EXITS=[18,20,22,24,27]
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
rows=[json.loads(l) for l in open(f"{D}/data/{MAPS}")]
rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
def fit_at(Lmax):
    """refit DWA using only layers <= Lmax; returns {qid: cell}"""
    X=[];Y=[];G=[];meta=[]
    for i,r in enumerate(rows):
        gh,gw=r["grid"]; n=gh*gw
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(Lmax+1)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(n-1,1)).T
        hi=min(Lmax+1,27); lo=min(BLK0,max(0,hi-1))
        dep=a[lo:hi].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo])
        Y.append(np.array([cov(float(fx[j]),float(fy[j]),r["gt_box_frac"]) for j in range(n)]))
        G.append(np.full(n,i)); meta.append((r["question_id_full"],gh,gw))
    X=np.vstack(X);Y=np.concatenate(Y);G=np.concatenate(G)
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in (700,701,702):
        rng=np.random.default_rng(s); perm={g:k for k,g in enumerate(rng.permutation(gs))}
        Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    P/=3; out={}; off=0
    for qid,gh,gw in meta:
        n=gh*gw; s=P[off:off+n].reshape(gh,gw); off+=n
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        j=int(np.argmax(np.where(rm.ravel(),s.ravel(),-1e9)))
        out[qid]=[float((j%gw+.5)/gw),float((j//gw+.5)/gh)]
    return out
CELLS={L:fit_at(L) for L in EXITS}
print("refits done:",{L:len(v) for L,v in CELLS.items()},flush=True)
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
def fit_img(img,target=300,refine=6,tol=0.08):
    W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/r)**0.5
    return best[0]
def ans(im,t):
    inp=build(fit_img(im),t).to(model.device)
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset")
ex={f"{e['category']}/{e['question_id']}":e for e in load_dataset("craigwu/vstar_bench")["test"]}
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
t0,n=time.time(),0
with open(OUT,"a") as f:
    for r in rows:
        qid=r["question_id_full"]
        if qid in done or qid not in ex: continue
        e=ex[qid]; img=Image.open(os.path.join(root,e["image"])).convert("RGB")
        lab="ABCD".index(e["label"]) if isinstance(e["label"],str) else int(e["label"])
        rec={"qid":qid,"category":r["category"],"label":lab,"probs":{},
             "tl":{f"L{L}":300*(L+1) for L in EXITS}}
        for L in EXITS: rec["probs"][f"exit{L}"]=ans(crop(img,*CELLS[L][qid]),e["text"])
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
