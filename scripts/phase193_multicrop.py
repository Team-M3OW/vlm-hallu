"""
Phase 193: MULTI-CROP ON THE RIDGE -- and the first measurement of the crop's TOKEN-COUNT curve.
WHY IT IS OPEN. Multi-crop won with a weak proposer (§14M/phase 58: argmax + 4 crops +7.9) and lost with the tree head
(-3.1); phase 168's proportional top-3 was -2.6 vs the tree on Qwen3 but +7.3 ✔ on Qwen2 (landing at 70.2 pooled --
exactly what the ridge single crop reaches there). It has never been run on the RIDGE, and the quantity that decides it
has never been measured: what does a crop at 150 or 100 tokens cost versus 300? §27 showed 450 buys nothing over 300,
but says nothing about going down.
DECOMPOSITION (every arm: localise@300 with the ridge, answer pass totals 300 tokens, total 600 = the bar)
    ridge1@300   incumbent (§25)
    ridge1@150   ONE crop, half the tokens        -> isolates the MAGNIFICATION TAX
    ridge2@150   TWO crops (NMS-separated top-2)  -> tax + COVERAGE GAIN;  (ridge2@150 - ridge1@150) = the gain alone
    ridge1@100 / ridge3@100   same at k=3
    oracle1@300  ceiling
Also relevant to §32: two crops are both high-resolution and both plausibly relevant, unlike the low-res scene that
diluted the crop by 8.7pp at oracle placement. ridge2@150 - ridge1@150 tests whether that dilution generalises.
PRE-REGISTERED
    P1     ridge2@150 - ridge1@300, pooled, CI clear of zero on BOTH models
    S1     ridge2@150 - ridge1@150   (coverage gain from the second crop, magnification held fixed)
    S2     ridge1@150 - ridge1@300   (magnification tax; expected <=0, magnitude unknown)
    RELATIONAL, reported as a secondary with its own CI: two crops can cover two objects, which no single crop can.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
NL=28; BLK=(16,27) if WHICH=="qwen3" else (15,27); OUT=f"{D}/data/phase193_multicrop_{WHICH}.jsonl"
B0,W,MINSEP=300,0.25,3; Image.MAX_IMAGE_PIXELS=None
def ridge_scores(rows,lam=1.0):
    """OOF ridge (§21). Returns per item: the ranked cell list (NMS-separated) and the gt box."""
    Xr=[];Yr=[];Gr=[]
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; n=q["n_img_tokens"]
        A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        dep=A[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge"); nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),(xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        Xr.append(np.c_[LA,R,geo]); Yr.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),q["gt_box_frac"]) for i in range(n)])); Gr.append(np.full(n,gi))
    Xr=np.vstack(Xr); Yr=np.concatenate(Yr); Gr=np.concatenate(Gr); Pr=np.zeros(len(Yr)); gs=np.unique(Gr)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(Gr)
        for tr,te in GroupKFold(5).split(Xr,Yr,Gp):
            mu,sd=Xr[tr].mean(0),Xr[tr].std(0)+1e-9; Xt=np.c_[(Xr[tr]-mu)/sd,np.ones(len(tr))]; A_=Xt.T@Xt+lam*np.eye(Xt.shape[1]); A_[-1,-1]-=lam
            w=np.linalg.solve(A_,Xt.T@Yr[tr]); Pr[te]+=np.c_[(Xr[te]-mu)/sd,np.ones(len(te))]@w
    Pr/=3; out={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; s=np.where(rm.ravel(),Pr[Gr==gi],-1e9)
        order=np.argsort(-s); picks=[]
        for j in order:                                   # NMS: >= MINSEP cells apart in Chebyshev distance
            if s[j]<=-1e8: break
            r_,c_=divmod(int(j),gw)
            if all(max(abs(r_-rr),abs(c_-cc))>=MINSEP for rr,cc in picks): picks.append((r_,c_))
            if len(picks)>=3: break
        while len(picks)<3: picks.append(picks[-1])
        out[q["question_id_full"]]={"cells":[[float((c+.5)/gw),float((r+.5)/gh)] for r,c in picks],"gt":q["gt_box_frac"]}
    return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    P70.W=W; rows=(P70.build() if WHICH=="qwen3" else P80.build())[4]; PL=ridge_scores(rows); print(f"ridge multi-cells for {len(PL)}",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(nimg,t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"} for _ in range(nimg)]+[{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
    def build(imgs,t): return pr(images=imgs,text=chat(len(imgs),t),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build([i],"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def win(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih); return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(imgs,t):
        inp=build(imgs,t); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); gt=PL[qid]["gt"]; C=PL[qid]["cells"]
            crops=[win(img,cx,cy,W) for cx,cy in C]; ocrop=win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W)
            arms={"uniform@600":[fit(img,2*B0)],
                  "ridge1@300":[fit(crops[0],300)],
                  "ridge1@150":[fit(crops[0],150)],
                  "ridge2@150":[fit(crops[0],150),fit(crops[1],150)],
                  "ridge1@100":[fit(crops[0],100)],
                  "ridge3@100":[fit(c,100) for c in crops],
                  "oracle1@300":[fit(ocrop,300)]}
            rec={"question_id_full":qid,"category":ex["category"],"label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),
                 "cells":C,"probs":{},"realized_tokens":{}}
            for nm,ims in arms.items():
                p,rz=answer(ims,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz+(0 if nm=="uniform@600" else B0)
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
