"""
Phase 184: DOES THE RIDGE DEPTH FILTER CONVERT? (§21 claims ridge == GBT as a ranker; this tests the end task)
§21: ridge ties the GBT on coverage on 4 architectures. Phase 182 check: the two agree on the SAME CELL only
38%/34% of the time (within 2 cells on 84%), so equal coverage does NOT imply equal accuracy -- §14W is the
precedent where a coverage-equal arm converted 3.1pp worse. Same pipeline as phase 179 (localise@300 -> crop
W=0.25 -> answer@300, bar uniform@600); the only new arm is the ridge placement.
PRE-REGISTERED
    P1  ridge - head(GBT), pooled: |CI| must EXCLUDE a 3pp loss on BOTH models for ridge to replace the tree
        (equivalence test, not superiority: we adopt ridge only if it is not meaningfully worse)
    context: ridge - bar, and per stratum.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase184_allarms_{WHICH}.jsonl"; B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None
NL,BLK=(28,(16,27)) if WHICH=="qwen3" else (28,(15,27))
def placements():
    build=P70.build if WHICH=="qwen3" else P80.build
    P70.W=W; r=build(); X,Y,G,rows=r[0],r[1],r[2],r[4]
    Pt=P70.oof(X,Y,G)
    Xr=[];Yr=[];Gr=[]
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; n=q["n_img_tokens"]
        A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        dep=A[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        Xr.append(np.c_[LA,R,geo]); Yr.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),q["gt_box_frac"]) for i in range(n)])); Gr.append(np.full(n,gi))
    Xr=np.vstack(Xr); Yr=np.concatenate(Yr); Gr=np.concatenate(Gr); Pr=np.zeros(len(Yr)); gs=np.unique(Gr)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(Gr)
        for tr,te in GroupKFold(5).split(Xr,Yr,Gp):
            mu,sd=Xr[tr].mean(0),Xr[tr].std(0)+1e-9; Xt=np.c_[(Xr[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Yr[tr]); Pr[te]+=np.c_[(Xr[te]-mu)/sd,np.ones(len(te))]@w
    Pr/=3
    Pr2=np.zeros(len(Yr))
    for s2 in range(3):
        rng=np.random.default_rng(800+s2); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(Gr)
        for tr,te in GroupKFold(5).split(Xr,Yr,Gp):
            mu,sd=Xr[tr].mean(0),Xr[tr].std(0)+1e-9; Xt=np.c_[(Xr[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Yr[tr]); Pr2[te]+=np.c_[(Xr[te]-mu)/sd,np.ones(len(te))]@w
    Pr2/=3; out={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; m=Gr==gi; rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel()
        jt=int(np.argmax(np.where(rm,Pt[G==gi],-1e9))); jr=int(np.argmax(np.where(rm,Pr[m],-1e9))); jr2=int(np.argmax(np.where(rm,Pr2[m],-1e9)))
        c=lambda j:[float((j%gw+.5)/gw),float((j//gw+.5)/gh)]
        out[q["question_id_full"]]={"gt":q["gt_box_frac"],"head":c(jt),"ridge":c(jr),"ridge_seed2":c(jr2)}
    return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    PL=placements()
    B=json.load(open(f"{D}/data/phase179_placements_{WHICH}.json"))
    for q,v in PL.items():
        for k in ("vicrop_block","vicrop_L14","gatemax","laser"):
            if q in B and k in B[q]["cells"]: v[k]=B[q]["cells"][k]
    print(f"placements for {len(PL)} items; arms per item: {sorted(set(PL[list(PL)[0]])-{'gt'})}",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(n,t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"}]*n+[{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(len(i),t),return_tensors="pt")
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
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih)
        return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(imgs,t):
        inp=build(imgs,t); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            p=PL[qid]; img=Image.open(ip).convert("RGB"); gt=p["gt"]
            arms={"uniform@600":[fit(img,2*B0)],"oracle":[fit(win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)]}
            for k in ("head","ridge","ridge_seed2","vicrop_block","vicrop_L14","gatemax","laser"):
                if k in p: arms[k]=[fit(win(img,*p[k],W),B0)]
            rec={"question_id_full":qid,"category":ex["category"],
                 "label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{}}
            for nm,ims in arms.items():
                pr_,rz=answer(ims,ex["text"]); rec["probs"][nm]=pr_; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
