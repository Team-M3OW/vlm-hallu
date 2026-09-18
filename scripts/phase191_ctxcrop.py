"""
Phase 191: CONTEXT-CONTINUED CROP -- make depth re-ranking work on RELATIONAL questions without a router.
WHY IT SHOULD: §22 -- a tight crop removes the second object; relational responds to PIXELS, not crops (uniform@900
relational +9.2 on Qwen3). §20C tried crop + a 64-token thumbnail in a 300-token answer pass: failed (0/2, guard breach).
THE CHANGE: the localiser pass already encoded the FULL scene at 300 tokens. Keep it. The answer pass CONTINUES that
sequence with the crop: [global@300, question][crop@300, question, answer]. The model answers with both the scene
(relational) and the magnified crop (single) in context. Compute: the prefix is bit-identical to the localiser pass, so
deployment caches its KV and appends 300 crop tokens + one question: 600 image tokens + 2 questions = the bar (+~40
text tokens). Implemented here as one forward over the full sequence (identical outputs to cache+append).
ARMS  uniform@600 (bar) | ridge300 (incumbent: crop only) | ridge_ctx (global@300 + ridge crop@300, one sequence; METHOD)
      | oracle_ctx (global + oracle crop) | oracle300
PRE-REGISTERED
    P1     ridge_ctx - bar on RELATIONAL, CI clear of zero on BOTH models
    GUARD  ridge_ctx - ridge300 on SINGLE not significantly negative on both
    P2     ridge_ctx - bar pooled, both.   Budget gate: measured image tokens; text overhead reported.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
NL=28; BLK=(16,27) if WHICH=="qwen3" else (15,27); OUT=f"{D}/data/phase191_ctxcrop_{WHICH}.jsonl"; B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None
def ridge_cells(rows,lam=1.0):
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
        gh,gw=q["grid"]; rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; j=int(np.argmax(np.where(rm.ravel(),Pr[Gr==gi],-1e9)))
        out[q["question_id_full"]]={"cell":[float((j%gw+.5)/gw),float((j//gw+.5)/gh)],"gt":q["gt_box_frac"]}
    return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    P70.W=W; rows=(P70.build() if WHICH=="qwen3" else P80.build())[4]; PL=ridge_cells(rows); print(f"ridge placements for {len(PL)}",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(content): return pr.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
    def build(imgs,content): return pr(images=imgs,text=chat(content),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build([i],[{"type":"image"},{"type":"text","text":"x"}])["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def win(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih); return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(imgs,content):
        inp=build(imgs,content); rz=int((inp["input_ids"][0]==itid).sum()); ntext=int(inp["input_ids"].shape[1]-rz); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz,ntext
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); gt=PL[qid]["gt"]; cx,cy=PL[qid]["cell"]; T=ex["text"]
            g300=fit(img,B0); crop=fit(win(img,cx,cy,W),B0); ocrop=fit(win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)
            one=lambda im:([im],[{"type":"image"},{"type":"text","text":T}])
            two=lambda a,b:([a,b],[{"type":"image"},{"type":"text","text":T},{"type":"image"},{"type":"text","text":T}])
            arms={"uniform@600":one(fit(img,2*B0)),"ridge300":one(crop),"ridge_ctx":two(g300,crop),"oracle_ctx":two(g300,ocrop),"oracle300":one(ocrop)}
            rec={"question_id_full":qid,"category":ex["category"],"label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{},"text_tokens":{}}
            for nm,(ims,content) in arms.items():
                p,rz,nt=answer(ims,content); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz+(0 if nm=="uniform@600" or "ctx" in nm else B0); rec["text_tokens"][nm]=nt
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
