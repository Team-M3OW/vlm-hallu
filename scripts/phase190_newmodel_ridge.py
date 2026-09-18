"""
Phase 190a: RIDGE replication on the extra checkpoints (Qwen2.5-VL-7B = q25_7b, Qwen3-VL-8B = q3_8b).
Phase 154 ran the full pipeline with the GBT head; every breadth number in the paper uses those tree placements. Here the
§21 ridge is fit OOF on the same phase-154 maps and the end task is re-run with BOTH placements in one run:
    uniform@300 | uniform@600 (bar) | head@0.25 (phase-154 tree cells, re-answered) | ridge@0.25 | oracle@0.25
PRE-REGISTERED per checkpoint: P1 ridge - bar (pooled and single); P2 ridge - head.  usage: <tag> <MODEL_ID>
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts"); import phase70_rerank_head as P70
from sklearn.model_selection import GroupKFold
TAG,MODEL_ID=sys.argv[1],sys.argv[2]; ATTN=f"{D}/data/phase154_{TAG}_attn.jsonl"; PROP154=f"{D}/data/phase154_{TAG}_proposals.json"; OUT=f"{D}/data/phase190_{TAG}_ridge_endtask.jsonl"
B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None; P70.W=W
def ridge_cells(rows,NL,BLK,lam=1.0):
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
    Pr/=3; out={}; hits=[]
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; j=int(np.argmax(np.where(rm.ravel(),Pr[Gr==gi],-1e9)))
        out[q["question_id_full"]]=[float((j%gw+.5)/gw),float((j//gw+.5)/gh)]; hits.append(float(Yr[Gr==gi][j]>=0.5))
    print(f"{TAG}: ridge OOF coverage {np.mean(hits)*100:.1f}%",flush=True); return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    rows=[json.loads(l) for l in open(ATTN)]; NL=len([k for k in rows[0]["attn"] if k.startswith("L")]); BLK=(int(.57*NL),int(.93*NL)+1)
    RC=ridge_cells(rows,NL,BLK); P154=json.load(open(PROP154))
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval(); pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    MS=getattr(pr.image_processor,"merge_size",2); opt=[sorted({tok(s,add_special_tokens=False)["input_ids"][-1] for s in [c,f" {c}"]}) for c in "ABCD"]
    def build(img,text): return pr(images=img,text=pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//(MS*MS) for g in build(img,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=5,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def window(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih); return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(img,text):
        inp=build(img,text); rz=int(sum(g[1]*g[2]//(MS*MS) for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0=time.time(); n=0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in RC or qid not in P154 or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); gt=P154[qid]["gt_box_frac"]
            arms={"uniform@300":fit(img,B0),"uniform@600":fit(img,2*B0),"head@0.25":fit(window(img,*P154[qid]["head"],W),B0),
                  "ridge@0.25":fit(window(img,*RC[qid],W),B0),"oracle@0.25":fit(window(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)}
            rec={"question_id_full":qid,"category":ex["category"],"label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{}}
            for nm,im in arms.items():
                p,rz=answer(im,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"done {n} -> {OUT}",flush=True)
main()
