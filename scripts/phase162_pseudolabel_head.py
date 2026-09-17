"""
Phase 162 (architecture fix 3): a head trained WITHOUT BOXES -- pseudo-labels from confidence gain.

The head regresses coverage, which needs ground-truth boxes on the training benchmark. Phase 32 /
SS6E found the two-pass confidence comparison is an unsupervised COVERAGE detector (AUROC 0.885 on
Qwen2-VL). So label each candidate cell by what cropping there does to the model's own confidence:

    for each training item: candidates = top-K cells of the label-free gated-max map (K=6)
    for each candidate c: crop@0.25 at c, re-encode at 300, record max-softmax over options
    pseudo-target(c) = conf(crop at c) - conf(uniform@300)        <- no answer labels, no boxes

Train the same GBT on the same 65 features against pseudo-targets (cells not in the candidate set
get target 0 as weak negatives). Evaluate OUT-OF-FOLD by item on the REAL coverage metric and on the
end task (the OOF head's crop is answered in the same run, so no second GPU pass).

PRE-REGISTERED
    P1  pseudo-label head OOF coverage vs deployed argmax (46.1 / 39.3) -- CI clear on both models
    P2  pseudo-label head end-task on single-object vs uniform@600 -- the box-free method's bar test
    context  the boxed head (63.4 / 54.5 coverage; +15.7 / +11.3 end-task) is the ceiling for this recipe
Cost: 6 crops per item = 1,146 passes per model, once, and it uses only the benchmark's images and
questions -- no boxes and no answers.
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1] if len(sys.argv)>1 else "qwen3"
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
ATTN={"qwen3":f"{D}/data/phase30c_attn_maps_all.jsonl","qwen2":f"{D}/data/phase74_Qwen2_VL_7B_Instruct.jsonl"}[WHICH]
OUT=f"{D}/data/phase162_pseudo_{WHICH}.jsonl"; B0,W,K=300,0.25,6
GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[WHICH]   # phase-95 divergence gate (label-free)
Image.MAX_IMAGE_PIXELS=None

def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    P70.W=W
    if WHICH=="qwen3": X,Y,G,DEP,rows,_=P70.build()
    else: X,Y,G,DEP,rows=P80.build()
    # candidates from the label-free gated max
    cand={}; ringm={}
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(28)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        s=A[GATE].max(0); rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        s=np.where(rm.ravel(),s,-1e9); cand[gi]=list(np.argsort(-s)[:K]); ringm[gi]=rm.ravel()
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}); model.eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def build(img,text):
        m=[{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}]
        return pr(images=img,text=pr.apply_chat_template(m,tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build(img,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=5,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best
    def window(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih)
        return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(img,text):
        inp=build(img,text); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); return p.tolist()
    byq={f"{ex['category']}/{ex['question_id']}":ex for ex in ds}
    # ---- stage 1: pseudo-labels (GPU)
    pseudo=np.zeros(len(Y)); conf_u={}; probs_u={}; imgs={}
    t0=time.time()
    for gi,r in enumerate(rows):
        qid=r["question_id_full"]; ex=byq[qid]; ip=os.path.join(root,ex["image"]); img=Image.open(ip).convert("RGB"); imgs[gi]=img
        gh,gw=r["grid"]; pu=answer(fit(img,B0)[0],ex["text"]); conf_u[gi]=max(pu); probs_u[gi]=pu
        base=np.where(G==gi)[0][0]
        for c in cand[gi]:
            cx,cy=((c%gw)+.5)/gw,((c//gw)+.5)/gh; pc=answer(fit(window(img,cx,cy,W),B0)[0],ex["text"])
            pseudo[base+c]=max(pc)-conf_u[gi]
        if (gi+1)%20==0: print(f"  pseudo [{gi+1}/{len(rows)}] {(gi+1)/(time.time()-t0):.2f} it/s",flush=True)
    # ---- stage 2: OOF head on pseudo-labels; eval REAL coverage; end-task of the OOF pick (GPU)
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            # training rows: all candidates (pseudo-labelled) + 30 random non-candidates per item as weak zeros
            cset=set(); 
            for gi in np.unique(G[tr]): cset.update(np.where(G==gi)[0][0]+np.array(cand[gi]))
            cidx=np.array(sorted(cset)); noncand=np.setdiff1d(tr,cidx); neg=rng.choice(noncand,size=min(len(noncand),30*len(np.unique(G[tr]))),replace=False)
            sub=np.concatenate([cidx,neg]); m=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(X[sub],pseudo[sub])
            P[te]+=m.predict(X[te])
    P/=3
    hit=[]; res=[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; m=G==gi; j=int(np.argmax(np.where(ringm[gi],P[m],-1e9))); jd=int(np.argmax(np.where(ringm[gi],DEP[gi],-1e9)))
        y=Y[m]; hit.append((float(y[j]>=P70.COV_HIT),float(y[jd]>=P70.COV_HIT)))
        ex=byq[r["question_id_full"]]; label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
        cx,cy=((j%gw)+.5)/gw,((j//gw)+.5)/gh; pcrop=answer(fit(window(imgs[gi],cx,cy,W),B0)[0],ex["text"]); p600=answer(fit(imgs[gi],2*B0)[0],ex["text"])
        res.append({"question_id_full":r["question_id_full"],"category":r["category"],"label":label,"pseudo_cov":float(y[j]),"argmax_cov":float(y[jd]),
                    "probs":{"uniform@300":probs_u[gi],"uniform@600":p600,"pseudo@0.25":pcrop}})
    with open(OUT,"w") as f:
        for r in res: f.write(json.dumps(r)+"\n")
    h=np.array(hit); print(f"\n{WHICH}: OOF coverage  pseudo-label head {h[:,0].mean()*100:.1f}%  deployed argmax {h[:,1].mean()*100:.1f}%",flush=True)
    cat=np.array([r["category"] for r in res]); acc=lambda k: np.array([int(np.argmax(r["probs"][k])==r["label"]) for r in res],float)
    rng=np.random.default_rng(162)
    for s_ in ["direct_attributes","relative_position","ALL"]:
        mm=np.ones(len(res),bool) if s_=="ALL" else cat==s_; d=(acc("pseudo@0.25")-acc("uniform@600"))[mm]; n=len(d)
        b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)])
        print(f"  {s_:>18}: pseudo@0.25 {acc('pseudo@0.25')[mm].mean()*100:5.1f}%  bar {acc('uniform@600')[mm].mean()*100:5.1f}%  delta {d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}] {'CLEARS' if np.percentile(b,2.5)>0 else ''}",flush=True)
main()
