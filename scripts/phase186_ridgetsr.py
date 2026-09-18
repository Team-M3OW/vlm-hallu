"""
Phase 186: RIDGE x TSR -- spend the pruning saving on the CROP pass, where resolution already converts on both models.
§26C: pruning 90% of visual tokens at L16 is free (2 of 2) but funds resolution whose conversion is model-dependent
(whole image: Qwen3 +5.8 ok, Qwen2 +3.1 n.s.). §25: the ridge crop converts on BOTH models (+8.9/+12.0 vs bar).
So: localise@300 unpruned (the ridge reads all 28 layers) -> ridge cell -> crop encoded at 450 tokens, pruned to
10% at L16 -> answer. Token-layers 8,400 + (450*17 + 45*11) = 16,545 <= uniform@600 = 16,800.
ARMS  uniform@600 (bar) | ridge300 (incumbent: crop@300 unpruned) | ridge450p (crop@450 pruned L16 k=.10; METHOD)
      | ridge600p (crop@600 pruned; 19,260 TL = 115%, DIAGNOSTIC only) | oracle450p (ceiling)
PRE-REGISTERED
    P1     ridge450p - ridge300, pooled, CI clear of zero on BOTH models   (the crop-resolution gain)
    GUARD  ridge450p - ridge300 on RELATIONAL not significantly negative on both
    P2     ridge450p - bar, both models (must hold if §25 holds)
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase185_tsr_{WHICH}.jsonl"; NL=28; P=16; Image.MAX_IMAGE_PIXELS=None
ARMS={"tsr900":(900,16,0.10,"win"),"tsr900_atP":(900,16,0.10,"atP"),"tsr900_rand":(900,16,0.10,"rand"),
      "fastv900":(900,2,0.627,"atP"),"tsr600":(600,16,0.10,"win")}
def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM3.eager_attention_forward=make_patched(QM3); QM2.eager_attention_forward=make_patched(QM2)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase186_ridgetsr_{WHICH}.jsonl"; B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None
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
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    layers=model.model.language_model.layers; NL=len(layers); itid=model.config.image_token_id
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
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def answer(imgs,t,prune=None):
        """prune=(P,k): read last-token attention (mean L(P-4)..P) at the prune point of THIS pass, keep top k of image tokens for layers P+1.."""
        inp=build(imgs,t); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device); clear()
        tl=rz*NL
        if prune is not None:
            P,k=prune; pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
            with torch.no_grad(): o=model(**inp,output_attentions=True)
            A=np.stack([o.attentions[L][0,:,-1,base:base+nt].float().mean(0).cpu().numpy() for L in range(max(0,P-4),P+1)]); del o
            A=A/np.maximum(A.sum(1,keepdims=True),1e-12); s=A.mean(0); keep=max(1,int(round(k*nt)))
            drop=(base+np.argsort(-s)[keep:]).tolist(); b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(P+1,NL): layers[li].self_attn._prune_bias=b
            tl=nt*(P+1)+keep*(NL-1-P)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        clear(); p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz,tl
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            p=PL[qid]; img=Image.open(ip).convert("RGB"); gt=p["gt"]
            cr=win(img,*p["ridge"],W); ocr=win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W)
            arms={"uniform@600":([fit(img,2*B0)],None),"ridge300":([fit(cr,B0)],None),
                  "ridge450p":([fit(cr,450)],(16,0.10)),"ridge600p":([fit(cr,600)],(16,0.10)),"oracle450p":([fit(ocr,450)],(16,0.10))}
            rec={"question_id_full":qid,"category":ex["category"],
                 "label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{},"token_layers":{}}
            for nm,(ims,pz) in arms.items():
                pr_,rz,tl=answer(ims,ex["text"],pz); rec["probs"][nm]=pr_; rec["realized_tokens"][nm]=rz
                rec["token_layers"][nm]=tl+(0 if nm=="uniform@600" else B0*NL)      # + the unpruned 300-token localiser pass
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
