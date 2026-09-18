"""
Phase 194: FUNDED MULTI-CROP -- two FULL-resolution ridge crops, paid for by composing three earlier results.
§35: the second crop converts (+7.8 ✔ single at fixed magnification) but halving the crop costs -11.3, so 2x150 nets
     -3.5. At full magnification two crops are worth +7.0 [+0.0,+13.9] on single -- at 149% of the bar.
§26C: pruning 90% of visual tokens at L16 costs 0.0-0.5pp, both models, any keep ratio (2 of 2).
§29:  the read-out needs layers only through L20 (exit there: -0.5 / +1.6 coverage); pruning the LOCALISER destroys it,
     but EXITING it does not.
BUDGET (token-layers; bar = 600x28 = 16,800)
    loc@300 all 28 layers + 2 crops@300 pruned at L16 = 8,400 + 10,860 = 19,260  (115%)  X
    loc@300 EXIT at L20  + 2 crops@300 pruned at L16 = 6,300 + 10,860 = 17,160  (102%)  <= inside the 10% gate
The ridge for the method arm is therefore refit on layers 0..20 only (that IS the early exit: L21-27 never computed).
ARMS  uniform@600 (bar) | ridge1@300 (incumbent, loc@300 all layers) | mc2p (METHOD, 102%)
      | mc2p_noexit (loc@300 all layers + 2 pruned crops; 115%, DIAGNOSTIC: what the exit costs)
      | oracle1@300
PRE-REGISTERED
    P1     mc2p - ridge1@300 on SINGLE-instance, CI clear of zero on BOTH models
    S1     mc2p - mc2p_noexit  (cost of the early exit, isolated)
    GUARD  cross-instance not significantly negative vs ridge1@300
    (§35 established cross-instance is NOT a coverage problem -- two full-res crops gave -2.6 -- so no cross gain is
     predicted; the guard only checks the second crop does not actively hurt there.)
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.model_selection import GroupKFold
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
NL=28; BLK=(16,27) if WHICH=="qwen3" else (15,27); EXIT=20; PRUNE_L=16; KEEP=0.10; MINSEP=3
OUT=f"{D}/data/phase194_funded_{WHICH}.jsonl"; B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None
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
def ridge_cells(rows,Lmax,k=2,lam=1.0):
    """OOF ridge using layers 0..Lmax only; returns top-k NMS cells per item."""
    X=[];Y=[];G=[]
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; n=q["n_img_tokens"]
        A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(Lmax+1)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        b0,b1=min(BLK[0],Lmax),min(BLK[1],Lmax+1); dep=A[b0:b1].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),(xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo]); Y.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),q["gt_box_frac"]) for i in range(n)])); G.append(np.full(n,gi))
    X=np.vstack(X); Y=np.concatenate(Y); G=np.concatenate(G); P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]; A_=Xt.T@Xt+lam*np.eye(Xt.shape[1]); A_[-1,-1]-=lam
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    P/=3; out={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; s=np.where(rm.ravel(),P[G==gi],-1e9); picks=[]
        for j in np.argsort(-s):
            if s[j]<=-1e8: break
            r_,c_=divmod(int(j),gw)
            if all(max(abs(r_-rr),abs(c_-cc))>=MINSEP for rr,cc in picks): picks.append((r_,c_))
            if len(picks)>=k: break
        while len(picks)<k: picks.append(picks[-1])
        out[q["question_id_full"]]={"cells":[[float((c+.5)/gw),float((r+.5)/gh)] for r,c in picks],"gt":q["gt_box_frac"]}
    return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    P70.W=W; rows=(P70.build() if WHICH=="qwen3" else P80.build())[4]
    C_exit=ridge_cells(rows,EXIT,2); C_full=ridge_cells(rows,NL-1,2)
    print(f"cells: exit-L{EXIT} ridge and full-depth ridge for {len(C_exit)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id; layers=model.model.language_model.layers
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
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def answer(imgs,t,prune=False,loc_tl=0):
        inp=build(imgs,t); rz=int((inp["input_ids"][0]==itid).sum()); inp=inp.to(model.device); clear(); tl=rz*NL
        if prune:
            pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base=int(pos[0]); n=int(len(pos))
            with torch.no_grad(): o=model(**inp,output_attentions=True)
            A=np.stack([o.attentions[L][0,:,-1,base:base+n].float().mean(0).cpu().numpy() for L in range(PRUNE_L-3,PRUNE_L+1)]); del o
            A=A/np.maximum(A.sum(1,keepdims=True),1e-12); s=A.mean(0); keep=max(1,int(round(KEEP*n)))
            drop=(base+np.argsort(-s)[keep:]).tolist(); b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(PRUNE_L+1,NL): layers[li].self_attn._prune_bias=b
            tl=n*(PRUNE_L+1)+keep*(NL-1-PRUNE_L)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        clear(); p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz,tl+loc_tl
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in C_exit or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); gt=C_exit[qid]["gt"]
            ce=[fit(win(img,cx,cy,W),300) for cx,cy in C_exit[qid]["cells"]]
            cf=[fit(win(img,cx,cy,W),300) for cx,cy in C_full[qid]["cells"]]
            ocrop=fit(win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),300)
            LOC_FULL=300*NL; LOC_EXIT=300*(EXIT+1)
            plan={"uniform@600":([fit(img,600)],False,0),"ridge1@300":([cf[0]],False,LOC_FULL),
                  "mc2p":(ce,True,LOC_EXIT),"mc2p_noexit":(cf,True,LOC_FULL),"oracle1@300":([ocrop],False,LOC_FULL)}
            rec={"question_id_full":qid,"category":ex["category"],"label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{},"token_layers":{}}
            for nm,(ims,pz,lt) in plan.items():
                p,rz,tl=answer(ims,ex["text"],pz,lt); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz; rec["token_layers"][nm]=tl
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
