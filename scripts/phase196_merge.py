"""
Phase 196: MERGE ridge + TSR -- the ridge decides WHERE, pruning decides HOW MUCH, and nothing is cropped.
§40: cropping wins single-instance and loses cross-instance (frame destroyed); TSR wins cross-instance and gives up
     the crop's magnification. The two have never been combined without a crop.
§26C: at L16 the token RANKING is irrelevant (random == attention) -- transport is done, tokens are inert.
§28/§31: at L4-L12 ranking SHOULD matter (in-window pruning costs when unguided; a probe at L4 retains the box 87/75%).
THE MERGE: localise@300 (exit L20, §29) -> ridge score map -> encode the FULL image at 900 tokens -> prune at L4
keeping the top 25% by the ridge map (nearest-neighbour upsampled from the 300-grid) -> answer. No crop, so the frame
and both objects survive; the retained tokens are at 1.5x the bar's resolution, so the evidence is magnified.
BUDGET (token-layers; bar = 600*NL)
    loc@300 exit L20 = 300*21 = 6,300   +   900*5 + 225*23 = 9,675   =   15,975   (95% of 16,800)
ARMS  uniform@600 (BAR) | ridge1@300 (crop incumbent) | tsr900 (no-crop incumbent, attention-pruned at L16)
      | merge (METHOD: ridge-guided prune at L4, keep 25%)
      | merge_attn (same but ranked by attention at L4 -- isolates the RIDGE)
      | merge_L16 (ridge-guided but pruned at L16 -- isolates the DEPTH; §26C predicts it degenerates to tsr900)
PRE-REGISTERED
    P1     merge - bar, pooled, CI clear of zero on BOTH models
    P2     merge - tsr900 on SINGLE-instance (does ridge targeting recover the crop's magnification advantage?)
    GUARD  merge - tsr900 on CROSS-instance not significantly negative (the frame must survive)
    S1     merge - merge_attn (the ridge is the variable)   S2  merge - merge_L16 (the depth is the variable)
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.model_selection import GroupKFold
import importlib
_QMODS=[]
for _m in ("qwen3_vl.modeling_qwen3_vl","qwen2_vl.modeling_qwen2_vl","qwen2_5_vl.modeling_qwen2_5_vl"):
    try: _QMODS.append(importlib.import_module(f"transformers.models.{_m}"))
    except Exception: pass
from transformers import AutoProcessor, AutoModelForImageTextToText
WHICH=sys.argv[1]; MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
BLK=(16,27) if WHICH=="qwen3" else (15,27); OUT=f"{D}/data/phase196_merge_{WHICH}.jsonl"
B0,W,E_ANS,KEEP=300,0.25,900,0.25; Image.MAX_IMAGE_PIXELS=None
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
for _M in _QMODS:
    if hasattr(_M,"eager_attention_forward"): _M.eager_attention_forward=make_patched(_M)
def ridge_maps(rows,Lmax,lam=1.0):
    """OOF ridge using layers 0..Lmax; returns the FULL per-cell score map (not just the argmax) + grid + top cell."""
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
        gh,gw=q["grid"]; sc=P[G==gi].copy(); rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True
        j=int(np.argmax(np.where(rm.ravel(),sc,-1e9)))
        out[q["question_id_full"]]={"grid":[gh,gw],"score":[float(v) for v in sc],
                                   "cell":[float((j%gw+.5)/gw),float((j//gw+.5)/gh)],"gt":q["gt_box_frac"]}
    return out
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    P70.W=W; rows=(P70.build() if WHICH=="qwen3" else P80.build())[4]
    RM=ridge_maps(rows,20)          # exit at L20 -> this IS the early-exit localiser
    print(f"ridge maps (layers<=20) for {len(RM)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
    layers=model.model.language_model.layers; NL=len(layers); MS=getattr(pr.image_processor,"merge_size",2)
    own=type(layers[0].self_attn).__module__
    assert any(own==M.__name__ for M in _QMODS), f"attention module {own} NOT patched"
    PL=4; PL16=int(round(0.57*NL))
    print(f"{WHICH}: NL={NL}, prune-early L{PL}, prune-late L{PL16}, patch verified on {own}",flush=True)
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def build(img,t): return pr(images=img,text=pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//(MS*MS) for g in build(i,"x")["image_grid_thw"].tolist()))
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
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def upsample(score,g_src,g_dst):
        """nearest-neighbour the 300-grid ridge map onto the 900-grid"""
        sh,sw=g_src; dh,dw=g_dst; S=np.asarray(score,float).reshape(sh,sw)
        yi=np.clip((np.arange(dh)+.5)*sh/dh,0,sh-1e-6).astype(int); xi=np.clip((np.arange(dw)+.5)*sw/dw,0,sw-1e-6).astype(int)
        return S[np.ix_(yi,xi)].ravel()
    def answer(img,t,prune=None,loc_tl=0):
        inp=build(img,t); rz=int((inp["input_ids"][0]==itid).sum()); g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//MS,g[2]//MS
        inp=inp.to(model.device); clear(); tl=rz*NL
        if prune is not None:
            mode,P_,src=prune; pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,n=int(pos[0]),int(len(pos))
            if mode=="attn":
                with torch.no_grad(): o=model(**inp,output_attentions=True)
                A=np.stack([o.attentions[L][0,:,-1,base:base+n].float().mean(0).cpu().numpy() for L in range(max(0,P_-3),P_+1)]); del o
                A=A/np.maximum(A.sum(1,keepdims=True),1e-12); s=A.mean(0); torch.cuda.empty_cache()
            else:
                s=upsample(src["score"],src["grid"],[gh,gw]) if gh*gw==n else np.resize(np.asarray(src["score"],float),n)
            keep=max(1,int(round(KEEP*n))); drop=(base+np.argsort(-s)[keep:]).tolist()
            b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(P_+1,NL): layers[li].self_attn._prune_bias=b
            tl=n*(P_+1)+keep*(NL-1-P_)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        clear(); p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz,tl+loc_tl
    LOC_EXIT=B0*21; LOC_FULL=B0*NL
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in RM or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); r=RM[qid]
            big=fit(img,E_ANS); crop=fit(win(img,*r["cell"],W),B0)
            plan={"uniform@600":(fit(img,2*B0),None,0),
                  "ridge1@300":(crop,None,LOC_FULL),
                  "tsr900":(big,("attn",int(round(0.57*NL)),None),0),
                  "merge":(big,("ridge",4,r),LOC_EXIT),
                  "merge_attn":(big,("attn",4,None),0),
                  "merge_L16":(big,("ridge",int(round(0.57*NL)),r),LOC_EXIT)}
            rec={"question_id_full":qid,"category":ex["category"],"label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{},"token_layers":{}}
            for nm,(im,pz,lt) in plan.items():
                p,rz,tl=answer(im,ex["text"],pz,lt); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz; rec["token_layers"][nm]=tl
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
