"""
Phase 226: DWA-guided contrastive decoding, in the NO-HEADROOM regime.

WHY RUN THIS AGAIN. Phase 77 (§14H) already killed contrastive decoding on V*: the ORACLE arm
(masking the ground-truth box) reached only +2.6 [-0.5,+5.8] n.s., and the method lost -5.5 to its
compute-matched bar because two passes spent on resolution bought +7.4 instead. A better localiser
cannot beat that oracle, so swapping in DWA changes nothing THERE.

What is untested is the regime where that economic argument collapses. On CV-Bench the resolution
headroom is ~0 (uniform@hi - uniform@lo = +0.9), so two passes spent on resolution buy almost
nothing: the hurdle a 2-pass method must clear is ~+1 rather than +7.4. And §14H's diagnosis
("the region is read, but what is there is too impoverished to answer from") is specific to V*'s
SUB-TOKEN targets; CV-Bench's objects -- things to be counted, compared for depth -- are plainly
visible and multi-token.

ARMS, per item. The z+ pass doubles as the localise pass (attention is captured from it), so the
method costs exactly two forward passes.
  bar         uniform@E_LO, one pass                              (the incumbent baseline)
  hi2x        uniform@(2*E_LO), one pass                          (the COMPUTE-MATCHED bar for 2 passes)
  cd_dwa      z+ = bar pass; z- = same budget, DWA top-K cells masked from attention;
              s = z+ + alpha*(z+ - z-)                            (the method)
  cd_rand     identical but a RANDOM cell set of the same size    (the control that killed §10B arms)
alpha swept over {0.5, 1.0, 2.0}; LASER uses alpha=1.

PRE-REGISTERED, before any number is seen:
  Adopt only if (cd_dwa - hi2x) excludes zero at some alpha AND (cd_dwa - cd_rand) excludes zero at
  the same alpha. Beating `bar` alone is NOT sufficient: cd_dwa costs 2x and bar costs 1x.
  This is the EIGHTH internal-intervention attempt; the previous seven failed.

Usage: phase226_cd_dwa.py <model-key> <benchmark> [n]
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
from transformers import AutoProcessor, AutoModelForImageTextToText
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Image.MAX_IMAGE_PIXELS=None
MODELS={"qwen3_2b":("Qwen/Qwen3-VL-2B-Instruct","qwen3"),"qwen2_7b":("Qwen/Qwen2-VL-7B-Instruct","qwen2")}
MK,BK=sys.argv[1],sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 0
mid,wtag=MODELS[MK]; OUT=f"{D}/data/phase226_cd_{MK}_{BK}.jsonl"
E_LO=600; ALPHAS=[0.5,1.0,2.0]; KFRAC=0.0625      # = the W=0.25 window's share of cells
Wv=np.load(f"{D}/data/fig_ridge_w_{wtag}.npy")
model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
layers=None
for f in (lambda m:m.model.language_model.layers, lambda m:m.language_model.model.layers, lambda m:m.model.layers):
    try: layers=f(model); break
    except Exception: pass
NL=len(layers); P=round(0.57*NL); BLK=(P,NL-1)
itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
QM=importlib.import_module(type(layers[0].self_attn).__module__)
def mk(QM):
    def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        if getattr(module,"_capture",False):
            module._lastrow=w[0,:,-1,:].float().mean(0).detach().cpu().numpy()
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM.eager_attention_forward=mk(QM)
print(f"{MK}/{BK}: NL={NL} band L{BLK[0]}-{BLK[1]-1}",flush=True)
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0]==itid).sum())
def fit(img,target,refine=6,tol=0.10):
    W_,H_=img.size; sc=(target/max(ntok(build(img,"x")),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(32,int(W_*sc)),max(32,int(H_*sc))),Image.BICUBIC); r=ntok(build(cur,"x"))
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/max(r,1))**0.5
    return best[0]
def clear():
    for l in layers:
        if hasattr(l.self_attn,"_prune_bias"): delattr(l.self_attn,"_prune_bias")
def logits_of(inp,mask_pos=None):
    clear()
    if mask_pos is not None and len(mask_pos):
        bias=torch.zeros(inp["input_ids"].shape[1],device=model.device); bias[list(mask_pos)]=-1e4
        for l in layers: l.self_attn._prune_bias=bias
    with torch.no_grad(): out=model(**inp)
    lg=out.logits[0,-1].float().cpu().numpy(); del out; clear(); torch.cuda.empty_cache()
    return lg
def grid_of(inp):
    if "image_grid_thw" in inp:
        g=inp["image_grid_thw"].tolist()[0]; return g[1]//2,g[2]//2,0
    n=ntok(inp); s=int(round(n**0.5)); return (s,s,0) if s*s==n else (0,0,0)
def dwa_rank(A,gh,gw):
    a=A/np.maximum(A.sum(1,keepdims=True),1e-12); nc=gh*gw
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(nc-1,1)).T
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    X=np.c_[LA,R,geo]
    if X.shape[1]!=len(Wv)-1: return None
    mu,sd=X.mean(0),X.std(0)+1e-9
    return np.c_[(X-mu)/sd,np.ones(len(X))]@Wv
def softmax_pick(lg,ids):
    e=np.array([np.logaddexp.reduce(lg[i]) for i in ids]); e=e-e.max()
    p=np.exp(e); return (p/p.sum()).tolist()
items=B.load(BK,N or None)
import collections as _c
print(f"  {len(items)} items  kinds={dict(_c.Counter(i[5] for i in items))}",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
rng=np.random.default_rng(226); t0=time.time(); n=0
with open(OUT,"a") as f:
    for qid,img,q,gold,strat,kind in items:
        if qid in done or not kind.startswith("mcq"): continue
        K=int(kind[3:]); ids=B.letter_ids(tok,K)
        img=img.convert("RGB")
        lo=fit(img,E_LO); inp=build(lo,q).to(model.device)
        gh,gw,off=grid_of(inp); pos=(inp["input_ids"][0]==itid).nonzero().flatten()
        if gh*gw==0 or len(pos)<off+gh*gw: continue
        for l in layers: l.self_attn._capture=True
        zp=logits_of(inp)                                    # z+ AND the localise pass
        A=np.stack([l.self_attn._lastrow for l in layers])
        for l in layers: l.self_attn._capture=False; l.self_attn._lastrow=None
        b=int(pos[0])+off; A=A[:,b:b+gh*gw]
        sc=dwa_rank(A,gh,gw)
        if sc is None: continue
        nk=max(1,int(round(KFRAC*gh*gw)))
        top=np.argsort(-sc)[:nk]; rnd=rng.choice(gh*gw,size=nk,replace=False)
        zm_dwa=logits_of(inp,[b+int(j) for j in top])
        zm_rnd=logits_of(inp,[b+int(j) for j in rnd])
        nlo=ntok(inp); del inp; torch.cuda.empty_cache()
        hi=fit(img,2*E_LO); inp2=build(hi,q).to(model.device)
        zhi=logits_of(inp2); nhi=ntok(inp2); del inp2; torch.cuda.empty_cache()
        rec={"qid":qid,"stratum":strat,"kind":kind,"gold":gold,
             "tokens":{"bar":nlo,"hi2x":nhi,"cd_total":2*nlo},
             "probs":{"bar":softmax_pick(zp,ids),"hi2x":softmax_pick(zhi,ids)}}
        for a in ALPHAS:
            rec["probs"][f"cd_dwa@{a}"]=softmax_pick(zp+a*(zp-zm_dwa),ids)
            rec["probs"][f"cd_rand@{a}"]=softmax_pick(zp+a*(zp-zm_rnd),ids)
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        if n%50==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
