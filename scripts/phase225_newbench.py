"""
Phase 225: AVR and DWA on the two replacement benchmarks, same items, one model load.

POPE is dropped (its "no" items have no object to crop toward). CV-Bench and RealworldQA are used
UNFILTERED -- the scorer adapts to each item (mcq2..mcq6 letter logits, or short greedy generation
with normalised exact match) rather than the item set being cut down to whatever the scorer
happened to support.

ARMS
  uniform@lo   the equal-compute bar
  uniform@hi   resolution-headroom diagnostic
  avr          encode@hi, prune 90% of visual tokens at round(0.57*NL)
  block        crop at the depth-block attention arg-max   (the incumbent placement rule)
  dwa_t        crop at the TRANSFERRED ridge arg-max (weights fitted on V*, never refitted here)
Every arm records its visual-token count, so nothing is quoted as equal-compute that is not.

Usage: phase225_newbench.py <model-key> <cvbench|realworldqa> [n]
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
from transformers import AutoProcessor, AutoModelForImageTextToText
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Image.MAX_IMAGE_PIXELS=None; W=0.25
MODELS={"qwen3_2b":("Qwen/Qwen3-VL-2B-Instruct","qwen3"),"qwen2_7b":("Qwen/Qwen2-VL-7B-Instruct","qwen2"),
        "llava_ov":("llava-hf/llava-onevision-qwen2-7b-ov-hf","llava_ov"),
        "internvl3_8b":("OpenGVLab/InternVL3-8B-hf","internvl3_8b"),
        "gemma3_4b":("google/gemma-3-4b-it","gemma3_4b")}
MK,BK=sys.argv[1],sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 0
mid,wtag=MODELS[MK]
OUT=f"{D}/data/phase225_{MK}_{BK}.jsonl"
E_LO,E_HI=600,900
LAYOUT=json.load(open(f"{D}/data/phase224_layout.json")) if os.path.exists(f"{D}/data/phase224_layout.json") else {}
WP=f"{D}/data/fig_ridge_w_{wtag}.npy"
Wv=np.load(WP) if os.path.exists(WP) else None
print(f"{MK}/{BK}: weights={'yes' if Wv is not None else 'NO -> DWA arms skipped'}",flush=True)

model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
layers=None
for f in (lambda m:m.model.language_model.layers, lambda m:m.language_model.model.layers,
          lambda m:m.model.text_model.layers, lambda m:m.model.layers):
    try: layers=f(model); break
    except Exception: pass
assert layers is not None
NL=len(layers); P=round(0.57*NL); BLK=(P,NL-1)
itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
modname=type(layers[0].self_attn).__module__; QM=importlib.import_module(modname)
assert hasattr(QM,"eager_attention_forward"), f"{modname} has no eager_attention_forward"
def make_patched(QM):
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
QM.eager_attention_forward=make_patched(QM)
print(f"  NL={NL} prune@L{P} attn={modname}",flush=True)

def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                           tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0]==itid).sum())
def fit_budget(img,target,refine=6,tol=0.10):
    # NO id()-KEYED CACHE. An earlier revision memoised on (id(img),target). CPython reuses the
    # addresses of freed objects, so the `block` crop and the `dwa_t` crop -- built and released in
    # sequence -- collided, and the second arm silently received the FIRST arm's resized image.
    # Symptom: block and dwa_t returned identical probability vectors on 96% of items, i.e. DWA
    # collapsed onto the incumbent. The safe reuse (uniform@hi and avr share one encode) is done
    # explicitly in the item loop instead of through a cache.
    W_,H_=img.size; r0=max(ntok(build(img,"x")),1); sc=(target/r0)**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(32,int(W_*sc)),max(32,int(H_*sc))),Image.BICUBIC); r=ntok(build(cur,"x"))
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/max(r,1))**0.5
    return best
def clear():
    for l in layers:
        if hasattr(l.self_attn,"_prune_bias"): delattr(l.self_attn,"_prune_bias")
def run_arm(img,q,kind,B_=None,prune=None,src=None):
    clear()
    if src is None: src=fit_budget(img,B_)[0] if B_ else img
    inp=build(src,q).to(model.device); n=ntok(inp)
    if prune is not None:
        pos=(inp["input_ids"][0]==itid).nonzero().flatten()
        if len(pos):
            bias=torch.zeros(inp["input_ids"].shape[1],device=model.device)
            keep=set(prune(len(pos)))
            drop=[int(pos[i]) for i in range(len(pos)) if i not in keep]
            bias[drop]=-1e4
            for l in layers[P:]: l.self_attn._prune_bias=bias
    p,txt=B.score_item(model,pr,tok,inp,kind)
    del inp; clear(); torch.cuda.empty_cache()
    return p,txt,n
def grid_of(inp):
    if "image_grid_thw" in inp:
        g=inp["image_grid_thw"].tolist()[0]; return g[1]//2,g[2]//2,0
    n=int((inp["input_ids"][0]==itid).sum()); L=LAYOUT.get(MK)
    if L:
        gh,gw=L["gh"],L["gw"]; off=n-gh*gw if L.get("mode","suffix")=="suffix" else 0
        if off>=0 and gh*gw<=n: return gh,gw,off
    s=int(round(n**0.5)); return (s,s,0) if s*s==n else (0,0,0)
def localise(img,q):
    L=LAYOUT.get(MK) or {}
    src=img.resize((L["resize"],L["resize"]),Image.BICUBIC) if L.get("resize") else fit_budget(img,CROP_B)[0]
    inp=build(src,q).to(model.device)
    gh,gw,off=grid_of(inp)
    if gh*gw==0: del inp; torch.cuda.empty_cache(); return None
    pos=(inp["input_ids"][0]==itid).nonzero().flatten()
    if len(pos)<off+gh*gw: del inp; torch.cuda.empty_cache(); return None
    for l in layers: l.self_attn._capture=True
    with torch.no_grad(): model(**inp)
    A=np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture=False; l.self_attn._lastrow=None
    b=int(pos[0])+off; A=A[:,b:b+gh*gw]; nt=ntok(inp)
    del inp; torch.cuda.empty_cache()
    return (A,gh,gw,nt) if A.shape[1]==gh*gw else None
def feats(A,gh,gw):
    a=A/np.maximum(A.sum(1,keepdims=True),1e-12); nc=gh*gw
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(nc-1,1)).T
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    return np.c_[LA,R,geo],dep
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))

# ---- per-architecture budget discovery -------------------------------------------------------
# A fixed 600/900 pair is meaningless on a quantised ladder. InternVL emits 256 tokens for one tile
# then jumps to 1792/2304/3328, so targeting 600 lands on 256 and 900 overshoots to 3328 -- a 13x
# gap masquerading as a headroom step, with the crops starved at 256 tokens (both placement arms
# then scored 10 points BELOW the bar). Probe the achievable counts and pick the nearest pair.
_probe=Image.new("RGB",(1200,900),(120,140,160))
# Probe SIZE at fixed aspect. If the token count never changes, the architecture has NO RESOLUTION
# LADDER: InternVL3 resizes to a tile grid chosen by ASPECT RATIO alone, returning 3328 tokens for
# a 300px and a 4000px 4:3 image alike. Such a model cannot convert compute into resolution, so AVR
# is structurally inapplicable (as on Gemma-3), and a W=0.25 crop -- which preserves aspect -- costs
# exactly as much as the full image, making DWA unavoidably 2x the bar.
_ach=sorted({ntok(build(_probe.resize((max(32,int(1200*f)),max(32,int(900*f)))),"x"))
             for f in (0.12,0.2,0.28,0.4,0.56,0.8,1.0,1.4,1.8,2.2)})
_ach=[a for a in _ach if a>0]
NO_LADDER=(len(_ach)<=1)
_lo_probe=fit_budget(_probe,E_LO)[1]; _hi_probe=fit_budget(_probe,E_HI)[1]
_cont=(not NO_LADDER) and (abs(_lo_probe-E_LO)/E_LO<=0.15 and abs(_hi_probe-E_HI)/E_HI<=0.15)
if NO_LADDER:
    E_LO=E_HI=None          # native encoding everywhere; AVR arms are skipped as infeasible
    print(f"  NO RESOLUTION LADDER (token count is size-invariant at {_ach[0]} tokens): AVR is",flush=True)
    print(f"  structurally infeasible here; bar and crops run at NATIVE resolution, DWA costs 2x.",flush=True)
elif _cont:
    print(f"  resolution axis CONTINUOUS (probe hit {_lo_probe}/{_hi_probe}); using {E_LO}/{E_HI}",flush=True)
else:
    # Choose an ADJACENT rung pair (lo,hi) with hi/lo <= 2.5, preferring lo nearest the intended
    # budget. Without the ratio guard InternVL picks lo=256 with the next rung at 1792 -- a 7x jump
    # that is not a headroom step, and it starves the crops (both placement arms then scored 10
    # points below the bar because a 256-token crop cannot resolve the detail being asked about).
    _pairs=[(a,b) for a,b in zip(_ach,_ach[1:]) if b/max(a,1)<=2.5]
    if _pairs:
        E_LO,E_HI=min(_pairs,key=lambda ab:abs(ab[0]-E_LO))
    else:
        E_LO=min(_ach,key=lambda a:abs(a-E_LO)); _up=[a for a in _ach if a>E_LO]
        E_HI=_up[0] if _up else E_LO
    print(f"  resolution axis QUANTISED {_ach}; using lo={E_LO} hi={E_HI} (ratio {E_HI/max(E_LO,1):.2f})",flush=True)
CROP_B=(E_LO//2 if _cont else E_LO) if not NO_LADDER else None  # None => native
print(f"  budgets: bar={E_LO} hi={E_HI} crop={CROP_B}  (DWA total = crop+localise)",flush=True)

items=B.load(BK, N or None)
import collections as _c
print(f"  {len(items)} items  kinds={dict(_c.Counter(i[5] for i in items))}  strata={dict(_c.Counter(i[4] for i in items))}",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
rng=np.random.default_rng(225); t0=time.time(); n=0; _ident=0; _cmp=0
FEAT=[]
with open(OUT,"a") as f:
    for qid,img,q,gold,stratum,kind in items:
        if qid in done: continue
        img=img.convert("RGB")
        rec={"qid":qid,"stratum":stratum,"kind":kind,"gold":gold,
             "probs":{},"preds":{},"tokens":{}}
        def put(name,res):
            p,t,nt=res
            if p is not None: rec["probs"][name]=p
            if t is not None: rec["preds"][name]=t
            rec["tokens"][name]=nt
        put("uniform@lo",run_arm(img,q,kind,E_LO))
        if NO_LADDER:
            rec["avr_infeasible"]=True          # no headroom exists; recorded, not faked
        else:
            hi_src=fit_budget(img,E_HI)[0]      # computed ONCE, passed explicitly to both arms
            put("uniform@hi",run_arm(img,q,kind,src=hi_src))
            put("avr",run_arm(img,q,kind,src=hi_src,
                              prune=lambda m: rng.choice(m,size=max(1,int(0.10*m)),replace=False).tolist()))
        if Wv is not None:
            L=localise(img,q)
            if L is not None:
                A,gh,gw,nloc=L
                X,dep=feats(A,gh,gw)
                if X.shape[1]==len(Wv)-1:
                    mu,sd=X.mean(0),X.std(0)+1e-9
                    s=np.c_[(X-mu)/sd,np.ones(len(X))]@Wv
                    rm=np.zeros((gh,gw),bool)
                    if gh>2 and gw>2: rm[1:-1,1:-1]=True
                    else: rm[:]=True
                    j=int(np.argmax(np.where(rm.ravel(),s,-1e9)))
                    k=int(np.argmax(np.where(rm.ravel(),dep.ravel(),-1e9)))
                    put("block",run_arm(crop(img,(k%gw+.5)/gw,(k//gw+.5)/gh),q,kind,CROP_B))
                    put("dwa_t",run_arm(crop(img,(j%gw+.5)/gw,(j//gw+.5)/gh),q,kind,CROP_B))
                    rec["tokens"]["localise"]=nloc
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        # Self-check for the §79 class of fault: two arms that should differ returning IDENTICAL
        # outputs. That bug (an id()-keyed resize cache) made dwa_t collapse onto block on 96% of
        # items and would have read as a believable null, not a crash. Cheap to detect, so detect it.
        if "dwa_t" in rec["probs"] and "block" in rec["probs"]:
            _ident += (rec["probs"]["dwa_t"]==rec["probs"]["block"]); _cmp += 1
        elif "dwa_t" in rec["preds"] and "block" in rec["preds"]:
            _ident += (rec["preds"]["dwa_t"]==rec["preds"]["block"]); _cmp += 1
        if _cmp and _cmp%50==0:
            fr=_ident/_cmp
            print(f"  [check] dwa_t==block on {_ident}/{_cmp} ({fr:.0%})"
                  + ("   <<< SUSPECT: arms may be sharing an input" if fr>0.60 else ""),flush=True)
        if n%50==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
