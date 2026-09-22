"""
Phase 224: DWA WITHOUT boxes at test time -- same-model, cross-benchmark weight transfer.

WHY THIS ARM EXISTS. DWA fits its ridge against GT-box coverage. V* has boxes; HR-Bench does not
(schema: index/question/answer/category/A-D/cycle_category/image -- no bbox field). So DWA cannot be
OOF-fitted on HR-Bench at all. Transfer is the only route, and it is also the honest deployment
story: fit once where boxes exist, run anywhere. This is NOT the cross-MODEL transfer already
rejected -- weights stay inside one checkpoint and only the benchmark changes.

Feature standardisation is refitted on the TARGET benchmark. That is label-free (mean/sd of the
features, no boxes), so it leaks nothing; only the 63 ridge coefficients come from V*.

ARMS, all equal-compute at 600 tokens:
  bar      uniform@600, no crop
  block    crop at the depth-block attention arg-max      <- the baseline DWA must beat
  dwa_t    crop at the transferred-ridge arg-max          <- the method

PRE-REGISTERED (written before any HR-Bench number was seen):
  Adopt the transfer arm iff (dwa_t - block) excludes zero on >=2 of 4 models.
  Beating `bar` alone is NOT sufficient: that would only show cropping helps, not that DWA's
  placement helps. On V* the block-relative margin is +10.5/+11.0 (Qwen3/Qwen2); if transfer
  collapses to block on HR-Bench, the honest report is that DWA does not transfer off V*.

Also reports the single/cross strata separately (HR-Bench ships `category`), which tests the scope
law -- cropping helps single-instance, hurts cross-instance -- on a SECOND benchmark instead of
asserting it from V* alone.

Usage: phase224_dwa_transfer.py <model-key> <hr4k|hr8k> [n]
"""
import json, os, sys, io, time, base64, importlib, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None; W=0.25
MODELS={"qwen3_2b":("Qwen/Qwen3-VL-2B-Instruct","qwen3"),"qwen2_7b":("Qwen/Qwen2-VL-7B-Instruct","qwen2"),
        "llava_ov":("llava-hf/llava-onevision-qwen2-7b-ov-hf","llava_ov"),
        "llava_next":("llava-hf/llava-v1.6-vicuna-7b-hf","llava_next")}
MK,BK = sys.argv[1], sys.argv[2]
N=int(sys.argv[3]) if len(sys.argv)>3 else 300
mid,wtag=MODELS[MK]
OUT=f"{D}/data/phase224_dwat_{MK}_{BK}.jsonl"
WPATH=f"{D}/data/fig_ridge_w_{wtag}.npy"
if not os.path.exists(WPATH):
    raise SystemExit(f"no V*-fitted weights at {WPATH}; fit them on V* first (fig_ridge_scores.py / phase214)")
Wv=np.load(WPATH)     # [63 coefs + bias], fitted on standardised V* features
print(f"{MK}/{BK}: loaded V* weights dim={len(Wv)}",flush=True)

model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
layers=None
for f in (lambda m:m.model.language_model.layers, lambda m:m.language_model.model.layers, lambda m:m.model.layers):
    try: layers=f(model); break
    except Exception: pass
assert layers is not None, "decoder layers not found"
NL=len(layers)
assert len(Wv)==2*NL+7+1, f"weight dim {len(Wv)} != 2*{NL}+7+1; weights and model disagree"
BLK=(round(0.57*NL),NL-1)
itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
modname=type(layers[0].self_attn).__module__; QM=importlib.import_module(modname)
assert hasattr(QM,"eager_attention_forward"), f"{modname} has no eager_attention_forward"
def make_patched(QM):
    def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        if getattr(module,"_capture",False):
            module._lastrow=w[0,:,-1,:].float().mean(0).detach().cpu().numpy()
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM.eager_attention_forward=make_patched(QM)

# --- token layout per family (llava_next corrected by phase220b: row stride includes the newline) ---
LAYOUT=json.load(open(f"{D}/data/phase224_layout.json")) if os.path.exists(f"{D}/data/phase224_layout.json") else {}
def grid_of(inp):
    if "image_grid_thw" in inp:
        g=inp["image_grid_thw"].tolist()[0]; return g[1]//2,g[2]//2,0
    n=int((inp["input_ids"][0]==itid).sum())
    L=LAYOUT.get(MK)
    if L:
        gh,gw,mode=L["gh"],L["gw"],L.get("mode","suffix")
        off=n-gh*gw if mode=="suffix" else 0
        if off>=0: return gh,gw,off
    s=int(round(n**0.5))
    return (s,s,0) if s*s==n else (0,0,0)

MCQ=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                           tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0]==itid).sum())
def fit_budget(img,target,refine=6,tol=0.08):
    W_,H_=img.size; sc=(target/max(ntok(build(img,"x")),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=ntok(build(cur,"x"))
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/r)**0.5
    return best[0]
def answer(img,q,B):
    inp=build(fit_budget(img,B),q).to(model.device)
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in MCQ]),0)
    del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
def localise(img,q,B=300):
    """one forward at the localise budget; return per-layer last-row attention over the image grid"""
    inp=build(fit_budget(img,B),q).to(model.device)
    gh,gw,off=grid_of(inp)
    if gh*gw==0: del inp; torch.cuda.empty_cache(); return None
    pos=(inp["input_ids"][0]==itid).nonzero().flatten()
    for l in layers: l.self_attn._capture=True
    with torch.no_grad(): model(**inp)
    A=np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture=False; l.self_attn._lastrow=None
    b=int(pos[0])+off
    A=A[:,b:b+gh*gw]
    del inp; torch.cuda.empty_cache()
    return (A,gh,gw) if A.shape[1]==gh*gw else None
def feats(A,gh,gw):
    a=A/np.maximum(A.sum(1,keepdims=True),1e-12); nc=gh*gw
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(nc-1,1)).T
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    return np.c_[LA,R,geo], dep
def ringmask(gh,gw):
    rm=np.zeros((gh,gw),bool)
    if gh>2 and gw>2: rm[1:-1,1:-1]=True
    else: rm[:]=True
    return rm
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))

from datasets import load_dataset
cfg="hrbench_4k" if BK=="hr4k" else "hrbench_8k"
ds=load_dataset("DreamMr/hr-bench","hrbench_version_split")[cfg]
items=[]
for e in ds:
    if len(items)>=N: break
    q=e["question"]+"\n"+"\n".join(f"({c}) {e[c]}" for c in "ABCD" if c in e)+\
      "\nAnswer with the option's letter from the given choices directly."
    items.append((str(e.get("index",len(items))),e["image"],q,"ABCD".index(str(e["answer"]).strip()[0]),
                  e.get("category","all")))
print(f"{BK}: {len(items)} items",flush=True)

def as_img(v):
    if isinstance(v,Image.Image): return v.convert("RGB")
    if isinstance(v,dict) and "bytes" in v: return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    if isinstance(v,str): return Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB")
    raise ValueError(f"unhandled image type {type(v)}")

# ---- pass 1: features for every item (needed to standardise on the TARGET distribution) ----
FE=[]; keep=[]
for i,(qid,imv,q,lab,cat) in enumerate(items):
    try: img=as_img(imv)
    except Exception as ex: print(f"  item {qid}: image decode failed: {ex}",flush=True); continue
    r=localise(img,q)
    if r is None:
        print(f"  item {qid}: no usable grid; skipped",flush=True); continue
    A,gh,gw=r; X,dep=feats(A,gh,gw)
    FE.append((X,dep,gh,gw)); keep.append((qid,img,q,lab,cat))
    if (i+1)%25==0: print(f"  feats [{i+1}/{len(items)}]",flush=True)
if not FE: raise SystemExit("no items produced a usable grid -- layout is wrong for this family")
ALL=np.vstack([x for x,_,_,_ in FE]); mu,sd=ALL.mean(0),ALL.std(0)+1e-9
print(f"standardised on target: {len(FE)} items, {ALL.shape[0]} cells",flush=True)

done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
t0=time.time()
with open(OUT,"a") as f:
    for (X,dep,gh,gw),(qid,img,q,lab,cat) in zip(FE,keep):
        if qid in done: continue
        rm=ringmask(gh,gw)
        s=np.c_[(X-mu)/sd,np.ones(len(X))]@Wv
        j=int(np.argmax(np.where(rm.ravel(),s,-1e9)))
        c_t=((j%gw+.5)/gw,(j//gw+.5)/gh)
        k=int(np.argmax(np.where(rm.ravel(),dep.ravel(),-1e9)))
        c_b=((k%gw+.5)/gw,(k//gw+.5)/gh)
        rec={"qid":qid,"category":cat,"label":lab,"grid":[gh,gw],
             "cell":{"dwa_t":list(c_t),"block":list(c_b)},
             "probs":{"bar":answer(img,q,600),
                      "block":answer(crop(img,*c_b),q,300),
                      "dwa_t":answer(crop(img,*c_t),q,300)}}
        f.write(json.dumps(rec)+"\n"); f.flush()
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
