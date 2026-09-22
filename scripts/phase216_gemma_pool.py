"""
Phase 216: Gemma-3 run BOTH WAYS.

  mode=stock     the released artifact. 256 visual tokens whatever the input size, so there is no
                 resolution axis and AVR is not budget-feasible. We still measure the BOUNDARY here.
  mode=adaptive  replace the projector's fixed-stride AvgPool2d with AdaptiveAvgPool2d((s,s)) and set
                 the processor's image_seq_length to s*s. Both are PARAMETER-FREE: no trained weight is
                 touched. This supplies the continuous resolution axis Prop. 2 needs, and BOTH the bar
                 and AVR run under the same modification, so the comparison is internally fair.
                 At s=16 the adaptive pool is exactly the stock pool, so the bar is unchanged.

Budget: bar = 256*34 = 8704 token-layers. AVR at 400 tokens, prune at L19, keep k:
        400*20 + k*400*14 <= 8704  ->  k <= 0.1257. We use k=0.125 (99.95% of bar).
Caveat recorded in the output: 400 tokens is outside Gemma's training distribution, so absolute
accuracy may fall. The identity is still testable because headroom is measured under the same change.
"""
import json, os, sys, time, random, importlib, io, base64, numpy as np, torch
import torch.nn as nn
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
MODE=sys.argv[1]; BK=sys.argv[2] if len(sys.argv)>2 else "vstar"; N=int(sys.argv[3]) if len(sys.argv)>3 else 200
MID="google/gemma-3-4b-it"; OUT=f"{D}/data/phase216_gemma_{MODE}_{BK}.jsonl"
S_LO,S_HI,KEEP=16,20,0.125           # 256 and 400 tokens
model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(MID); tok=pr.tokenizer
layers=model.model.language_model.layers; NL=len(layers); P=round(0.57*NL)
QM=importlib.import_module(type(layers[0].self_attn).__module__)
assert hasattr(QM,"eager_attention_forward"), "not patchable"
def make_patched(QM):
    def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM.eager_attention_forward=make_patched(QM)
proj=model.model.multi_modal_projector
itid=model.config.image_token_id
def set_tokens(s):
    """parameter-free: swap the pool and the processor's placeholder count together"""
    proj.avg_pool=nn.AdaptiveAvgPool2d((s,s))
    n=s*s; pr.image_seq_length=n
    exp="".join([tok.image_token]*n)
    pr.full_image_sequence=f"\n\n{tok.boi_token}{exp}{tok.eoi_token}\n\n"
    return n
print(f"gemma3 mode={MODE} NL={NL} prune@L{P}",flush=True)
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def clear():
    for l in layers:
        if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
def run(inp,drop=None,frm=None,want=False):
    clear()
    if drop is not None and len(drop):
        b=torch.zeros(inp["input_ids"].shape[1],device=model.device)
        b[torch.as_tensor(drop,device=model.device)]=-1e4
        for li in range(frm,NL): layers[li].self_attn._prune_bias=b
    for l in layers: l.self_attn._capture=bool(want)
    with torch.no_grad(): out=model(**inp)
    lg=out.logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    A=None
    if want: A=np.stack([layers[L].self_attn._lastrow for L in range(NL)])
    for l in layers: l.self_attn._capture=False
    del out; torch.cuda.empty_cache(); clear()
    return [round(float(v),6) for v in p.tolist()], A
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset")
ds=load_dataset("craigwu/vstar_bench")["test"]
items=[(f"{e['category']}/{e['question_id']}",os.path.join(root,e["image"]),e["text"],
        "ABCD".index(e["label"]) if isinstance(e["label"],str) else int(e["label"]),e["category"]) for e in ds]
rng=random.Random(216)
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
t0,n=time.time(),0
with open(OUT,"a") as f:
    for qid,ip,text,lab,cat in items:
        if n>=N: break
        if qid in done or not os.path.exists(ip): continue
        img=Image.open(ip).convert("RGB")
        rec={"qid":qid,"category":cat,"label":lab,"mode":MODE,"nl":NL,"P":P,"probs":{},"tokens":{}}
        nlo=set_tokens(S_LO); inp=build(img,text).to(model.device)
        rec["probs"]["uniform@lo"],_=run(inp); rec["tokens"]["uniform@lo"]=int((inp["input_ids"][0]==itid).sum())
        pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
        rec["probs"]["suffix_mask"],_=run(inp,drop=list(range(base,base+nt)),frm=P)
        if MODE=="adaptive":
            nhi=set_tokens(S_HI); inp=build(img,text).to(model.device)
            rec["probs"]["uniform@hi"],_=run(inp); rec["tokens"]["uniform@hi"]=int((inp["input_ids"][0]==itid).sum())
            pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
            _,A=run(inp,want=True)
            Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12)
            s=Ai[max(0,P-4):P+1].mean(0); keep=max(1,int(round(KEEP*nt)))
            rec["probs"]["avr"],_=run(inp,drop=(base+np.argsort(-s)[keep:]).tolist(),frm=P+1)
            sr=np.array([rng.random() for _ in range(nt)])
            rec["probs"]["avr_rand"],_=run(inp,drop=(base+np.argsort(-sr)[keep:]).tolist(),frm=P+1)
            rec["tl"]={"bar":rec["tokens"]["uniform@lo"]*NL,"avr":nt*(P+1)+keep*(NL-1-P)}
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
