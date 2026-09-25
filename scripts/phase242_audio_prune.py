"""
Phase 242 (Track A2): is pruning FREE past the audio transport boundary?

WHY THIS AND NOT AVR-AS-ALLOCATION. Qwen2-Audio's encoder emits a FIXED 25 tokens/second (2s->50,
10s->250, 30s->750, capped at 750) and accepts exactly one sample rate. There is no resolution axis,
so AVR-as-allocation is undefined here, as on Gemma-3 and InternVL3 (§80). DWA is undefined for the
same reason from the other side: a temporal crop keeps 25 tok/s, so cropping to 25% of the clip
yields 25% of the tokens at IDENTICAL density -- it buys no resolution, only loss. Both exclusions
are structural and are recorded, not measured as failures.

What survives is AVR's MECHANISM without its reinvestment: if modality tokens are causally inert
past the boundary, dropping them should cost ~nothing. That is a pure efficiency claim (compute
saved, not accuracy gained) and it is worth something in a modality where a 30s clip costs 750
tokens of context.

ARMS (keep fraction k=0.10 of audio tokens, applied from layer L onward via an additive -1e4 bias
on the audio columns -- the same mechanism as the image AVR arm):
  bar          no pruning
  prune@P      P = the measured transport boundary (phase241)
  prune@early  the same keep ratio at P//2  -- the control that SHOULD cost points
  prune_rand@P random keep set of the same size -- tests keep-choice irrelevance, as in vision

PRE-REGISTERED, before the run:
  H2a  prune@P - bar is within +-2 points (free).
  H2b  prune@early - bar is negative and its CI excludes zero (the cut must matter).
  H2c  prune@P - prune_rand@P is within +-2 points (at the boundary the choice of survivor is
       irrelevant, as at the image boundary).
  If H2a fails the boundary does not license pruning in audio. If H2b fails the measurement has no
  contrast and the arm is uninformative -- report, do not reinterpret.

SANITY: an exact +0.0 on any contrast is a pipeline fault (§93b), never a result.

Usage: phase242_audio_prune.py <boundary_layer> [n]
"""
import json, os, sys, time, numpy as np, torch
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID="Qwen/Qwen2-Audio-7B-Instruct"; LET="ABCD"; KEEP=0.10
P=int(sys.argv[1]); NCAL=int(sys.argv[2]) if len(sys.argv)>2 else 200
OUT=f"{D}/data/phase242_audio_prune.jsonl"

model=Qwen2AudioForConditionalGeneration.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},
                                                         attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(MID); tok=pr.tokenizer
layers=model.language_model.model.layers; NL=len(layers)
atid=model.config.audio_token_index; SR=pr.feature_extractor.sampling_rate
letters=[tok.encode(c,add_special_tokens=False)[0] for c in LET]
EARLY=max(1,P//2)
print(f"NL={NL}  boundary P=L{P} ({P/NL:.2f} depth)  early control L{EARLY}  keep={KEEP}",flush=True)

def _resample(w,src,dst):
    if src==dst: return w
    import math; g=math.gcd(int(src),int(dst))
    from scipy.signal import resample_poly
    return resample_poly(w,int(dst)//g,int(src)//g).astype(np.float32)
def as_wave(a):
    if isinstance(a,dict) and "array" in a:
        w=np.asarray(a["array"],dtype=np.float32); sr=int(a.get("sampling_rate",SR))
    elif hasattr(a,"get_all_samples"):
        s=a.get_all_samples(); w=s.data
        w=w.numpy() if hasattr(w,"numpy") else np.asarray(w)
        if w.ndim>1: w=w.mean(0)
        w=w.astype(np.float32); sr=int(s.sample_rate)
    else: raise TypeError(str(type(a)))
    return _resample(w,sr,SR), SR
def build(wave,sr,q,ch):
    body=q+"\n"+"\n".join(f"({LET[i]}) {c}" for i,c in enumerate(ch))+\
         "\nAnswer with the option's letter from the given choices directly."
    chat=pr.apply_chat_template([{"role":"user","content":[{"type":"audio"},{"type":"text","text":body}]}],
                                tokenize=False,add_generation_prompt=True)
    return pr(text=chat,audio=[wave],sampling_rate=sr,return_tensors="pt",padding=True)
def clear():
    for l in layers:
        if hasattr(l.self_attn,"_prune_bias"): delattr(l.self_attn,"_prune_bias")
import importlib
QM=importlib.import_module(type(layers[0].self_attn).__module__)
def mk(QM):
    def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM.eager_attention_forward=mk(QM)

def run(inp,drop=None,frm=None):
    clear()
    if drop is not None and len(drop):
        bias=torch.zeros(inp["input_ids"].shape[1],device=model.device); bias[list(drop)]=-1e4
        for l in layers[frm:]: l.self_attn._prune_bias=bias
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    clear()
    p=torch.softmax(lg[letters],0)
    return [round(float(v),6) for v in p.tolist()]

from datasets import load_dataset
import random as _rnd, ast
ds=load_dataset("TwinkStart/MMAU")["v05.15.25"]
_by={}
for i,t in enumerate(ds["task"]): _by.setdefault(t,[]).append(i)
_r=_rnd.Random(242)
order=[]
for t in sorted(_by): _r.shuffle(_by[t]); order+=_by[t][:NCAL//len(_by)+12]
order.sort(); ds=ds.select(order)
print(f"  stratified pool {len(order)} over {sorted(_by)}",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["id"] for l in open(OUT)}
rng=np.random.default_rng(242); n=0; t0=time.time(); seen={}
with open(OUT,"a") as f:
    for ex in ds:
        if n>=NCAL: break
        if ex["id"] in done: continue
        if seen.get(ex["task"],0)>=NCAL//len(_by)+1: continue
        ch=ex["choices"]
        if isinstance(ch,str):
            try: ch=ast.literal_eval(ch)
            except Exception: continue
        if len(ch)!=4: continue
        try: gold=[str(c).strip() for c in ch].index(str(ex["answer"]).strip())
        except ValueError: continue
        try: wave,sr=as_wave(ex["audio"])
        except Exception: continue
        inp=build(wave,sr,ex["question"],ch).to(model.device)
        pos=(inp["input_ids"][0]==atid).nonzero().flatten()
        if len(pos)<20: continue
        m=len(pos); nk=max(1,int(KEEP*m))
        keep=set(rng.choice(m,size=nk,replace=False).tolist())
        drop=[int(pos[i]) for i in range(m) if i not in keep]
        keep2=set(rng.choice(m,size=nk,replace=False).tolist())
        drop2=[int(pos[i]) for i in range(m) if i not in keep2]
        rec={"id":ex["id"],"task":ex["task"],"sub_category":ex["sub_category"],"gold":gold,
             "n_audio":int(m),"kept":nk,
             "probs":{"bar":run(inp),
                      f"prune@{P}":run(inp,drop,P),
                      f"prune@{EARLY}":run(inp,drop,EARLY),
                      f"prune_rand@{P}":run(inp,drop2,P)}}
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        seen[ex["task"]]=seen.get(ex["task"],0)+1
        del inp; torch.cuda.empty_cache()
        if n%25==0: print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
