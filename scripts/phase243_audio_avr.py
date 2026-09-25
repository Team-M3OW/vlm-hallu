"""
Phase 243 (Track A3): AVR for audio, using TIME-STRETCH as the resolution axis.

THE ANALOGUE. Qwen2-Audio emits a fixed 25 tokens/second, so I first concluded audio had no
resolution axis. That was wrong, and the error was looking on the wrong axis: vision's tokens/pixel
is fixed too -- resolution comes from changing PIXELS PER UNIT OF CONTENT (resizing). The audio
analogue is SECONDS PER UNIT OF CONTENT, i.e. time-stretching. Verified: 6s of content gives 150
tokens at x1, 300 at x2, 450 at x3 -- same content, linearly more tokens. Pitch-preserving stretch
(librosa phase vocoder) keeps the content identical, unlike naive resampling.

BUDGET, in audio token-layers, from the MEASURED boundary (phase241: P=L24 of 32, 0.75 depth):
    bar = N*NL = 32N
    AVR stretched xk, keeping 10% past P = k*N*P + 0.1*k*N*(NL-P) = 24.8kN
    equal-compute  =>  k = 32/24.8 = 1.29
Vision's affordable multiplier was 1.65 (P=16 of 28). Audio's is SMALLER because its boundary is
LATER -- less depth left to save on. That is a quantitative consequence of the 0.75 result.

ARMS (audio token-layers recorded for every arm; nothing quoted as equal-compute unless it is):
  bar          native clip
  hi@1.29      stretched to the affordable multiplier, no pruning     (equal tokens? NO -- over budget)
  hi@2.0       stretched x2, no pruning    -- the HEADROOM DIAGNOSTIC, deliberately over budget
  avr@1.29     stretched x1.29, keep 10% from L24                     (equal-compute arm)
  avr@2.0      stretched x2,    keep 10% from L24                     (over budget, upper reach)
  avr_rand@2.0 same as avr@2.0 with a RANDOM keep set                 (keep-choice control)
  early@2.0    stretched x2,    keep 10% from L12 (0.38 depth)        (the cut must matter)

PRE-REGISTERED, before the first number:
  H4a  headroom exists iff hi@2.0 - bar > 0 with CI clear. If it does NOT, audio has no headroom to
       convert and AVR is inapplicable for want of headroom (not for want of a ladder) -- report it.
  H4b  the identity: avr@k - bar should track hi@k - bar (vision slope 1.01, r=0.966).
  H4c  avr@2.0 - avr_rand@2.0 within +-2 points (keep-choice irrelevant at the boundary, as in vision).
  H4d  early@2.0 - avr@2.0 negative with CI clear (pruning before the boundary must cost).
SANITY: exact +0.0 on any contrast is a pipeline fault (§93b), never a result.

Usage: phase243_audio_avr.py [n]
"""
import json, os, sys, time, importlib, numpy as np, torch
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID="Qwen/Qwen2-Audio-7B-Instruct"; LET="ABCD"; KEEP=0.10
P=24; EARLY=12; NCAL=int(sys.argv[1]) if len(sys.argv)>1 else 240
OUT=f"{D}/data/phase243_audio_avr.jsonl"
model=Qwen2AudioForConditionalGeneration.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},
                                                         attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(MID); tok=pr.tokenizer
layers=model.language_model.model.layers; NL=len(layers)
atid=model.config.audio_token_index; SR=pr.feature_extractor.sampling_rate
letters=[tok.encode(c,add_special_tokens=False)[0] for c in LET]
KAFF=NL/(P+KEEP*(NL-P))
print(f"NL={NL} P=L{P} ({P/NL:.2f})  affordable multiplier k={KAFF:.2f}  keep={KEEP}",flush=True)
import librosa
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
def clear():
    for l in layers:
        if hasattr(l.self_attn,"_prune_bias"): delattr(l.self_attn,"_prune_bias")
def _rs(w,src,dst):
    if src==dst: return w
    import math; g=math.gcd(int(src),int(dst))
    from scipy.signal import resample_poly
    return resample_poly(w,int(dst)//g,int(src)//g).astype(np.float32)
def as_wave(a):
    if isinstance(a,dict) and "array" in a:
        w=np.asarray(a["array"],dtype=np.float32); sr=int(a.get("sampling_rate",SR))
    else:
        s=a.get_all_samples(); w=s.data
        w=w.numpy() if hasattr(w,"numpy") else np.asarray(w)
        if w.ndim>1: w=w.mean(0)
        w=w.astype(np.float32); sr=int(s.sample_rate)
    return _rs(w,sr,SR)
def stretch(w,k):
    """k>1 = longer = more tokens for the SAME content. Pitch preserved (phase vocoder)."""
    if abs(k-1.0)<1e-3: return w
    return librosa.effects.time_stretch(w,rate=1.0/k).astype(np.float32)
def build(w,q,ch):
    body=q+"\n"+"\n".join(f"({LET[i]}) {c}" for i,c in enumerate(ch))+\
         "\nAnswer with the option's letter from the given choices directly."
    chat=pr.apply_chat_template([{"role":"user","content":[{"type":"audio"},{"type":"text","text":body}]}],
                                tokenize=False,add_generation_prompt=True)
    return pr(text=chat,audio=[w],sampling_rate=SR,return_tensors="pt",padding=True)
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
_r=_rnd.Random(243); order=[]
for t in sorted(_by): _r.shuffle(_by[t]); order+=_by[t][:NCAL//len(_by)+15]
order.sort(); ds=ds.select(order)
print(f"  stratified pool {len(order)} over {sorted(_by)}",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["id"] for l in open(OUT)}
rng=np.random.default_rng(243); n=0; t0=time.time(); seen={}
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
        try: w0=as_wave(ex["audio"])
        except Exception: continue
        if len(w0)/SR < 1.0: continue
        rec={"id":ex["id"],"task":ex["task"],"sub_category":ex["sub_category"],"gold":gold,
             "dur":round(len(w0)/SR,2),"probs":{},"tokens":{},"tl":{}}
        ok=True
        for nm,k in (("bar",1.0),("hi@1.29",KAFF),("hi@2.0",2.0)):
            w=stretch(w0,k); inp=build(w,ex["question"],ch).to(model.device)
            m=int((inp["input_ids"][0]==atid).sum())
            if m<10: ok=False; del inp; break
            rec["probs"][nm]=run(inp); rec["tokens"][nm]=m; rec["tl"][nm]=m*NL
            if nm=="bar": pass
            del inp; torch.cuda.empty_cache()
        if not ok: continue
        for nm,k,frm,randm in (("avr@1.29",KAFF,P,False),("avr@2.0",2.0,P,False),
                               ("avr_rand@2.0",2.0,P,True),("early@2.0",2.0,EARLY,False)):
            w=stretch(w0,k); inp=build(w,ex["question"],ch).to(model.device)
            pos=(inp["input_ids"][0]==atid).nonzero().flatten(); m=len(pos)
            nk=max(1,int(KEEP*m))
            keep=set(rng.choice(m,size=nk,replace=False).tolist())
            drop=[int(pos[i]) for i in range(m) if i not in keep]
            rec["probs"][nm]=run(inp,drop,frm); rec["tokens"][nm]=int(m)
            rec["tl"][nm]=int(m*frm + nk*(NL-frm))
            del inp; torch.cuda.empty_cache()
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        seen[ex["task"]]=seen.get(ex["task"],0)+1
        if n%25==0: print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item  dur={rec['dur']}s tok={rec['tokens']['bar']}",flush=True)
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
