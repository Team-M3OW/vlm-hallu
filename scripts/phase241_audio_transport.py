"""
Phase 241 (Track A1): does the transport boundary exist for AUDIO tokens?

The image protocol (phase176c) applied unchanged to Qwen2-Audio-7B-Instruct on MMAU. At selected
layers we block every text row after the audio span from attending to the audio-token columns, then
measure KL at the output and answer flips. Single-layer, prefix, suffix and all-layer schedules,
exactly as on images.

WHY THIS IS THE RIGHT FIRST EXPERIMENT. Both policies in the paper are DERIVED from the boundary:
DWA is "read the map across all depths rather than one", AVR is "prune where it is causally free".
If the boundary is not there, neither policy is motivated for audio and no amount of engineering a
temporal-crop analogue would rescue it.

PRE-REGISTERED, written before the first run (PREREG_CROSSMODAL.md H1):
  Qwen2-Audio has NL=32 decoder layers. The vision law puts transport completion at 0.57*depth,
  which predicts L18 (0.57*32 = 18.2). The suffix mask should be near-inert from about there.
  Specifically: suffix KL at L18 should be within an order of magnitude of the image case
  (0.014/0.029 at the boundary) and answer flips should be near zero by L20-L22.
  If the boundary sits elsewhere, or there is no boundary, THAT IS THE RESULT and the 0.57 law is
  vision-specific -- which is itself worth reporting.

SANITY (the exactly-zero rule): masking ALL layers must give large KL and flip answers; per-layer
KL must be > 0. An exact +0.0 anywhere is a pipeline fault, not a finding.

Benchmark: MMAU (TwinkStart/MMAU, 1000 items, 4-way MCQ). Strata come from its own sub_category:
temporal/ordering/counting items are the relational analogue, single-event items the V* analogue.

Usage: phase241_audio_transport.py [n]
"""
import json, os, sys, time, numpy as np, torch
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID="Qwen/Qwen2-Audio-7B-Instruct"
NCAL=int(sys.argv[1]) if len(sys.argv)>1 else 40
OUT=f"{D}/data/phase241_audio_transport.json"   # stratified over MMAU task
LET="ABCD"

model=Qwen2AudioForConditionalGeneration.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},
                                                         attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(MID)
tok=pr.tokenizer
layers=model.language_model.model.layers; NL=len(layers)
atid=model.config.audio_token_index
SR=getattr(pr.feature_extractor,"sampling_rate",16000)
letters=[tok.encode(c,add_special_tokens=False)[0] for c in LET]
print(f"Qwen2-Audio: NL={NL}  audio_token_index={atid}  sr={SR}",flush=True)
print(f"PRE-REGISTERED prediction: boundary at 0.57*{NL} = L{0.57*NL:.1f}",flush=True)

def _resample(w,src,dst):
    """MMAU ships 32 kHz; Whisper's feature extractor requires exactly 16 kHz."""
    if src==dst: return w
    import math
    g=math.gcd(int(src),int(dst))
    try:
        from scipy.signal import resample_poly
        return resample_poly(w,int(dst)//g,int(src)//g).astype(np.float32)
    except Exception:
        import torch as _t, torchaudio
        return torchaudio.functional.resample(_t.from_numpy(w),int(src),int(dst)).numpy().astype(np.float32)

def as_wave(a):
    """MMAU ships an AudioDecoder; normalise to (float32 mono array at SR)."""
    if isinstance(a,dict) and "array" in a:
        w=np.asarray(a["array"],dtype=np.float32); sr=int(a.get("sampling_rate",SR))
    elif hasattr(a,"get_all_samples"):
        s=a.get_all_samples(); w=s.data
        w=w.numpy() if hasattr(w,"numpy") else np.asarray(w)
        if w.ndim>1: w=w.mean(0)
        w=w.astype(np.float32); sr=int(s.sample_rate)
    else:
        raise TypeError(f"unhandled audio type {type(a)}")
    return _resample(w,sr,SR), SR

def build(wave,sr,q,choices):
    body=q+"\n"+"\n".join(f"({LET[i]}) {c}" for i,c in enumerate(choices))+\
         "\nAnswer with the option's letter from the given choices directly."
    chat=pr.apply_chat_template([{"role":"user","content":[{"type":"audio"},{"type":"text","text":body}]}],
                                tokenize=False,add_generation_prompt=True)
    return pr(text=chat,audio=[wave],sampling_rate=sr,return_tensors="pt",padding=True)

# ---- the mask: every text row AFTER the audio span loses the audio columns, at chosen layers ----
state={"layers":set(),"span":None}
def make_hook(l):
    def pre(mod,args,kwargs):
        if l not in state["layers"]: return None
        am=kwargs.get("attention_mask")
        if am is None: am=args[1] if len(args)>1 else None
        if am is None or am.dtype==torch.bool:
            raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}; cannot ablate")
        am=am.clone(); a,b=state["span"]
        am[:,:,b:,a:b]=torch.finfo(am.dtype).min
        kwargs["attention_mask"]=am; return (args,kwargs)
    return pre
hooks=[layers[l].self_attn.register_forward_pre_hook(make_hook(l),with_kwargs=True) for l in range(NL)]

def logits(inp):
    with torch.no_grad(): return model(**inp).logits[0,-1].float()
def kl(p,q):
    p=torch.softmax(p,-1); lq=torch.log_softmax(q,-1)
    return float((p*(torch.log(p+1e-12)-lq)).sum())

from datasets import load_dataset
ds=load_dataset("TwinkStart/MMAU")["v05.15.25"]
# MMAU is ORDERED BY TASK (333 sound / 333 speech / 334 music). Taking a prefix returns one task
# only -- the first 40 items are all `sound`, and the temporal/counting sub-categories that form the
# relational stratum never appear. Sample STRATIFIED over task, deterministically, so the boundary
# estimate is for MMAU rather than for one of its thirds.
import random as _rnd
_by={}
for i,_t in enumerate(ds["task"]): _by.setdefault(_t,[]).append(i)
_r=_rnd.Random(241); _order=[]
for _t in sorted(_by): _r.shuffle(_by[_t])
_per=max(1,NCAL//max(len(_by),1))
for _t in sorted(_by): _order+=_by[_t][:_per+8]        # headroom for skips
_order.sort()
ds=ds.select(_order)
print(f"  stratified pool: {len(_order)} items over tasks {sorted(_by)}",flush=True)
out=[]; n=0; t0=time.time()
_seen={}
for ex in ds:
    if n>=NCAL: break
    # keep the mixture balanced even when items are skipped
    if _seen.get(ex["task"],0) >= (NCAL//max(len(_by),1))+1: continue
    ch=ex["choices"]
    if isinstance(ch,str):
        import ast
        try: ch=ast.literal_eval(ch)
        except Exception: continue
    if len(ch)!=4: continue
    try: gold=[str(c).strip() for c in ch].index(str(ex["answer"]).strip())
    except ValueError: continue
    try: wave,sr=as_wave(ex["audio"])
    except Exception as e:
        print(f"  skip {ex['id']}: {e}",flush=True); continue
    inp=build(wave,sr,ex["question"],ch).to(model.device)
    pos=(inp["input_ids"][0]==atid).nonzero().flatten()
    if len(pos)<8:
        print(f"  skip {ex['id']}: only {len(pos)} audio tokens",flush=True); continue
    state["span"]=(int(pos[0]),int(pos[-1])+1)
    state["layers"]=set(); base=logits(inp); bl=base[letters]
    state["layers"]=set(range(NL)); allab=logits(inp)
    rec={"id":ex["id"],"task":ex["task"],"sub_category":ex["sub_category"],"difficulty":ex["difficulty"],
         "gold":gold,"n_audio":int(len(pos)),
         "kl_all":kl(base,allab),"kl_all_letters":kl(bl,allab[letters]),
         "flip_all":int(int(torch.argmax(bl))!=int(torch.argmax(allab[letters]))),
         "kl":[],"kl_letters":[],"flip":[],"prefix":{},"suffix":{}}
    for l in range(NL):
        state["layers"]={l}; q=logits(inp)
        rec["kl"].append(kl(base,q)); rec["kl_letters"].append(kl(bl,q[letters]))
        rec["flip"].append(int(int(torch.argmax(bl))!=int(torch.argmax(q[letters]))))
    for l in range(0,NL,2):                       # step 2: the prediction is L18, resolve it finely
        state["layers"]=set(range(0,l+1)); q=logits(inp)
        rec["prefix"][str(l)]=[kl(base,q),int(int(torch.argmax(bl))!=int(torch.argmax(q[letters])))]
        state["layers"]=set(range(l,NL)); q=logits(inp)
        rec["suffix"][str(l)]=[kl(base,q),int(int(torch.argmax(bl))!=int(torch.argmax(q[letters])))]
    state["layers"]=set()
    out.append(rec); n+=1; _seen[ex["task"]]=_seen.get(ex["task"],0)+1
    del inp; torch.cuda.empty_cache()
    if n%5==0:
        print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item  kl_all={rec['kl_all']:.3f} "
              f"peak L{int(np.argmax(rec['kl']))}  n_audio={rec['n_audio']}",flush=True)
json.dump(out,open(OUT,"w"))
K=np.array([r["kl"] for r in out])
print(f"\nn={len(out)}  mean audio tokens={np.mean([r['n_audio'] for r in out]):.0f}")
print(f"SANITY  kl_all={np.mean([r['kl_all'] for r in out]):.3f}  flip_all={np.mean([r['flip_all'] for r in out]):.2f}"
      f"   (must be large; exact 0.0 = pipeline fault)")
print("mean per-layer KL: "+" ".join(f"L{l}:{K[:,l].mean():.4f}" for l in range(NL)))
for nm in ("prefix","suffix"):
    ks=sorted(out[0][nm],key=int)
    print(f"\n{nm} masks:")
    for k in ks:
        print(f"   L{k:>2}  KL {np.mean([r[nm][k][0] for r in out]):8.4f}   flips {np.mean([r[nm][k][1] for r in out]):.2f}")
print(f"\nPREDICTED boundary L{0.57*NL:.1f}; read the suffix column for where KL and flips go to ~0.")
print(f"Done -> {OUT}",flush=True)
