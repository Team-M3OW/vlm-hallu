"""
Phase 244 (Track A4): CAN temporal placement pay on audio at all? Oracle ceiling first.

DWA-audio is well defined once time-stretch is recognised as the resolution axis (phase243):
crop W=25% of the DURATION and stretch x4 -> the original token count at 4x temporal density on the
chosen window. That is exactly vision's trade (same tokens, more density on a selected region),
which a naive crop could never deliver because tokens/second is fixed.

But fitting the DWA ridge needs ground-truth temporal segments, which MMAU does not have -- we would
have to fit on an annotated set (AudioSet-strong / DESED) and transfer. Before paying that cost we
measure the CEILING, exactly as the oracle-gate discipline demands: if the BEST POSSIBLE window
barely beats the bar, no read-out can help and the ridge is not worth fitting.

ARMS (equal token-layers unless marked; W=0.25 of duration, stretched x4 back to the bar's budget):
  bar            native clip, no crop
  crop_block     window at the block-mean attention arg-max  (the INCUMBENT, label-free)
  crop_rand      a random window of the same width           (the control)
  crop_oracle    best of K evenly-spaced windows, scored with the LABEL -> the placement CEILING
                 (not a method; an upper bound on what any read-out could achieve)

PRE-REGISTERED:
  C1  if crop_oracle - bar does NOT clear zero, temporal placement cannot pay on this benchmark and
      DWA-audio is not worth fitting -- report and stop.
  C2  crop_block - crop_rand tells whether the INCUMBENT read-out carries any signal at all.
  C3  the gap crop_oracle - crop_block is the headroom a fitted read-out could capture.
SANITY: exact +0.0 on any contrast is a pipeline fault (§93b).

Usage: phase244_audio_dwa_ceiling.py [n] [K]
"""
import json, os, sys, time, numpy as np, torch
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID="Qwen/Qwen2-Audio-7B-Instruct"; LET="ABCD"; W=0.25; P=24
NCAL=int(sys.argv[1]) if len(sys.argv)>1 else 200
K=int(sys.argv[2]) if len(sys.argv)>2 else 8
OUT=f"{D}/data/phase244_audio_dwa_ceiling.jsonl"
model=Qwen2AudioForConditionalGeneration.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},
                                                         attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(MID); tok=pr.tokenizer
layers=model.language_model.model.layers; NL=len(layers)
atid=model.config.audio_token_index; SR=pr.feature_extractor.sampling_rate
letters=[tok.encode(c,add_special_tokens=False)[0] for c in LET]
BLK=(P,NL-1)
print(f"NL={NL}  read-out band L{BLK[0]}-{BLK[1]-1}  W={W}  K={K}",flush=True)
import librosa
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
    if abs(k-1.0)<1e-3: return w
    return librosa.effects.time_stretch(w,rate=1.0/k).astype(np.float32)
def build(w,q,ch):
    body=q+"\n"+"\n".join(f"({LET[i]}) {c}" for i,c in enumerate(ch))+\
         "\nAnswer with the option's letter from the given choices directly."
    chat=pr.apply_chat_template([{"role":"user","content":[{"type":"audio"},{"type":"text","text":body}]}],
                                tokenize=False,add_generation_prompt=True)
    return pr(text=chat,audio=[w],sampling_rate=SR,return_tensors="pt",padding=True)
def answer(w,q,ch,want_attn=False):
    inp=build(w,q,ch).to(model.device)
    if want_attn:
        for l in layers: l.self_attn._capture=True
    with torch.no_grad(): o=model(**inp,output_attentions=want_attn)
    lg=o.logits[0,-1].float(); p=torch.softmax(lg[letters],0)
    A=None
    if want_attn:
        pos=(inp["input_ids"][0]==atid).nonzero().flatten()
        A=np.stack([o.attentions[l][0,:,-1,:].float().mean(0).cpu().numpy() for l in range(NL)])
        A=A[:,int(pos[0]):int(pos[-1])+1]
    n=int((inp["input_ids"][0]==atid).sum())
    del inp,o; torch.cuda.empty_cache()
    return [round(float(v),6) for v in p.tolist()], n, A
def window(w,frac_start):
    L=len(w); wl=int(W*L); s=int(min(max(0,frac_start*L),L-wl))
    return w[s:s+wl]
from datasets import load_dataset
import random as _rnd, ast
ds=load_dataset("TwinkStart/MMAU")["v05.15.25"]
_by={}
for i,t in enumerate(ds["task"]): _by.setdefault(t,[]).append(i)
_r=_rnd.Random(244); order=[]
for t in sorted(_by): _r.shuffle(_by[t]); order+=_by[t][:NCAL//len(_by)+15]
order.sort(); ds=ds.select(order)
print(f"  stratified pool {len(order)} over {sorted(_by)}",flush=True)
done=set()
if os.path.exists(OUT): done={json.loads(l)["id"] for l in open(OUT)}
rng=np.random.default_rng(244); n=0; t0=time.time(); seen={}
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
        if len(w0)/SR < 2.0: continue
        pb,nb,A=answer(w0,ex["question"],ch,want_attn=True)      # bar + the localise map
        if A is None or A.shape[1]<8: continue
        a=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        dep=a[BLK[0]:BLK[1]].mean(0)                              # block-mean read-out over time
        m=len(dep); wl=max(1,int(W*m))
        # incumbent: window centred on the block-mean arg-max, expressed as a start fraction
        j=int(np.argmax(dep)); s_block=min(max(0.0,(j-wl/2)/m),1.0-W)
        s_rand=float(rng.uniform(0,1.0-W))
        rec={"id":ex["id"],"task":ex["task"],"sub_category":ex["sub_category"],"gold":gold,
             "dur":round(len(w0)/SR,2),"probs":{"bar":pb},"tokens":{"bar":nb},"starts":{}}
        for nm,s in (("crop_block",s_block),("crop_rand",s_rand)):
            seg=stretch(window(w0,s),1.0/W)                       # crop 25% then stretch x4
            p,nn,_=answer(seg,ex["question"],ch); rec["probs"][nm]=p; rec["tokens"][nm]=nn
            rec["starts"][nm]=round(float(s),4)
        best=None
        for i in range(K):
            s=i*(1.0-W)/max(K-1,1)
            seg=stretch(window(w0,s),1.0/W)
            p,nn,_=answer(seg,ex["question"],ch)
            corr=int(int(np.argmax(p))==gold)
            if best is None or corr>best[0]: best=(corr,p,nn,s)
        rec["probs"]["crop_oracle"]=best[1]; rec["tokens"]["crop_oracle"]=best[2]
        rec["starts"]["crop_oracle"]=round(float(best[3]),4)
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        seen[ex["task"]]=seen.get(ex["task"],0)+1
        if n%20==0: print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item dur={rec['dur']}s",flush=True)
print(f"Done -> {OUT}  ({(time.time()-t0)/60:.1f} min)",flush=True)
