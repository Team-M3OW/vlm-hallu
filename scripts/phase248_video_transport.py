"""
Phase 248 -- WHERE DOES VIDEO REACH THE TEXT?  (Track V boundary, on REAL video)
The phase176c protocol on TempCompass with Qwen3-VL-2B. Supersedes phase240, which used synthetic
video (one V* image repeated 8x) and sat AT CHANCE (27.5/30.0 vs 25.0) -- that validated the
plumbing, not competence, and a boundary measured on a model that cannot answer is meaningless.
TempCompass gives a competent baseline (hi 66.7 vs chance 29.7, phase247).

DO NOT ASSUME 0.57. Vision put the boundary at 0.57 of depth (L16/28); AUDIO came out at 0.75
(L24/32) and falsified its own pre-registered L18. The depth is measured here, not ported.

Prefix/suffix schedules as in vision:
    prefix L0..l   large KL only while l is inside the transport window
    suffix Ll..end KL ~ 0 once l is past it  -> the depth at which transport is COMPLETE
TWO QWEN3-VL-SPECIFIC FAULTS, both fixed here (see phase240 history):
  1. get_rope_index StopIteration -- the processor emits ONE grid row [[t,h,w]] while
     mm_token_type_ids has t video runs (interleaved frame timestamps). Expand to [[1,h,w]]*t.
  2. The interleaved TIMESTAMP text rows sit INSIDE the video span. Masking only rows after the
     last frame leaves them free to read earlier frames and relay that onward -- the boundary would
     be an underestimate with no error raised. Mask every NON-VIDEO row from the first video token.
SANITY (the SS93b rule): mask-all must give large KL and flips; per-layer KL > 0. A weak kl_all is
the signature of fault 2 above.
"""
import os, sys, json, glob, time, re, collections, importlib, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
MID="Qwen/Qwen3-VL-2B-Instruct"; PX=int(os.environ.get("PX","224")); NFR=int(os.environ.get("NFR","16"))
ROOT=glob.glob("/media/kavinder/hdd2/hf_cache/datasets--lmms-lab--TempCompass/snapshots/*")[0]
VIDEODIR=os.environ.get("TEMPCOMPASS_VIDEOS","/media/kavinder/hdd2/tempcompass")

def read_frames(path,k,px=PX):
    import decord
    vr=decord.VideoReader(path,num_threads=2); n=len(vr)
    idx=np.linspace(0,n-1,k).round().astype(int)
    return [Image.fromarray(f).convert("RGB").resize((px,px),Image.BICUBIC) for f in vr.get_batch(idx).asnumpy()]

def main(NCAL=40):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    vids={os.path.splitext(os.path.basename(p))[0]:p for p in glob.glob(VIDEODIR+"/**/*.mp4",recursive=True)}
    assert vids, "no videos"
    df=pd.read_parquet(glob.glob(ROOT+"/multi-choice/*.parquet")[0]).sample(frac=1.0,random_state=248)
    pr=AutoProcessor.from_pretrained(MID)
    model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    tok=pr.tokenizer
    layers=model.model.language_model.layers if hasattr(model.model,"language_model") else model.model.layers
    NL=len(layers); print(f"  NL={NL}",flush=True)
    state={"layers":set(),"cols":None,"rows":None}
    def mk(l):
        def pre(mod,args,kwargs):
            if l not in state["layers"]: return None
            am=kwargs.get("attention_mask")
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
            am=am.clone()
            am[:,:,state["rows"].unsqueeze(1),state["cols"].unsqueeze(0)]=torch.finfo(am.dtype).min
            kwargs["attention_mask"]=am; return (args,kwargs)
        return pre
    for l in range(NL): layers[l].self_attn.register_forward_pre_hook(mk(l),with_kwargs=True)
    def build(frames,text):
        chat=pr.apply_chat_template([{"role":"user","content":[{"type":"video","video":frames},
             {"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True)
        inp=pr(text=[chat],videos=[frames],do_sample_frames=False,return_tensors="pt")
        g=inp["video_grid_thw"]; t,h,w=[int(x) for x in g[0]]
        if len(g)==1 and t>1: inp["video_grid_thw"]=torch.tensor([[1,h,w]]*t,dtype=g.dtype)
        return inp
    def lg(inp):
        with torch.no_grad(): return model(**inp).logits[0,-1].float()
    def kl(p,q):
        p=torch.softmax(p,-1); lq=torch.log_softmax(q,-1)
        return float((p*(torch.log(p+1e-12)-lq)).sum())
    out=[]; t0=time.time()
    for _,r in df.iterrows():
        if len(out)>=NCAL: break
        vid=str(r['video_id'])
        if vid not in vids: continue
        opts=re.findall(r'^([A-E])\.\s',r['question'],flags=re.M)
        if len(opts)<2: continue
        text=r['question']+"\nAnswer with the option's letter only."
        try: frames=read_frames(vids[vid],NFR)
        except Exception: continue
        inp=build(frames,text).to(model.device)
        m=inp["mm_token_type_ids"][0]; cols=(m==2).nonzero().flatten()
        if not len(cols): continue
        vset=set(int(x) for x in cols)
        state["cols"]=cols
        state["rows"]=torch.tensor([i for i in range(int(cols[0]),inp["input_ids"].shape[1]) if i not in vset],device=model.device)
        L=[tok.encode(x,add_special_tokens=False)[0] for x in opts]
        state["layers"]=set(); base=lg(inp); bl=base[L]
        state["layers"]=set(range(NL)); alla=lg(inp)
        rec={"id":vid,"dim":str(r['dim']),"vtok":int(len(cols)),
             "kl_all":kl(base,alla),"flip_all":int(int(torch.argmax(bl))!=int(torch.argmax(alla[L]))),
             "kl":[],"flip":[],"prefix":{},"suffix":{}}
        for l in range(NL):
            state["layers"]={l}; q=lg(inp)
            rec["kl"].append(kl(base,q)); rec["flip"].append(int(int(torch.argmax(bl))!=int(torch.argmax(q[L]))))
        for l in range(0,NL,2):
            state["layers"]=set(range(0,l+1)); q=lg(inp)
            rec["prefix"][str(l)]=[kl(base,q),int(int(torch.argmax(bl))!=int(torch.argmax(q[L])))]
            state["layers"]=set(range(l,NL)); q=lg(inp)
            rec["suffix"][str(l)]=[kl(base,q),int(int(torch.argmax(bl))!=int(torch.argmax(q[L])))]
        out.append(rec)
        if len(out)%5==0:
            print(f"  [{len(out)}/{NCAL}] {(time.time()-t0)/len(out):.1f}s/it kl_all={rec['kl_all']:.3f} "
                  f"peak L{int(np.argmax(rec['kl']))} vtok={rec['vtok']}",flush=True)
    json.dump(out,open("data/phase248_video_transport.json","w")); report(out,NL)

def report(out,NL=None):
    NL=NL or len(out[0]["kl"])
    K=np.array([r["kl"] for r in out])
    print(f"\n=== PHASE 248 VIDEO BOUNDARY  n={len(out)} NL={NL} ===")
    print(f"  SANITY kl_all={np.mean([r['kl_all'] for r in out]):.3f}  flip_all={np.mean([r['flip_all'] for r in out]):.2f}"
          f"   {'ok' if np.mean([r['kl_all'] for r in out])>0.3 else 'WEAK -> suspect a mask leak'}")
    pk=int(np.argmax(K.mean(0)))
    print(f"  per-layer KL peak: L{pk} = {pk/NL:.2f} of depth   (vision 0.39, audio 0.38)")
    ks=sorted(out[0]["suffix"],key=int)
    def bnd(thr=0.03):
        for k in ks:
            if np.mean([r["suffix"][k][0] for r in out])<=thr: return int(k)
    b=bnd()
    print(f"  suffix KL<=0.03 -> transport COMPLETE by L{b} = {b/NL:.2f} of depth" if b is not None
          else "  suffix never reaches 0.03")
    print("    vision 0.57 (L16/28) | audio 0.75 (L24/32) | video measured above -- NOT assumed")
    print("  suffix: "+"  ".join(f"L{k}:{np.mean([r['suffix'][k][0] for r in out]):.3f}" for k in ks[:14]))
    print("  prefix: "+"  ".join(f"L{k}:{np.mean([r['prefix'][k][0] for r in out]):.3f}" for k in ks[:14]))
    for dd in sorted({r['dim'] for r in out}):
        s=[r for r in out if r['dim']==dd]
        bb=None
        for k in ks:
            if np.mean([r["suffix"][k][0] for r in s])<=0.03: bb=int(k); break
        print(f"    {dd:18s} n={len(s):3d}  boundary {'L'+str(bb)+f' = {bb/NL:.2f}' if bb is not None else 'not reached'}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report": report(json.load(open("data/phase248_video_transport.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 40)
