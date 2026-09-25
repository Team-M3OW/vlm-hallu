"""
Phase 247 -- VIDEO: token ladder (G0), baseline sanity, and HEADROOM on the frame axis (G1).
Track V of the cross-modal extension, run with the SAME gate order as audio Route B.

WHY FRAMES. Video gives the LM three tokenized axes (T,H,W) against audio's one (time). The frame
count is a genuine RESOLUTION knob, not a token knob: sampling 32 frames instead of 4 delivers
temporal detail that 4-frame sampling destroyed. That is the property audio's time-stretch lacked,
and the headroom identity (AVR gain = acc@hi - acc@lo, vision r=0.966) says it is the ONLY thing
that decides whether AVR can pay.

THE BAR IS UNIFORM SUBSAMPLING, NOT A PREFIX. bar takes 4 frames spread across the WHOLE video.
Taking the first 4 frames would destroy coverage -- the SS81 straw bar, and a pipeline fault.
  bar        4 frames, uniformly spaced over the full duration
  hi        32 frames, uniformly spaced
  bar_latent 32 frames encoded, but only every 8th frame's tokens VISIBLE (2D attention mask):
             the same information budget as bar with NO change to the input. Guards the gap from
             being an artefact of how few-frame inputs are processed rather than lost information.
  headroom = acc(hi) - max(acc(bar), acc(bar_latent))   must be CI-clear > 0, else STOP.
SANITY, required before any of it means anything: hi must be CI-clear ABOVE CHANCE. In the synthetic
V*-as-video check both paths sat at chance (27.5/30.0 vs 25.0), which validated the plumbing only.
STRATIFY by TempCompass `dim` afterwards (order/speed/direction are multi-frame by construction;
action/attribute_change need not be). This is the video reading of the scope law. No router.
Requires the phase240 grid fix (get_rope_index StopIteration) -- applied here too.
"""
import os, sys, json, glob, time, collections, re, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
MID="Qwen/Qwen3-VL-2B-Instruct"; PX=int(os.environ.get("PX","224"))
ROOT=glob.glob("/media/kavinder/hdd2/hf_cache/datasets--lmms-lab--TempCompass/snapshots/*")[0]

VIDEODIR=os.environ.get("TEMPCOMPASS_VIDEOS","/media/kavinder/hdd2/tempcompass")
def find_videos():
    """tempcompass_videos.zip is extracted to VIDEODIR; ids match the parquet exactly (1580/1580)."""
    cand=glob.glob(VIDEODIR+"/**/*.mp4",recursive=True)
    return {os.path.splitext(os.path.basename(p))[0]: p for p in cand}

def read_frames(path, k, px=PX):
    import decord
    vr=decord.VideoReader(path, num_threads=2)
    n=len(vr)
    idx=np.linspace(0, n-1, k).round().astype(int)          # UNIFORM over the whole video
    fr=vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f).convert("RGB").resize((px,px), Image.BICUBIC) for f in fr]

def main(nmax=0, FEW=4, MANY=32):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    vids=find_videos(); print(f"videos found: {len(vids)}",flush=True)
    assert vids, "no mp4 found -- did tempcompass_videos.zip download AND extract?"
    df=pd.read_parquet(glob.glob(ROOT+"/multi-choice/*.parquet")[0])
    df=df.sample(frac=1.0, random_state=247)                 # ordered by dim -> shuffle (CV-Bench trap)
    if nmax: df=df.head(nmax)
    pr=AutoProcessor.from_pretrained(MID)
    # EAGER, not sdpa: bar_latent hides video tokens by masking their COLUMNS inside attention.
    # The obvious trick -- zeroing them in the 2D attention_mask -- CORRUPTS THE MODEL: Qwen3-VL
    # derives its 3D rope positions from attention_mask, so dropping entries changes the position
    # count and the forward dies (shape mismatch [3,955] vs [3,269]). Masking at the layer leaves
    # rope untouched.
    model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    tok=pr.tokenizer
    layers=model.model.language_model.layers if hasattr(model.model,"language_model") else model.model.layers
    st={"cols":None}
    def mk(l):
        def pre(mod,args,kwargs):
            if st["cols"] is None: return None
            am=kwargs.get("attention_mask")
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
            am=am.clone(); am[:,:,:,st["cols"]]=torch.finfo(am.dtype).min
            kwargs["attention_mask"]=am; return (args,kwargs)
        return pre
    for l in range(len(layers)): layers[l].self_attn.register_forward_pre_hook(mk(l),with_kwargs=True)
    def build(frames,text):
        chat=pr.apply_chat_template([{"role":"user","content":[{"type":"video","video":frames},
             {"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True)
        inp=pr(text=[chat],videos=[frames],do_sample_frames=False,return_tensors="pt")
        g=inp["video_grid_thw"]; t,h,w=[int(x) for x in g[0]]
        if len(g)==1 and t>1: inp["video_grid_thw"]=torch.tensor([[1,h,w]]*t,dtype=g.dtype)
        return inp
    def vcols(inp):
        m=inp.get("mm_token_type_ids")
        return (m[0]==2).nonzero().flatten()
    def fwd(inp,L,keep_every=0):
        cols=vcols(inp); nt=len(cols)
        inp={k:(v.to(model.device) if hasattr(v,"to") else v) for k,v in inp.items()}
        if keep_every>1:
            drop=torch.tensor([int(c) for i,c in enumerate(cols) if i%keep_every],device=model.device)
            st["cols"]=drop; nt=nt-len(drop)
        else: st["cols"]=None
        try:
            with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        finally: st["cols"]=None
        ids=[tok.encode(x,add_special_tokens=False)[0] for x in L]
        return torch.softmax(lg[ids],-1).tolist(), nt
    out=[]; skips=collections.Counter(); t0=time.time(); shown=False
    fo=open("data/phase247_video_g1.jsonl","w")
    for _,r in df.iterrows():
        vid=str(r['video_id'])
        if vid not in vids: skips["no_video"]+=1; continue
        opts=re.findall(r'^([A-E])\.\s', r['question'], flags=re.M)
        if len(opts)<2: skips["no_options"]+=1; continue
        g=re.match(r'\s*([A-E])\.', str(r['answer']))
        if not g: skips["ungradable_answer"]+=1; continue
        L="".join(opts); gold=L.index(g.group(1))
        text=r['question']+"\nAnswer with the option's letter only."
        try:
            f_many=read_frames(vids[vid],MANY); f_few=read_frames(vids[vid],FEW)
        except Exception as e: skips["decode_fail"]+=1; continue
        rec={"id":vid,"dim":str(r['dim']),"nch":len(L),"gold":gold,"probs":{},"ntok":{}}
        i_hi=build(f_many,text)
        p,nt=fwd(i_hi,L); rec["probs"]["hi"]=p; rec["ntok"]["hi"]=nt
        p,n2=fwd(i_hi,L,keep_every=MANY//FEW); rec["probs"]["bar_latent"]=p; rec["ntok"]["bar_latent"]=n2
        p,n3=fwd(build(f_few,text),L); rec["probs"]["bar"]=p; rec["ntok"]["bar"]=n3
        if not shown:
            print(f"  [selfcheck] {MANY}f -> {nt} video tokens ({nt/MANY:.0f}/frame); "
                  f"{FEW}f -> {n3}; latent-masked -> {n2}  ratio hi/bar={nt/max(n3,1):.1f}x",flush=True)
            assert nt>n3, "more frames did not give more tokens -- no ladder"
            shown=True
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            a=lambda A:100*np.mean([int(np.argmax(x["probs"][A]))==x["gold"] for x in out])
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/it | hi={a('hi'):.1f} lat={a('bar_latent'):.1f} bar={a('bar'):.1f}",flush=True)
    fo.close(); json.dump(out,open("data/phase247_video_g1.json","w")); report(out,skips)

def report(out,skips=None):
    rng=np.random.default_rng(0)
    cor=lambda r,a: float(int(np.argmax(r["probs"][a]))==r["gold"])
    def ci(a,b):
        d=np.array([cor(r,a)-cor(r,b) for r in out]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
        return d.mean()*100, float(np.percentile(m,2.5)), float(np.percentile(m,97.5))
    print(f"\n=== VIDEO G1  n={len(out)} ===")
    if skips: print("  skips:",dict(skips))
    for a in ("hi","bar_latent","bar"):
        print(f"  {a:11s} acc {100*np.mean([cor(r,a) for r in out]):5.1f}  video tokens {np.mean([r['ntok'][a] for r in out]):6.0f}")
    d=np.array([cor(r,'hi')-1.0/r['nch'] for r in out]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
    ch=100*np.mean([1.0/r['nch'] for r in out])
    print(f"\n  SANITY hi vs chance {100*np.mean([cor(r,'hi') for r in out]):.1f} vs {ch:.1f}: {d.mean()*100:+.1f} "
          f"[{np.percentile(m,2.5):+.1f},{np.percentile(m,97.5):+.1f}]  {'ok' if np.percentile(m,2.5)>0 else 'AT FLOOR -> uninterpretable'}")
    accs={a:100*np.mean([cor(r,a) for r in out]) for a in ("bar","bar_latent")}
    st=max(accs,key=accs.get); print(f"  strongest bar: {st} ({accs[st]:.1f})")
    m_,lo,hi_=ci("hi",st)
    print(f"\n  HEADROOM  hi - {st:10s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]  {'PASS -> AVR/DWA video' if lo>0 else 'FAIL -> stop'}")
    m_,lo,hi_=ci("bar_latent","bar"); print(f"  latent vs resampled bar  = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]")
    print("\n  by dim (the scope law, video reading):")
    for dd in sorted({r['dim'] for r in out}):
        s=[r for r in out if r['dim']==dd]
        dl=np.array([cor(r,'hi')-cor(r,st) for r in s]); mm=dl[rng.integers(0,len(dl),(4000,len(dl)))].mean(1)*100
        print(f"    {dd:18s} n={len(s):4d}  hi {100*np.mean([cor(r,'hi') for r in s]):5.1f}  bar {100*np.mean([cor(r,st) for r in s]):5.1f}"
              f"  headroom {dl.mean()*100:+5.1f} [{np.percentile(mm,2.5):+5.1f},{np.percentile(mm,97.5):+5.1f}]")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report": report(json.load(open("data/phase247_video_g1.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
