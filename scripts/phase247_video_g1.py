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
    def vruns(cols):
        """Contiguous groups of video columns = ONE TEMPORAL PATCH each (Qwen3-VL puts a timestamp
        text block between frame blocks). 32 frames -> 16 runs of 49 tokens."""
        c=[int(x) for x in cols]; runs=[]; st=0
        for i in range(1,len(c)+1):
            if i==len(c) or c[i]!=c[i-1]+1: runs.append(c[st:i]); st=i
        return runs
    def fwd(inp,L,mode=None,keep_every=8):
        """mode=None native; 'spatial' keeps every k-th TOKEN (full temporal coverage, 1/k spatial);
        'temporal' keeps whole RUNS (= whole temporal patches), the true few-frame-equivalent bar."""
        cols=vcols(inp); nt=len(cols)
        inp={k:(v.to(model.device) if hasattr(v,"to") else v) for k,v in inp.items()}
        if mode=="spatial":
            drop=[int(c) for i,c in enumerate(cols) if i%keep_every]
        elif mode=="temporal":
            runs=vruns(cols); nk=max(1,len(runs)//keep_every)
            keep=set(np.linspace(0,len(runs)-1,nk).round().astype(int).tolist())
            drop=[c for j,r in enumerate(runs) if j not in keep for c in r]
        else: drop=None
        if drop:
            st["cols"]=torch.tensor(drop,device=model.device); nt=nt-len(drop)
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
        p,n2=fwd(i_hi,L,mode="temporal",keep_every=MANY//FEW); rec["probs"]["bar_latent_temporal"]=p; rec["ntok"]["bar_latent_temporal"]=n2
        p,n2b=fwd(i_hi,L,mode="spatial",keep_every=MANY//FEW); rec["probs"]["bar_latent_spatial"]=p; rec["ntok"]["bar_latent_spatial"]=n2b
        p,n3=fwd(build(f_few,text),L); rec["probs"]["bar"]=p; rec["ntok"]["bar"]=n3
        if not shown:
            print(f"  [selfcheck] {MANY}f -> {nt} video tokens ({nt/MANY:.0f}/frame); "
                  f"{FEW}f -> {n3}; latent_temporal -> {n2}; latent_spatial -> {n2b}  ratio hi/bar={nt/max(n3,1):.1f}x",flush=True)
            assert nt>n3, "more frames did not give more tokens -- no ladder"
            shown=True
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            a=lambda A:100*np.mean([int(np.argmax(x["probs"][A]))==x["gold"] for x in out])
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/it | hi={a('hi'):.1f} latT={a('bar_latent_temporal'):.1f} latS={a('bar_latent_spatial'):.1f} bar={a('bar'):.1f}",flush=True)
    fo.close(); json.dump(out,open("data/phase247_video_g1.json","w")); report(out,skips)

def report(out,skips=None):
    rng=np.random.default_rng(0)
    cor=lambda r,a: float(int(np.argmax(r["probs"][a]))==r["gold"])
    # CLUSTER-BOOTSTRAP BY BASE VIDEO. 1580 questions come from 410 videos, and the _reverse /
    # _concat_N variants share source content, so item-level CIs are too narrow (the HR-Bench
    # 4-item-cycle lesson).
    base=lambda r: re.sub(r'_(reverse|concat_\d+)$','',r["id"])
    groups=collections.defaultdict(list)
    for r in out: groups[base(r)].append(r)
    gk=list(groups)
    def ci(a,b,sub=None):
        g=gk if sub is None else sub
        per=np.array([np.mean([cor(r,a)-cor(r,b) for r in groups[k]]) for k in g])
        m=per[rng.integers(0,len(per),(10000,len(per)))].mean(1)*100
        return per.mean()*100, float(np.percentile(m,2.5)), float(np.percentile(m,97.5))
    ARMS=("hi","bar_latent_temporal","bar_latent_spatial","bar")
    print(f"\n=== VIDEO G1  n={len(out)} questions / {len(gk)} base videos ===")
    if skips: print("  skips:",dict(skips))
    for a in ARMS:
        print(f"  {a:21s} acc {100*np.mean([cor(r,a) for r in out]):5.1f}  video tokens {np.mean([r['ntok'][a] for r in out]):6.0f}")
    perg=np.array([np.mean([cor(r,'hi')-1.0/r['nch'] for r in groups[k]]) for k in gk])
    m=perg[rng.integers(0,len(perg),(10000,len(perg)))].mean(1)*100
    ch=100*np.mean([1.0/r['nch'] for r in out])
    print(f"\n  SANITY hi vs chance {100*np.mean([cor(r,'hi') for r in out]):.1f} vs {ch:.1f}: {perg.mean()*100:+.1f} "
          f"[{np.percentile(m,2.5):+.1f},{np.percentile(m,97.5):+.1f}]  {'ok' if np.percentile(m,2.5)>0 else 'AT FLOOR -> uninterpretable'}")
    accs={a:100*np.mean([cor(r,a) for r in out]) for a in ARMS[1:]}
    st=max(accs,key=accs.get)
    print(f"  bars: "+"  ".join(f"{k} {v:.1f}" for k,v in accs.items())+f"   -> strongest = {st}")
    m_,lo,hi_=ci("hi",st)
    print(f"\n  HEADROOM  hi - {st:21s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]  {'PASS -> AVR/DWA video' if lo>0 else 'FAIL -> stop'}")
    for a,b,lab in (("bar_latent_temporal","bar","temporal-latent vs re-encoded few frames"),
                    ("bar_latent_spatial","bar_latent_temporal","spatial-latent vs temporal-latent")):
        m_,lo,hi_=ci(a,b); print(f"  {lab:42s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]")
    print("\n  by dim (the scope law, video reading; clustered CIs):")
    for dd in sorted({r['dim'] for r in out}):
        sub=[k for k in gk if any(r['dim']==dd for r in groups[k])]
        g2=collections.defaultdict(list)
        for k in sub:
            for r in groups[k]:
                if r['dim']==dd: g2[k].append(r)
        sv=groups; groups.update(g2)
        per=np.array([np.mean([cor(r,'hi')-cor(r,st) for r in g2[k]]) for k in sub])
        mm=per[rng.integers(0,len(per),(4000,len(per)))].mean(1)*100
        items=[r for r in out if r['dim']==dd]
        print(f"    {dd:18s} n={len(items):4d}/{len(sub):3d}vid  hi {100*np.mean([cor(r,'hi') for r in items]):5.1f}  bar {100*np.mean([cor(r,st) for r in items]):5.1f}"
              f"  headroom {per.mean()*100:+5.1f} [{np.percentile(mm,2.5):+5.1f},{np.percentile(mm,97.5):+5.1f}]")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report": report(json.load(open("data/phase247_video_g1.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
