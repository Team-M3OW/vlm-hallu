"""
Phase 249 -- DWA-VIDEO CEILING (Track V gate 2). ORACLE FIRST, placer only if the ceiling clears.
The audio lesson: on MMAU the oracle crop scored -3.3 [-8.3,+2.2] against the bar, i.e. the BEST
window choosable WITH THE LABEL still lost. Placement had no room, so a better placer was pointless
and the pre-registered rule said stop -- even though the incumbent signal was real (block-rand=+5.0)
and oracle-block was +10.0. Measure the ceiling BEFORE building anything.

THE VIDEO ANALOGUE OF A CROP IS TEMPORAL REALLOCATION, at equal tokens:
    bar          NFR frames spread uniformly over the WHOLE video
    crop_*       NFR frames inside a window of FRAC of the duration -> same token budget, 1/FRAC
                 the temporal density. This is DWA's move: same compute, spent in one place.
    crop_rand    window chosen at random           (the control every internal intervention must beat)
    crop_block   window with the largest mean video->text attention at the read-out layers (label-free)
    crop_oracle  best of K evenly spaced windows, scored WITH THE LABEL -> the CEILING
RULE, pre-registered: if crop_oracle - bar is not CI-clear positive, STOP. No placer is written.
Also reports oracle-block (how much a perfect placer could add over the incumbent) and block-rand
(whether the label-free statistic carries any signal at all) -- both are diagnostics, not gates.
Spatial crops are NOT tested here: TempCompass is a temporal benchmark, so temporal reallocation is
the move its questions can reward. A spatial ceiling belongs with a spatial benchmark.
CIs cluster-bootstrapped by base video (_reverse/_concat_N share source content).
"""
import os, sys, json, glob, time, re, collections, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
MID="Qwen/Qwen3-VL-2B-Instruct"; PX=int(os.environ.get("PX","224"))
NFR=int(os.environ.get("NFR","8")); FRAC=float(os.environ.get("FRAC","0.25")); K=int(os.environ.get("K","4"))
ROOT=glob.glob("/media/kavinder/hdd2/hf_cache/datasets--lmms-lab--TempCompass/snapshots/*")[0]
VIDEODIR=os.environ.get("TEMPCOMPASS_VIDEOS","/media/kavinder/hdd2/tempcompass")

def frames_from(path, k, lo=0.0, hi=1.0, px=PX):
    """k frames uniformly spaced inside the [lo,hi] FRACTION of the video."""
    import decord
    vr=decord.VideoReader(path,num_threads=2); n=len(vr)
    a,b=int(lo*(n-1)), int(hi*(n-1))
    idx=np.linspace(a,max(a,b),k).round().astype(int)
    return [Image.fromarray(f).convert("RGB").resize((px,px),Image.BICUBIC) for f in vr.get_batch(idx).asnumpy()]

def main(nmax=200):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    vids={os.path.splitext(os.path.basename(p))[0]:p for p in glob.glob(VIDEODIR+"/**/*.mp4",recursive=True)}
    df=pd.read_parquet(glob.glob(ROOT+"/multi-choice/*.parquet")[0]).sample(frac=1.0,random_state=249)
    pr=AutoProcessor.from_pretrained(MID)
    model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    tok=pr.tokenizer
    def build(frames,text):
        chat=pr.apply_chat_template([{"role":"user","content":[{"type":"video","video":frames},
             {"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True)
        inp=pr(text=[chat],videos=[frames],do_sample_frames=False,return_tensors="pt")
        g=inp["video_grid_thw"]; t,h,w=[int(x) for x in g[0]]
        if len(g)==1 and t>1: inp["video_grid_thw"]=torch.tensor([[1,h,w]]*t,dtype=g.dtype)
        return inp
    def probs(frames,text,L,want_attn=False):
        inp=build(frames,text).to(model.device)
        with torch.no_grad(): o=model(**inp,output_attentions=want_attn)
        lg=o.logits[0,-1].float(); ids=[tok.encode(x,add_special_tokens=False)[0] for x in L]
        nt=int((inp["mm_token_type_ids"][0]==2).sum())
        att=None
        if want_attn:
            m=inp["mm_token_type_ids"][0]; cols=(m==2).nonzero().flatten()
            vset=set(int(x) for x in cols)
            rows=[i for i in range(int(cols[0]),inp["input_ids"].shape[1]) if i not in vset]
            # read-out layers: the per-layer KL peak sits at ~0.4 depth (phase248)
            A=torch.stack([o.attentions[l][0].mean(0) for l in range(int(0.3*len(o.attentions)),int(0.5*len(o.attentions)))]).mean(0)
            att=A[rows][:,cols].mean(0).float().cpu().numpy()
        return torch.softmax(lg[ids],-1).tolist(), nt, att
    windows=[(i/K,(i+1)/K) for i in range(K)] if abs(FRAC-1.0/K)<1e-6 else \
            [(s, min(1.0,s+FRAC)) for s in np.linspace(0,1-FRAC,K)]
    out=[]; skips=collections.Counter(); t0=time.time(); shown=False
    fo=open("data/phase249_video_dwa.jsonl","w")
    rng=np.random.default_rng(249)
    for _,r in df.iterrows():
        if len(out)>=nmax: break
        vid=str(r['video_id'])
        if vid not in vids: skips["no_video"]+=1; continue
        opts=re.findall(r'^([A-E])\.\s',r['question'],flags=re.M)
        g=re.match(r'\s*([A-E])\.',str(r['answer']))
        if len(opts)<2 or not g: skips["ungradable"]+=1; continue
        L="".join(opts); gold=L.index(g.group(1))
        text=r['question']+"\nAnswer with the option's letter only."
        try:
            fb=frames_from(vids[vid],NFR,0.0,1.0)
            fw=[frames_from(vids[vid],NFR,a,b) for a,b in windows]
        except Exception: skips["decode"]+=1; continue
        rec={"id":vid,"dim":str(r['dim']),"nch":len(L),"gold":gold,"probs":{},"ntok":{}}
        p,nt,att=probs(fb,text,L,want_attn=True); rec["probs"]["bar"]=p; rec["ntok"]["bar"]=nt
        pw=[]
        for j,(a,b) in enumerate(windows):
            p2,nt2,_=probs(fw[j],text,L); pw.append(p2); rec["ntok"][f"w{j}"]=nt2
        # block: window whose frames carry the most video->text attention in the bar encoding
        nvt=len(att); per=np.array_split(np.arange(nvt),K)
        blk=int(np.argmax([att[ix].mean() for ix in per]))
        rec["probs"]["crop_block"]=pw[blk]
        rec["probs"]["crop_rand"]=pw[int(rng.integers(0,K))]
        best=max(range(K),key=lambda j: pw[j][gold])          # ORACLE: uses the label
        rec["probs"]["crop_oracle"]=pw[best]
        rec["blk"]=blk; rec["best"]=best
        if not shown:
            print(f"  [selfcheck] bar {nt} video tokens over the whole clip; window {rec['ntok']['w0']} "
                  f"over {FRAC:.0%} -> {1/FRAC:.0f}x temporal density at {'EQUAL' if abs(nt-rec['ntok']['w0'])<=2 else 'UNEQUAL'} budget",flush=True)
            assert abs(nt-rec["ntok"]["w0"])<=2, "crop and bar budgets differ -- not equal-compute"
            shown=True
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            a_=lambda A:100*np.mean([int(np.argmax(x["probs"][A]))==x["gold"] for x in out])
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/it | bar={a_('bar'):.1f} blk={a_('crop_block'):.1f} "
                  f"rand={a_('crop_rand'):.1f} oracle={a_('crop_oracle'):.1f}",flush=True)
    fo.close(); json.dump(out,open("data/phase249_video_dwa.json","w")); report(out,skips)

def report(out,skips=None):
    rng=np.random.default_rng(0)
    cor=lambda r,a: float(int(np.argmax(r["probs"][a]))==r["gold"])
    base=lambda r: re.sub(r'_(reverse|concat_\d+)$','',r["id"])
    G=collections.defaultdict(list)
    for r in out: G[base(r)].append(r)
    gk=list(G)
    def ci(a,b):
        per=np.array([np.mean([cor(r,a)-cor(r,b) for r in G[k]]) for k in gk])
        m=per[rng.integers(0,len(per),(10000,len(per)))].mean(1)*100
        lo,hi=float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
        return per.mean()*100,lo,hi,("*" if (lo>0 or hi<0) else " ")
    print(f"\n=== PHASE 249 DWA-VIDEO CEILING  n={len(out)} / {len(gk)} base videos ===")
    if skips: print("  skips:",dict(skips))
    for a in ("bar","crop_rand","crop_block","crop_oracle"):
        print(f"  {a:12s} acc {100*np.mean([cor(r,a) for r in out]):5.1f}")
    m,lo,hi,s=ci("crop_oracle","bar")
    print(f"\n  CEILING     crop_oracle - bar   = {m:+.1f} [{lo:+.1f},{hi:+.1f}]{s}"
          f"  {'PASS -> a placer is worth building' if lo>0 else 'UNDERWATER -> STOP, no placer (the audio outcome)'}")
    for a,b,lab in (("crop_block","crop_rand","incumbent signal (block - rand)"),
                    ("crop_oracle","crop_block","headroom for a better placer (oracle - block)"),
                    ("crop_block","bar","incumbent vs bar")):
        m,lo,hi,s=ci(a,b); print(f"  {lab:40s} = {m:+.1f} [{lo:+.1f},{hi:+.1f}]{s}")
    print("\n  by dim:")
    for dd in sorted({r['dim'] for r in out}):
        sub=[r for r in out if r['dim']==dd]
        print(f"    {dd:18s} n={len(sub):4d}  bar {100*np.mean([cor(r,'bar') for r in sub]):5.1f}"
              f"  oracle {100*np.mean([cor(r,'crop_oracle') for r in sub]):5.1f}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report": report(json.load(open("data/phase249_video_dwa.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
