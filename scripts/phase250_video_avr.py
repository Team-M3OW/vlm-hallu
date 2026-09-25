"""
Phase 250 -- AVR-VIDEO: does the policy actually PAY at equal compute?  (Track V, the deliverable)
G1 (phase247) established only the PRECONDITION -- that a resolution knob exists on video
(headroom +4.9 [+3.0,+6.9] over an 8x frame span). It did NOT measure AVR's gain. This does.

AVR: encode MANY frames, then prune to KEEP of the modality tokens AT THE TRANSPORT BOUNDARY,
which phase248 measured at L16/28 = 0.57 of depth on this model (video-used subset). Not assumed.

EQUAL COMPUTE, in token-layers. Cost of N modality tokens with a cut at layer P keeping fraction k:
    TL = N*P + N*k*(NL-P)
so AVR can afford a multiplier over the bar of
    KAFF = NL / (P + k*(NL-P))
With NL=28, P=16, k=0.1 -> KAFF = 1.63x. NOTE: this is set by the BOUNDARY DEPTH. Audio's deeper
0.75 boundary affords only 1.29x. A deeper boundary leaves fewer layers to save on. Video's 1.63x
buys ~6 frames against the bar's 4, while the +4.9 headroom was measured across 4->32 frames -- so
a SMALL gain is the honest expectation here, and a null is a real possible outcome.

ARMS (token-layers recorded for every one; frames chosen to land on the multiplier):
    bar        FEW frames, uniformly spaced, no pruning      -- the equal-compute reference
    hi         MANY frames, no pruning                        -- the ceiling, NOT affordable
    avr        KAFF x bar tokens, pruned to KEEP at L16       -- the policy
    avr_rand   same budget/cut, RANDOM keep set               -- the control it must beat
    early      same budget/keep, cut at L8 (BEFORE the boundary) -- tests that the DEPTH matters
Keep-choice is by mean modality->text attention at the read-out layers, label-free.
CIs cluster-bootstrapped by base video (_reverse/_concat_N share source content).
"""
import os, sys, json, glob, time, re, collections, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
MID="Qwen/Qwen3-VL-2B-Instruct"; PX=int(os.environ.get("PX","224"))
FEW=int(os.environ.get("FEW","4")); MANY=int(os.environ.get("MANY","32"))
KEEP=float(os.environ.get("KEEP","0.10")); PCUT=int(os.environ.get("PCUT","16")); EARLY=int(os.environ.get("EARLY","8"))
ROOT=glob.glob("/media/kavinder/hdd2/hf_cache/datasets--lmms-lab--TempCompass/snapshots/*")[0]
VIDEODIR=os.environ.get("TEMPCOMPASS_VIDEOS","/media/kavinder/hdd2/tempcompass")

def frames_from(path,k,px=PX):
    import decord
    vr=decord.VideoReader(path,num_threads=2); n=len(vr)
    idx=np.linspace(0,n-1,k).round().astype(int)
    return [Image.fromarray(f).convert("RGB").resize((px,px),Image.BICUBIC) for f in vr.get_batch(idx).asnumpy()]

def main(nmax=400):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    vids={os.path.splitext(os.path.basename(p))[0]:p for p in glob.glob(VIDEODIR+"/**/*.mp4",recursive=True)}
    df=pd.read_parquet(glob.glob(ROOT+"/multi-choice/*.parquet")[0]).sample(frac=1.0,random_state=250)
    pr=AutoProcessor.from_pretrained(MID)
    model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    tok=pr.tokenizer
    layers=model.model.language_model.layers if hasattr(model.model,"language_model") else model.model.layers
    NL=len(layers)
    KAFF=NL/(PCUT+KEEP*(NL-PCUT))
    st={"cut":None,"cols":None}
    def mk(l):
        def pre(mod,args,kwargs):
            if st["cut"] is None or l<st["cut"] or st["cols"] is None: return None
            am=kwargs.get("attention_mask")
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
            am=am.clone(); am[:,:,:,st["cols"]]=torch.finfo(am.dtype).min
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
    # Video tokens are QUANTISED BY TEMPORAL PATCH (2 frames/patch, 49 tok each at 224px), so the
    # affordable multiplier cannot be applied to the FRAME count: 4f->98 tok but 7f rounds up to 4
    # patches = 196 tok = 2.0x, over the 1.63x budget. Pick the largest frame count whose TOKEN
    # count stays within KAFF. Probed on a real clip, not assumed.
    _probe=frames_from(list(vids.values())[0], MANY)
    def _ntok(k):
        i=build(_probe[:k] if k<=len(_probe) else _probe,"x")
        return int((i["mm_token_type_ids"][0]==2).sum())
    _nbar=_ntok(FEW); NFR_AVR=FEW
    for k in range(FEW+1, MANY+1):
        if _ntok(k) <= KAFF*_nbar: NFR_AVR=k
        else: break
    print(f"  NL={NL} boundary L{PCUT}={PCUT/NL:.2f} keep={KEEP:.0%} -> KAFF={KAFF:.2f}x | "
          f"bar {FEW}f={_nbar}tok -> avr {NFR_AVR}f={_ntok(NFR_AVR)}tok "
          f"({_ntok(NFR_AVR)/_nbar:.2f}x tokens)",flush=True)
    assert NFR_AVR>FEW, f"no affordable frame count above the bar at KAFF={KAFF:.2f}"

    def run(frames,text,L,cut=None,drop=None,want_attn=False):
        inp=build(frames,text).to(model.device)
        m=inp["mm_token_type_ids"][0]; cols=(m==2).nonzero().flatten(); N=len(cols)
        st["cut"]=cut; st["cols"]=drop
        try:
            with torch.no_grad(): o=model(**inp,output_attentions=want_attn)
        finally: st["cut"]=None; st["cols"]=None
        lg=o.logits[0,-1].float(); ids=[tok.encode(x,add_special_tokens=False)[0] for x in L]
        nd=0 if drop is None else len(drop)
        tl = N*NL if cut is None else N*cut + (N-nd)*(NL-cut)
        att=None
        if want_attn:
            vset=set(int(x) for x in cols)
            rows=[i for i in range(int(cols[0]),inp["input_ids"].shape[1]) if i not in vset]
            A=torch.stack([o.attentions[l][0].mean(0) for l in range(int(0.3*NL),int(0.5*NL))]).mean(0)
            att=A[rows][:,cols].mean(0).float().cpu().numpy()
        return torch.softmax(lg[ids],-1).tolist(), N, tl, cols, att
    out=[]; skips=collections.Counter(); t0=time.time(); shown=False
    rng=np.random.default_rng(250)
    fo=open("data/phase250_video_avr.jsonl","w")
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
            fb=frames_from(vids[vid],FEW); fh=frames_from(vids[vid],MANY); fa=frames_from(vids[vid],NFR_AVR)
        except Exception: skips["decode"]+=1; continue
        rec={"id":vid,"dim":str(r['dim']),"nch":len(L),"gold":gold,"probs":{},"tl":{},"ntok":{}}
        p,N,tl,_,_ = run(fb,text,L);            rec["probs"]["bar"]=p; rec["tl"]["bar"]=tl; rec["ntok"]["bar"]=N
        p,Nh,tlh,_,_= run(fh,text,L);           rec["probs"]["hi"]=p;  rec["tl"]["hi"]=tlh; rec["ntok"]["hi"]=Nh
        # AVR budget encode + label-free keep choice from its own attention
        p,Na,tla,cols,att = run(fa,text,L,want_attn=True)
        # SAVE IT. This unpruned KAFF-budget forward is the headroom-identity arm: without it a
        # null is uninterpretable -- "6 frames ~ 4 frames, so no headroom at 1.5x" cannot be told
        # apart from "pruning at L16 destroyed the gain". NOT equal-compute (no cut).
        rec["probs"]["hi_at_budget"]=p; rec["tl"]["hi_at_budget"]=tla
        rec["ntok"]["avr_encode"]=Na
        ndrop=int(round((1-KEEP)*Na))
        order=np.argsort(att)                     # lowest attention dropped first
        drop_avr=cols[torch.tensor(order[:ndrop].copy(),device=cols.device)]
        perm=rng.permutation(Na)
        drop_rnd=cols[torch.tensor(perm[:ndrop].copy(),device=cols.device)]
        p,_,tl2,_,_ = run(fa,text,L,cut=PCUT,drop=drop_avr); rec["probs"]["avr"]=p; rec["tl"]["avr"]=tl2
        p,_,tl3,_,_ = run(fa,text,L,cut=PCUT,drop=drop_rnd); rec["probs"]["avr_rand"]=p; rec["tl"]["avr_rand"]=tl3
        p,_,tl4,_,_ = run(fa,text,L,cut=EARLY,drop=drop_avr); rec["probs"]["early"]=p; rec["tl"]["early"]=tl4
        if not shown:
            print(f"  [selfcheck] bar {N} tok TL={tl} | avr encode {Na} tok cut L{PCUT} keep {KEEP:.0%} "
                  f"TL={tl2} ({tl2/tl:.2f}x bar) | hi {Nh} tok TL={tlh} ({tlh/tl:.1f}x bar)",flush=True)
            assert tl2<=tl*1.15, f"AVR is not equal-compute: {tl2/tl:.2f}x the bar"
            shown=True
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            a=lambda A:100*np.mean([int(np.argmax(x["probs"][A]))==x["gold"] for x in out])
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/it | bar={a('bar'):.1f} avr={a('avr'):.1f} "
                  f"rand={a('avr_rand'):.1f} early={a('early'):.1f} hi={a('hi'):.1f}",flush=True)
    fo.close(); json.dump(out,open("data/phase250_video_avr.json","w")); report(out,skips)

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
    print(f"\n=== PHASE 250 AVR-VIDEO  n={len(out)} / {len(gk)} base videos ===")
    if skips: print("  skips:",dict(skips))
    b=np.mean([r["tl"]["bar"] for r in out])
    print("  arm          acc    TL/bar")
    ident=sum(1 for r in out if r["probs"]["avr"]==r["probs"]["avr_rand"])
    if ident: print(f"  !! avr == avr_rand EXACTLY on {ident}/{len(out)} items -- arms may share a keep set")
    for a in ("bar","avr","avr_rand","early","hi_at_budget","hi"):
        print(f"    {a:10s} {100*np.mean([cor(r,a) for r in out]):5.1f}   {np.mean([r['tl'][a] for r in out])/b:.2f}x")
    print()
    for a,bb,lab in (("avr","bar","AVR GAIN at equal compute (avr - bar)"),
                     ("avr","avr_rand","keep-choice matters (avr - rand)"),
                     ("early","avr","cut DEPTH matters (early - avr)"),
                     ("hi_at_budget","bar","headroom AT THE AVR BUDGET (hi@1.5x - bar), unpruned"),
                     ("avr","hi_at_budget","cost of pruning (avr - hi@1.5x)"),
                     ("hi","bar","headroom ceiling (hi@8x - bar), NOT affordable")):
        m,lo,hi,s=ci(a,bb); print(f"  {lab:44s} = {m:+.1f} [{lo:+.1f},{hi:+.1f}]{s}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report": report(json.load(open("data/phase250_video_avr.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 400)
