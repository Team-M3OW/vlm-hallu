"""
Phase 251 -- DWA-VIDEO: fit the placer ON VIDEO, out-of-fold. (Track V, gate 2 passed)
GATE: phase249 bias-matched ceiling crop_oracle - bar_oracle = +5.5 [+2.7,+8.5]* -> placement has
genuine value on video. The INCUMBENT baseline (single block-mean attention) does not find it:
block - rand = -0.7 [-5.2,+3.8], block - bar = -9.8*. The gap oracle - block = +19.0* is the prize.

WHAT THIS IS. DWA proper: a ridge over PER-LAYER attention features that scores each candidate
window, label-free at inference. This is the video analogue of the vision ridge over 63 per-layer
features -- REFIT HERE, not ported. The vision weights are not even dimensionally transferable
(per-layer features, different depth/architecture), and nothing vision-fitted is loaded anywhere
in this pipeline.

FEATURES (label-free, from ONE bar encoding of the whole clip):
  for each layer l in 0..NL-1 and each window j: mean attention from non-video rows to the video
  columns belonging to window j, plus that value normalised across windows within the layer.
  -> 2*NL features per window. The normalisation matters: raw attention mass drifts with depth, so
  an unnormalised ridge can learn "prefer deep layers" instead of "prefer this window".
TARGET: per-window gold probability (the window's quality). At inference pick argmax_j w.f_j.

LEAK DISCIPLINE (SS78: phase225's V* dwa_t was fitted on V* and only caught because it reproduced
the OOF number EXACTLY -- exact agreement between two supposedly independent estimators is the
signature):
  * GroupKFold by BASE VIDEO, stripping _reverse/_concat_N. Question-level folds would leak,
    because those variants share source content (the HR-Bench 4-item-cycle lesson).
  * Both IN-SAMPLE and OOF accuracy are reported. If they agree exactly, that is a LEAK, not a
    result, and the run says so.
CIs cluster-bootstrapped by base video.
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
BASE=lambda i: re.sub(r'_(reverse|concat_\d+)$','',i)

def frames_from(path,k,lo=0.0,hi=1.0,px=PX):
    import decord
    vr=decord.VideoReader(path,num_threads=2); n=len(vr)
    a,b=int(lo*(n-1)),int(hi*(n-1))
    idx=np.linspace(a,max(a,b),k).round().astype(int)
    return [Image.fromarray(f).convert("RGB").resize((px,px),Image.BICUBIC) for f in vr.get_batch(idx).asnumpy()]

def collect(nmax):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    vids={os.path.splitext(os.path.basename(p))[0]:p for p in glob.glob(VIDEODIR+"/**/*.mp4",recursive=True)}
    df=pd.read_parquet(glob.glob(ROOT+"/multi-choice/*.parquet")[0]).sample(frac=1.0,random_state=251)
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
    def fwd(frames,text,L,want_attn=False):
        inp=build(frames,text).to(model.device)
        with torch.no_grad(): o=model(**inp,output_attentions=want_attn)
        ids=[tok.encode(x,add_special_tokens=False)[0] for x in L]
        p=torch.softmax(o.logits[0,-1].float()[ids],-1).tolist()
        feats=None
        if want_attn:
            m=inp["mm_token_type_ids"][0]; cols=(m==2).nonzero().flatten()
            vset=set(int(x) for x in cols)
            rows=[i for i in range(int(cols[0]),inp["input_ids"].shape[1]) if i not in vset]
            NL=len(o.attentions); per=np.array_split(np.arange(len(cols)),K)
            raw=np.zeros((NL,K),dtype=np.float32)
            for l in range(NL):
                A=o.attentions[l][0].mean(0)[rows][:,cols].mean(0).float().cpu().numpy()
                for j in range(K): raw[l,j]=A[per[j]].mean()
            nrm=raw/ (raw.sum(axis=1,keepdims=True)+1e-12)   # share within layer
            feats=np.concatenate([raw,nrm],axis=0)           # (2*NL, K)
        return p, feats
    out=[]; t0=time.time(); skips=collections.Counter()
    fo=open("data/phase251_video_dwa_fit.jsonl","w")
    windows=[(i/K,(i+1)/K) for i in range(K)]
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
            fb=frames_from(vids[vid],NFR); fw=[frames_from(vids[vid],NFR,a,b) for a,b in windows]
            fb2=frames_from(vids[vid],2*NFR)     # bar_matched: 2x frames = 2x tokens = DWA's true cost
        except Exception: skips["decode"]+=1; continue
        pbar,feats=fwd(fb,text,L,want_attn=True)
        pw=[fwd(f,text,L)[0] for f in fw]
        pbar2,_=fwd(fb2,text,L)
        rec={"id":vid,"base":BASE(vid),"dim":str(r['dim']),"nch":len(L),"gold":gold,
             "p_bar":pbar,"p_bar_matched":pbar2,"p_win":pw,"feats":feats.tolist()}
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            print(f"  [{len(out)}/{nmax}] {(time.time()-t0)/len(out):.1f}s/it",flush=True)
    fo.close(); json.dump(out,open("data/phase251_video_dwa_fit.json","w")); print("collected",len(out),dict(skips))
    return out

def fit_and_eval(out):
    """Mirrors the VISION DWA estimator (scripts/fit_ridge_weights.py): per-candidate features are
    log-attention per layer + within-layer RANK + geometry; OOF GroupKFold(5) x 3 group
    permutations; lambda=1; features standardised PER FOLD; unpenalised intercept.
    TWO NAMED DEVIATIONS, forced by the benchmark rather than chosen:
      * Vision's target is ground-truth BBOX COVERAGE. TempCompass has no temporal annotation, so
        the target here is the window's gold probability, DEMEANED WITHIN ITEM -- only the
        within-item ranking matters, and without demeaning the ridge mostly learns item difficulty.
      * Geometry collapses 2-D -> 1-D: 6 terms instead of 7 (no separate fx/fy).
    """
    from sklearn.model_selection import GroupKFold
    rng=np.random.default_rng(0)
    raw=np.array([r["feats"] for r in out])[:, :len(out[0]["feats"])//2, :]   # (n, NL, K) raw attn
    n,NL,Kk=raw.shape
    groups=np.array([r["base"] for r in out])
    a=raw/np.maximum(raw.sum(axis=2,keepdims=True),1e-12)      # share within layer, per item
    LA=np.log(a+1e-12)                                          # (n,NL,K)
    R=np.argsort(np.argsort(-a,axis=2),axis=2)/max(Kk-1,1)      # within-layer rank across windows
    BLK=round(0.57*NL)                                          # vision's depth band, same rule
    dep=a[:,BLK:,:].mean(axis=1)                                # (n,K)
    pad=np.pad(dep,((0,0),(1,1)),mode="edge")
    nb=(pad[:,:-2]+pad[:,1:-1]+pad[:,2:])/3.0                   # 1-D neighbourhood of the depth band
    t=(np.arange(Kk)+0.5)/Kk
    geo=np.stack([np.log(nb+1e-12),
                  np.broadcast_to(t,(n,Kk)),
                  np.broadcast_to(np.abs(t-0.5),(n,Kk)),
                  np.broadcast_to(np.minimum(t,1-t),(n,Kk)),
                  np.broadcast_to((np.arange(Kk)==0).astype(float),(n,Kk)),
                  np.broadcast_to((np.arange(Kk)==Kk-1).astype(float),(n,Kk))],axis=1)  # (n,6,K)
    X=np.concatenate([LA,R,geo],axis=1)                         # (n, 2NL+6, K)
    F=X.shape[1]
    q=np.array([[r["p_win"][j2][r["gold"]] for j2 in range(Kk)] for r in out])
    y=q-q.mean(axis=1,keepdims=True)                            # within-item demean (see docstring)
    Xf=np.transpose(X,(0,2,1)).reshape(n*Kk,F); yf=y.reshape(-1)
    item=np.repeat(np.arange(n),Kk); gf=np.repeat(groups,Kk)
    cw=np.array([[float(int(np.argmax(r["p_win"][j2]))==r["gold"]) for j2 in range(Kk)] for r in out])
    cbar=np.array([float(int(np.argmax(r["p_bar"]))==r["gold"]) for r in out])
    cbm=np.array([float(int(np.argmax(r.get("p_bar_matched",r["p_bar"])))==r["gold"]) for r in out])
    S=np.zeros((n,Kk)); nrep=0
    gs=np.unique(groups)
    for rep in range(3):
        rr=np.random.default_rng(700+rep); perm={g:i2 for i2,g in enumerate(rr.permutation(gs))}
        Gp=np.vectorize(perm.get)(gf)
        Sr=np.zeros((n,Kk))
        for tr,te in GroupKFold(5).split(Xf,yf,Gp):
            mu,sd=Xf[tr].mean(0),Xf[tr].std(0)+1e-9
            Xt=np.c_[(Xf[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0     # unpenalised intercept
            w=np.linalg.solve(A_,Xt.T@yf[tr])
            pr_=np.c_[(Xf[te]-mu)/sd,np.ones(len(te))]@w
            for idx,v in zip(te,pr_): Sr[item[idx],idx%Kk]=v
        S+=Sr; nrep+=1
    S/=nrep
    pick=np.argmax(S,axis=1)
    # IN-SAMPLE twin: exact agreement with OOF would be the SS78 leak signature.
    mu,sd=Xf.mean(0),Xf.std(0)+1e-9; Xt=np.c_[(Xf-mu)/sd,np.ones(len(Xf))]
    A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
    w_in=np.linalg.solve(A_,Xt.T@yf); pick_in=np.argmax((Xt@w_in).reshape(n,Kk),axis=1)
    # incumbent EXACTLY as phase249 measured it: raw attention over layers 0.3..0.5 of depth
    blk=np.argmax(raw[:,int(.3*NL):int(.5*NL),:].mean(axis=1),axis=1)
    rnd=rng.integers(0,Kk,n); orc=np.argmax(q,axis=1)
    acc=lambda pk: 100*np.mean(cw[np.arange(n),pk])
    print(f"\n=== PHASE 251 DWA-VIDEO (refit ON VIDEO, OOF by base video)  n={n} / {len(gs)} base videos ===")
    print(f"  features {F} = 2*{NL} + 6 (vision: 2*NL + 7)")
    for k_,v in (("bar",100*cbar.mean()),("bar_matched(2x)",100*cbm.mean()),("crop_rand",acc(rnd)),
                 ("crop_block",acc(blk)),("DWA oof",acc(pick)),("DWA in-sample",acc(pick_in)),
                 ("crop_oracle",acc(orc))):
        print(f"  {k_:18s} {v:5.1f}")
    if abs(acc(pick)-acc(pick_in))<1e-9 and (pick==pick_in).all():
        print("  !! OOF == IN-SAMPLE EXACTLY -> LEAK, not a result (SS78 signature)")
    else:
        print(f"  in-sample - OOF = {acc(pick_in)-acc(pick):+.1f}; picks differ on "
              f"{100*(pick!=pick_in).mean():.0f}% of items (exact agreement would be a leak)")
    def ci(A,B):
        d=A-B; per=np.array([d[groups==gg].mean() for gg in gs])
        mm=per[rng.integers(0,len(per),(10000,len(per)))].mean(1)*100
        lo,hi=float(np.percentile(mm,2.5)),float(np.percentile(mm,97.5))
        return per.mean()*100,lo,hi,("*" if (lo>0 or hi<0) else " ")
    sel=cw[np.arange(n),pick]
    print("\n  EQUAL-COMPUTE comparisons (DWA pays localisation + window = 2x the bar):")
    for nm,B in (("DWA - bar_matched(2x)",cbm),("DWA - crop_block",cw[np.arange(n),blk])):
        m_,lo,hi,st_=ci(sel,B); print(f"    {nm:26s} = {m_:+.1f} [{lo:+.1f},{hi:+.1f}]{st_}")
    print("  diagnostics (NOT equal compute):")
    for nm,B in (("DWA - bar(1x)",cbar),("DWA - crop_rand",cw[np.arange(n),rnd])):
        m_,lo,hi,st_=ci(sel,B); print(f"    {nm:26s} = {m_:+.1f} [{lo:+.1f},{hi:+.1f}]{st_}")
    m_,lo,hi,st_=ci(cw[np.arange(n),orc],sel)
    print(f"    {'oracle - DWA (remaining)':26s} = {m_:+.1f} [{lo:+.1f},{hi:+.1f}]{st_}  (oracle is best-of-K, itself worth ~+3.7)")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="fit":
        fit_and_eval(json.load(open("data/phase251_video_dwa_fit.json")))
    else:
        fit_and_eval(collect(int(sys.argv[1]) if len(sys.argv)>1 else 600))
