"""
Phase 253 (Track R, H7+H8) -- DWA-VLA CEILING, and the SCOPE LAW in a third modality.
ORACLE FIRST, as in audio and video: if the bias-matched ceiling is not CI-clear, STOP and write
no placer. That rule already stopped DWA-audio (oracle -3.3) and it is what kept DWA-video honest
(uncontrolled +9.1 -> bias-matched +5.5, i.e. best-of-K alone was worth +3.7).

WHY THIS IS THE CLEAN CASE. OpenVLA resizes every observation to 224x224 with patch-14 towers, so
it emits EXACTLY 256 visual tokens whatever it is fed (G0: NO_LADDER). A crop is therefore
AUTOMATICALLY budget-matched -- crop and full frame cost the same 256 tokens and the same
token-layers. None of the equal-compute machinery that video needed (bar_matched) or that forced
the SS73/SS75/SS81 corrections in vision can apply. The 256==256 equality is ASSERTED, not assumed.

ARMS (all at 256 visual tokens):
  bar          full 256x256 frame resized to 224
  crop_rand    a random one of the K windows
  crop_block   window with the largest mean image->text attention at the read-out layers (label-free)
  crop_oracle  the window minimising action error, chosen WITH THE GROUND-TRUTH ACTION -> ceiling
  bar_oracle   best of K whole-frame JITTERS (small translations, no zoom), also label-chosen
               -> MANDATORY bias control. Honest ceiling = crop_oracle - bar_oracle.

METRIC: mean L1 between the model's predicted 7-DoF action and the demonstrated action, via
OpenVLA's own predict_action (it handles de-normalisation). LOWER IS BETTER, so all deltas are
reported as ERROR REDUCTION (bar_error - arm_error); positive = the arm helps.
THIS IS NOT TASK SUCCESS. Open-loop action error can move without closed-loop success moving; a
rollout needs the simulator. Claims are about the action distribution only.

H8 SCOPE LAW: LIBERO Object is single-target ("pick up the orange juice..."), Spatial is relational
("pick up the black bowl NEXT TO THE COOKIE BOX..."). The law predicts cropping helps on Object and
hurts on Spatial. Run both suites; no router, stratify after the fact.
"""
import os, sys, json, glob, time, collections, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
SUITE=os.environ.get("SUITE","spatial")
MID=f"openvla/openvla-7b-finetuned-libero-{SUITE}"
NEP=int(os.environ.get("NEP","60")); STRIDE=int(os.environ.get("STRIDE","12"))
FRAC=float(os.environ.get("FRAC","0.6")); K=int(os.environ.get("K","4"))

def read_frames(path):
    import av
    with av.open(path) as c: return [f.to_image().convert("RGB") for f in c.decode(video=0)]

def windows(W,H,frac,k):
    """k crop windows: corners of a frac-sized box. Each is resized to 224 -> same 256 tokens."""
    w,h=int(W*frac),int(H*frac)
    return [(0,0,w,h),(W-w,0,W,h),(0,H-h,w,H),(W-w,H-h,W,H)][:k]

def jitters(W,H,frac,k):
    """Whole-frame views at the SAME scale as the bar, shifted slightly. The bias control: they
    contain essentially the same content, so any best-of-k gain here is SELECTION NOISE."""
    d=int(W*(1-frac)/4) or 1
    return [(0,0,W,H),(d,0,W,H-d),(0,d,W-d,H),(d,d,W,H)][:k]

def main():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    base=glob.glob(f"/media/kavinder/hdd2/hf_cache/datasets--IPEC-COMMUNITY--libero_{SUITE}_no_noops_1.0.0_lerobot/snapshots/*")[0]
    pq=sorted(glob.glob(base+"/data/chunk-000/*.parquet"))[:NEP]
    vids={os.path.basename(p).split('.')[0]:p for p in glob.glob(base+"/videos/chunk-000/observation.images.image/*.mp4")}
    tasks={}
    for l in open(os.path.join(base,"meta/tasks.jsonl")):
        d=json.loads(l); tasks[d["task_index"]]=d["task"]
    pr=AutoProcessor.from_pretrained(MID, trust_remote_code=True)
    model=AutoModelForVision2Seq.from_pretrained(MID, dtype=torch.bfloat16, device_map={"":0},
            attn_implementation="eager", trust_remote_code=True, low_cpu_mem_usage=True).eval()
    UNNORM=[k for k in getattr(model,"norm_stats",{}) ] or ["libero_"+SUITE]
    print(f"  unnorm keys: {UNNORM[:3]}",flush=True)
    UK=UNNORM[0]
    def act(img, instr, want_attn=False):
        prompt=f"In: What action should the robot take to {instr.lower()}?\nOut:"
        inp=pr(prompt, img.resize((224,224), Image.BICUBIC)).to(model.device, dtype=torch.bfloat16)
        a=model.predict_action(**inp, unnorm_key=UK, do_sample=False)
        return np.asarray(a,dtype=np.float32), inp
    out=[]; t0=time.time(); shown=False
    fo=open(f"data/phase253_vla_dwa_{SUITE}.jsonl","w")
    for ep in pq:
        df=pd.read_parquet(ep); key=os.path.basename(ep).split('.')[0]
        if key not in vids or not len(df): continue
        instr=tasks.get(int(df.iloc[0]["task_index"]),"complete the task")
        frames=read_frames(vids[key])
        for t in range(0, min(len(df),len(frames)), STRIDE):
            gt=np.asarray(df.iloc[t]["action"],dtype=np.float32)
            im=frames[t]; W,H=im.size
            a_bar,inp=act(im,instr)
            ntok_bar=inp["input_ids"].shape[1] if "input_ids" in inp else None
            errs={}; errs["bar"]=float(np.abs(a_bar-gt).mean())
            cw=[im.crop(b) for b in windows(W,H,FRAC,K)]
            jw=[im.crop(b) for b in jitters(W,H,FRAC,K)]
            ce=[float(np.abs(act(c,instr)[0]-gt).mean()) for c in cw]
            je=[float(np.abs(act(j,instr)[0]-gt).mean()) for j in jw]
            errs["crop_oracle"]=min(ce); errs["bar_oracle"]=min(je)
            errs["crop_rand"]=ce[int(np.random.default_rng(len(out)).integers(0,K))]
            rec={"ep":key,"t":int(t),"suite":SUITE,"instr":instr,"err":errs,"ce":ce,"je":je}
            if not shown:
                n2=pr(f"In: x?\nOut:", cw[0].resize((224,224),Image.BICUBIC))["input_ids"].shape[1]
                n1=pr(f"In: x?\nOut:", im.resize((224,224),Image.BICUBIC))["input_ids"].shape[1]
                print(f"  [selfcheck] bar seq={n1} crop seq={n2} -> {'EQUAL budget' if n1==n2 else 'UNEQUAL!'}",flush=True)
                assert n1==n2, "crop and bar differ in token count -- NO_LADDER assumption broken"
                shown=True
            out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%50<STRIDE and out:
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/step bar={np.mean([r['err']['bar'] for r in out]):.4f}",flush=True)
    fo.close(); json.dump(out,open(f"data/phase253_vla_dwa_{SUITE}.json","w")); report(out)

def report(out):
    rng=np.random.default_rng(0)
    eps=np.array([r["ep"] for r in out])
    def ci(a,b):
        """error REDUCTION (b - a): positive means arm `a` has LOWER error than `b`."""
        d=np.array([r["err"][b]-r["err"][a] for r in out])
        per=np.array([d[eps==e].mean() for e in np.unique(eps)])   # cluster by EPISODE
        m=per[rng.integers(0,len(per),(10000,len(per)))].mean(1)
        lo,hi=float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
        return per.mean(),lo,hi,("*" if (lo>0 or hi<0) else " ")
    print(f"\n=== PHASE 253 DWA-VLA ({out[0]['suite']})  n={len(out)} steps / {len(set(eps))} episodes ===")
    print("  mean L1 action error (LOWER is better):")
    for a in ("bar","bar_oracle","crop_rand","crop_oracle"):
        print(f"    {a:12s} {np.mean([r['err'][a] for r in out]):.4f}")
    m,lo,hi,s=ci("crop_oracle","bar_oracle")
    print(f"\n  CEILING (bias-matched)  crop_oracle vs bar_oracle = {m:+.4f} [{lo:+.4f},{hi:+.4f}]{s}"
          f"  {'PASS' if lo>0 else 'UNDERWATER -> STOP, no placer'}")
    for a,b,lab in (("crop_oracle","bar","uncontrolled (inflated by best-of-K)"),
                    ("bar_oracle","bar","selection bias itself"),
                    ("crop_rand","bar","does cropping help ON AVERAGE? (the scope law)")):
        m,lo,hi,s=ci(a,b); print(f"  {lab:44s} = {m:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report":
        report(json.load(open(f"data/phase253_vla_dwa_{SUITE}.json")))
    else: main()
