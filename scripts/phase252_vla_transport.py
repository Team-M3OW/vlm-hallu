"""
Phase 252 (Track R) -- WHERE DOES THE IMAGE REACH THE ACTION?  OpenVLA-7B on LIBERO.
The phase176c protocol in a third modality. PRE-REGISTERED (PREREG_CROSSMODAL H6): transport
completes at 0.57 of depth = L18/32 for the Llama-2-7B backbone -- the value measured in BOTH
vision and video. Audio's 0.75 is the outlier. If it lands elsewhere, that is the result: audio
already falsified one such prediction.

WHY VLA IS THE CLEAN CASE. OpenVLA resizes every observation to 224x224 with patch-14 towers, so
it emits EXACTLY 256 visual tokens no matter what (G0: NO_LADDER). AVR is therefore not expressible
-- but a crop is AUTOMATICALLY budget-matched, which removes the equal-compute confound that forced
bar_matched in video and the SS73/SS75/SS81 corrections in vision.

WHAT IS MEASURED. OpenVLA emits 7 discrete action tokens (256 bins each) from the tail of the
vocabulary. We mask text->image attention at selected layers and measure KL on the FIRST action
token's distribution over those 256 bins, plus whether the arg-max bin changes. Single-layer,
prefix and suffix schedules, exactly as on images/audio/video.
The OFT checkpoints are NOT used: they decode actions continuously in parallel and expose no
action logits to read.

IMPLEMENTATION UNKNOWN, RESOLVED EMPIRICALLY: OpenVLA concatenates projected patch embeddings into
inputs_embeds rather than expanding a placeholder token, so there may be NO image token id in
input_ids. The image span is therefore discovered by differencing sequence lengths with and
without the image, and the discovery is ASSERTED (must be exactly 256) before anything is measured.
SANITY (SS93b): mask-all must give large KL and move the action bin; per-layer KL > 0.
"""
import json, os, sys, time, glob, collections, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import pandas as pd
from PIL import Image
SUITE=os.environ.get("SUITE","spatial")
MID=f"openvla/openvla-7b-finetuned-libero-{SUITE}"
NCAL=int(os.environ.get("NCAL","40"))

def episodes(suite, n):
    base=glob.glob(f"/media/kavinder/hdd2/hf_cache/datasets--IPEC-COMMUNITY--libero_{suite}_no_noops_1.0.0_lerobot/snapshots/*")[0]
    pq=sorted(glob.glob(base+"/data/chunk-000/*.parquet"))[:n]
    vids={os.path.basename(p).split('.')[0]: p for p in glob.glob(base+"/videos/chunk-000/observation.images.image/*.mp4")}
    tasks={}
    tj=os.path.join(base,"meta/tasks.jsonl")
    if os.path.exists(tj):
        for l in open(tj):
            d=json.loads(l); tasks[d["task_index"]]=d["task"]
    return pq, vids, tasks

def main():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import decord
    pq, vids, tasks = episodes(SUITE, NCAL)
    assert pq, "no episode parquet found -- did the asset fetch run?"
    print(f"  {len(pq)} episodes, {len(vids)} videos, {len(tasks)} task strings",flush=True)
    pr=AutoProcessor.from_pretrained(MID, trust_remote_code=True)
    model=AutoModelForVision2Seq.from_pretrained(MID, dtype=torch.bfloat16, device_map={"":0},
            attn_implementation="eager", trust_remote_code=True, low_cpu_mem_usage=True).eval()
    lm=model.language_model if hasattr(model,"language_model") else model.model
    layers=lm.model.layers if hasattr(lm,"model") else lm.layers
    NL=len(layers); print(f"  NL={NL}  predicted boundary L{round(0.57*NL)} = 0.57",flush=True)
    V=model.config.text_config.vocab_size if hasattr(model.config,"text_config") else model.config.vocab_size
    ACT=list(range(V-256, V))            # OpenVLA's action bins live at the tail of the vocab

    def build(img, instr):
        prompt=f"In: What action should the robot take to {instr.lower()}?\nOut:"
        return pr(prompt, img).to(model.device, dtype=torch.bfloat16)

    # --- discover the image span by DIFFERENCING sequence lengths (no placeholder token exists) ---
    probe_img=Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))
    i_with=build(probe_img,"pick up the object")
    n_txt=len(pr.tokenizer(f"In: What action should the robot take to pick up the object?\nOut:")["input_ids"])
    with torch.no_grad(): _o=model(**i_with, output_hidden_states=True)
    n_seq=_o.hidden_states[0].shape[1]
    n_img=n_seq-n_txt
    print(f"  seq={n_seq} text={n_txt} -> image tokens={n_img}",flush=True)
    assert n_img==256, f"expected 256 fixed visual tokens (G0: NO_LADDER), got {n_img}"
    IMG0=1; IMG1=1+n_img                 # patches are inserted right after BOS
    state={"layers":set()}
    def mk(l):
        def pre(mod,args,kwargs):
            if l not in state["layers"]: return None
            am=kwargs.get("attention_mask")
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
            am=am.clone(); am[:,:,IMG1:,IMG0:IMG1]=torch.finfo(am.dtype).min
            kwargs["attention_mask"]=am; return (args,kwargs)
        return pre
    for l in range(NL): layers[l].self_attn.register_forward_pre_hook(mk(l),with_kwargs=True)
    def logits(inp):
        with torch.no_grad(): return model(**inp).logits[0,-1].float()
    def kl(p,q):
        p=torch.softmax(p,-1); lq=torch.log_softmax(q,-1)
        return float((p*(torch.log(p+1e-12)-lq)).sum())

    out=[]; t0=time.time()
    for ep in pq:
        if len(out)>=NCAL: break
        df=pd.read_parquet(ep); key=os.path.basename(ep).split('.')[0]
        if key not in vids or not len(df): continue
        instr=tasks.get(int(df.iloc[0]["task_index"]), "complete the task")
        vr=decord.VideoReader(vids[key], num_threads=2)
        t=min(len(vr)-1, len(df)//2)                       # mid-episode frame
        img=Image.fromarray(vr[t].asnumpy()).convert("RGB").resize((224,224), Image.BICUBIC)
        inp=build(img, instr)
        state["layers"]=set(); base=logits(inp); ba=base[ACT]
        state["layers"]=set(range(NL)); alla=logits(inp)
        rec={"ep":key,"suite":SUITE,"instr":instr,"nimg":n_img,
             "kl_all":kl(base,alla),"kl_all_act":kl(ba,alla[ACT]),
             "flip_all":int(int(torch.argmax(ba))!=int(torch.argmax(alla[ACT]))),
             "kl":[],"flip":[],"prefix":{},"suffix":{}}
        for l in range(NL):
            state["layers"]={l}; q=logits(inp)
            rec["kl"].append(kl(base,q)); rec["flip"].append(int(int(torch.argmax(ba))!=int(torch.argmax(q[ACT]))))
        for l in range(0,NL,2):
            state["layers"]=set(range(0,l+1)); q=logits(inp)
            rec["prefix"][str(l)]=[kl(base,q),int(int(torch.argmax(ba))!=int(torch.argmax(q[ACT])))]
            state["layers"]=set(range(l,NL)); q=logits(inp)
            rec["suffix"][str(l)]=[kl(base,q),int(int(torch.argmax(ba))!=int(torch.argmax(q[ACT])))]
        out.append(rec)
        if len(out)%5==0:
            print(f"  [{len(out)}/{NCAL}] {(time.time()-t0)/len(out):.1f}s/it kl_all={rec['kl_all']:.3f} "
                  f"peak L{int(np.argmax(rec['kl']))}",flush=True)
    json.dump(out,open(f"data/phase252_vla_transport_{SUITE}.json","w")); report(out,NL)

def report(out,NL=None):
    NL=NL or len(out[0]["kl"])
    K=np.array([r["kl"] for r in out])
    print(f"\n=== PHASE 252 VLA BOUNDARY ({out[0]['suite']})  n={len(out)} NL={NL} ===")
    ka=np.mean([r['kl_all'] for r in out]); fa=np.mean([r['flip_all'] for r in out])
    print(f"  SANITY kl_all={ka:.3f} flip_all={fa:.2f}  {'ok' if ka>0.3 else 'WEAK -> suspect the image span'}")
    pk=int(np.argmax(K.mean(0)))
    print(f"  per-layer KL peak: L{pk} = {pk/NL:.2f} of depth   (vision 0.39, audio 0.38, video 0.43)")
    ks=sorted(out[0]["suffix"],key=int)
    b=None
    for k in ks:
        if np.mean([r["suffix"][k][0] for r in out])<=0.03: b=int(k); break
    print(f"  suffix KL<=0.03 -> transport COMPLETE by L{b} = {b/NL:.2f}" if b is not None else "  suffix never reaches 0.03")
    print(f"    PRE-REGISTERED 0.57 (L{round(0.57*NL)}) | vision 0.57 | audio 0.75 | video 0.57")
    print("  suffix: "+"  ".join(f"L{k}:{np.mean([r['suffix'][k][0] for r in out]):.3f}" for k in ks[:14]))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report":
        report(json.load(open(f"data/phase252_vla_transport_{SUITE}.json")))
    else: main()
