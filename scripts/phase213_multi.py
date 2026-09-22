"""
Phase 213: the transport boundary and AVR across MODEL FAMILIES and BENCHMARKS.

Per (model, benchmark) it runs:
  uniform@E_lo   the equal-compute bar
  uniform@E_hi   the resolution-headroom diagnostic (no method)
  avr            encode @E_hi, prune 90% of visual tokens at round(0.57*NL)   [AVR]
  avr_rand       same budget, RANDOM keep set                                 [the §42 control]
  suffix_mask    block text->image attention from the boundary on             [the §20B boundary check]
Everything here is label-free, so it runs on benchmarks without boxes.

Usage:  phase213_multi.py <model-key> <benchmark-key>
The attention patch is discovered from the LOADED model's own attention class module at runtime and asserted,
because hard-coding module names silently voided a run once (§38B fault 1).
"""
import json, os, sys, time, random, importlib, io, base64, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)  # the env token is invalid; use the stored login
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
MODELS={"qwen3_2b":"Qwen/Qwen3-VL-2B-Instruct","qwen2_7b":"Qwen/Qwen2-VL-7B-Instruct",
        "llava_ov":"llava-hf/llava-onevision-qwen2-7b-ov-hf","llava_next":"llava-hf/llava-v1.6-vicuna-7b-hf",
        "gemma3_4b":"google/gemma-3-4b-it","phi35v":"microsoft/Phi-3.5-vision-instruct"}
MK, BK = sys.argv[1], sys.argv[2]
N_ITEMS=int(sys.argv[3]) if len(sys.argv)>3 else 200
OUT=f"{D}/data/phase213_{MK}_{BK}.jsonl"
E_LO,E_HI,K=600,900,0.10

# ---------------- benchmarks ----------------
def load_bench(key):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    if key=="vstar":
        root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
        def it():
            for e in ds:
                yield (f"{e['category']}/{e['question_id']}", os.path.join(root,e["image"]), e["text"],
                       "ABCD".index(e["label"]) if isinstance(e["label"],str) else int(e["label"]), e["category"], "mcq4")
        return it
    if key=="pope":
        ds=load_dataset("lmms-lab/POPE")["test"]
        def it():
            for i,e in enumerate(ds):
                yield (f"{e['category']}/{e['question_id']}", e["image"],
                       e["question"]+"\nAnswer yes or no.", 0 if str(e["answer"]).strip().lower()=="yes" else 1,
                       e["category"], "yesno")
        return it
    if key in ("hr4k","hr8k"):
        cfg="hrbench_4k" if key=="hr4k" else "hrbench_8k"
        ds=load_dataset("DreamMr/hr-bench","hrbench_version_split")[cfg]
        def it():
            for e in ds:
                q=e["question"]+"\n"+"\n".join(f"({c}) {e[c]}" for c in "ABCD" if c in e)+\
                  "\nAnswer with the option's letter from the given choices directly."
                yield (str(e.get("index",e.get("id",0))), e["image"], q,
                       "ABCD".index(str(e["answer"]).strip()[0]), e.get("category","all"), "mcq4")
        return it
    raise SystemExit(f"unknown benchmark {key}")

def main():
    mid=MODELS[MK]
    model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                      attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
    layers=None
    for f in (lambda m:m.model.language_model.layers, lambda m:m.language_model.model.layers, lambda m:m.model.layers):
        try: layers=f(model); break
        except Exception: pass
    assert layers is not None, "decoder layers not found"
    NL=len(layers); P=round(0.57*NL)
    # --- patch the attention module the LOADED model actually uses, then assert it ---
    modname=type(layers[0].self_attn).__module__
    QM=importlib.import_module(modname)
    assert hasattr(QM,"eager_attention_forward"), f"{modname} has no eager_attention_forward to patch"
    def make_patched(QM):
        def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
            ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
            w=torch.matmul(query,ks.transpose(2,3))*scaling
            if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
            b=getattr(module,"_prune_bias",None)
            if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
            w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
            if getattr(module,"_capture",False):
                module._lastrow=w[0,:,-1,:].float().mean(0).detach().cpu().numpy()
            return torch.matmul(w,vs).transpose(1,2).contiguous(), w
        return patched
    QM.eager_attention_forward=make_patched(QM)
    print(f"{MK}: NL={NL} prune@L{P} attn-module={modname}",flush=True)
    itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
    assert itid is not None, "image token id not found"
    YES=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in ["Yes"," Yes","yes"," yes"]}),
         sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in ["No"," No","no"," no"]})]
    MCQ=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                               tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
    def ntok(x):
        inp=x if isinstance(x,dict) else x
        return int((inp["input_ids"][0]==itid).sum())
    def fit(img,target,refine=6,tol=0.10):
        W_,H_=img.size; best=None; sc=(target/max(ntok(build(img,"x")),1))**0.5
        for _ in range(refine):
            cur=img.resize((max(32,int(W_*sc)),max(32,int(H_*sc))),Image.BICUBIC); r=ntok(build(cur,"x"))
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/max(r,1))**0.5
        return best
    def clear():
        for l in layers:
            for a in ("_prune_bias","_mask_img"):
                if hasattr(l.self_attn,a): delattr(l.self_attn,a)
    def run(inp,opts,drop=None,frm=None,want=False):
        clear()
        if drop is not None and len(drop):
            b=torch.zeros(inp["input_ids"].shape[1],device=model.device)
            b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(frm,NL): layers[li].self_attn._prune_bias=b
        for l in layers: l.self_attn._capture=bool(want)
        with torch.no_grad(): out=model(**inp)
        lg=out.logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opts]),0)
        A=None
        if want: A=np.stack([layers[L].self_attn._lastrow for L in range(NL)])
        for l in layers: l.self_attn._capture=False
        del out; torch.cuda.empty_cache(); clear()
        return [round(float(v),6) for v in p.tolist()], A
    # ---- per-architecture budget discovery (Prop. 2) -------------------------------------------
    # Anyres tilers quantise resolution, so a fixed 600/900 pair is meaningless for them. Find the
    # achievable visual-token counts, then solve Prop. 2 for the largest keep fraction that still fits
    # the bar:  k = (E_lo*NL - E_hi*(P+1)) / (E_hi*(NL-1-P)).  If k < 0.05 the step is too coarse and
    # AVR is not budget-feasible on this architecture -- recorded, not fudged.
    probe=Image.new("RGB",(1200,900),(120,140,160))
    achievable=sorted({ntok(build(probe.resize((max(32,int(1200*f)),max(32,int(900*f)))),"x"))
                       for f in (0.12,0.2,0.28,0.4,0.56,0.8,1.0,1.6,2.2)})
    achievable=[a for a in achievable if a>0]
    E_lo_a=achievable[0]; best=None
    for E in achievable[1:]:
        k=(E_lo_a*NL - E*(P+1))/max(E*(NL-1-P),1)
        if k>=0.05 and (best is None or E>best[0]): best=(E,k)
    if best is None:
        E_hi_a,K_a,feasible=achievable[-1],K,False
    else:
        E_hi_a,K_a,feasible=best[0],min(best[1],0.5),True
    print(f"   achievable visual-token counts: {achievable}",flush=True)
    print(f"   bar E_lo={E_lo_a}  AVR E_hi={E_hi_a}  keep k={K_a:.3f}  "
          f"{'FEASIBLE' if feasible else 'NOT budget-feasible (resolution axis too coarse)'}",flush=True)
    rng=random.Random(213); it=load_bench(BK)
    done=set()
    if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
    t0,n=time.time(),0
    with open(OUT,"a") as fout:
        for qid,imsrc,text,lab,cat,kind in it():
            if n>=N_ITEMS: break
            if qid in done: continue
            img=None
            if isinstance(imsrc,str):
                # HR-Bench stores images as base64 strings, not paths. Try both, loudly.
                try: img=Image.open(io.BytesIO(base64.b64decode(imsrc)))
                except Exception:
                    try: img=Image.open(imsrc)
                    except Exception as ex: print(f"   image decode failed for {qid}: {type(ex).__name__}",flush=True)
            else: img=imsrc
            if img is None: continue
            img=img.convert("RGB")
            opts=MCQ if kind=="mcq4" else YES
            rec={"qid":qid,"category":cat,"label":lab,"kind":kind,"probs":{},"tokens":{},"nl":NL,"P":P}
            ok=True
            for nm,E in (("uniform@lo",E_lo_a),("uniform@hi",E_hi_a)):
                sm,rt=fit(img,E)
                if rt==0: ok=False; break
                inp=build(sm,text).to(model.device)
                rec["probs"][nm],_=run(inp,opts); rec["tokens"][nm]=ntok(inp)
            if not ok: continue
            sm,_=fit(img,E_hi_a); inp=build(sm,text).to(model.device)
            pos=(inp["input_ids"][0]==itid).nonzero().flatten()
            if len(pos)==0: continue
            base,nt=int(pos[0]),int(len(pos))
            _,A=run(inp,opts,want=True)
            Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12)
            s=Ai[max(0,P-4):P+1].mean(0); keep=max(1,int(round(K_a*nt)))
            rec["probs"]["avr"],_=run(inp,opts,drop=(base+np.argsort(-s)[keep:]).tolist(),frm=P+1)
            sr=np.array([rng.random() for _ in range(nt)])
            rec["probs"]["avr_rand"],_=run(inp,opts,drop=(base+np.argsort(-sr)[keep:]).tolist(),frm=P+1)
            rec["probs"]["suffix_mask"],_=run(inp,opts,drop=list(range(base,base+nt)),frm=P)
            rec["tokens"]["avr"]=nt; rec["keep_frac"]=K_a; rec["avr_feasible"]=feasible
            rec["tl"]={"bar":rec["tokens"]["uniform@lo"]*NL,
                       "avr":nt*(P+1)+keep*(NL-1-P)}
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
