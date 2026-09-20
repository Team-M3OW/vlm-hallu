"""
Phase 203 (GPU): two causal/interpretability probes of the transport boundary.

E1  REPRESENTATION PATCHING. Replace the IMAGE tokens' hidden state at layer l with those from a
    DIFFERENT image, and measure the change at the output. Attention masking (§20B) showed the
    text->image attention CHANNEL is inert after the boundary; this tests whether the image token
    VALUES are. Two are different claims.
    PRE-REGISTERED SANITY (must both hold or E1's null is uninterpretable):
      C1  patching image tokens at l=4 must give a LARGE effect  (the patch takes)
      C2  patching TEXT tokens at l=20 must give a LARGE effect  (late layers are not inert in general)
    MAIN PREDICTION: patching image tokens at l >= 16 gives ~0 KL and ~0 answer flips.

E3  LOGIT LENS across depth at the answer position, for the 4 option letters.
    §12D bug guard: HF returns hidden_states[-1] with the final norm ALREADY applied. We apply the norm
    to intermediate layers ONLY, and assert our last-layer read-out matches the true logits.
    PREDICTION: the correct option becomes decodable only AFTER the transport window, giving a two-stage
    picture: image -> text (L0-16, §20B), then text -> answer (L17+).
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 60
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase203_patchlens_{WHICH}.jsonl"; Image.MAX_IMAGE_PIXELS=None
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
lm=model.model.language_model; layers=lm.layers; NL=len(layers)
norm=lm.norm; head=model.lm_head if hasattr(model,"lm_head") else model.get_output_embeddings()
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                           tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
def fit(img,target=300,refine=6,tol=0.08):
    W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
    for _ in range(refine):
        cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
        if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
        if r==0 or abs(r-target)/target<=tol: break
        sc*=(target/r)**0.5
    return best[0]
def probs_from(lg):
    return torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
def run(inp,hook=None):
    h=layers[hook[0]].register_forward_hook(hook[1]) if hook else None
    try:
        with torch.no_grad(): out=model(**inp)
        return out.logits[0,-1].float()
    finally:
        if h is not None: h.remove()
def make_patch(donor,mask):
    def fn(mod,args,output):
        o=output[0] if isinstance(output,tuple) else output
        o=o.clone(); k=min(mask.sum().item(),donor.shape[0])
        idx=mask.nonzero().flatten()[:k]
        o[0,idx]=donor[:k].to(o.dtype)
        return (o,)+tuple(output[1:]) if isinstance(output,tuple) else o
    return fn
def hidden_at(inp,l):
    with torch.no_grad(): out=model(**inp,output_hidden_states=True)
    return out.hidden_states[l+1][0].detach()
lut=[(f"{e['category']}/{e['question_id']}",e) for e in ds]
done=set()
if os.path.exists(OUT): done={json.loads(l)["question_id_full"] for l in open(OUT)}
PATCH_L=[0,4,8,12,16,20,24]
t0=time.time(); n=0
with open(OUT,"a") as fout:
    for i,(qid,ex) in enumerate(lut):
        if qid in done or n>=N: continue
        ip=os.path.join(root,ex["image"])
        dq,dex=lut[(i+37)%len(lut)]                       # a fixed, different donor image
        dp=os.path.join(root,dex["image"])
        if not (os.path.exists(ip) and os.path.exists(dp)): continue
        img=fit(Image.open(ip).convert("RGB")); dimg=fit(Image.open(dp).convert("RGB"))
        lab="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
        inp=build(img,ex["text"]).to(model.device)
        dinp=build(dimg,ex["text"]).to(model.device)      # same question, different image
        ids=inp["input_ids"][0]; imask=(ids==itid); tmask=(~imask)
        base=probs_from(run(inp))
        rec={"question_id_full":qid,"category":ex["category"],"label":lab,
             "base":[round(float(v),6) for v in base.tolist()],"patch":{},"lens":{}}
        d_ids=dinp["input_ids"][0]; d_imask=(d_ids==itid)
        for L in PATCH_L:
            dh=hidden_at(dinp,L)
            dimg_h=dh[d_imask.nonzero().flatten()]
            p=probs_from(run(inp,hook=(L,make_patch(dimg_h,imask))))
            kl=float((base*(base.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum())
            rec["patch"][f"img_L{L}"]={"kl":round(kl,6),"flip":int(int(p.argmax())!=int(base.argmax()))}
        for L in (4,20):                                   # C2: text-token patching control
            dh=hidden_at(dinp,L); dtxt=dh[(~d_imask).nonzero().flatten()]
            p=probs_from(run(inp,hook=(L,make_patch(dtxt,tmask))))
            kl=float((base*(base.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum())
            rec["patch"][f"txt_L{L}"]={"kl":round(kl,6),"flip":int(int(p.argmax())!=int(base.argmax()))}
        with torch.no_grad(): out=model(**inp,output_hidden_states=True)
        true_lg=out.logits[0,-1].float()
        for L in range(NL+1):
            h=out.hidden_states[L][0,-1]
            lg=head(h if L==NL else norm(h)).float()       # §12D: no double norm on the last
            pv=probs_from(lg)
            rec["lens"][f"L{L}"]={"p_correct":round(float(pv[lab]),6),"argmax":int(pv.argmax())}
        chk=float((head(out.hidden_states[NL][0,-1]).float()-true_lg).abs().max())
        rec["lens_lastlayer_maxdiff"]=round(chk,4)
        del out; torch.cuda.empty_cache()
        fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
        if n%10==0: print(f"  [{n}/{N}] {(time.time()-t0)/n:.1f}s/item  lens-check maxdiff {chk:.3f}",flush=True)
print(f"Done -> {OUT}",flush=True)
