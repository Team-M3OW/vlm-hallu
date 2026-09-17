"""
Phase 176b (CONTROL) -- same as 176 but the mask removes image attention for ALL text query rows.
If this does not destroy the answer (large KL, flips), the mask machinery is a no-op and 176 is void.
Phase 176 -- RippleKV-style LABEL-FREE layer weights from READ-OUT SENSITIVITY (arXiv 2608.08684 ported).
RippleKV weights each layer by how much perturbing its cache moves the output distribution (KL), label-free, on a
calibration set. Port: at layer l, block the FINAL prompt token's attention to the image tokens (mask that one query
row over the image columns, at that layer only) and measure KL(p_answer || p_answer_ablated) over the option letters
and the full vocab. w_l = mean KL over calibration items. Then, on disk: read the map as sum_l w_l A_l (and max over
the top-k sensitive layers) and compare to block mean / gate max / head.
PRE-REGISTERED PREDICTION (from SS19): sensitivity should peak at answer-formation depth (L21+), i.e. on the layers the
head SUBTRACTS -- so a sensitivity-weighted read-out should be no better than gate max. If instead it peaks at the
gate, it merely rediscovers it. Either way the run measures whether "layers the answer reads" = "layers that localise".
SANITY (the +0.0 rule): ablating ALL layers at once must give large KL and flip answers; per-layer KL must be > 0.
Calibration: first 40 items of the V*Bench set (both categories), 1 + NL forwards each.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
MODELS={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}; B0=300; NCAL=12; Image.MAX_IMAGE_PIXELS=None
def main(tag):
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    mid=MODELS[tag]; root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(mid); itid=model.config.image_token_id; MS=getattr(pr.image_processor,"merge_size",1)
    layers=model.model.language_model.layers if hasattr(model.model,"language_model") else model.language_model.layers; NL=len(layers)
    tok=pr.tokenizer; letters=[tok.encode(x,add_special_tokens=False)[0] for x in ["A","B","C","D"]]
    def build(img,text):
        chat=pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}],tokenize=False,add_generation_prompt=True)
        return pr(images=img,text=chat,return_tensors="pt")
    def measure(img):
        i=build(img,"x"); return int(sum(g[1]*g[2]//(MS*MS) for g in i["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    state={"layers":set(),"span":None}
    def make_hook(l):
        def pre(mod,args,kwargs):
            if l not in state["layers"]: return None
            am=kwargs.get("attention_mask"); 
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask type {None if am is None else am.dtype}; cannot ablate")
            am=am.clone(); a,b=state["span"]; am[:,:,b:,a:b]=torch.finfo(am.dtype).min   # ALL text rows after the image
            kwargs["attention_mask"]=am; return (args,kwargs)
        return pre
    hooks=[layers[l].self_attn.register_forward_pre_hook(make_hook(l),with_kwargs=True) for l in range(NL)]
    def logits(inp):
        with torch.no_grad(): return model(**inp).logits[0,-1].float()
    def kl(p,q): p=torch.softmax(p,-1); lq=torch.log_softmax(q,-1); return float((p*(torch.log(p+1e-12)-lq)).sum())
    out=[]; n=0; t0=time.time()
    for ex in ds:
        if n>=NCAL: break
        ap=os.path.splitext(os.path.join(root,ex["image"]))[0]+".json"
        if not os.path.exists(ap) or not json.load(open(ap)).get("bbox"): continue
        img=fit(Image.open(os.path.join(root,ex["image"])).convert("RGB"),B0); inp=build(img,ex["text"]).to(model.device)
        pos=(inp["input_ids"][0]==itid).nonzero().flatten(); state["span"]=(int(pos[0]),int(pos[-1])+1)
        state["layers"]=set(); base=logits(inp); bl=base[letters]
        state["layers"]=set(range(NL)); allab=logits(inp)
        rec={"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],"kl_all":kl(base,allab),"kl_all_letters":kl(bl,allab[letters]),
             "flip_all":int(int(torch.argmax(bl))!=int(torch.argmax(allab[letters]))),"kl":[],"kl_letters":[],"flip":[]}
        for l in range(NL):
            state["layers"]={l}; q=logits(inp); rec["kl"].append(kl(base,q)); rec["kl_letters"].append(kl(bl,q[letters])); rec["flip"].append(int(int(torch.argmax(bl))!=int(torch.argmax(q[letters]))))
        out.append(rec); n+=1
        if n%5==0: print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item  kl_all={rec['kl_all']:.3f}  per-layer KL max={max(rec['kl']):.4f} at L{int(np.argmax(rec['kl']))}",flush=True)
    json.dump(out,open(f"data/phase176b_allrows_{tag}.json","w"))
    K=np.array([r["kl"] for r in out]); print("mean per-layer KL:"," ".join(f"L{l}:{K[:,l].mean():.4f}" for l in range(NL))); print(f"kl_all mean {np.mean([r['kl_all'] for r in out]):.3f}, flip_all {np.mean([r['flip_all'] for r in out]):.2f}")
if __name__=="__main__": main(sys.argv[1])
