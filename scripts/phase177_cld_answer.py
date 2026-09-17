"""
Phase 177 -- CONFIDENT LAYER DECODING applied to the ANSWER (arXiv 2606.21906 ported faithfully).
CLD: decode from the most confident near-final layer -- scan backwards from the final layer over the last K layers
and stop at the first entropy valley. Here: one forward per item with hidden states; lens_l = lm_head(final_norm(h_l))
at the answer position (final layer from model.logits, per §12D); store option-letter logits and full-vocab entropy per
layer. Rules are applied on disk (phase177_analyze): final layer (baseline), CLD valley (K=10), min-entropy layer,
DoLa-style contrast. Budgets: uniform@300 (deployed) and uniform@600 (bar), so we see whether CLD closes any
resolution gap. Per-layer lens accuracy is reported too (the "alignment tax" curve).
PRE-REGISTERED  P1  CLD − final at 300 tokens, CI clear of zero on both models.  CONTEXT  same at 600.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
MODELS={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}; BUDGETS=[300,600]; Image.MAX_IMAGE_PIXELS=None
def main(tag):
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    mid=MODELS[tag]; root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa").eval()
    pr=AutoProcessor.from_pretrained(mid); MS=getattr(pr.image_processor,"merge_size",1)
    lm=model.model.language_model if hasattr(model.model,"language_model") else model.language_model; norm,head=lm.norm,model.lm_head
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
        return best
    out=open(f"data/phase177_cld_{tag}.jsonl","w"); n=0; t0=time.time()
    for ex in ds:
        ap=os.path.splitext(os.path.join(root,ex["image"]))[0]+".json"
        if not os.path.exists(ap) or not json.load(open(ap)).get("bbox"): continue
        img=Image.open(os.path.join(root,ex["image"])).convert("RGB"); label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) and ex["label"] in "ABCD" else int(ex["label"])
        rec={"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],"label":label,"budgets":{}}
        for B in BUDGETS:
            small,ntok=fit(img,B); inp=build(small,ex["text"]).to(model.device)
            with torch.no_grad(): o=model(**inp,output_hidden_states=True)
            H=o.hidden_states; L=[]; E=[]
            with torch.no_grad():
                for l in range(1,len(H)):
                    lg=o.logits[0,-1].float() if l==len(H)-1 else head(norm(H[l][0,-1])).float()
                    p=torch.softmax(lg,-1); E.append(float(-(p*torch.log(p+1e-12)).sum())); L.append([float(v) for v in lg[letters]])
            rec["budgets"][str(B)]={"ntok":ntok,"letters":L,"entropy":E}
            del o,H; torch.cuda.empty_cache()
        out.write(json.dumps(rec)+"\n"); out.flush(); n+=1
        if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    out.close(); print(f"wrote {n}",flush=True)
if __name__=="__main__": main(sys.argv[1])
