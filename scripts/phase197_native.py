"""
Phase 197: THE BENCHMARK AS THE FIELD RUNS IT -- native dynamic resolution, no token cap.
Every result in this project is measured at an imposed 300/600-visual-token budget. Published V*Bench / HR-Bench
numbers use the model's NATIVE dynamic resolution (thousands of tokens). If our gains are artifacts of a starved
budget they must vanish here. This run measures, with no method at all:
    native      : the processor's own default resolution (what the benchmark harness does)
    uniform@600 : our bar, for calibration against every earlier phase
    uniform@N   : native token count, but through our fit() path -- checks our resizing harness is not the difference
and records the realized token count so the native budget is known rather than assumed.
usage: phase197_native.py <qwen3|qwen2|q25_7b|q3_8b> [MODEL_ID]
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID=sys.argv[2] if len(sys.argv)>2 else {"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase197_native_{WHICH}.jsonl"; Image.MAX_IMAGE_PIXELS=None
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
    MS=getattr(pr.image_processor,"merge_size",2)
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def build(img,t): return pr(images=img,text=pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//(MS*MS) for g in build(i,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def answer(img,t):
        inp=build(img,t); rz=int(sum(g[1]*g[2]//(MS*MS) for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            ip=os.path.join(root,ex["image"])
            if not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB")
            rec={"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],
                 "label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),
                 "img_wh":list(img.size),"probs":{},"realized_tokens":{}}
            try:
                p,rz=answer(img,ex["text"]); rec["probs"]["native"]=p; rec["realized_tokens"]["native"]=rz
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); rec["probs"]["native"]=None; rec["realized_tokens"]["native"]=-1; rz=0
            for nm,tg in (("uniform@600",600),):
                p,r2=answer(fit(img,tg),ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=r2
            if rz>0:
                p,r3=answer(fit(img,rz),ex["text"]); rec["probs"]["uniform@native_n"]=p; rec["realized_tokens"]["uniform@native_n"]=r3
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s  median native tokens so far",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
