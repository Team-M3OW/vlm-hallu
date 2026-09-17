"""
Phase 179b: END-TASK BASELINE TABLE at equal compute -- what a method-led paper needs and we never ran.
Every arm is the SAME pipeline (localise at 300 -> crop at W=0.25 -> answer at 300, total 600, bar uniform@600);
arms differ ONLY in which cell the crop is centred on (placements frozen offline by phase179_placements.py).
    uniform@300 | uniform@600 (BAR) | vicrop_block | vicrop_L14 | gatemax | laser | head (OURS) | oracle
PRE-REGISTERED
    P1     head - best published baseline (max of vicrop_block, vicrop_L14, laser), pooled, CI clear on BOTH models
    P2     head - gatemax (what the supervision buys over the best label-free rule), pooled, both models
    report per stratum; budget gate: tokens measured, >10% drift voids a contrast.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase179_baselines_{WHICH}.jsonl"; B0,W=300,0.25; Image.MAX_IMAGE_PIXELS=None
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    PL=json.load(open(f"{D}/data/phase179_placements_{WHICH}.json")); print(f"placements for {len(PL)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(n,t):
        return pr.apply_chat_template([{"role":"user","content":[{"type":"image"}]*n+[{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(len(i),t),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build([i],"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def win(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih)
        return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(imgs,t):
        inp=build(imgs,t); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            p=PL[qid]; img=Image.open(ip).convert("RGB"); gt=p["gt"]
            arms={"uniform@300":[fit(img,B0)],"uniform@600":[fit(img,2*B0)],
                  "oracle":[fit(win(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)]}
            for k,(cx,cy) in p["cells"].items(): arms[k]=[fit(win(img,cx,cy,W),B0)]
            rec={"question_id_full":qid,"category":ex["category"],
                 "label":"ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"]),"probs":{},"realized_tokens":{}}
            for nm,ims in arms.items():
                pr_,rz=answer(ims,ex["text"]); rec["probs"][nm]=pr_; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
