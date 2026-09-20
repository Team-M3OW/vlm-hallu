"""Phase 199: end-task accuracy of cropping at EVERY layer's arg-max.
The paper claims the read-out depth dominates every other design choice; this measures the whole curve
instead of one hand-picked layer. Cells are the ring-masked arg-max of each layer's map (same convention
as phase179/phase184). Answer pass @300 tokens on the crop, as in every two-pass arm."""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
MAPS={"qwen3":"phase30c_attn_maps_all.jsonl","qwen2":"phase74_Qwen2_VL_7B_Instruct.jsonl"}[WHICH]
OUT=f"{D}/data/phase199_layersweep_{WHICH}.jsonl"; W=0.25; Image.MAX_IMAGE_PIXELS=None
def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                               tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def crop(img,cx,cy):
        iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
        return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
    def answer(im,t):
        inp=build(fit(im,300),t).to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
    maps={}
    for l in open(f"{D}/data/{MAPS}"):
        r=json.loads(l)
        if "attn" in r and "grid" in r: maps[r["question_id_full"]]=r
    lut={f"{e['category']}/{e['question_id']}":e for e in ds}
    done=set()
    if os.path.exists(OUT): done={json.loads(l)["question_id_full"] for l in open(OUT)}
    t0,n=time.time(),0
    with open(OUT,"a") as fout:
        for qid,ex in lut.items():
            if qid in done or qid not in maps: continue
            ip=os.path.join(root,ex["image"])
            if not os.path.exists(ip): continue
            q=maps[qid]; gh,gw=q["grid"]; NL=len(q["attn"])
            A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NL)])
            A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
            rm=np.zeros((gh,gw),bool)
            if gh>2 and gw>2: rm[1:-1,1:-1]=True
            else: rm[:]=True
            rm=rm.ravel()
            img=Image.open(ip).convert("RGB")
            lab="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            rec={"question_id_full":qid,"category":ex["category"],"label":lab,"grid":[gh,gw],
                 "gt":q["gt_box_frac"],"cells":{},"probs":{}}
            for L in range(NL):
                j=int(np.argmax(np.where(rm,A[L],-1e9)))
                cx,cy=float((j%gw+.5)/gw),float((j//gw+.5)/gh)
                rec["cells"][f"L{L}"]=[cx,cy]
                rec["probs"][f"L{L}"]=answer(crop(img,cx,cy),ex["text"])
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%10==0:
                el=time.time()-t0
                print(f"  [{n}] {el/n:.1f}s/item  eta {(len(maps)-len(done)-n)*el/n/60:.0f}min",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
