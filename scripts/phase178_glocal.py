"""
Phase 178: GLOBAL + LOCAL IN ONE PASS -- question-type agnostic by construction.

THE CONSTRAINT THIS IS DESIGNED AGAINST (not another placement tweak)
    gt_box_frac is the UNION box of all annotated objects. On RELATIONAL items that union spans both
    objects, so a W=0.25 window cannot contain it even when perfectly centred. Hence oracle@0.25 on
    relational is only 75.0 (Qwen3, bar 65.8) and 77.6 (Qwen2, bar 59.2): with PERFECT placement a
    single tight crop has +9.2 / +18.4 of headroom, and on Qwen3 that is inside the CI at n~76.
    Eight designs (§14U, §18B, §18D, §18F, §18G, §18H, §18L, plus the banned §18E gate) all kept the
    single-crop primitive and moved the window. The primitive is what fails, not the placement.

THE CHANGE
    The second pass sees BOTH a downsampled global view AND the crop, as two images, at a MATCHED
    total budget. Identical computation on every item: no question text, no classifier, no gate,
    no per-item window. The model uses the stream it needs -- magnification from the crop for
    single-object, spatial layout from the global view for relational.

PRIOR ART, stated plainly: the composition is NOT ours. VisLens (2608.30705) feeds the crop "back in
alongside the original image"; FAVE (2609.04392) is a foveated global+local encoder. ViRGo
(2606.21968) reports our exact relational failure -- "patch-based zooming ... can destroy global
spatial context" -- and fixes it with a THREE-WAY ROUTER on object scale. The open question, and the
only thing claimed here, is whether a BUDGET-MATCHED global+local pass with the crop placed by depth
re-ranking removes the question-type dependence WITHOUT a router.

ARMS (bar = uniform@600; every crop arm is pass1@300 + a second pass totalling 300)
    uniform@300 | uniform@600 | head@0.25 (incumbent) | glocal_150_150 (PRIMARY)
    | glocal_100_200, glocal_200_100 (CONTEXT ONLY -- shape of the trade-off; NOT eligible to be
      reported as the method, cf. phase 97's rule for W=0.35) | oracle_glocal_150_150 | oracle@0.25
PRE-REGISTERED, written before the run
    P1     glocal_150_150 - bar on RELATIONAL, CI clear of zero on BOTH models
    GUARD  glocal_150_150 - head@0.25 on SINGLE-object not significantly negative on BOTH models
    P1 and GUARD both hold -> the question-type gap closes without a router. Either fails -> ninth
    failed adaptation design, logged as such; the composition is not the fix either.
Budget gate: tokens MEASURED from image_grid_thw for every arm; >10% drift voids a contrast.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
WHICH=sys.argv[1] if len(sys.argv)>1 else "qwen3"
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase178_glocal_{WHICH}.jsonl"
B0,W=300,0.25
Image.MAX_IMAGE_PIXELS=None

def proposals():
    """OOF head cell per item -- identical to phase 168's proposer, nothing refit here."""
    P70.W=W
    if WHICH=="qwen3": X,Y,G,DEP,rows,_=P70.build()
    else: X,Y,G,DEP,rows=P80.build()
    P=P70.oof(X,Y,G); out={}
    for gi,r in enumerate(rows):
        m=G==gi; gh,gw=r["grid"]
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        s=np.where(rm.ravel(),P[m],-1e9); j=int(np.argmax(s))
        out[r["question_id_full"]]={"cx":float((j%gw+.5)/gw),"cy":float((j//gw+.5)/gh),"gt":r["gt_box_frac"]}
    return out

def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    props=proposals(); print(f"proposals for {len(props)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(nimg,text):
        c=[{"type":"image"} for _ in range(nimg)]+[{"type":"text","text":text}]
        return pr.apply_chat_template([{"role":"user","content":c}],tokenize=False,add_generation_prompt=True)
    def build(imgs,text): return pr(images=imgs,text=chat(len(imgs),text),return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build([img],"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best
    def window(img,cx,cy,w):
        iw,ih=img.size; x0=min(max(0,(cx-w/2)*iw),iw-w*iw); y0=min(max(0,(cy-w/2)*ih),ih-w*ih)
        return img.crop((int(x0),int(y0),int(x0+w*iw),int(y0+w*ih)))
    def answer(imgs,text):
        inp=build(imgs,text); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in props or not os.path.exists(ip): continue
            pp=props[qid]; img=Image.open(ip).convert("RGB"); gt=pp["gt"]
            cx,cy=pp["cx"],pp["cy"]; ocx,ocy=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2
            crop=window(img,cx,cy,W); ocrop=window(img,ocx,ocy,W)
            label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            arms={"uniform@300":[fit(img,B0)[0]],
                  "uniform@600":[fit(img,2*B0)[0]],
                  "head@0.25":[fit(crop,B0)[0]],
                  "glocal_150_150":[fit(img,150)[0],fit(crop,150)[0]],
                  "glocal_100_200":[fit(img,100)[0],fit(crop,200)[0]],
                  "glocal_200_100":[fit(img,200)[0],fit(crop,100)[0]],
                  "oracle_glocal_150_150":[fit(img,150)[0],fit(ocrop,150)[0]],
                  "oracle@0.25":[fit(ocrop,B0)[0]]}
            rec={"question_id_full":qid,"category":ex["category"],"label":label,"probs":{},"realized_tokens":{}}
            for nm,ims in arms.items():
                p,rz=answer(ims,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
