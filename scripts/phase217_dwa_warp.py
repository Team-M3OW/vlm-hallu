"""
Phase 217: DWA-STEERED WARP -- the one route to a both-strata method the failure catalogue leaves open.

WHY THIS AND NOT ANOTHER CROP VARIANT
  §34's scope law was corrected: a W=0.25 window already contains BOTH boxes on 93.4% of cross-instance
  items. Cropping therefore does not fail by discarding the second object; it fails by destroying the
  global spatial frame a left/right question needs. Warping is the only operation tested that magnifies
  the target while KEEPING that frame -- the periphery is compressed, not removed.
  Phase 65 ran exactly this and got the mirror image of cropping: single -2.6, cross +5.3 at lambda=0.7
  (neither clear at n=191). But phase 65 predates the ridge: it steered the warp with the old block-mean
  read-out, which covers the evidence on 46.6% of items against DWA's 63.9%.

WHAT IS NEW HERE: the same warp, steered by DWA's out-of-fold score map, plus a finer lambda sweep in
the region phase 65 found best. Prediction, pre-registered: better steering lifts the SINGLE-instance
side (where warping currently loses, because the magnification lands off-target) without giving back
the cross-instance gain. A both-strata win needs roughly +8 on each stratum to clear at n=191.
PRIOR ART: AttWarp (arXiv:2510.09741) does attention-guided warping. The idea is theirs; what is ours
is the budget-matched, stratified evaluation and the corrected read-out depth.

ARMS (answer pass at 300 tokens; warp arms also pay the 300-token localiser, so 600 total = the bar)
  uniform@600            the bar
  crop_dwa               DWA cell, W=0.25 crop          (the incumbent single-instance winner)
  warp_dwa@{.5,.7,.8,.9} DWA-steered warp               (this file)
  warp_blk@0.7           block-mean-steered warp        (phase 65's steering, as the control)
"""
import json, os, time, numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)  # the env token is invalid; use the stored login
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"; OUT=f"{D}/data/phase217_dwawarp.jsonl"
B0,W,FLOOR=300,0.25,0.15; LAMS=[0.5,0.7,0.8,0.9]
def axis_src(s1d,n_out,lam,floor=FLOOR):
    s=np.clip(np.asarray(s1d,float),0,None); s=s/max(s.sum(),1e-12)
    u=np.ones_like(s)/len(s); s=(1-lam)*u+lam*s; s=(1-floor)*s+floor*u
    cdf=np.concatenate([[0.0],np.cumsum(s)]); cdf/=cdf[-1]
    return np.interp(np.linspace(0,1,n_out),cdf,np.linspace(0.0,1.0,len(cdf)))
def warp_image(img,sal,lam,out_size):
    Wd,Ht=img.size; ow,oh=out_size
    xs=(axis_src(sal.sum(0),ow,lam)*(Wd-1)).astype(np.float32)
    ys=(axis_src(sal.sum(1),oh,lam)*(Ht-1)).astype(np.float32)
    XX,YY=np.meshgrid(xs,ys); a=np.asarray(img,dtype=np.float32)
    out=np.stack([map_coordinates(a[...,c],[YY,XX],order=1,mode="nearest") for c in range(3)],-1)
    del XX,YY
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
# --- maps + the deployed DWA score surface (out-of-fold, ring-masked): reuse, do not refit ---
SC={json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_qwen3.jsonl")}
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}).eval()
pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
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
def ans(im,t):
    inp=build(fit(im,B0),t).to(model.device)
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    del inp; torch.cuda.empty_cache(); return [round(float(v),6) for v in p.tolist()]
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset")
ds=load_dataset("craigwu/vstar_bench")["test"]
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
t0,n=time.time(),0
with open(OUT,"a") as f:
    for e in ds:
        qid=f"{e['category']}/{e['question_id']}"
        if qid in done or qid not in SC: continue
        ip=os.path.join(root,e["image"])
        if not os.path.exists(ip): continue
        m=SC[qid]; gh,gw=m["grid"]
        img=Image.open(ip).convert("RGB")
        lab="ABCD".index(e["label"]) if isinstance(e["label"],str) else int(e["label"])
        dwa=np.asarray(m["score"],float).reshape(gh,gw)
        dwa=dwa-dwa.min(); dwa=gaussian_filter(dwa,1.0)
        blk=np.asarray(m["dep"],float).reshape(gh,gw); blk=gaussian_filter(blk,1.0)
        ow,oh=fit(img,B0).size
        rec={"qid":qid,"category":e["category"],"label":lab,"probs":{}}
        rec["probs"]["uniform@600"]=ans(fit(img,600),e["text"])
        rec["probs"]["crop_dwa"]=ans(crop(img,*m["ridge_cell"]),e["text"])
        for lam in LAMS:
            rec["probs"][f"warp_dwa@{lam}"]=ans(warp_image(img,dwa,lam,(ow,oh)),e["text"])
        rec["probs"]["warp_blk@0.7"]=ans(warp_image(img,blk,0.7,(ow,oh)),e["text"])
        f.write(json.dumps(rec)+"\n"); f.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
