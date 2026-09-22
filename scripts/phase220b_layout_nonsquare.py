"""phase220b: NON-SQUARE extension of the layout probe.
phase220 only tested g x g. LLaVA-NeXT anyres appends a row-separator token per row, so the true
row STRIDE can exceed the number of visual columns (24 cols + 1 newline -> stride 25). A square
sweep can only approximate that, so this sweeps gh x gw independently and also saves the attention
dump so the sweep can be rerun on CPU without reloading the model."""
import json, os, sys, glob, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
MK=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 24
MID={"llava_next":("llava-hf/llava-v1.6-vicuna-7b-hf",336),"gemma3_4b":("google/gemma-3-4b-it",None)}[MK]
mid,base_px=MID
model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(mid)
layers=model.model.language_model.layers; NL=len(layers)
itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
print(f"{MK}: NL={NL} itid={itid}  sliding_window={getattr(model.config.text_config,'sliding_window',None) if hasattr(model.config,'text_config') else None}",flush=True)
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
SNAP=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*"))[0]
BOX={json.loads(l)["question_id_full"]:json.loads(l)["gt_box_frac"] for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl")}
from datasets import load_dataset
ds=load_dataset("craigwu/vstar_bench")["test"]
recs=[]
for e in ds:
    qid=f"{e['category']}/{e['question_id']}"
    if qid not in BOX or len(recs)>=N: continue
    img=Image.open(os.path.join(SNAP,e["image"])).convert("RGB")
    if base_px: img=img.resize((base_px,base_px),Image.BICUBIC)
    inp=pr(images=img,text=chat(e["text"]),return_tensors="pt").to(model.device)
    pos=(inp["input_ids"][0]==itid).nonzero().flatten()
    with torch.no_grad(): out=model(**inp,output_attentions=True)
    A=np.stack([out.attentions[L][0,:,-1,:].float().mean(0).cpu().numpy() for L in range(NL)])
    recs.append({"qid":qid,"gt":BOX[qid],"n_img":int(len(pos)),"base":int(pos[0]),
                 "attn":A[:,int(pos[0]):int(pos[0])+int(len(pos))]})
    del out; torch.cuda.empty_cache()
print(f"dumped {len(recs)} items; n_img={recs[0]['n_img']}",flush=True)
W=0.25
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
n_img=recs[0]["n_img"]
np.save(f"{D}/data/phase220b_dump_{MK}.npy",np.stack([r["attn"] for r in recs]))
json.dump([{"qid":r["qid"],"gt":r["gt"],"n_img":r["n_img"]} for r in recs],open(f"{D}/data/phase220b_meta_{MK}.json","w"))
cands=[]
for gh in range(8,40):
    for gw in range(8,40):
        n=gh*gw
        if n<=n_img:
            cands.append(("prefix",gh,gw,0))
            cands.append(("suffix",gh,gw,n_img-n))
best=[]
for tag,gh,gw,off in cands:
    for lo in range(0,NL-2,2):
        for width in (1,3,6,11):
            hi=min(lo+width,NL); cs=[]
            for r in recs:
                a=r["attn"][lo:hi,off:off+gh*gw]
                if a.shape[1]<gh*gw: continue
                a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
                d=a.mean(0); j=int(np.argmax(d))
                cs.append(cov((j%gw+.5)/gw,(j//gh+.5)/gh if False else (j//gw+.5)/gh,r["gt"]))
            if cs: best.append((float(np.mean(cs)),tag,gh,gw,lo,hi))
best.sort(reverse=True)
ch=np.mean([cov(np.random.rand(),np.random.rand(),r["gt"]) for r in recs for _ in range(50)])
print(f"chance ~{ch:.3f}.  Top layouts/bands (non-square sweep):")
seen=set(); shown=0
for c,tag,gh,gw,lo,hi in best:
    k=(tag,gh,gw)
    if k in seen: continue
    seen.add(k); shown+=1
    print(f"   cov {c:.3f} ({c/max(ch,1e-9):.1f}x)   {tag} grid {gh}x{gw} (hxw)  layers L{lo}-{hi-1}")
    if shown>=12: break
print("square-only best for reference:")
sq=[b for b in best if b[2]==b[3]][:3]
for c,tag,gh,gw,lo,hi in sq:
    print(f"   cov {c:.3f} ({c/max(ch,1e-9):.1f}x)   {tag} grid {gh}x{gw}  layers L{lo}-{hi-1}")
