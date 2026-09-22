"""
Phase 214: DWA (Depth-Weighted Attention) across model families.

DWA needs a 2D grid of visual tokens to score. Families expose one differently:
  Qwen-VL      explicit image_grid_thw                      -> grid is given
  LLaVA-*      anyres: [base tile g*g][row newlines][tiles]  -> the BASE TILE is a contiguous PREFIX of
               length g*g and is a uniform low-res view of the whole image; we use it as the grid.
               Verified arithmetically (1176 = 576+24+576 ; 1485 = 729+27+729) and, below, EMPIRICALLY:
               the block-mean arg-max must cover ground-truth boxes far above chance or we abort.
  InstructBLIP 32 unordered Q-Former queries                -> NO spatial grid; DWA inapplicable.

Stage 1 dumps per-layer maps; stage 2 fits the ridge out-of-fold and answers on the crop.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
MODELS={"llava_ov":("llava-hf/llava-onevision-qwen2-7b-ov-hf",384,14),
        "llava_next":("llava-hf/llava-v1.6-vicuna-7b-hf",336,14),
        "gemma3_4b":("google/gemma-3-4b-it",None,None)}
MK=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 200
OUT=f"{D}/data/phase214_dwa_{MK}.jsonl"; W=0.25
mid,base_px,patch=MODELS[MK]
model=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0},
                                                  attn_implementation="eager").eval()
pr=AutoProcessor.from_pretrained(mid); tok=pr.tokenizer
layers=None
for f in (lambda m:m.model.language_model.layers, lambda m:m.language_model.model.layers, lambda m:m.model.layers):
    try: layers=f(model); break
    except Exception: pass
NL=len(layers)
itid=getattr(model.config,"image_token_id",getattr(model.config,"image_token_index",None))
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                           tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def grid_of(inp,img):
    if "image_grid_thw" in inp:
        g=inp["image_grid_thw"].tolist()[0]; return g[1]//2,g[2]//2,0
    if base_px:                      # LLaVA anyres: base tile is the prefix
        g=base_px//patch; return g,g,0
    n=int((inp["input_ids"][0]==itid).sum()); s=int(round(n**0.5))
    return (s,s,0) if s*s==n else (0,0,0)
def run(inp,want=False):
    with torch.no_grad(): out=model(**inp,output_attentions=want)
    lg=out.logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    A=None
    if want: A=np.stack([out.attentions[L][0,:,-1,:].float().mean(0).cpu().numpy() for L in range(NL)])
    del out; torch.cuda.empty_cache()
    return [round(float(v),6) for v in p.tolist()], A
def crop(img,cx,cy):
    iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
    return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
try:
    import re
    ann={}
    for e in ds:
        b=e.get("bbox") or e.get("gt_box") or None
        ann[f"{e['category']}/{e['question_id']}"]=b
except Exception: ann={}
BOX={json.loads(l)["question_id_full"]:json.loads(l)["gt_box_frac"]
     for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl")}
done=set()
if os.path.exists(OUT): done={json.loads(l)["qid"] for l in open(OUT)}
print(f"{MK}: NL={NL} itid={itid}",flush=True)
t0,n=time.time(),0
with open(OUT,"a") as fout:
    for e in ds:
        if n>=N: break
        qid=f"{e['category']}/{e['question_id']}"
        if qid in done or qid not in BOX: continue
        ip=os.path.join(root,e["image"])
        if not os.path.exists(ip): continue
        img=Image.open(ip).convert("RGB")
        if base_px: img_in=img.resize((base_px,base_px),Image.BICUBIC)
        else: img_in=img
        inp=build(img_in,e["text"]).to(model.device)
        gh,gw,off=grid_of(inp,img_in)
        if gh==0: print("no grid; abort"); break
        pos=(inp["input_ids"][0]==itid).nonzero().flatten()
        if len(pos)<gh*gw: continue
        base=int(pos[0])
        _,A=run(inp,want=True)
        Ai=A[:,base:base+gh*gw]                      # the base-tile prefix
        Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12)
        lab="ABCD".index(e["label"]) if isinstance(e["label"],str) else int(e["label"])
        rec={"qid":qid,"category":e["category"],"label":lab,"grid":[gh,gw],"nl":NL,
             "gt":BOX[qid],"attn":{f"L{l}":[round(float(v),8) for v in Ai[l]] for l in range(NL)}}
        fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
        if n%25==0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item",flush=True)
print(f"Done -> {OUT}",flush=True)
