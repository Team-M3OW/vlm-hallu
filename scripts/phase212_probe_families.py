"""Capability probe: for each family, can we (a) load it, (b) find the LM layers, (c) find image-token
positions, (d) patch its attention for pruning, (e) score an answer? Cheap; de-risks the full runs."""
import os, sys, json, torch, traceback
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
CANDS=[("llava-ov","llava-hf/llava-onevision-qwen2-7b-ov-hf"),
       ("llava-next","llava-hf/llava-v1.6-vicuna-7b-hf")]
if len(sys.argv)>1: CANDS=[(sys.argv[1],sys.argv[2])]
img=Image.new("RGB",(448,448),(120,140,160))
for tag,mid in CANDS:
    print(f"\n=== {tag}  {mid}")
    try:
        cfg=AutoConfig.from_pretrained(mid)
        print("   config:",type(cfg).__name__,"| image_token_id:",getattr(cfg,"image_token_id",getattr(cfg,"image_token_index","?")))
        m=AutoModelForImageTextToText.from_pretrained(mid,dtype=torch.bfloat16,device_map={"":0}).eval()
        pr=AutoProcessor.from_pretrained(mid)
        # locate decoder layers
        cand=[("model.language_model.layers",lambda m:m.model.language_model.layers),
              ("language_model.model.layers",lambda m:m.language_model.model.layers),
              ("model.layers",lambda m:m.model.layers)]
        layers=None
        for nm,f in cand:
            try:
                layers=f(m); print(f"   layers at {nm}: {len(layers)}"); break
            except Exception: pass
        if layers is None: print("   LAYERS NOT FOUND"); continue
        attn_mod=type(layers[0].self_attn).__module__
        print("   attention module:",attn_mod)
        itid=getattr(m.config,"image_token_id",getattr(m.config,"image_token_index",None))
        msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":"What is this?\n(A) a\n(B) b\n(C) c\n(D) d\nAnswer with the option's letter."}]}]
        txt=pr.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=pr(images=img,text=txt,return_tensors="pt").to(m.device)
        nimg=int((inp["input_ids"][0]==itid).sum()) if itid is not None else -1
        with torch.no_grad(): out=m(**inp,output_attentions=False)
        print(f"   image tokens in prompt: {nimg}   logits ok: {tuple(out.logits.shape)}")
        print("   VERDICT: usable" if nimg>0 else "   VERDICT: image-token id not resolving")
        del m,out; torch.cuda.empty_cache()
    except Exception as e:
        print("   FAILED:",type(e).__name__,str(e)[:160]); torch.cuda.empty_cache()
