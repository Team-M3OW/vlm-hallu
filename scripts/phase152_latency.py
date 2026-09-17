"""
Phase 152 (reviewer W4): wall-clock cost of DPR vs the single 600-token pass it is budget-matched to.

Tokens are matched (600 vs 300+300); the reviewer notes latency is not. Measure it, honestly:
    bar       uniform@600, one pass, sdpa attention
    DPR       localise@300 with eager attention + output_attentions (the read-out NEEDS the weights),
              head scoring on CPU, crop, answer@300 with sdpa
    DPR-hook  localise@300 with sdpa but a forward hook that materialises last-token attention only
              at the block layers (what a deployed implementation would do)
Batch size 1, 50 V*Bench items, CUDA-synchronised timings, warm-up excluded. Reported: median ms per
item per arm and the ratio; plus the fraction of DPR time spent in each stage.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL_ID = sys.argv[1] if len(sys.argv)>1 else "Qwen/Qwen3-VL-2B-Instruct"
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; B0=300; W=0.25; N=50; BLOCK=list(range(16,27))
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    def build(img,text):
        m=[{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}]
        return pr(images=img,text=pr.apply_chat_template(m,tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build(img,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=5,tol=0.06):
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
    items=[]
    for ex in ds:
        ip=os.path.join(root,ex["image"])
        if os.path.exists(ip): items.append((Image.open(ip).convert("RGB"),ex["text"]))
        if len(items)>=N+5: break
    res={}
    for impl in ["sdpa","eager"]:
        model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation=impl); model.eval()
        itid=model.config.image_token_id
        layers=model.model.language_model.layers
        def sync(): torch.cuda.synchronize()
        def t_answer(img,text):
            inp=build(fit(img,B0 if impl=="eager" else 2*B0)[0],text).to(model.device); sync(); t=time.perf_counter()
            with torch.no_grad(): model(**inp)
            sync(); return time.perf_counter()-t
        def t_localise_eager(img,text):
            small,_=fit(img,B0); inp=build(small,text); g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//2,g[2]//2; n=gh*gw
            base=int((inp["input_ids"][0]==itid).nonzero().flatten()[0]); inp=inp.to(model.device); sync(); t=time.perf_counter()
            with torch.no_grad(): out=model(**inp,output_attentions=True)
            acc=torch.zeros(n,device=model.device)
            for L in BLOCK:
                a=out.attentions[L][0,:,-1,base:base+n].float().mean(0); acc+=a/a.sum()
            i=int(acc.argmax()); sync(); dt=time.perf_counter()-t; del out
            return dt,((i%gw)+.5)/gw,((i//gw)+.5)/gh
        def t_localise_hook(img,text):
            small,_=fit(img,B0); inp=build(small,text); g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//2,g[2]//2; n=gh*gw
            base=int((inp["input_ids"][0]==itid).nonzero().flatten()[0]); inp=inp.to(model.device)
            store={}
            def mk(L):
                def h(mod,args,kw,out):
                    hs=args[0] if args else kw["hidden_states"]; q=mod.q_proj(hs); k=mod.k_proj(hs)
                    B,S,_=q.shape; H=mod.config.num_attention_heads; KV=mod.config.num_key_value_heads; d=q.shape[-1]//H
                    q=q.view(B,S,H,d)[:, -1]; k=k.view(B,S,KV,d); rep=H//KV; k=k.repeat_interleave(rep,dim=2)
                    a=torch.softmax((q.unsqueeze(1)*k).sum(-1).float()/d**0.5,dim=1)   # (B,S,H)
                    store[L]=a[0,base:base+n].mean(-1)
                return h
            hs=[layers[L].self_attn.register_forward_hook(mk(L),with_kwargs=True) for L in BLOCK]
            sync(); t=time.perf_counter()
            with torch.no_grad(): model(**inp)
            acc=sum(store[L]/store[L].sum() for L in BLOCK); i=int(acc.argmax()); sync(); dt=time.perf_counter()-t
            for h in hs: h.remove()
            return dt,((i%gw)+.5)/gw,((i//gw)+.5)/gh
        for img,text in items[:5]:   # warm-up
            t_answer(img,text)
        bar=[];loc=[];crop=[];lochook=[]
        for img,text in items[5:5+N]:
            if impl=="sdpa":
                bar.append(t_answer(img,text))
                dt,cx,cy=t_localise_hook(img,text); lochook.append(dt)
                inp=build(fit(window(img,cx,cy,W),B0)[0],text).to(model.device); sync(); t=time.perf_counter()
                with torch.no_grad(): model(**inp)
                sync(); crop.append(time.perf_counter()-t)
            else:
                dt,cx,cy=t_localise_eager(img,text); loc.append(dt)
        res[impl]={"bar":bar,"loc_hook":lochook,"crop":crop,"loc_eager":loc}
        del model; torch.cuda.empty_cache()
    med=lambda x: 1000*np.median(x) if len(x) else float("nan")
    b=med(res["sdpa"]["bar"]); lh=med(res["sdpa"]["loc_hook"]); c=med(res["sdpa"]["crop"]); le=med(res["eager"]["loc_eager"])
    print(f"\n{MODEL_ID}, batch 1, n={N}, median ms/item")
    print(f"  bar  uniform@600 (sdpa)                       {b:7.1f}")
    print(f"  DPR  localise@300 eager+output_attentions     {le:7.1f}")
    print(f"       localise@300 sdpa + last-token hook       {lh:7.1f}")
    print(f"       crop answer@300 (sdpa)                    {c:7.1f}")
    print(f"  DPR total (eager localiser)   {le+c:7.1f}   = {(le+c)/b:.2f}x the bar")
    print(f"  DPR total (hooked localiser)  {lh+c:7.1f}   = {(lh+c)/b:.2f}x the bar")
    json.dump({k:{kk:list(map(float,v)) for kk,v in d.items()} for k,d in res.items()},open(f"{D}/data/phase152_latency_{MODEL_ID.split('/')[-1]}.json","w"))
main()
