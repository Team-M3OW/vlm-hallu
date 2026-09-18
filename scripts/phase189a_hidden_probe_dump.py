"""
Phase 189a: can a LEARNED PRUNING HEAD rank visual tokens EARLY, where attention cannot?
§26C: at/after L16 ranking is irrelevant (random = attention). §29: attention-based ranking before ~L14 is question-blind
(ridge on L<=8: 33%, L<=12: 16-33% coverage). A learned pruning head that pays must therefore read the RESIDUAL STREAM at
an early depth, not attention. Dump, for the 191 V*Bench items at 300 tokens: image-token hidden states at L4/L8/L12/L16
(fp16), the last-prompt-token hidden state at the same layers (question conditioning), the grid, and the GT box.
Probe (189b, on disk): logistic regression on [h_l(token) ; h_l(last) ; h_l(token)*h_l(last)] -> token inside GT box,
GroupKFold by item; report OOF AUC and "keep top-25% covers >=50% of the box" vs attention ranking at the same depth and
vs the L16 attention ranking (the free-prune reference).
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]; LAYERS=[4,8,12,16]; B0=int(sys.argv[2]) if len(sys.argv)>2 else 300
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]; Image.MAX_IMAGE_PIXELS=None
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); itid=model.config.image_token_id; MS=getattr(pr.image_processor,"merge_size",1)
    def build(i,t): return pr(images=i,text=pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//(MS*MS) for g in build(i,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    H={f"L{l}":[] for l in LAYERS}; Q={f"L{l}":[] for l in LAYERS}; ATT={f"L{l}":[] for l in LAYERS}; meta=[]; n=0; t0=time.time()
    for ex in ds:
        ap=os.path.splitext(os.path.join(root,ex["image"]))[0]+".json"
        if not os.path.exists(ap): continue
        ann=json.load(open(ap))
        if not ann.get("bbox"): continue
        img=Image.open(os.path.join(root,ex["image"])).convert("RGB"); IW,IH=img.size
        gt=[min(b[0] for b in ann["bbox"])/IW,min(b[1] for b in ann["bbox"])/IH,max(b[0]+b[2] for b in ann["bbox"])/IW,max(b[1]+b[3] for b in ann["bbox"])/IH]
        inp=build(fit(img,B0),ex["text"]).to(model.device); pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
        g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//MS,g[2]//MS
        if gh*gw!=nt: continue
        with torch.no_grad(): o=model(**inp,output_hidden_states=True,output_attentions=True)
        for l in LAYERS:
            hs=o.hidden_states[l][0]                                  # residual stream entering layer l (after l blocks)
            H[f"L{l}"].append(hs[base:base+nt].to(torch.float16).cpu().numpy()); Q[f"L{l}"].append(hs[-1].to(torch.float16).cpu().numpy())
            a=o.attentions[l][0,:,-1,base:base+nt].float().mean(0); ATT[f"L{l}"].append((a/a.sum()).cpu().numpy().astype(np.float32))
        meta.append({"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],"grid":[gh,gw],"n":nt,"gt_box_frac":gt})
        del o; torch.cuda.empty_cache(); n+=1
        if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    np.savez_compressed(f"{D}/data/phase189a_hidden_{WHICH}"+("" if B0==300 else f"_{B0}")+".npz",**{f"h_{k}_{i}":v for k,vs in H.items() for i,v in enumerate(vs)},
                        **{f"q_{k}_{i}":v for k,vs in Q.items() for i,v in enumerate(vs)},**{f"a_{k}_{i}":v for k,vs in ATT.items() for i,v in enumerate(vs)})
    json.dump(meta,open(f"{D}/data/phase189a_hidden_{WHICH}"+("" if B0==300 else f"_{B0}")+"_meta.json","w")); print(f"wrote {n} items",flush=True)
main()
