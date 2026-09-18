"""
Phase 188a: the pruning saving's only remaining sink is the LOCALISATION pass (§27: crop resolution is above the
cliff and converts to nothing; the 16pp gap to the oracle is placement). Question: does a higher-resolution
localisation pass place better? Dump last-prompt-token attention maps (28 layers, head-mean, deployed convention)
for the 191 V*Bench items at E=450 tokens, (a) unpruned and (b) with the L16 k=.10 prune applied (bias from L17),
so a ridge re-fit on (b) is trained on exactly what deployment would see. Budget if adopted:
loc@450 pruned (450*17+45*11 = 8,145 TL) + crop@300 (8,400) = 16,545 <= 16,800. On-disk next: ridge OOF coverage
on 300-stored vs 450-unpruned vs 450-pruned maps; end-task (188b) only if 450-pruned >= 300 on BOTH models.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]; E=int(sys.argv[2]) if len(sys.argv)>2 else 450; PRUNED=len(sys.argv)<=3 or sys.argv[3]!="nopruned"; P=16; K=0.10; NL=28
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]; Image.MAX_IMAGE_PIXELS=None
def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
QM3.eager_attention_forward=make_patched(QM3); QM2.eager_attention_forward=make_patched(QM2)
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); itid=model.config.image_token_id; MS=getattr(pr.image_processor,"merge_size",1)
    layers=model.model.language_model.layers
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
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def maps(inp,base,nt,drop=None):
        clear()
        if drop is not None:
            b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(P+1,NL): layers[li].self_attn._prune_bias=b
        with torch.no_grad(): o=model(**inp,output_attentions=True)
        A=np.stack([o.attentions[L][0,:,-1,base:base+nt].float().mean(0).cpu().numpy() for L in range(NL)]); del o; torch.cuda.empty_cache(); clear()
        return A/np.maximum(A.sum(1,keepdims=True),1e-12)
    outs={k:open(f"{D}/data/phase188a_loc{E}_{k}_{WHICH}.jsonl","w") for k in (("unpruned","pruned") if PRUNED else ("unpruned",))}; n=0; t0=time.time()
    for ex in ds:
        ap=os.path.splitext(os.path.join(root,ex["image"]))[0]+".json"
        if not os.path.exists(ap): continue
        ann=json.load(open(ap))
        if not ann.get("bbox"): continue
        img=Image.open(os.path.join(root,ex["image"])).convert("RGB"); IW,IH=img.size
        gt=[min(b[0] for b in ann["bbox"])/IW,min(b[1] for b in ann["bbox"])/IH,max(b[0]+b[2] for b in ann["bbox"])/IW,max(b[1]+b[3] for b in ann["bbox"])/IH]
        inp=build(fit(img,E),ex["text"]).to(model.device); pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
        g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//MS,g[2]//MS
        if gh*gw!=nt: continue
        A0=maps(inp,base,nt)
        s=A0[max(0,P-4):P+1].mean(0); keep=max(1,int(round(K*nt))); drop=(base+np.argsort(-s)[keep:]).tolist()
        pairs=[("unpruned",A0)]
        if PRUNED: pairs.append(("pruned",maps(inp,base,nt,drop)))
        for k,A in pairs:
            outs[k].write(json.dumps({"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],"grid":[gh,gw],"n_img_tokens":nt,
                "gt_box_frac":gt,"attn":{f"L{i}":[round(float(v),8) for v in A[i]] for i in range(NL)}})+"\n"); outs[k].flush()
        n+=1
        if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    for f in outs.values(): f.close()
    print(f"wrote {n}",flush=True)
main()
