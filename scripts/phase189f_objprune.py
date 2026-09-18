"""
Phase 189f: OBJECTNESS-PRUNE EARLY, PLACE LATE -- end task at equal compute.
§31: a linear probe on L4 hidden states (token-only, trained on V*Bench boxes) retains the evidence box at 25% keep far
better than attention at any depth. Use it as a LEARNED PRUNING HEAD at L4: encode at E, run L0-L4 full width, score
image tokens with the probe (fit on the 300-token dump, applied to this pass's L4 hidden states), keep 25%, run L5-L27
narrow. TL = E*5 + 0.25*E*23 = 10.75 E.  E=600 -> 6,450 (38% of bar: efficiency arm); E=1500 -> 16,125 (96%: the
resolution arm). Controls at E=1500: random keep 25% at L4; attention-at-L4 keep 25% (known bad). Bar uniform@600.
No crop: single pass, question-type neutral.
PRE-REGISTERED
    S0  objprune600 - bar  not significantly negative on BOTH (75% of tokens droppable at L4 for free)
    P1  objprune1500 - bar, pooled, CI clear of zero on BOTH models
    S1  objprune1500 - random1500 (the head matters at L4)     GUARD relational not sig. negative
Honest prior (§28B): Qwen2's V*Bench resolution curve is flat beyond 600 -> P1 likely 1 of 2; S0/S1 are the 2-of-2 tests.
Probe scoring uses the SAME pass's hidden states at L4 (pass 1: full forward with hidden states -> scores; pass 2: pruned).
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.linear_model import LogisticRegression
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]; NL=28; PL=4; KEEP=0.25
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]; OUT=f"{D}/data/phase189f_objprune_{WHICH}.jsonl"; Image.MAX_IMAGE_PIXELS=None
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
def fit_probe():
    Z=np.load(f"{D}/data/phase189a_hidden_{WHICH}.npz"); meta=json.load(open(f"{D}/data/phase189a_hidden_{WHICH}_meta.json")); N=len(meta)
    X=[];Y=[]
    for i,m in enumerate(meta):
        gh,gw=m["grid"]; yy,xx=np.mgrid[0:gh,0:gw]; fx,fy=((xx+.5)/gw).ravel(),((yy+.5)/gh).ravel(); x0,y0,x1,y1=m["gt_box_frac"]
        Y.append(((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(int)); X.append(Z[f"h_L{PL}_{i}"].astype(np.float32))
    X=np.vstack(X); Y=np.concatenate(Y); mu,sd=X.mean(0),X.std(0)+1e-6
    clf=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit((X-mu)/sd,Y); return clf,mu,sd
def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    clf,mu,sd=fit_probe(); print("probe fit on 300-token dump (token-only, L4)",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id; MS=getattr(pr.image_processor,"merge_size",1); layers=model.model.language_model.layers
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
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
    def probs_of(lg): return [round(float(v),6) for v in torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0).tolist()]
    rng=random.Random(189); n=0; t0=time.time()
    with open(OUT,"w") as fout:
        for ex in ds:
            ap=os.path.splitext(os.path.join(root,ex["image"]))[0]+".json"
            if not os.path.exists(ap) or not json.load(open(ap)).get("bbox"): continue
            img=Image.open(os.path.join(root,ex["image"])).convert("RGB"); label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            rec={"question_id_full":f"{ex['category']}/{ex['question_id']}","category":ex["category"],"label":label,"probs":{},"realized_tokens":{},"token_layers":{}}
            inp=build(fit(img,600),ex["text"]).to(model.device); nt=int((inp["input_ids"][0]==itid).sum()); clear()
            with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
            rec["probs"]["uniform@600"]=probs_of(lg); rec["realized_tokens"]["uniform@600"]=nt; rec["token_layers"]["uniform@600"]=nt*NL
            for E in (600,1500):
                inp=build(fit(img,E),ex["text"]).to(model.device); pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos)); clear()
                with torch.no_grad(): o=model(**inp,output_hidden_states=True,output_attentions=(E==1500))
                h=o.hidden_states[PL][0,base:base+nt].float().cpu().numpy(); s_probe=clf.decision_function((h-mu)/sd)
                arms={f"objprune{E}":s_probe}
                if E==1500:
                    a=o.attentions[PL][0,:,-1,base:base+nt].float().mean(0).cpu().numpy(); arms["attnprune1500"]=a
                    arms["random1500"]=np.array([rng.random() for _ in range(nt)])
                del o; torch.cuda.empty_cache()
                keep=max(1,int(round(KEEP*nt)))
                for nm,s in arms.items():
                    clear(); drop=(base+np.argsort(-s)[keep:]).tolist(); b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
                    for li in range(PL+1,NL): layers[li].self_attn._prune_bias=b
                    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
                    clear(); rec["probs"][nm]=probs_of(lg); rec["realized_tokens"][nm]=nt; rec["token_layers"][nm]=nt*(PL+1)+keep*(NL-1-PL)
                del inp; torch.cuda.empty_cache()
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
