"""
Phase 187: LAYER-WISE (PYRAMID) TOKEN PRUNING with a CAUSALLY-DERIVED SCHEDULE, at equal compute.
§20B suffix masks give the share of image->text transport still ahead at depth l (Qwen3: L4 .76, L8 .69, L12 .25,
L16 .01; Qwen2 similar with the window ~2 layers later). A causal pyramid keeps visual tokens in proportion:
    causal:   keep 1.00 for L0-8 | 0.50 for L9-12 | 0.25 for L13-16 | 0.10 for L17-27   -> TL = 13.1 E
    fixed:    PyramidDrop-style lambda=0.5 at L7/L14/L21: 1.0 | .5 | .25 | .125            -> TL = 13.1 E   (published control)
Both afford E ~ 1250 tokens under the uniform@600 bar (16,800 TL); single-cut TSR (§26) afforded only 900.
Ranking at each stage = last-prompt-token attention averaged over the 4 layers preceding the cut, read from the
unpruned pass (pruned tokens carry ~0 attention afterwards, so survivor ranking is unchanged); keep sets are NESTED.
ARMS  uniform@600 (bar) | tsr900 (single cut, §26) | pyr_causal1250 | pyr_fixed1250 | uniform@1250 (DIAGNOSTIC, 208%)
PRE-REGISTERED
    P1  pyr_causal1250 - bar, pooled, CI clear on BOTH models
    S1  pyr_causal1250 - tsr900      (does tapering earlier to afford more resolution pay?)
    S2  pyr_causal1250 - pyr_fixed1250 (does the causal schedule beat PyramidDrop's at equal compute?)
    GUARD relational not significantly negative on both.  Budget gate: measured tokens, >10% drift voids.
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase187_pyramid_{WHICH}.jsonl"; NL=28; P=16; Image.MAX_IMAGE_PIXELS=None
# schedule = list of (cut layer P, keep fraction k): tokens outside the keep set get the bias from layer P+1 on
SCHED={"tsr900":(900,[(16,0.10)]),
       "pyr_causal1250":(1250,[(8,0.50),(12,0.25),(16,0.10)]),
       "pyr_fixed1250":(1250,[(7,0.50),(14,0.25),(21,0.125)])}
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
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
    layers=model.model.language_model.layers; assert len(layers)==NL
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def run(inp,drop=None,from_layer=None,want_attn=False):
        clear()
        if drop is not None and len(drop):
            b=torch.zeros(inp["input_ids"].shape[1],device=model.device); b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(from_layer,NL): layers[li].self_attn._prune_bias=b
        with torch.no_grad(): out=model(**inp,output_attentions=want_attn)
        lg=out.logits[0,-1].float(); assert torch.isfinite(lg).all()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        A=None
        if want_attn: A=np.stack([out.attentions[L][0,:,-1,:].float().mean(0).cpu().numpy() for L in range(NL)])
        del out; torch.cuda.empty_cache(); clear()
        return [round(float(v),6) for v in p.tolist()],A
    rng=random.Random(185); n=0; t0=time.time(); done=set()
    if os.path.exists(OUT): done={json.loads(l)["question_id_full"] for l in open(OUT)}
    with open(OUT,"a") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid in done or not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB"); label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            rec={"question_id_full":qid,"category":ex["category"],"label":label,"probs":{},"realized_tokens":{},"token_layers":{}}
            for E in (600,):
                sm,_=fit(img,E); inp=build(sm,ex["text"]).to(model.device); nt=int((inp["input_ids"][0]==itid).sum())
                p,_=run(inp); rec["probs"][f"uniform@{E}"]=p; rec["realized_tokens"][f"uniform@{E}"]=nt; rec["token_layers"][f"uniform@{E}"]=nt*NL
            cache={}
            for E in (1250,):
                sm,_=fit(img,E); inp=build(sm,ex["text"]).to(model.device); nt=int((inp["input_ids"][0]==itid).sum())
                p,_=run(inp); rec["probs"]["uniform@1250"]=p; rec["realized_tokens"]["uniform@1250"]=nt; rec["token_layers"]["uniform@1250"]=nt*NL
            for name,(E,sched) in SCHED.items():
                if E not in cache:
                    sm,_=fit(img,E); inp=build(sm,ex["text"]).to(model.device); pos=(inp["input_ids"][0]==itid).nonzero().flatten()
                    base,nt=int(pos[0]),int(len(pos)); _,A=run(inp,want_attn=True)
                    Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12); cache[E]=(inp,base,nt,Ai)
                inp,base,nt,Ai=cache[E]
                clear(); alive=np.arange(nt); tl=0; prevP=-1
                for (Pp,k) in sched:
                    s_=Ai[max(0,Pp-3):Pp+1].mean(0); keep=max(1,int(round(k*nt)))
                    alive=alive[np.argsort(-s_[alive])[:keep]]                 # nested keep set
                    drop=np.setdiff1d(np.arange(nt),alive); b=torch.zeros(inp["input_ids"].shape[1],device=model.device)
                    b[torch.as_tensor(base+drop,device=model.device)]=-1e4
                    nxt=[q for q,_ in sched if q>Pp]; endL=(nxt[0] if nxt else NL-1)
                    for li in range(Pp+1,endL+1): layers[li].self_attn._prune_bias=b
                    tl+=nt*(Pp-prevP) if prevP<0 else nt*0     # full width up to the first cut
                    prevP=Pp
                # token-layers: full width through first cut, then keep_i for each segment
                tl=nt*(sched[0][0]+1); pp=sched[0][0]
                for i,(Pp,k) in enumerate(sched):
                    nxt=sched[i+1][0] if i+1<len(sched) else NL-1; tl+=max(1,int(round(k*nt)))*(nxt-Pp)
                with torch.no_grad(): out=model(**inp)
                lg=out.logits[0,-1].float(); pz=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
                del out; torch.cuda.empty_cache(); clear()
                rec["probs"][name]=[round(float(v),6) for v in pz.tolist()]; rec["realized_tokens"][name]=nt; rec["token_layers"][name]=tl
            del cache; torch.cuda.empty_cache()
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
