"""
Phase 185: TRANSPORT-SCHEDULED RESOLUTION (TSR) -- depth-wise pruning that BUYS ACCURACY at equal compute.

THE MECHANISM IT IS BUILT ON (§20B, two models): image->text transport is complete by ~L16; visual tokens are
causally inert afterwards (suffix-mask KL ~0 from L16). §14L(b)/phase 87 measured the same thing from the
pruning side: dropping 90% of visual tokens at L16 costs ~2pp, at L14 costs 9-15pp, at L2 is below random.
THE TRADE: spend the compute that the inert late-layer visual tokens would have used on RESOLUTION during the
window where the image is actually read. Encode at E tokens, run FULL width through layers 0..P, at layer P
rank image tokens by the question-conditioned read-out available at that depth and keep fraction k for
layers P+1..27. Compute proxy (token-layers): TL = E*(P+1) + k*E*(NL-1-P).
    TSR  E=900, P=16, k=0.10 : 900*17 + 90*11 = 16,290   <  uniform@600 = 600*28 = 16,800  (the bar)
No crop, so nothing spatial is removed before L16 -- the property every cropping method lacks on relational items.

PRIOR ART, stated: PyramidDrop (2410.17247) drops tokens in stages and notes it can afford higher resolution;
FastV (2403.06764) prunes at L2; HiRED (2408.10945) prunes pre-LLM under a budget. What is new here is (i) the
prune depth is DERIVED from a causal measurement (§20B) rather than tuned, (ii) the comparison is accuracy at
MATCHED compute against the plain higher-resolution baseline, stratified by question type, and (iii) a
matched-compute FastV control isolates depth as the variable.

ARMS (tokens measured; TL computed from measured E)
    uniform@300 | uniform@600 (BAR, 16,800 TL) | tsr900 (P=16, k=.10, rank = mean attn L12..L16 at the last
    prompt token) | tsr900_atP (rank = L16 only) | tsr900_rand (random keep, same k) |
    fastv900 (P=2, k=.627 -> 16,290 TL, rank = L2 attention: FastV at EQUAL compute) |
    tsr600 (E=600, P=16, k=.10 -> 10,860 TL: the efficiency arm, ~65% of the bar's compute)
PRE-REGISTERED
    P1     tsr900 - uniform@600, pooled, CI clear of zero on BOTH models
    GUARD  tsr900 - uniform@600 on RELATIONAL not significantly negative on both (the no-crop property)
    S1     tsr900 - fastv900 (equal compute): depth is the variable
    S2     tsr600 - uniform@600 not significantly negative (equal accuracy at 65% compute)
Pruning is implemented as a -1e4 key bias from layer P+1 on (phase 83/93 machinery), which is accuracy-
equivalent to token removal; the ranking attention is read at layer P of the SAME 900-token pass.
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase185_tsr_{WHICH}.jsonl"; NL=28; P=16; Image.MAX_IMAGE_PIXELS=None
ARMS={"tsr900":(900,16,0.10,"win"),"tsr900_atP":(900,16,0.10,"atP"),"tsr900_rand":(900,16,0.10,"rand"),
      "fastv900":(900,2,0.627,"atP"),"tsr600":(600,16,0.10,"win")}
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
            for E in (300,600):
                sm,_=fit(img,E); inp=build(sm,ex["text"]).to(model.device); nt=int((inp["input_ids"][0]==itid).sum())
                p,_=run(inp); rec["probs"][f"uniform@{E}"]=p; rec["realized_tokens"][f"uniform@{E}"]=nt; rec["token_layers"][f"uniform@{E}"]=nt*NL
            cache={}
            for name,(E,Pp,k,rank) in ARMS.items():
                if E not in cache:
                    sm,_=fit(img,E); inp=build(sm,ex["text"]).to(model.device); pos=(inp["input_ids"][0]==itid).nonzero().flatten()
                    base,nt=int(pos[0]),int(len(pos)); _,A=run(inp,want_attn=True)
                    Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12); cache[E]=(inp,base,nt,Ai)
                inp,base,nt,Ai=cache[E]
                if rank=="win": s=Ai[max(0,Pp-4):Pp+1].mean(0)
                elif rank=="atP": s=Ai[Pp]
                else: s=np.array([rng.random() for _ in range(nt)])
                keep=max(1,int(round(k*nt))); drop=(base+np.argsort(-s)[keep:]).tolist()
                p,_=run(inp,drop=drop,from_layer=Pp+1)
                rec["probs"][name]=p; rec["realized_tokens"][name]=nt; rec["token_layers"][name]=nt*(Pp+1)+keep*(NL-1-Pp)
            del cache; torch.cuda.empty_cache()
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
