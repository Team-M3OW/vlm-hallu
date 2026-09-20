"""
Phase 198: DOES TSR STACK ON TOP OF EXISTING PLACEMENT RULES?

TSR is a resolution converter (§39: gain = acc@E_hi - acc@E_lo, r=0.966, slope 1.01) and is orthogonal to
WHERE the crop goes. If that is right, TSR should fund a higher-resolution crop for ANY placement rule at
the same token-layer budget -- including the published ones we compete with.

  two-pass bar   : localise@300 (300x28) + crop@300 (300x28)            = 16,800 TL = the bar
  TSR-funded     : localise@300 (300x28) + crop@460 pruned 90% at L16   = 16,726 TL = 99.6% of bar
                   (460x17 + 46x11 = 8,326 for the answer pass vs 8,400 plain)
  diagnostic     : localise@300 + crop@460 UNPRUNED                     = 21,280 TL = 127% (over budget)

This is NOT the merge rejected in §44/§196. That merged TWR's RANKING with pruning in the LOCALISATION pass,
which is structurally impossible (§29: pruning the localiser destroys the read-out). Here the pruning is in
the ANSWER pass, which §37 already showed composes mechanically.

PRE-REGISTERED
  P1      for each placement rule P: P_tsr - P_plain pooled, on BOTH models. "TSR stacks" iff positive on
          both models for every rule that has resolution headroom.
  PRED    from the §39 identity: P_tsr - P_plain ~= P_460 - P_plain (within ~2pp).
  GUARD   pruning is free: |P_tsr - P_460| <= 1.0pp (else the answer-pass prune is NOT free at this budget).
  S1      does the gain depend on placement quality? correlate gain with the rule's coverage@0.25.
  S2      contrast stacking on a weak rule (vicrop_block) vs the strongest (ridge/TWR).
Placement cells are the SAME ones the paper evaluates (phase179 + phase184 ridge, outer ring masked).
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; WHICH=sys.argv[1]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase198_stack_{WHICH}.jsonl"; Image.MAX_IMAGE_PIXELS=None
W=0.25; E_LO,E_HI,K=300,460,0.10
def make_patched(QM):
    def patched(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
for M in (QM3,QM2): M.eager_attention_forward=make_patched(M)
def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
                                                      attn_implementation="eager").eval()
    mod=type(model.model.language_model.layers[0].self_attn).__module__
    assert any(k in mod for k in ("qwen3_vl","qwen2_vl","qwen2_5_vl")), f"unpatched attention module: {mod}"
    print("attention module:",mod,flush=True)
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
    layers=model.model.language_model.layers; NL=len(layers); P=round(0.57*NL)
    print(f"NL={NL}  prune layer P={P}",flush=True)
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],
                                               tokenize=False,add_generation_prompt=True)
    def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
    def measure(i): return int(sum(g[1]*g[2]//4 for g in build(i,"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.06):
        W_,H_=img.size; sc=(target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur=img.resize((max(28,int(W_*sc)),max(28,int(H_*sc))),Image.BICUBIC); r=measure(cur)
            if best is None or abs(r-target)<abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc*=(target/r)**0.5
        return best[0]
    def crop(img,cx,cy):
        iw,ih=img.size; x0=min(max(0,(cx-W/2)*iw),iw-W*iw); y0=min(max(0,(cy-W/2)*ih),ih-W*ih)
        return img.crop((int(x0),int(y0),int(x0+W*iw),int(y0+W*ih)))
    def clear():
        for l in layers:
            if hasattr(l.self_attn,"_prune_bias"): del l.self_attn._prune_bias
    def run(inp,drop=None,want_attn=False):
        clear()
        if drop is not None and len(drop):
            b=torch.zeros(inp["input_ids"].shape[1],device=model.device)
            b[torch.as_tensor(drop,device=model.device)]=-1e4
            for li in range(P+1,NL): layers[li].self_attn._prune_bias=b
        with torch.no_grad(): out=model(**inp,output_attentions=want_attn)
        lg=out.logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
        A=None
        if want_attn: A=np.stack([out.attentions[L][0,:,-1,:].float().mean(0).cpu().numpy() for L in range(NL)])
        del out; torch.cuda.empty_cache(); clear()
        return [round(float(v),6) for v in p.tolist()], A
    def answer_plain(img,txt,E):
        inp=build(fit(img,E),txt); nt=int((inp["input_ids"][0]==itid).sum()); inp=inp.to(model.device)
        p,_=run(inp); return p,nt,nt*NL
    def answer_tsr(img,txt,E):
        sm=fit(img,E); inp=build(sm,txt); pos=(inp["input_ids"][0]==itid).nonzero().flatten()
        base,nt=int(pos[0]),int(len(pos)); inp=inp.to(model.device)
        _,A=run(inp,want_attn=True)
        Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12)
        s=Ai[max(0,P-4):P+1].mean(0)
        keep=max(1,int(round(K*nt))); drop=(base+np.argsort(-s)[keep:]).tolist()
        p,_=run(inp,drop=drop); return p,nt,nt*(P+1)+keep*(NL-1-P)
    # ---- placements: exactly the cells the paper evaluates ----
    PL=json.load(open(f"{D}/data/phase179_placements_{WHICH}.json"))
    RG={json.loads(l)["qid"]:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_{WHICH}.jsonl")} \
        if os.path.exists(f"{D}/data/fig_ridge_scores_{WHICH}.jsonl") else {}
    RULES=["vicrop_block","laser","ridge","oracle"]
    lut={f"{e['category']}/{e['question_id']}":e for e in ds}
    LOC=E_LO*NL
    done=set()
    if os.path.exists(OUT): done={json.loads(l)["question_id_full"] for l in open(OUT)}
    t0=time.time(); n=0
    with open(OUT,"a") as fout:
        for qid,ex in lut.items():
            if qid in done or qid not in PL: continue
            ip=os.path.join(root,ex["image"])
            if not os.path.exists(ip): continue
            img=Image.open(ip).convert("RGB")
            lab="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            gt=PL[qid]["gt"]
            cells=dict(PL[qid]["cells"])
            if qid in RG: cells["ridge"]=RG[qid]["ridge_cell"]
            cells["oracle"]=[(gt[0]+gt[2])/2,(gt[1]+gt[3])/2]
            rec={"question_id_full":qid,"category":ex["category"],"label":lab,"gt":gt,
                 "cells":{k:cells[k] for k in RULES if k in cells},
                 "probs":{},"realized_tokens":{},"token_layers":{}}
            p,nt,tl=answer_plain(img,ex["text"],600)
            rec["probs"]["uniform@600"]=p; rec["realized_tokens"]["uniform@600"]=nt; rec["token_layers"]["uniform@600"]=tl
            for r in RULES:
                if r not in cells: continue
                cimg=crop(img,*cells[r])
                for name,fn,E in ((f"{r}_plain",answer_plain,E_LO),(f"{r}_460",answer_plain,E_HI),(f"{r}_tsr",answer_tsr,E_HI)):
                    p,nt,tl=fn(cimg,ex["text"],E)
                    rec["probs"][name]=p; rec["realized_tokens"][name]=nt; rec["token_layers"][name]=LOC+tl
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%10==0:
                el=time.time()-t0
                print(f"  [{n}] {el/n:.1f}s/item  eta {(len(lut)-len(done)-n)*el/n/60:.0f}min",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
