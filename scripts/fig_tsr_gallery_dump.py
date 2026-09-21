"""Capture, for three V*Bench items, exactly what AVR keeps at the transport boundary and what the model
answers under attention-keep vs random-keep. Reuses phase185_tsr.py's pruning machinery verbatim."""
import json, os, sys, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"; NL=28; P=16; K=0.10
Image.MAX_IMAGE_PIXELS=None
PICKS=json.load(open("/tmp/tsr_picks.json")) if __import__("os").path.exists("/tmp/tsr_picks.json") else ["direct_attributes/9"]
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
QM3.eager_attention_forward=make_patched(QM3); QM2.eager_attention_forward=make_patched(QM2)
from huggingface_hub import snapshot_download
from datasets import load_dataset
root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager").eval()
assert "qwen3_vl" in type(model.model.language_model.layers[0].self_attn).__module__, "unpatched module"
pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer; itid=model.config.image_token_id
layers=model.model.language_model.layers; assert len(layers)==NL
opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
def chat(t): return pr.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":t}]}],tokenize=False,add_generation_prompt=True)
def build(i,t): return pr(images=i,text=chat(t),return_tensors="pt")
def gridof(i):
    g=build(i,"x")["image_grid_thw"].tolist()[0]; return g[1]//2, g[2]//2
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
    lg=out.logits[0,-1].float()
    p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0)
    A=None
    if want_attn: A=np.stack([out.attentions[L][0,:,-1,:].float().mean(0).cpu().numpy() for L in range(NL)])
    del out; torch.cuda.empty_cache(); clear()
    return [round(float(v),6) for v in p.tolist()], A
lut={f"{r['category']}/{r['question_id']}":r for r in ds}
rng=random.Random(185); out=[]
for qid in PICKS:
    ex=lut[qid]; img=Image.open(os.path.join(root,ex["image"])).convert("RGB")
    lab="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
    rec={"qid":qid,"category":ex["category"],"label":lab,"text":ex["text"],"image":ex["image"],"probs":{}}
    sm,_=fit(img,600); inp=build(sm,ex["text"]).to(model.device)
    rec["probs"]["uniform@600"],_=run(inp)
    sm9,_=fit(img,900); inp9=build(sm9,ex["text"]).to(model.device)
    pos=(inp9["input_ids"][0]==itid).nonzero().flatten(); base,nt=int(pos[0]),int(len(pos))
    gh,gw=gridof(sm9); rec["grid"]=[gh,gw]; rec["n_tokens"]=nt
    _,A=run(inp9,want_attn=True)
    Ai=A[:,base:base+nt]; Ai=Ai/np.maximum(Ai.sum(1,keepdims=True),1e-12)
    s=Ai[max(0,P-4):P+1].mean(0)                      # the AVR ranking: mean attention L12..L16
    keep=max(1,int(round(K*nt)))
    ord_attn=np.argsort(-s); drop_attn=(base+ord_attn[keep:]).tolist()
    rec["probs"]["tsr900"],_=run(inp9,drop=drop_attn,from_layer=P+1)
    sr=np.array([rng.random() for _ in range(nt)]); ord_rand=np.argsort(-sr)
    rec["probs"]["tsr900_rand"],_=run(inp9,drop=(base+ord_rand[keep:]).tolist(),from_layer=P+1)
    rec["probs"]["uniform@900"],_=run(inp9)
    rec["keep_attn"]=sorted(int(i) for i in ord_attn[:keep])
    rec["keep_rand"]=sorted(int(i) for i in ord_rand[:keep])
    rec["rank_score"]=[float(v) for v in s]
    rec["overlap"]=len(set(rec["keep_attn"])&set(rec["keep_rand"]))/keep
    out.append(rec)
    print(qid,"grid",gh,gw,"n",nt,"keep",keep,"overlap %.3f"%rec["overlap"],
          "ans",{k:int(np.argmax(v)) for k,v in rec["probs"].items()},"label",lab,flush=True)
json.dump(out,open(f"{D}/data/fig_tsr_gallery.json","w"))
print("wrote data/fig_tsr_gallery.json")
