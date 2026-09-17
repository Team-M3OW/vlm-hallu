"""
Phase 168: PROPORTIONAL BUDGET ALLOCATION -- continuous, question-agnostic, one pass.

WHY. Phases 58 and 81 are not in conflict: multi-crop trades union coverage for per-crop
magnification, so it wins with a weak proposer (argmax top-1 covers 39.3% -> +7.9pp) and loses with a
strong one (head top-1 covers 52.9% -> -3.1pp). The head is weak exactly on RELATIONAL items (top-1
52.6/47.4%, top-5 75/61%), which is the regime where splitting pays. A fixed k forces the same trade
on every item; a CONTINUOUS split does not.

MECHANISM (no gate, no question text, no threshold on peaks)
    scores  = head's OOF predicted coverage, NMS >= 3 cells apart, top-3 candidates
    weights = predicted coverage, normalised over the candidates
    tokens_i = round(w_i * B0)   (crops under 50 tokens are dropped and the rest renormalised)
    window_i = W * sqrt(w_i)     <- magnification-preserving: tokens per pixel is constant across crops
    one forward pass with the surviving crops as multiple images, B0 tokens in total
Degenerate by construction: a dominant peak takes ~all the budget at W=0.25 -- that IS today's DPR.
A bimodal map splits into two crops at W~0.18 each, same magnification, covering both regions.

ARMS at matched budget (bar = uniform@600; every crop arm is pass1@300 + crops totalling 300)
    uniform@300 | uniform@600 | head@0.25 (incumbent) | prop (magnification-preserving)
    | prop_fixedW (same split, window fixed at 0.25 -- isolates the sqrt rule) | oracle@0.25
PRE-REGISTERED
    P1     prop - bar on RELATIONAL, CI clear of zero on BOTH models   <- the cell DPR loses
    GUARD  prop - head@0.25 on SINGLE-object not significantly negative (the split must degenerate)
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
WHICH=sys.argv[1] if len(sys.argv)>1 else "qwen3"
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase168_prop_{WHICH}.jsonl"
B0,W,KMAX,MINSEP,MINTOK=300,0.25,3,3,50
Image.MAX_IMAGE_PIXELS=None

def proposals():
    P70.W=W
    if WHICH=="qwen3": X,Y,G,DEP,rows,_=P70.build()
    else: X,Y,G,DEP,rows=P80.build()
    P=P70.oof(X,Y,G,seeds=3); out={}
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; m=G==gi; rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        s=np.where(rm.ravel(),P[m],-1e9).reshape(gh,gw)
        cand=[]
        for i in range(gh):
            for j in range(gw):
                v=s[i,j]
                if v<=-1e8: continue
                if v>=s[max(0,i-1):i+2,max(0,j-1):j+2].max(): cand.append((float(v),i,j))
        cand.sort(reverse=True); kept=[]
        for v,i,j in cand:
            if all(max(abs(i-ii),abs(j-jj))>=MINSEP for _,ii,jj in kept): kept.append((v,i,j))
            if len(kept)==KMAX: break
        cells=[(((j)+.5)/gw,((i)+.5)/gh,max(v,0.0)) for v,i,j in kept]
        out[r["question_id_full"]]={"cells":cells,"gt":r["gt_box_frac"],"cat":r["category"]}
    return out

def split(cells):
    """weights ∝ predicted coverage; drop crops under MINTOK and renormalise."""
    w=np.array([c[2] for c in cells],float)
    if w.sum()<=0: w=np.ones(len(cells))
    w=w/w.sum(); keep=np.ones(len(w),bool)
    for _ in range(len(w)):
        tok=w*B0; bad=(tok<MINTOK)&keep
        if not bad.any(): break
        keep[np.argmin(np.where(keep,tok,np.inf))]=False
        w=np.where(keep,w,0); w=w/max(w.sum(),1e-9)
    return [(cells[i][0],cells[i][1],w[i]) for i in range(len(w)) if keep[i] and w[i]>0]

def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    props=proposals(); print(f"proposals for {len(props)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}); model.eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
    def chat(nimg,text):
        c=[{"type":"image"} for _ in range(nimg)]+[{"type":"text","text":text}]
        return pr.apply_chat_template([{"role":"user","content":c}],tokenize=False,add_generation_prompt=True)
    def build(imgs,text): return pr(images=imgs,text=chat(len(imgs),text),return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build([img],"x")["image_grid_thw"].tolist()))
    def fit(img,target,refine=6,tol=0.08):
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
    def answer(imgs,text):
        inp=build(imgs,text); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in props or not os.path.exists(ip): continue
            pp=props[qid]; img=Image.open(ip).convert("RGB"); gt=pp["gt"]
            parts=split(pp["cells"])
            prop_imgs=[fit(window(img,cx,cy,W*np.sqrt(w)),max(MINTOK,int(round(w*B0))))[0] for cx,cy,w in parts]
            propF_imgs=[fit(window(img,cx,cy,W),max(MINTOK,int(round(w*B0))))[0] for cx,cy,w in parts]
            cx0,cy0,_=pp["cells"][0]
            label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            arms={"uniform@300":[fit(img,B0)[0]],"uniform@600":[fit(img,2*B0)[0]],
                  "head@0.25":[fit(window(img,cx0,cy0,W),B0)[0]],"prop":prop_imgs,"prop_fixedW":propF_imgs,
                  "oracle@0.25":[fit(window(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)[0]]}
            rec={"question_id_full":qid,"category":ex["category"],"label":label,
                 "k":len(parts),"weights":[round(float(w),3) for _,_,w in parts],"probs":{},"realized_tokens":{}}
            for nm,ims in arms.items():
                p,rz=answer(ims,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
