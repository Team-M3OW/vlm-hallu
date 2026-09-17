"""
Phase 163 (architecture fix 2b): MASS-CONTAINMENT window -- question-agnostic, no threshold on peaks.

PATCHED before running (phase-161 diagnostic): the GBT score map is a smooth coverage regression whose
second peak is >=0.5x the first on 94% of SINGLE-object items, so peaks/mass on it cannot see question
type. The RAW gated-max attention map can: top-1 share separates single from relational at AUROC
0.735 / 0.805 (Qwen3 / Qwen2). So the mass box is grown on the raw gated-max map (label-free layers
L17-20 / L19-22), and if the box collapses to W (concentrated map) the crop is the head's W=0.25
window (the accurate single-object proposer). Question-agnostic: the map's dispersion decides.

Window = the smallest axis-aligned box (grown greedily from the top cell of the RAW gated map) that
contains a fraction q=0.6 of its ring-masked mass; padded to at least W=0.25 per side.
Concentrated maps (single-object) give a small box; bimodal maps (relational) give a box spanning
both modes. No peak threshold, no question text, no router. Pre-registered q=0.6, untuned.

Original phase-161 header follows.



DPR loses on relational questions in every cell because their evidence set is a union of regions
7.9x larger than a single target's, and a fixed window centred on one peak cannot cover it (SS14T,
SS15G). Multi-crop (four separate crops in one pass) failed (-3.1pp) because three of the views are
distractors. This is different: ONE crop, whose window is the bounding box of the top-k distinct
peaks of the head's score map, padded by half a cell. Single-object questions have one dominant
peak -> k collapses to 1 and the window is the usual W. Relational questions have two -> the window
spans both. No router, no question text, no labels: the map decides.

RULE (fixed, pre-registered, not tuned):
    peaks = local maxima of the ring-masked head score map, non-maximum-suppressed at >= 3 cells apart
    keep peak 2 iff score2 >= 0.5 * score1            (a second region of comparable evidence)
    window = bbox(peaks) padded to at least W=0.25 per side; re-encode at 300 tokens
ARMS at matched budget (uniform@600 bar):  head@0.25 (incumbent) | span@k<=2 | oracle@0.25
PRIMARY   span - bar on RELATIONAL questions, both models (the cell DPR loses today)
GUARD     span - head@0.25 on SINGLE-object must not be significantly negative (k should collapse to 1)
Per-cell head scores are needed: recomputed OOF here with phase 70's machinery (Qwen3: phase30c maps;
Qwen2: phase74 maps), so the proposals are the same fold-honest scores the paper uses.
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
WHICH=sys.argv[1] if len(sys.argv)>1 else "qwen3"
GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[WHICH]
MODEL_ID={"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT=f"{D}/data/phase163_mass_{WHICH}.jsonl"; B0,W,RATIO,MINSEP=300,0.25,0.5,3
Image.MAX_IMAGE_PIXELS=None

def score_maps():
    P70.W=W
    if WHICH=="qwen3": X,Y,G,DEP,rows,_=P70.build()
    else: X,Y,G,DEP,rows=P80.build()
    P=P70.oof(X,Y,G,seeds=3); out={}
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; m=G==gi; rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        s=np.where(rm.ravel(),P[m],-1e9).reshape(gh,gw)
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(28)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        raw=np.where(rm.ravel(),A[GATE].max(0),-1e9).reshape(gh,gw)
        out[r["question_id_full"]]=(s,gh,gw,r["gt_box_frac"],raw)
    return out

def peaks(s,gh,gw):
    cand=[]
    for i in range(gh):
        for j in range(gw):
            v=s[i,j]
            if v<=-1e8: continue
            nb=s[max(0,i-1):i+2,max(0,j-1):j+2]
            if v>=nb.max(): cand.append((v,i,j))
    cand.sort(reverse=True); kept=[]
    for v,i,j in cand:
        if all(max(abs(i-ii),abs(j-jj))>=MINSEP for _,ii,jj in kept): kept.append((v,i,j))
        if len(kept)==2: break
    if len(kept)==2 and kept[1][0] < RATIO*kept[0][0]: kept=kept[:1]
    return kept

def main():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    maps=score_maps(); print(f"score maps for {len(maps)} items",flush=True)
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0}); model.eval()
    pr=AutoProcessor.from_pretrained(MODEL_ID); tok=pr.tokenizer
    opt=[sorted({tok(x,add_special_tokens=False)["input_ids"][-1] for x in [c,f" {c}"]}) for c in "ABCD"]
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
    def crop_frac(img,x0,y0,x1,y1):
        iw,ih=img.size; return img.crop((int(x0*iw),int(y0*ih),int(x1*iw),int(y1*ih)))
    def window_box(cx,cy,w):
        x0=min(max(0,cx-w/2),1-w); y0=min(max(0,cy-w/2),1-w); return x0,y0,x0+w,y0+w
    def answer(img,text):
        inp=build(img,text); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); return [round(float(v),6) for v in p.tolist()],rz
    t0,n=time.time(),0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in maps or not os.path.exists(ip): continue
            s,gh,gw,gt,raw=maps[qid]; img=Image.open(ip).convert("RGB")
            pk=peaks(s,gh,gw); cells=[(((j)+.5)/gw,((i)+.5)/gh) for _,i,j in pk]
            cx,cy=cells[0]; head_box=window_box(cx,cy,W)
            # ---- mass containment (phase 163): grow a box from the top cell until it holds q of the mass
            sm=np.clip(np.where(raw>-1e8,raw,0.0),0,None); tot=sm.sum()+1e-12
            ti,tj=np.unravel_index(int(np.argmax(sm)),sm.shape); i0=i1=ti; j0=j1=tj; q=0.6
            while sm[i0:i1+1,j0:j1+1].sum()/tot < q:
                grow=[]
                if i0>0: grow.append((sm[i0-1,j0:j1+1].sum(),'u'))
                if i1<gh-1: grow.append((sm[i1+1,j0:j1+1].sum(),'d'))
                if j0>0: grow.append((sm[i0:i1+1,j0-1].sum(),'l'))
                if j1<gw-1: grow.append((sm[i0:i1+1,j1+1].sum(),'r'))
                if not grow: break
                g=max(grow)[1]
                if g=='u': i0-=1
                elif g=='d': i1+=1
                elif g=='l': j0-=1
                else: j1+=1
            mx0,mx1,my0,my1=j0/gw,(j1+1)/gw,i0/gh,(i1+1)/gh
            if mx1-mx0<W: c=(mx0+mx1)/2; mx0=min(max(0,c-W/2),1-W); mx1=mx0+W
            if my1-my0<W: c=(my0+my1)/2; my0=min(max(0,c-W/2),1-W); my1=my0+W
            mass_box=(mx0,my0,mx1,my1)
            concentrated = (mx1-mx0 <= W+1e-9) and (my1-my0 <= W+1e-9)
            if concentrated: mass_box=head_box
            if len(cells)==2:
                xs=[c[0] for c in cells]; ys=[c[1] for c in cells]; padx,pady=0.5/gw,0.5/gh
                x0,x1=max(0,min(xs)-padx),min(1,max(xs)+padx); y0,y1=max(0,min(ys)-pady),min(1,max(ys)+pady)
                # enforce at least W per side, centred on the span
                if x1-x0<W: c=(x0+x1)/2; x0,y_=min(max(0,c-W/2),1-W),0; x1=x0+W
                if y1-y0<W: c=(y0+y1)/2; y0=min(max(0,c-W/2),1-W); y1=y0+W
                span_box=(x0,y0,x1,y1)
            else: span_box=head_box
            label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            arms={"uniform@300":fit(img,B0)[0],"uniform@600":fit(img,2*B0)[0],
                  "head@0.25":fit(crop_frac(img,*head_box),B0)[0],"span":fit(crop_frac(img,*span_box),B0)[0],"mass":fit(crop_frac(img,*mass_box),B0)[0],
                  "oracle@0.25":fit(crop_frac(img,*window_box((gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W)),B0)[0]}
            rec={"question_id_full":qid,"category":ex["category"],"label":label,"k":len(cells),
                 "span_box":[round(v,4) for v in span_box],"mass_box":[round(v,4) for v in mass_box],"mass_area":round((mass_box[2]-mass_box[0])*(mass_box[3]-mass_box[1]),4),"mass_concentrated":bool(concentrated),"span_area":round((span_box[2]-span_box[0])*(span_box[3]-span_box[1]),4),
                 "probs":{},"realized_tokens":{}}
            for nm,im in arms.items():
                p,rz=answer(im,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"Done -> {OUT}",flush=True)
main()
