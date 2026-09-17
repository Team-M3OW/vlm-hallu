"""
Phase 154 (reviewer W6): the full DPR pipeline on a NEW checkpoint, parameterised.
    stage extract : V*Bench attention maps (phase-30c columns) with the full prompt, all LM layers
    stage head    : OOF GBT head on those maps (phase-70 machinery, block = same stack fraction)
    stage endtask : uniform@300/600, argmax@W, head@W, rand@W, oracle@W  (phase-97 arms)
usage: python3 phase154_newmodel_pipeline.py <tag> <MODEL_ID> <stage>
Pre-registered primary: head@0.25 - uniform@600 on single-object (direct_attributes), CI clear of
zero. W=0.25 TRANSFERRED from Qwen3-VL-2B, not selected. Block = layers [.57*NL, .93*NL].
"""
import io, json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; sys.path.insert(0,f"{D}/scripts")
TAG, MODEL_ID, STAGE = sys.argv[1], sys.argv[2], sys.argv[3]
ATTN=f"{D}/data/phase154_{TAG}_attn.jsonl"; PROP=f"{D}/data/phase154_{TAG}_proposals.json"; OUT=f"{D}/data/phase154_{TAG}_endtask.jsonl"
B0, W = 300, 0.25
Image.MAX_IMAGE_PIXELS=None

def load_model(eager):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    kw=dict(dtype=torch.bfloat16, device_map={"":0})
    if eager: kw["attn_implementation"]="eager"
    m=AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw); m.eval()
    return m, AutoProcessor.from_pretrained(MODEL_ID)

def helpers(model, pr):
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
    return build, fit, window

def vstar():
    from huggingface_hub import snapshot_download; from datasets import load_dataset
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset"); ds=load_dataset("craigwu/vstar_bench")["test"]
    return root, ds

if STAGE=="extract":
    root,ds=vstar(); model,pr=load_model(True); build,fit,window=helpers(model,pr); itid=model.config.image_token_id
    done=set()
    if os.path.exists(ATTN): done={json.loads(l)["question_id_full"] for l in open(ATTN)}
    t0=time.time(); n=0
    with open(ATTN,"a") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"]); ap=os.path.splitext(ip)[0]+".json"
            if qid in done or not os.path.exists(ap): continue
            ann=json.load(open(ap))
            if not ann.get("bbox"): continue
            img=Image.open(ip).convert("RGB"); Wd,Ht=img.size
            gx0=min(b[0] for b in ann["bbox"])/Wd; gy0=min(b[1] for b in ann["bbox"])/Ht
            gx1=max(b[0]+b[2] for b in ann["bbox"])/Wd; gy1=max(b[1]+b[3] for b in ann["bbox"])/Ht
            small,rz=fit(img,B0); inp=build(small,ex["text"]); g=inp["image_grid_thw"][0].tolist(); gh,gw=g[1]//2,g[2]//2; n_img=gh*gw
            pos=(inp["input_ids"][0]==itid).nonzero().flatten(); base=int(pos[0].item()); inp=inp.to(model.device)
            with torch.no_grad(): out=model(**inp,output_attentions=True)
            rec={"question_id_full":qid,"category":ex["category"],"label":ex["label"],"grid":[gh,gw],"n_img_tokens":n_img,
                 "gt_box_frac":[gx0,gy0,gx1,gy1],"attn":{}}
            for L in range(len(out.attentions)):
                a=out.attentions[L][0,:,-1,base:base+n_img].float().mean(0); rec["attn"][f"L{L}"]=[round(float(v),8) for v in a.tolist()]
            del out; fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
            torch.cuda.empty_cache()
    print(f"extract done: {n} -> {ATTN}",flush=True)

elif STAGE=="head":
    import phase70_rerank_head as P70
    P70.W=W
    rows=[json.loads(l) for l in open(ATTN)]; NL=len([k for k in rows[0]["attn"] if k.startswith("L")]); b0,b1=int(.57*NL),int(.93*NL)+1
    X,Y,G,DEP=[],[],[],[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        M=A.reshape(NL,gh,gw); dep=M[b0:b1].mean(0); R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).reshape(NL,gh,gw)
        pad=np.pad(dep,1,mode="edge"); nb=sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=(yy+.5)/gh,(xx+.5)/gw
        X.append(np.concatenate([M.reshape(NL,-1).T,R.reshape(NL,-1).T,nb.reshape(-1,1),dep.reshape(-1,1),fx.reshape(-1,1),fy.reshape(-1,1),
            np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1,1),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)).reshape(-1,1),
            (xx==gw-1).astype(float).reshape(-1,1),(yy==gh-1).astype(float).reshape(-1,1),(xx==0).astype(float).reshape(-1,1)],1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]),float(fy.flat[i]),r["gt_box_frac"]) for i in range(n)])); G.append(np.full(n,gi)); DEP.append(dep.ravel())
    X=np.vstack(X); Y=np.concatenate(Y); G=np.concatenate(G)
    P=P70.oof(X,Y,G); out={}; hc=[]; ac=[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; m=G==gi; rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rmf=rm.ravel(); j=int(np.argmax(np.where(rmf,P[m],-1e9))); jd=int(np.argmax(np.where(rmf,DEP[gi],-1e9)))
        cell=lambda i:(((i%gw)+.5)/gw,((i//gw)+.5)/gh); y=Y[m]
        out[r["question_id_full"]]={"head":cell(j),"argmax":cell(jd),"head_cov":float(y[j]),"argmax_cov":float(y[jd]),"gt_box_frac":r["gt_box_frac"],"category":r["category"]}
        hc.append(float(y[j]>=P70.COV_HIT)); ac.append(float(y[jd]>=P70.COV_HIT))
    json.dump(out,open(PROP,"w"),indent=1)
    print(f"\n{TAG}: NL={NL} block L{b0}-{b1-1}  OOF coverage head {100*np.mean(hc):.1f}%  deployed argmax {100*np.mean(ac):.1f}%  (W={W})",flush=True)

elif STAGE=="endtask":
    root,ds=vstar(); props=json.load(open(PROP)); model,pr=load_model(False); build,fit,window=helpers(model,pr); tok=pr.tokenizer
    opt=[sorted({tok(s,add_special_tokens=False)["input_ids"][-1] for s in [c,f" {c}"]}) for c in "ABCD"]
    def answer(img,text):
        inp=build(img,text); rz=int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist())); inp=inp.to(model.device)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in opt]),0); return [round(float(v),6) for v in p.tolist()],rz
    t0=time.time(); n=0
    with open(OUT,"w") as fout:
        for ex in ds:
            qid=f"{ex['category']}/{ex['question_id']}"; ip=os.path.join(root,ex["image"])
            if qid not in props or not os.path.exists(ip): continue
            pp=props[qid]; img=Image.open(ip).convert("RGB"); gt=pp["gt_box_frac"]
            rng=random.Random(7100+hash(qid)%100000); rcx,rcy=rng.uniform(.1,.9),rng.uniform(.1,.9)
            label="ABCD".index(ex["label"]) if isinstance(ex["label"],str) else int(ex["label"])
            arms={"uniform@300":fit(img,B0)[0],"uniform@600":fit(img,2*B0)[0],
                  f"argmax@{W}":fit(window(img,*pp["argmax"],W),B0)[0],f"head@{W}":fit(window(img,*pp["head"],W),B0)[0],
                  f"rand@{W}":fit(window(img,rcx,rcy,W),B0)[0],f"oracle@{W}":fit(window(img,(gt[0]+gt[2])/2,(gt[1]+gt[3])/2,W),B0)[0]}
            rec={"question_id_full":qid,"category":ex["category"],"label":label,"probs":{},"realized_tokens":{}}
            for nm,im in arms.items():
                p,rz=answer(im,ex["text"]); rec["probs"][nm]=p; rec["realized_tokens"][nm]=rz
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%25==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s",flush=True)
    print(f"endtask done: {n} -> {OUT}",flush=True)
