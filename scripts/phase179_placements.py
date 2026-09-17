"""
Phase 179a (offline): crop PLACEMENTS for every baseline, from maps already on disk.
Each rule picks one cell; phase179_baselines.py then runs the identical crop->answer pass for all of them,
so the arms differ ONLY in where they look. Ring mask and W=0.25 throughout.
  vicrop_block   argmax of the deployed block-mean map          (the incumbent read-out; ViCrop 'rel-attention')
  vicrop_L{k}    argmax of ONE fixed mid layer                  (ViCrop's actual recipe: a hand-picked layer)
  gatemax        argmax of max over the divergence-gated layers (§16D, label-free, no training)
  laser          LASER (2602.04304): l*=argmax_l ||ReLU(A_q-A_noq)||, score=ReLU(A_q-A_noq)[l*]  (phase173 maps)
  head           ours: OOF learned head over the 28-layer profile
  oracle         GT-box centre (upper bound)
"""
import json, sys, numpy as np
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
P70.W=0.25; NL=28
CFG={"qwen3":(P70.build,list(range(17,21)),(16,27),14,"data/phase173_laser_qwen3.jsonl"),
     "qwen2":(P80.build,list(range(19,23)),(15,27),14,"data/phase173_laser_qwen2.jsonl")}
for tag,(build,GATE,BLK,FIXED,laserf) in CFG.items():
    r=build(); X,Y,G,rows=r[0],r[1],r[2],r[4]
    P=P70.oof(X,Y,G); lz={json.loads(l)["question_id_full"]:json.loads(l) for l in open(laserf)}
    out={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; n=q["n_img_tokens"]; m=G==gi
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel()
        A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        pick=lambda s: int(np.argmax(np.where(rm,s,-1e9)))
        cells={"vicrop_block":pick(A[BLK[0]:BLK[1]].mean(0)),f"vicrop_L{FIXED}":pick(A[FIXED]),
               "gatemax":pick(A[GATE].max(0)),"head":pick(P[m])}
        qid=q["question_id_full"]
        if qid in lz and lz[qid]["grid"]==[gh,gw]:
            L=lambda k: (lambda M: M/np.maximum(M.sum(1,keepdims=True),1e-12))(np.stack([np.asarray(lz[qid]["attn"][k][f"L{i}"],float) for i in range(NL)]))
            C=np.maximum(L("q")-L("noq"),0); cells["laser"]=pick(C[int(np.argmax(np.linalg.norm(C,axis=1)))])
        gt=q["gt_box_frac"]
        out[qid]={"grid":[gh,gw],"gt":gt,"cells":{k:[float((v%gw+.5)/gw),float((v//gw+.5)/gh)] for k,v in cells.items()}}
    json.dump(out,open(f"data/phase179_placements_{tag}.json","w"))
    agree={k:np.mean([o["cells"][k]==o["cells"]["head"] for o in out.values() if k in o["cells"]]) for k in ("vicrop_block",f"vicrop_L{FIXED}","gatemax","laser")}
    print(f"{tag}: {len(out)} items, laser on {sum('laser' in o['cells'] for o in out.values())}; cell agreement with head: "+", ".join(f"{k} {v*100:.0f}%" for k,v in agree.items()))
