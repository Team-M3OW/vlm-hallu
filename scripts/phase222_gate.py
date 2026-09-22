"""
Phase 222 (CPU): (1) a SELF-confidence gate -- crop or fall back to no-crop, decided from the map alone --
and (3) the depth profile as a predictor of the model being wrong.

The gate is NOT a question-type router: it never sees the question type, only the read-out's own map.
Threshold chosen OUT-OF-FOLD (GroupKFold 5, seeds 700-702), never on the evaluated item.
Answers are reused from phase184 (ridge crop and uniform@600), so this costs no GPU.
"""
import json, numpy as np, random
from sklearn.model_selection import KFold
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; random.seed(222); B=8000
def boot(d):
    m=sum(d)/len(d); s=sorted(sum(random.choice(d) for _ in range(len(d)))/len(d) for _ in range(B))
    return 100*m,100*s[int(.025*B)],100*s[int(.975*B)]
def tag(lo,hi): return " OK" if lo>0 else (" NEG" if hi<0 else " n.s.")
for w,BLK in (("qwen3",(16,27)),("qwen2",(15,27))):
    SC={json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_{w}.jsonl")}
    ARM={json.loads(l)['question_id_full']:json.loads(l) for l in open(f"{D}/data/phase184_allarms_{w}.jsonl")}
    ids=sorted(set(SC)&set(ARM))
    crop=np.array([int(np.argmax(ARM[q]['probs']['ridge'])==ARM[q]['label']) for q in ids])
    bar =np.array([int(np.argmax(ARM[q]['probs']['uniform@600'])==ARM[q]['label']) for q in ids])
    cat =np.array([SC[q]['category'] for q in ids])
    feats={}
    for q in ids:
        m=SC[q]; gh,gw=m['grid']
        s=np.asarray(m['score'],float).ravel(); dep=np.asarray(m['dep'],float).ravel()
        rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; rm=rm.ravel()
        sv=np.sort(s[rm])[::-1]; dv=dep[rm]/max(dep[rm].sum(),1e-12)
        ent=-(dv*np.log(dv+1e-12)).sum()/np.log(len(dv))
        peak=dep[rm].max()/max(dep[rm].mean(),1e-12)
        feats.setdefault('margin',[]).append(float(sv[0]-sv[1]))
        feats.setdefault('top5gap',[]).append(float(sv[0]-sv[:5].mean()))
        feats.setdefault('neg_entropy',[]).append(float(-ent))
        feats.setdefault('peakiness',[]).append(float(peak))
    print(f"=== {w}  n={len(ids)}   bar {100*bar.mean():.1f}   crop {100*crop.mean():.1f}")
    # oracle gate: the ceiling of ANY gate
    orc=np.maximum(crop,bar)
    print(f"   oracle gate (ceiling of any gate): {100*orc.mean():.1f}  "
          f"(+{100*(orc.mean()-max(crop.mean(),bar.mean())):.1f} over the better fixed policy)")
    for fname,v in feats.items():
        v=np.array(v); pick=np.zeros(len(ids),int)
        for tr,te in KFold(5,shuffle=True,random_state=700).split(v):
            best=(-1,None)
            for thr in np.quantile(v[tr],np.linspace(0.05,0.95,19)):
                g=np.where(v[tr]>=thr,crop[tr],bar[tr]).mean()
                if g>best[0]: best=(g,thr)
            pick[te]=(v[te]>=best[1]).astype(int)
        gated=np.where(pick==1,crop,bar)
        m,lo,hi=boot(list(gated-crop))
        m2,lo2,hi2=boot(list(gated-bar))
        s=cat=="direct_attributes"; x=~s
        print(f"   gate[{fname:11s}] acc {100*gated.mean():5.1f}  vs crop {m:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{tag(lo,hi):5s}"
              f" vs bar {m2:+5.1f}[{lo2:+5.1f},{hi2:+5.1f}]{tag(lo2,hi2):5s}"
              f"  single {100*gated[s].mean():5.1f} cross {100*gated[x].mean():5.1f}  (crops {100*pick.mean():.0f}%)")
    # (3) does the map predict the model being WRONG at the bar?
    Xf=np.stack([np.array(feats[k]) for k in feats],1)
    Xf=(Xf-Xf.mean(0))/(Xf.std(0)+1e-9)
    from sklearn.linear_model import LogisticRegression
    oof=np.zeros(len(ids))
    for tr,te in KFold(5,shuffle=True,random_state=700).split(Xf):
        lr=LogisticRegression(max_iter=1000).fit(Xf[tr],bar[tr]); oof[te]=lr.predict_proba(Xf[te])[:,1]
    from sklearn.metrics import roc_auc_score
    print(f"   (3) depth-profile features predict CORRECTNESS at the bar: AUROC {roc_auc_score(bar,oof):.3f}")
