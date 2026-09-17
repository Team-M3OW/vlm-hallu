"""
Phase 151 (reviewer W2): how many boxed items does the head need?
Train on k items (k in 25,50,100,150,191-in-OOF-folds), evaluate OOF on the rest, V*Bench, both
models, W=0.25 coverage. If 50 items already recover most of the gain, the annotation cost of
'boxed supervision' is a one-off afternoon per task type, not per domain.
"""
import json, sys, numpy as np
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor
P70.W=0.25
def run(build, name):
    r=build(); X,Y,G,DEP,rows=r[0],r[1],r[2],r[3],r[4]
    gs=np.unique(G); N=len(gs)
    ring={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; m=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: m[1:-1,1:-1]=True
        else: m[:]=True
        ring[gi]=m.ravel()
    def top1(P,items):
        return np.array([float(Y[G==gi][int(np.argmax(np.where(ring[gi],P[G==gi],-1e9)))]>=P70.COV_HIT) for gi in items])
    dep=np.array([float(Y[G==gi][int(np.argmax(np.where(ring[gi],DEP[gi],-1e9)))]>=P70.COV_HIT) for gi in gs])
    print(f"\n{name}: deployed argmax {dep.mean()*100:.1f}%")
    for k in [10,25,50,100,150]:
        accs=[]
        for s in range(10):
            rng=np.random.default_rng(1510+s); perm=rng.permutation(gs); tr_items=set(perm[:k]); te_items=perm[k:]
            tr=np.where(np.isin(G,list(tr_items)))[0]; ytr=Y[tr]; pos=tr[ytr>0]; negpool=tr[ytr<=0]
            neg=rng.choice(negpool,size=min(len(negpool),30*k),replace=False); sub=np.concatenate([pos,neg])
            m=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(X[sub],Y[sub])
            P=np.zeros(len(Y)); te=np.where(np.isin(G,te_items))[0]; P[te]=m.predict(X[te])
            accs.append(top1(P,te_items).mean() - dep[np.isin(gs,te_items)].mean())
        a=np.array(accs)
        print(f"   k={k:3d} boxed items: head - argmax on held-out = {a.mean()*100:+5.1f}pp  (10 draws, sd {a.std()*100:.1f})")
    print(f"   k=191 (5-fold OOF, reference): {top1(P70.oof(X,Y,G,seeds=1),gs).mean()*100 - dep.mean()*100:+5.1f}pp")
run(P70.build,"Qwen3-VL"); run(P80.build,"Qwen2-VL")
