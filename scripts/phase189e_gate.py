"""Pre-specified gate for 189f: the 300-token-trained token-only L4 probe, applied to the 900-token dump, must retain the box
(box-hit@25%) at least as well as attention@L16 ranking on the SAME 900-token dump, on BOTH models. exit 0 = PASS."""
import json,sys,numpy as np
from sklearn.linear_model import LogisticRegression
ok=True
for tag in ("qwen3","qwen2"):
    Z3=np.load(f"data/phase189a_hidden_{tag}.npz"); m3=json.load(open(f"data/phase189a_hidden_{tag}_meta.json"))
    Z9=np.load(f"data/phase189a_hidden_{tag}_900.npz"); m9=json.load(open(f"data/phase189a_hidden_{tag}_900_meta.json"))
    def yb(m):
        gh,gw=m["grid"]; yy,xx=np.mgrid[0:gh,0:gw]; fx,fy=((xx+.5)/gw).ravel(),((yy+.5)/gh).ravel(); x0,y0,x1,y1=m["gt_box_frac"]; return ((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(int)
    X=np.vstack([Z3[f"h_L4_{i}"].astype(np.float32) for i in range(len(m3))]); Y=np.concatenate([yb(m) for m in m3]); mu,sd=X.mean(0),X.std(0)+1e-6
    clf=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit((X-mu)/sd,Y)
    def hit(scores):
        h=[]
        for i,m in enumerate(m9):
            y=yb(m); s=scores[i]; k=max(1,int(round(.25*len(y)))); kept=np.zeros(len(y),bool); kept[np.argsort(-s)[:k]]=True
            if y.sum()>0: h.append(float((kept&(y==1)).sum()/y.sum()>=0.5))
        return np.mean(h)*100
    hp=hit([clf.decision_function((Z9[f"h_L4_{i}"].astype(np.float32)-mu)/sd) for i in range(len(m9))]); ha=hit([Z9[f"a_L16_{i}"] for i in range(len(m9))])
    print(f"{tag} @900: probe(trained@300) box-hit {hp:.1f}%  vs attention@L16 {ha:.1f}%  -> {'ok' if hp>=ha else 'FAIL'}"); ok&=(hp>=ha)
print("GATE","PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
