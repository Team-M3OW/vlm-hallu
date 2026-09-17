"""Phase 140b: which layers does the divergence gate select, and is the result sensitive to the gate
constant? Also dumps per-item proposals for the two best fixed rules so phase 141 can run end-task.
The 0.5 gate was pre-registered; 0.3/0.7 are reported for robustness only, never selected."""
import json, sys, numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase140_trainfree_ladder as L   # re-runs the ladder on import (prints), then we reuse loaders
D = L.D
props = {}
for which, label, dk in [("qwen3","Qwen3-VL-2B","Qwen3-VL-2B"),("qwen2","Qwen2-VL-7B","Qwen2-VL-7B")]:
    items, (b0,b1), div = L.load_qwen(which)
    n=len(items); rng=np.random.default_rng(1401)
    def hits(maps): return np.array([float(it["cov"][int(np.argmax(np.where(it["rm"],m,-1e9)))]>=L.COV) for it,m in zip(items,maps)])
    def ci(d):
        b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(4000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
    ref=hits([it["nw"][b0:b1].max(0) for it in items])
    print(f"\n=== {label}: divergence per layer (phase 95) ===")
    print("  "+" ".join(f"L{i}:{v:.3f}" for i,v in enumerate(div)))
    for g in [0.3,0.5,0.7]:
        gate=div>=g*div.max(); layers=np.where(gate)[0]
        r=hits([it["raw"][gate].max(0) for it in items]); c=hits([it["hs_nw"][gate].max(0) for it in items])
        m1,l1,h1=ci(r-ref); m2,l2,h2=ci(c-ref)
        print(f"  gate {g}: layers {layers.min()}-{layers.max()} (n={len(layers)})  raw-div-max {r.mean()*100:5.1f}% ({m1:+.1f} [{l1:+.1f},{h1:+.1f}] vs nw-max)   composite-div-max {c.mean()*100:5.1f}% ({m2:+.1f} [{l2:+.1f},{h2:+.1f}])")
    gate=div>=0.5*div.max()
    for it in items:
        gh,gw=it["gh"],it["gw"]
        def cell(m):
            j=int(np.argmax(np.where(it["rm"],m,-1e9))); return [((j%gw)+.5)/gw, ((j//gw)+.5)/gh]
        props[it["q"]]={"rawdiv":cell(it["raw"][gate].max(0)),"compdiv":cell(it["hs_nw"][gate].max(0)),
                        "rawdiv_cov":float(it["cov"][int(np.argmax(np.where(it["rm"],it["raw"][gate].max(0),-1e9)))]),
                        "compdiv_cov":float(it["cov"][int(np.argmax(np.where(it["rm"],it["hs_nw"][gate].max(0),-1e9)))])}
    json.dump(props, open(f"{D}/phase140_proposals_{which}.json","w")); props={}
    print(f"  wrote {D}/phase140_proposals_{which}.json")
