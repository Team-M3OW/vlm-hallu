"""Does a single W=0.25 window CONTAIN all GT boxes of a relational V* item?
Settles the paper's scope-law claim. CPU only, no model.
Containment (union extent <= W in both axes) is NOT the same as the coverage-style
statistic that produced the earlier 93.4% figure -- that is the whole point."""
import json,glob,os,numpy as np
from PIL import Image
SNAP=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*/"))[0]
W=0.25
for cat in ("relative_position","direct_attributes"):
    tot=multi=ok=0; nb=[]
    for jf in sorted(glob.glob(os.path.join(SNAP,cat,"*.json"))):
        d=json.load(open(jf)); iw,ih=Image.open(jf[:-5]+".jpg").size
        bx=d.get("bbox") or d.get("boxes") or []
        if bx and not isinstance(bx[0],(list,tuple)): bx=[bx]
        if not bx: continue
        tot+=1; nb.append(len(bx))
        fb=[(b[0]/iw,b[1]/ih,(b[0]+b[2])/iw,(b[1]+b[3])/ih) for b in bx]
        if len(fb)<2: continue
        multi+=1
        if (max(b[2] for b in fb)-min(b[0] for b in fb))<=W and \
           (max(b[3] for b in fb)-min(b[1] for b in fb))<=W: ok+=1
    print(f"{cat}: {tot} items, mean boxes {np.mean(nb):.2f}, {multi} multi-box")
    if multi: print(f"   union fits in one W={W} window: {ok}/{multi} = {100*ok/multi:.1f}%")
