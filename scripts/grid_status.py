"""4x4 completion tracker: 4 models x 4 benchmarks, both methods, one harness (phase225)."""
import json,os,glob
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS=["qwen3_2b","qwen2_7b","internvl3_8b","llava_ov"]
BENCH=["vstar","hr4k","cvbench","realworldqa"]
TARGET={"vstar":191,"hr4k":800,"cvbench":2638,"realworldqa":765}   # FULL splits
print(f"{'model':14s}"+"".join(f"{b:>16s}" for b in BENCH))
tot=done=0
for m in MODELS:
    row=f"{m:14s}"
    for b in BENCH:
        f=f"{D}/data/phase225_{m}_{b}.jsonl"
        tot+=1
        if os.path.exists(f):
            rows=[json.loads(l) for l in open(f) if l.strip()]
            n=len(rows)
            has_dwa=sum(1 for r in rows if "dwa_t" in r.get("probs",{}) or "dwa_t" in r.get("preds",{}))
            t=TARGET[b]
            # A cell counts as complete only when its runner logged "Done ->". A 95% item threshold
            # marked cells DONE while they were still running their last items.
            lg=f"{D}/logs/grid_phase225_newbench_{m}_{b}_0.log"
            fin=os.path.exists(lg) and "Done ->" in open(lg,errors="ignore").read()
            if fin: done+=1; row+=f"{'DONE '+str(n)+'/'+str(has_dwa):>16s}"
            else: row+=f"{str(n)+'/'+str(t):>16s}"
        else: row+=f"{'-':>16s}"
    print(row)
print(f"\ncells complete: {done}/{tot}   (cell = model x benchmark, each carrying BOTH AVR and DWA arms)")
print("format: DONE <items>/<items with DWA arms>")
