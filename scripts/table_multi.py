"""The multi-model x multi-benchmark table, recomputed from the raw per-item probs.
Paired percentile bootstrap (B=10000, seed 0) on the per-item correctness difference.
Flip counts are printed alongside every delta: an exact +0.0 with a [+0.0,+0.0] CI is a
no-op (0 flips -> arms are the same computation), NOT a null result (sec 93b).
Usage: python3 scripts/table_multi.py
"""
import json, numpy as np, os
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
def load(f):
    p=f"{D}/data/{f}.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
def acc(rows,a): return np.array([1.0 if int(np.argmax(r["probs"][a]))==r["label"] else 0.0 for r in rows])
def ci(d,B=10000):
    n=len(d); m=d[rng.integers(0,n,(B,n))].mean(1)
    return d.mean()*100,np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
def flips(rows,a,b):
    n=0
    for r in rows:
        if int(np.argmax(r["probs"][a]))!=int(np.argmax(r["probs"][b])): n+=1
    return n
def delta(rows,a,b):
    m,lo,hi=ci(acc(rows,a)-acc(rows,b)); return f"{m:+5.1f}[{lo:+5.1f},{hi:+5.1f}] f={flips(rows,a,b):<3d}"

print("="*118); print("AVR  (bar = uniform@600 equal-compute; hi = uniform@900 headroom diagnostic)"); print("="*118)
print(f"{'model':15s} {'bench':6s} {'n':>4s} {'bar':>6s} {'hi':>6s} {'AVR':>6s} | {'AVR - bar':>26s} {'headroom hi - bar':>26s}")
QWEN=[("Qwen3-VL-2B","V*",'phase185_tsr_qwen3'),("Qwen2-VL-7B","V*",'phase185_tsr_qwen2'),
      ("Qwen3-VL-8B","V*",'phase192_tsr_q3_8b'),("Qwen2.5-VL-7B","V*",'phase192_tsr_q25_7b'),
      ("Qwen3-VL-2B","HR-4k",'phase195_tsr_hr4k_qwen3'),("Qwen2-VL-7B","HR-4k",'phase195_tsr_hr4k_qwen2'),
      ("Qwen3-VL-2B","HR-8k",'phase195_tsr_hr8k_qwen3')]
for mk,bk,f in QWEN:
    r=load(f)
    if not r: print(f"{mk:15s} {bk:6s}  -- missing --"); continue
    hi="uniform@900" if "uniform@900" in r[0]["probs"] else None
    hv=f"{acc(r,hi).mean()*100:6.1f}" if hi else "   n/a"
    d2=delta(r,hi,"uniform@600") if hi else "n/a"
    print(f"{mk:15s} {bk:6s} {len(r):4d} {acc(r,'uniform@600').mean()*100:6.1f} {hv} "
          f"{acc(r,'tsr900').mean()*100:6.1f} | {delta(r,'tsr900','uniform@600'):>26s} {d2:>26s}")
NAMES={"qwen3_2b":"Qwen3-VL-2B","qwen2_7b":"Qwen2-VL-7B","llava_ov":"LLaVA-OV-7B",
       "llava_next":"LLaVA-NeXT-7B","gemma3_4b":"Gemma-3-4B"}
for f in sorted(os.listdir(f"{D}/data")):
    if not f.startswith("phase213_") or not f.endswith(".jsonl"): continue
    r=load(f[:-6])
    if not r: print(f"{f[:-6].replace('phase213_',''):22s} -- EMPTY (job did not complete) --"); continue
    mk,bk=f[:-6].replace("phase213_","").rsplit("_",1)
    tk=r[0].get("tokens",{}); noladder=tk.get("uniform@lo")==tk.get("uniform@hi")
    note=" NO LADDER (E_lo==E_hi)" if noladder else ""
    if not r[0].get("avr_feasible",True): note+=" AVR-INFEASIBLE"
    print(f"{NAMES.get(mk,mk):15s} {bk:6s} {len(r):4d} {acc(r,'uniform@lo').mean()*100:6.1f} "
          f"{acc(r,'uniform@hi').mean()*100:6.1f} {acc(r,'avr').mean()*100:6.1f} | "
          f"{delta(r,'avr','uniform@lo'):>26s} {delta(r,'uniform@hi','uniform@lo'):>26s}{note}")

print(); print("="*118); print("AVR controls (phase213 only): keep-choice and the transport boundary"); print("="*118)
print(f"{'model':15s} {'bench':6s} | {'AVR - AVR_random':>26s} {'suffix_mask - bar':>26s}")
for f in sorted(os.listdir(f"{D}/data")):
    if not f.startswith("phase213_") or not f.endswith(".jsonl"): continue
    r=load(f[:-6])
    if not r: continue
    mk,bk=f[:-6].replace("phase213_","").rsplit("_",1)
    print(f"{NAMES.get(mk,mk):15s} {bk:6s} | {delta(r,'avr','avr_rand'):>26s} {delta(r,'suffix_mask','uniform@lo'):>26s}")

print(); print("="*118); print("DWA  (bar = uniform@600; block = single depth-block argmax; DWA = 63-feature ridge)"); print("="*118)
print(f"{'model':15s} {'bench':6s} {'n':>4s} {'bar':>6s} {'block':>6s} {'DWA':>6s} | {'DWA - bar':>26s} {'DWA - block':>26s}")
for mk,bk,f,d in [("LLaVA-OV-7B","V*",'phase215_dwaeval_llava_ov','dwa'),
                  ("Qwen3-VL-2B","V*",'phase223_logonly_qwen3','dwa63'),
                  ("Qwen2-VL-7B","V*",'phase223_logonly_qwen2','dwa63')]:
    r=load(f)
    if not r: print(f"{mk:15s} {bk:6s}  -- missing --"); continue
    hb="block" in r[0]["probs"]
    bv=f"{acc(r,'block').mean()*100:6.1f}" if hb else "   n/a"
    d2=delta(r,d,"block") if hb else "n/a"
    print(f"{mk:15s} {bk:6s} {len(r):4d} {acc(r,'bar').mean()*100:6.1f} {bv} {acc(r,d).mean()*100:6.1f} | "
          f"{delta(r,d,'bar'):>26s} {d2:>26s}")
print("\nDWA on LLaVA-NeXT / Gemma-3: map extraction lands at chance (1.14x / 1.07x); the phase214 guard")
print("aborts rather than reporting a number. phase220_layout_probe is testing grid layout as the cause.")
