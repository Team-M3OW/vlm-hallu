"""
Phase 236b: replay the map-only policy on every (model, benchmark) cell that has both the phase-225
arm outcomes and the disp dump.

Policy: tau = global median of disp (label-free, transductive); disp > tau -> DWA, else -> AVR where
the checkpoint has a resolution ladder, otherwise the equal-compute bar (recorded as fallback=bar).
Reports pooled and per-stratum deltas vs the bar, with always-DWA / always-AVR / random-route controls.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(236)
MODELS = [("qwen3_2b", "Qwen3-VL-2B"), ("qwen2_7b", "Qwen2-VL-7B"), ("llava_ov", "LLaVA-OV-7B"), ("internvl3_8b", "InternVL3-8B")]
BENCH = [("vstar", "V*"), ("hr4k", "HR-4K"), ("cvbench", "CV-Bench"), ("realworldqa", "RWQA")]


def ok(rec, a):
    if a in rec.get("probs", {}):
        return 1.0 if int(np.argmax(rec["probs"][a])) == int(rec["gold"]) else 0.0
    if a in rec.get("preds", {}):
        p = rec["preds"][a]; gs = rec["gold"] if isinstance(rec["gold"], (list, tuple)) else [rec["gold"]]
        return 1.0 if any(B.norm(g) and B.norm(g) == B.norm(p) for g in gs) else 0.0
    return None


def boot(d, Bn=8000):
    d = np.asarray(d, float); n = len(d)
    if n < 8: return float("nan"), float("nan"), float("nan"), n
    m = d[rng.integers(0, n, (Bn, n))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100, n


TAU_STAR = {}
for mk, mn in MODELS:
    f235 = f"{D}/data/phase235_{mk}_vstar.jsonl"
    if os.path.exists(f235):
        TAU_STAR[mk] = float(np.median([json.loads(l)["disp"] for l in open(f235)]))
print("thresholds calibrated on V* (label-free):", {k: round(v,4) for k,v in TAU_STAR.items()})
print(f"{'model':15s}{'bench':9s}{'n':>6s}{'route%':>7s}{'fallback':>9s}{'policy-bar':>24s}{'DWA-bar':>10s}{'AVR-bar':>10s}{'random':>10s}")
summary = {}
for mk, mn in MODELS:
    for bk, bn in BENCH:
        f25 = f"{D}/data/phase225_{mk}_{bk}.jsonl"
        fd = f"{D}/data/phase236_disp_{mk}_{bk}.jsonl"
        f235 = f"{D}/data/phase235_{mk}_vstar.jsonl"
        if not os.path.exists(f25): continue
        rows = [json.loads(l) for l in open(f25)]
        if os.path.exists(fd):
            disp = {json.loads(l)["qid"]: json.loads(l)["disp"] for l in open(fd)}
        elif bk == "vstar" and os.path.exists(f235):
            disp = {json.loads(l)["qid"]: json.loads(l)["disp"] for l in open(f235)}
        else:
            continue
        rows = [r for r in rows if r["qid"] in disp]
        if len(rows) < 50: continue
        arms = set(a for r in rows for a in list(r.get("probs", {})) + list(r.get("preds", {})))
        alt = "avr" if "avr" in arms else "uniform@lo"
        tau = float(np.median([disp[r["qid"]] for r in rows]))
        # fixed threshold calibrated once on V* (label-free), transferred to this benchmark
        tau_star = TAU_STAR.get(mk)
        polB, routeB = [], []
        pol, bar, dwa, avr, rnd = [], [], [], [], []
        for r in rows:
            if tau_star is not None:
                routeB.append(disp[r["qid"]] > tau_star)
            ch = disp[r["qid"]] > tau
            b = ok(r, "uniform@lo"); d = ok(r, "dwa_t"); a = ok(r, alt)
            if b is None or d is None or a is None: continue
            bar.append(b); dwa.append(d); avr.append(a)
            pol.append(d if ch else a)
            rnd.append(d if rng.random() < 0.5 else a)
        m, lo, hi, n = boot(np.array(pol) - np.array(bar))
        if tau_star is not None:
            polB = np.array([ok(r, "dwa_t") if routeB[i] else ok(r, alt) for i, r in enumerate(rows)])
            mB, loB, hiB, _ = boot(polB - np.array(bar))
            print(f"    [fixed tau from V*={tau_star:.3f}] routed {100*np.mean(routeB):4.0f}%  policyB {mB:+5.1f} [{loB:+5.1f},{hiB:+5.1f}]{' *' if loB>0 or hiB<0 else ''}")
        # policy C: three actions -- crop if disp above the item-set median; else the better of AVR and
        # the bar by two baseline runs (label-free at test time; here the cell's measured accuracies).
        if "uniform@hi" in arms:
            use_avr = (np.mean([ok(r, alt) for r in rows]) >= np.mean([ok(r, "uniform@lo") for r in rows]))
            fb = alt if use_avr else "uniform@lo"
            polC = np.array([ok(r, "dwa_t") if disp[r["qid"]] > tau else ok(r, fb) for r in rows])
            mC, loC, hiC, _ = boot(polC - np.array(bar))
            print(f"    [three-action: crop else {'AVR' if use_avr else 'BAR'}] policyC {mC:+5.1f} [{loC:+5.1f},{hiC:+5.1f}]{' *' if loC>0 or hiC<0 else ''}")
            for st, lab in [("direct_attributes","single"),("relative_position","cross"),("single","single"),("cross","cross")]:
                m2 = np.array([r.get("stratum")==st for r in rows])
                if m2.sum() < 20: continue
                mm, lo2, hi2, _ = boot((polC-np.array(bar))[m2])
                print(f"        {lab}: {mm:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}] n={m2.sum()}")
        dm, _, _, _ = boot(np.array(dwa) - np.array(bar))
        am, _, _, _ = boot(np.array(avr) - np.array(bar))
        rm, _, _, _ = boot(np.array(rnd) - np.array(bar))
        route = 100 * np.mean([disp[r["qid"]] > tau for r in rows])
        ms = f"{m:+5.1f}[{lo:+4.1f},{hi:+4.1f}]" + ("*" if lo > 0 or hi < 0 else " ")
        print(f"{mn:15s}{bn:9s}{n:6d}{route:7.0f}{alt:>9s}{ms:>24s}{dm:+10.1f}{am:+10.1f}{rm:+10.1f}")
        summary[(mn, bn)] = (m, lo, hi, route)
print("\nper-stratum policy - bar (single/cross where the benchmark defines them):")
for mk, mn in MODELS:
    for bk, tn, keys in [("vstar", "V*", ("direct_attributes", "relative_position")), ("hr4k", "HR-4K", ("single", "cross"))]:
        f25 = f"{D}/data/phase225_{mk}_{bk}.jsonl"; fd = f"{D}/data/phase236_disp_{mk}_{bk}.jsonl"
        f235 = f"{D}/data/phase235_{mk}_vstar.jsonl"
        if not os.path.exists(f25): continue
        disp = {}
        if os.path.exists(fd): disp = {json.loads(l)["qid"]: json.loads(l)["disp"] for l in open(fd)}
        elif bk == "vstar" and os.path.exists(f235): disp = {json.loads(l)["qid"]: json.loads(l)["disp"] for l in open(f235)}
        if not disp: continue
        rows = [json.loads(l) for l in open(f25) if json.loads(l)["qid"] in disp]
        arms = set(a for r in rows for a in list(r.get("probs", {})) + list(r.get("preds", {})))
        alt = "avr" if "avr" in arms else "uniform@lo"
        tau = float(np.median([disp[r["qid"]] for r in rows]))
        out = []
        for key in keys:
            sub = [r for r in rows if r.get("stratum") == key]
            d = np.array([(ok(r, "dwa_t") if disp[r["qid"]] > tau else ok(r, alt)) - ok(r, "uniform@lo") for r in sub])
            mm, lo, hi, nn = boot(d)
            out.append(f"{key[:6]}: {mm:+5.1f}[{lo:+5.1f},{hi:+5.1f}] n={nn}")
        print(f"  {mn:15s}{tn:8s}" + "   ".join(out))
