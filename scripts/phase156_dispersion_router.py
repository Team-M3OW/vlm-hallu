"""
Phase 156: A FREE, QUESTION-BLIND ROUTER FROM THE MAP'S OWN DISPERSION.

WHY
---
SS17A's router is a keyword rule. It works (pooled +9.4 / +6.8 over the equal-compute bar on
V*Bench, both models) but it agrees with V*Bench's own `category` annotation on 100% of items, so on
that benchmark it may BE the annotation; and on HR-Bench its agreement falls to 51.5% and the pooled
contrast goes null. The paper's central weakness -- DPR does not clear the bar pooled -- therefore
rests on a rule that reads the question text and was validated where the text happens to match the
labels.

The phase-163 diagnostic says the signal is already in the attention map: on the RAW gated-max map
the top-1 mass share separates single-object from relational questions at AUROC 0.735 / 0.805
(Qwen3 / Qwen2). SS14T separately found a `top1_frac` routing gate worth +9.6pp [+2.0,+17.7] on
Qwen3-VL that was never tested on a second model because PREREG_DCR_SIZER.md barred substituting it
for the sizer under test. It has never been evaluated AS A ROUTER across models and benchmarks.

This routes on that scalar. No question text, no annotations, no extra forward pass: the statistic is
a function of the localisation map DPR already computes.

COMPOSITION (identical to phase 150, so the numbers are comparable line for line)
    routed item    -> localise@300 + crop@300   (DPR)
    unrouted item  -> uniform@600               (the bar)
    scored against uniform@600 on every item.  Tokens are matched by construction.

ROUTERS
    R0  none                 : DPR everywhere                               (Table 1's pooled row)
    R1  keyword rule         : SS17A's free text rule                       (the incumbent router)
    R3  dispersion (THIS)    : route iff top1_frac >= tau, tau fitted OUT-OF-FOLD on training folds
    R2  oracle category      : the benchmark's own labels                   (the ceiling)

PRE-REGISTERED, before the run
    PRIMARY   R3 pooled - bar, CI clear of zero on BOTH models on V*Bench -> a question-blind router
              carries the pooled claim and the keyword rule can be retired to a robustness row.
    SECONDARY R3 - R1 (does it match the text rule?) and R3's AUROC against the category label.
    STATISTIC top1_frac of the ring-masked map is PRIMARY; peak_over_median, top5_frac, entropy_norm
              are reported as secondary and are NOT eligible to replace it if it fails.
    MAP       gated-max (L17-20 Qwen3 / L19-22 Qwen2), the label-free layer set of phases 95/141,
              where the AUROC was measured. The deployed block-mean map is reported alongside.
    FOLDS     GroupKFold(5) grouped by ITEM (V*Bench) / INSTANCE (HR-Bench, whose 4 CircularEval
              cycles share an image, a question and therefore a map). tau is chosen on training
              folds only, by maximising routed accuracy there.
    NULL      if R3 does not clear, the keyword rule stays the paper's router and SS17A's caveat
              stands as written. That outcome is reported, not re-shopped over the secondaries.

COVERAGE OF THIS RUN
    V*Bench Qwen3 / Qwen2  : maps on disk (phase30c / phase74), outcomes on disk (phase78 / phase97m)
    HR-Bench Qwen3         : dispersion scalars on disk (phase46, BLOCK-MEAN map only -- the gated-max
                             arm needs a re-extraction), outcomes phase72c
    HR-Bench Qwen2         : NOT RUNNABLE -- no dispersion scalars on disk. Queued for the GPU.

CPU ONLY, and deliberately streamed: the map files are read one line at a time and reduced to six
scalars per item, so nothing larger than a single item's 28 x n map is ever resident.
"""
import json, os, re, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
REL = re.compile(r"\b(left|right|above|below|next to|between|behind|in front|closer|farther|further|"
                 r"top of|bottom of|beside|near|nearer|far from|under|over|side of|adjacent|opposite|"
                 r"facing|toward)\b", re.I)
GATE = {"Qwen3-VL": list(range(17, 21)), "Qwen2-VL": list(range(19, 23))}
BLOCK = list(range(16, 27))
STATS = ["top1_frac", "peak_over_median", "top5_frac", "entropy_norm"]


def dispersion(vec):
    """Six scalars from one ring-masked, L1-normalised map. Input is already ring-masked."""
    v = np.asarray(vec, float)
    v = v / max(v.sum(), 1e-12)
    s = np.sort(v)[::-1]
    p = v[v > 0]
    ent = float(-(p * np.log(p)).sum())
    return {"peak": float(s[0]),
            "peak_over_median": float(s[0] / max(np.median(v), 1e-12)),
            "top1_frac": float(s[0]),
            "top5_frac": float(s[:5].sum()),
            "entropy": ent,
            "entropy_norm": float(ent / max(np.log(len(v)), 1e-12))}


def stream_maps(path, model):
    """Yield (question_id_full, gated-max stats, block-mean stats). One item resident at a time."""
    gate = GATE[model]
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            gh, gw = r["grid"]
            n = r["n_img_tokens"]
            A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(28)])
            A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
            rm = np.zeros((gh, gw), bool)
            if gh > 2 and gw > 2:
                rm[1:-1, 1:-1] = True
            else:
                rm[:] = True
            rm = rm.ravel()[:n]
            yield r["question_id_full"], dispersion(A[gate].max(0)[rm]), dispersion(A[BLOCK].mean(0)[rm])
            del A, r


def oof_threshold(stat, acc_dpr, acc_bar, groups, seed=156):
    """Route iff stat >= tau. tau chosen on TRAINING folds only, by maximising routed accuracy there.
    Returns the out-of-fold route mask."""
    route = np.zeros(len(stat), bool)
    gs = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(gs))}
    Gp = np.vectorize(perm.get)(groups)
    for tr, te in GroupKFold(5).split(stat.reshape(-1, 1), acc_dpr, Gp):
        grid = np.unique(np.quantile(stat[tr], np.linspace(0, 1, 41)))
        best, btau = -1.0, grid[0]
        for tau in grid:
            m = stat[tr] >= tau
            a = np.where(m, acc_dpr[tr], acc_bar[tr]).mean()
            if a > best:
                best, btau = a, tau
        route[te] = stat[te] >= btau
    return route


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(d, rng, B=8000):
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean() * 100, float(np.percentile(b, 2.5)) * 100, float(np.percentile(b, 97.5)) * 100


def load_text():
    os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import glob, pyarrow.parquet as pq
    from datasets import load_dataset
    vs = load_dataset("craigwu/vstar_bench")["test"]
    # read the arrow columns directly: iterating the dataset would decode every high-res image
    tb = vs.data
    cat, qid, txt = tb.column("category"), tb.column("question_id"), tb.column("text")
    vtext = {f"{cat[i].as_py()}/{qid[i].as_py()}": txt[i].as_py().split("\n")[0]
             for i in range(vs.num_rows)}
    f = glob.glob(f"{os.environ['HF_HUB_CACHE']}/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet")[0]
    t = pq.read_table(f, columns=["index", "question"])
    htext = {int(t.column("index")[i].as_py()): t.column("question")[i].as_py() for i in range(t.num_rows)}
    return vtext, htext


def report(name, rows, stats, harm, single_cat, idk, groupk, textmap, mapname, out):
    n = len(rows)
    bar = np.array([int(np.argmax(r["probs"]["uniform@600"]) == r["label"]) for r in rows], float)
    dpr = np.array([int(np.argmax(r["probs"][harm]) == r["label"]) for r in rows], float)
    groups = np.array([r[groupk] for r in rows])
    is_single = np.array([r["category"] == single_cat for r in rows], bool)
    q = [textmap.get(r[idk], "") for r in rows]
    rng = np.random.default_rng(156)

    routers = {"R0 always DPR": np.ones(n, bool),
               "R1 keyword rule": np.array([not REL.search(s) for s in q])}
    S = {k: np.array([s[k] for s in stats], float) for k in STATS}
    routers[f"R3 top1_frac OOF [{mapname}]"] = oof_threshold(S["top1_frac"], dpr, bar, groups)
    routers["R2 oracle category"] = is_single

    print(f"\n{name}   n={n}   arm={harm}   map={mapname}")
    print(f"    {'router':>30} {'routed':>7} {'pooled':>7} {'bar':>6}   pooled - bar")
    res = {}
    for rn, route in routers.items():
        acc = np.where(route, dpr, bar)
        m, lo, hi = ci(acc - bar, rng)
        flag = "CLEARS" if lo > 0 else ""
        print(f"    {rn:>30} {100*route.mean():6.1f}% {100*acc.mean():6.1f}% {100*bar.mean():5.1f}%   "
              f"{m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {flag}")
        res[rn] = {"routed": float(route.mean()), "pooled": float(acc.mean()),
                   "delta": m, "lo": lo, "hi": hi}
    r3 = routers[f"R3 top1_frac OOF [{mapname}]"]
    r1, r2 = routers["R1 keyword rule"], routers["R2 oracle category"]
    d, lo, hi = ci((np.where(r3, dpr, bar) - np.where(r1, dpr, bar)), rng)
    print(f"    R3 - R1 {d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]   "
          f"agreement with oracle category: R3 {100*np.mean(r3==r2):.1f}%  R1 {100*np.mean(r1==r2):.1f}%")
    print("    AUROC of each statistic against the single-object label (chance 0.500):  " +
          "  ".join(f"{k} {auroc(S[k], is_single):.3f}" for k in STATS))
    res["_auroc"] = {k: auroc(S[k], is_single) for k in STATS}
    res["_r3_minus_r1"] = {"delta": d, "lo": lo, "hi": hi}
    out[f"{name} [{mapname}]"] = res


def main():
    vtext, htext = load_text()
    out = {}
    VS = [("Qwen3-VL", f"{D}/phase30c_attn_maps_all.jsonl", f"{D}/phase78_w_sweep.jsonl", "head@0.25"),
          ("Qwen2-VL", f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", f"{D}/phase97m_merged_qwen2vl.jsonl", "head@0.25")]
    for model, mappath, outpath, harm in VS:
        gated, blockm = {}, {}
        for qid, g, b in stream_maps(mappath, model):
            gated[qid], blockm[qid] = g, b
        rows = [json.loads(l) for l in open(outpath)]
        rows = [r for r in rows if r["question_id_full"] in gated]
        rows = [dict(r, label=int(r["label"]) if not isinstance(r["label"], int) else r["label"]) for r in rows]
        for mapname, table in [("gated-max", gated), ("block-mean", blockm)]:
            report(f"{model}  V*Bench", rows, [table[r["question_id_full"]] for r in rows], harm,
                   "direct_attributes", "question_id_full", "question_id_full", vtext, mapname, out)

    # ---- HR-Bench, Qwen3 only: block-mean scalars precomputed in phase 46, keyed by instance
    p46 = {r["instance"]: r for r in (json.loads(l) for l in open(f"{D}/phase46_hrbench_peak.jsonl"))}
    rows = [json.loads(l) for l in open(f"{D}/phase72c_hrbench_fullprompt.jsonl")]
    rows = [r for r in rows if r["instance"] in p46]
    for r in rows:
        r["question_id_full"] = r["row_id"]
    report("Qwen3-VL  HR-Bench", rows, [p46[r["instance"]] for r in rows], "head@0.15",
           "single", "row_id", "instance", htext, "block-mean (phase46)", out)
    print("\n  Qwen2-VL HR-Bench: NOT RUNNABLE -- no dispersion scalars on disk; needs one "
          "localisation pass per instance. Queued for the GPU.")

    json.dump(out, open(f"{D}/phase156_dispersion_router.json", "w"), indent=1)
    print(f"\nwrote {D}/phase156_dispersion_router.json")


if __name__ == "__main__":
    main()
