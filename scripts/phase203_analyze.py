import json, sys, numpy as np

PATCH_L = [0, 4, 8, 12, 16, 20, 24]


def load(w):
    return [json.loads(l) for l in open(f"data/phase203_patchlens_{w}.jsonl")]


def boot(vals, seed, B=8000):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def paired(a, b, seed, B=8000):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


for w in ["qwen3", "qwen2"]:
    rows = load(w)
    n = len(rows)
    print(f"\n================ {w}  n={n} ================")
    for c in ["direct_attributes", "relative_position"]:
        m = [r for r in rows if r["category"] == c]
        if not m: continue
        km = np.mean([r["patch"]["img_L16"]["kl"] for r in m])
        fm = 100 * np.mean([r["patch"]["img_L16"]["flip"] for r in m])
        kd = np.mean([r["patch"]["img_L8"]["kl"] for r in m])
        lens = [np.mean([r["lens"][f"L{l}"]["p_correct"] for r in m]) for l in range(len(m[0]["lens"]))]
        onset = next((l for l in range(len(lens)) if lens[l] > np.mean(lens[:16]) + 0.10), None)
        print(f"  stratum {c:18s} n={len(m):3d}: img_L8 KL {kd:.3f} | img_L16 KL {km:.4f} flips {fm:4.1f}% | lens onset L{onset}")
    print("lens last-layer check (must be ~0): max %.4f" % max(r["lens_lastlayer_maxdiff"] for r in rows))
    print("base acc %.1f%%" % (100 * np.mean([int(np.argmax(r["base"])) == r["label"] for r in rows])))

    print("\nE1 representation patching -- KL at output and answer flips")
    kl = {}
    for k in [f"img_L{L}" for L in PATCH_L] + ["txt_L4", "txt_L20"]:
        kls = [r["patch"][k]["kl"] for r in rows]
        fl = [r["patch"][k]["flip"] for r in rows]
        km, klo, khi = boot(kls, 203)
        fm, flo, fhi = boot(fl, 203)
        kl[k] = kls
        tag = ""
        if k == "img_L4":
            tag = "  <- C1 sanity: must be LARGE"
        if k == "txt_L20":
            tag = "  <- C2 sanity: must be LARGE"
        print(f"  {k:8s} KL {km:7.4f} [{klo:7.4f},{khi:7.4f}]   flips {100*fm:5.1f}% [{100*flo:5.1f},{100*fhi:5.1f}]{tag}")

    print("\nE1 pre-registered checks")
    c1 = kl["img_L4"]
    c2 = kl["txt_L20"]
    print(f"  C1 img_L4 large:  KL {np.mean(c1):.3f}, flips {100*np.mean([r['patch']['img_L4']['flip'] for r in rows]):.1f}%"
          + ("  PASS" if np.mean(c1) > 0.1 else "  FAIL"))
    print(f"  C2 txt_L20 large: KL {np.mean(c2):.3f}, flips {100*np.mean([r['patch']['txt_L20']['flip'] for r in rows]):.1f}%"
          + ("  PASS" if np.mean(c2) > 0.1 else "  FAIL"))
    late = kl["img_L16"] + kl["img_L20"] + kl["img_L24"]
    print(f"  MAIN img L>=16 inert: mean KL {np.mean(late):.4f}; "
          f"flips {100*np.mean([r['patch'][k]['flip'] for r in rows for k in ('img_L16','img_L20','img_L24')]):.1f}%")
    dm, dlo, dhi = paired(kl["img_L16"], kl["img_L4"], 203)
    print(f"  paired KL(L16 - L4): {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}]")

    print("\nE3 logit lens at the answer position")
    NL = len(rows[0]["lens"])
    pc = [np.mean([r["lens"][f"L{l}"]["p_correct"] for r in rows]) for l in range(NL)]
    am = [100 * np.mean([r["lens"][f"L{l}"]["argmax"] == r["label"] for r in rows]) for l in range(NL)]
    print("  p_correct by L: " + " ".join(f"{v:.2f}" for v in pc))
    print("  acc%% by L:     " + " ".join(f"{v:.0f}" for v in am))
    pre = np.mean(pc[:16])
    onset = next(l for l in range(NL) if pc[l] > pre + 0.10)
    print(f"  pre-boundary mean p_correct (L0-15) {pre:.3f}; onset of decodability L{onset} (first layer > pre+0.10)")
    print(f"  last-layer lens acc {am[-1]:.0f}% vs base acc {100*np.mean([int(np.argmax(r['base']))==r['label'] for r in rows]):.0f}%")

    # agreement between end-task correctness and lens decodability, late band vs transport band
    late_acc = np.mean([am[l] for l in range(onset, NL)])
    print(f"  mean lens acc from onset to end: {late_acc:.1f}%")
