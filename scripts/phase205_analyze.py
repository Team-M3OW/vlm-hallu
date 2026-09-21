import json, os, numpy as np

for w in ["qwen3", "qwen2"]:
    f = f"data/phase205_tsr_probe_{w}.jsonl"
    if not os.path.exists(f):
        print(f"\n===== phase205 {w}: no data yet ====="); continue
    rows = [json.loads(l) for l in open(f)]
    n = len(rows)
    print(f"\n================ phase205 {w}  n={n} ================")
    base = np.array([r["base"] for r in rows]); tsr = np.array([r["tsr"] for r in rows])
    rand = np.array([r["tsr_rand"] for r in rows]); lab = np.array([r["label"] for r in rows])
    acc = lambda P, L=None: 100 * np.mean(P.argmax(1) == (lab if L is None else L))
    print(f"base acc {acc(base):5.1f}%   tsr acc {acc(tsr):5.1f}%   rand acc {acc(rand):5.1f}%")
    print(f"KL(base||tsr)  mean {np.mean([r['kl_tsr'] for r in rows]):.4f}  median {np.median([r['kl_tsr'] for r in rows]):.4f}")
    print(f"KL(base||rand) mean {np.mean([r['kl_rand'] for r in rows]):.4f}  median {np.median([r['kl_rand'] for r in rows]):.4f}")
    print(f"argmax agreement: attn keep {100*np.mean(tsr.argmax(1)==base.argmax(1)):.1f}%   "
          f"random keep {100*np.mean(rand.argmax(1)==base.argmax(1)):.1f}%")
    print("value patching (KL / flips):")
    for k in ("drop_L16", "keep_L16", "drop_L8", "drop_L24"):
        kl = np.mean([r["patch"][k]["kl"] for r in rows]); fl = 100 * np.mean([r["patch"][k]["flip"] for r in rows])
        print(f"   {k:9s} KL {kl:8.4f}   flips {fl:5.1f}%")
    d16 = np.mean([r["patch"]["drop_L16"]["kl"] for r in rows])
    k16 = np.mean([r["patch"]["keep_L16"]["kl"] for r in rows])
    d8 = np.mean([r["patch"]["drop_L8"]["kl"] for r in rows])
    print(f"   P-T1a drop@L16 {d16:.4f} <= 0.01: {'PASS' if d16 <= 0.01 else 'FAIL'}")
    print(f"   P-T1b drop@L8  {d8:.4f} >= 0.10: {'PASS' if d8 >= 0.10 else 'FAIL'}")
    print(f"   P-T1c keep@L16 {k16:.4f} >= 0.10: {'PASS' if k16 >= 0.10 else 'FAIL'}  "
          f"(if FAIL: all boundary values are inert, not only the pruned ones -- stronger premise)")
    kl_tsr = np.mean([r["kl_tsr"] for r in rows]); kl_rand = np.mean([r["kl_rand"] for r in rows])
    ag = 100 * np.mean(tsr.argmax(1) == base.argmax(1))
    print(f"   P-T2 KL<=0.02 {kl_tsr:.4f}: {'PASS' if kl_tsr <= 0.02 else 'FAIL'}; "
          f"agreement>=97% {ag:.1f}: {'PASS' if ag >= 97 else 'FAIL'}; "
          f"|attn-rand| {abs(kl_tsr-kl_rand):.4f} <= 0.01: {'PASS' if abs(kl_tsr-kl_rand) <= 0.01 else 'FAIL'}")
    for c in ("direct_attributes", "relative_position"):
        m = np.array([r["category"] == c for r in rows])
        if m.sum(): print(f"   stratum {c:18s} n={m.sum():3d}  base {acc(base[m], lab[m]):5.1f}  tsr {acc(tsr[m], lab[m]):5.1f}  rand {acc(rand[m], lab[m]):5.1f}")
