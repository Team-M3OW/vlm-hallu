"""Incremental read of audio G1 (phase246) while it runs. The gate is binary -- headroom > 0 or
not -- so it usually resolves long before the full 1219-item split finishes at ~60s/item."""
import json,sys,collections,numpy as np
rows=[json.loads(l) for l in open("data/phase246_longaudio_g1.jsonl") if l.strip()]
if len(rows)<5: print(f"only {len(rows)} items yet"); sys.exit()
rng=np.random.default_rng(0)
cor=lambda r,a: float(int(np.argmax(r["probs"][a]))==r["gold"])
def ci(a,b):
    d=np.array([cor(r,a)-cor(r,b) for r in rows]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
    lo,hi=float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
    return d.mean()*100,lo,hi,("*" if (lo>0 or hi<0) else " ")
print(f"=== AUDIO G1 (running)  n={len(rows)}  mean dur {np.mean([r['dur'] for r in rows]):.0f}s  "
      f"MAXDUR-truncated {100*np.mean([r['maxdur_truncated'] for r in rows]):.0f}% ===")
print("  len_type",dict(collections.Counter(r['len'] for r in rows)),"| cats",dict(collections.Counter(r['cat'] for r in rows)))
for a in ("hi","bar_latent","bar_voc","roundtrip","bar_trunc"):
    print(f"  {a:11s} acc {100*np.mean([cor(r,a) for r in rows]):5.1f}  audio tokens {np.mean([r['ntok'][a] for r in rows]):6.0f}")
ch=100*np.mean([1.0/r['nch'] for r in rows]); hi=100*np.mean([cor(r,'hi') for r in rows])
d=np.array([cor(r,'hi')-1.0/r['nch'] for r in rows]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
print(f"\n  SANITY hi {hi:.1f} vs chance {ch:.1f}: {d.mean()*100:+.1f} [{np.percentile(m,2.5):+.1f},{np.percentile(m,97.5):+.1f}]"
      f"  {'ok' if np.percentile(m,2.5)>0 else 'AT FLOOR -> headroom uninterpretable'}")
accs={a:100*np.mean([cor(r,a) for r in rows]) for a in ("bar_latent","bar_voc")}
st=max(accs,key=accs.get); print(f"  strongest honest bar: {st} ({accs[st]:.1f})   [trunc {100*np.mean([cor(r,'bar_trunc') for r in rows]):.1f} is the straw bar, diagnostic only]")
m_,lo,hi_,s=ci("hi",st); print(f"\n  HEADROOM  hi - {st:10s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]{s}  {'PASS' if lo>0 else ('FAIL' if hi_<0 else 'undecided')}")
for a,b,lab in (("hi","roundtrip","vocoder damage only (hi - roundtrip)"),
                ("bar_voc","bar_latent","vocoder bar vs latent bar"),
                (st,"bar_trunc","honest bar - straw bar")):
    m_,lo,hi_,s=ci(a,b); print(f"  {lab:38s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]{s}")
