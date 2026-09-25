"""AVR-audio results, with the 750-token cap handled explicitly.

Qwen2-Audio truncates at 30s / 750 tokens. A 29s clip stretched x2 is 58s -> truncated, so its
REALISED multiplier is ~1.03, not 2.0. Averaging over capped and uncapped items would dilute any
headroom effect toward zero and invite the wrong conclusion. We therefore report the realised
multiplier per arm and split on whether the stretch was actually delivered.
"""
import json,sys,numpy as np,collections
rows=[json.loads(l) for l in open('data/phase243_audio_avr.jsonl')]
rng=np.random.default_rng(0)
def corr(r,a): return 1.0 if int(np.argmax(r['probs'][a]))==r['gold'] else 0.0
def acc(rs,a): return 100*np.mean([corr(r,a) for r in rs]) if rs else float('nan')
def ci(rs,a,b):
    if len(rs)<15: return '      n/a       '
    d=np.array([corr(r,a)-corr(r,b) for r in rs])
    m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    lo,hi=np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
    return f'{d.mean()*100:+5.1f}[{lo:+5.1f},{hi:+5.1f}]'+('*' if (lo>0 or hi<0) else ' ')
print(f'n={len(rows)}  tasks={dict(collections.Counter(r["task"] for r in rows))}')
# realised multiplier: tokens of the stretched arm over the bar's
rm2=np.array([r['tokens']['hi@2.0']/r['tokens']['bar'] for r in rows])
rm1=np.array([r['tokens']['hi@1.29']/r['tokens']['bar'] for r in rows])
print(f'REALISED multiplier  hi@1.29: mean {rm1.mean():.2f} (min {rm1.min():.2f})   hi@2.0: mean {rm2.mean():.2f} (min {rm2.min():.2f})')
FULL=[r for r in rows if r['tokens']['hi@2.0']/r['tokens']['bar']>=1.9]
CAP =[r for r in rows if r['tokens']['hi@2.0']/r['tokens']['bar']< 1.9]
print(f'  stretch DELIVERED (>=1.9x): n={len(FULL)}     CAPPED (<1.9x): n={len(CAP)}')
tl={k:np.mean([r['tl'][k] for r in rows]) for k in rows[0]['tl']}; b=tl['bar']
print()
print(f"{'arm':14s}{'acc(all)':>10s}{'acc(full)':>11s}{'acc(cap)':>10s}{'TL/bar':>9s}")
for a in ('bar','hi@1.29','hi@2.0','avr@1.29','avr@2.0','avr_rand@2.0','early@2.0'):
    print(f'  {a:12s}{acc(rows,a):10.1f}{acc(FULL,a):11.1f}{acc(CAP,a):10.1f}{tl[a]/b:9.2f}x')
print()
print('PRE-REGISTERED TESTS                    all items          stretch delivered')
tests=[('H4a headroom  hi@2.0 - bar','hi@2.0','bar'),
       ('    headroom  hi@1.29 - bar','hi@1.29','bar'),
       ('H4b identity  avr@1.29 - bar (=cost)','avr@1.29','bar'),
       ('    identity  avr@2.0 - bar','avr@2.0','bar'),
       ('H4c keepchoice avr@2 - rand@2','avr@2.0','avr_rand@2.0'),
       ('H4d cut depth early@2 - avr@2','early@2.0','avr@2.0')]
for nm,a,bb in tests:
    print(f'  {nm:38s}{ci(rows,a,bb):>18s}{ci(FULL,a,bb):>20s}')
print()
print('by task (avr@1.29 - bar, the equal-compute arm):')
for t in sorted({r['task'] for r in rows}):
    s=[r for r in rows if r['task']==t]
    print(f'   {t:7s} n={len(s):3d}  {ci(s,"avr@1.29","bar")}   headroom {ci(s,"hi@2.0","bar")}')
