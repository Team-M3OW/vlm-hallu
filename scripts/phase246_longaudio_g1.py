"""
Phase 246 G1 -- IS THERE HEADROOM IN LONG AUDIO?  (Route B gate 1)
WHY. On 10s MMAU clips AVR-audio had no gain and the headroom identity said exactly why: AVR's gain
IS acc@hi - acc@lo (vision r=0.966, slope 1.01), and in audio it was NEGATIVE (hi@2.0-bar = -3.8).
Time-stretch mints tokens without adding acoustic detail -- a token knob, not a resolution knob.
Long audio is the candidate REAL knob: past the encoder window the model MUST compress, so native
encoding recovers information the compressed version destroyed.
G0 (phase245) settled who can do this: Qwen2-Audio PLATEAUS at 750 tok (30s window) -> out.
Qwen2.5-Omni is LINEAR, 25 tok/s to 6000 tok at 240s -> in. ONE MODEL for this modality (user's
scope decision) -- so Route B yields a demonstration, not a multi-model claim.

THE BAR IS THE HARD PART. Three ways to spend only B=750 tokens on a long clip:
  bar_trunc  first 30s only. Destroys COVERAGE. This is the SS81 InternVL straw bar and a pipeline
             fault. DIAGNOSTIC ONLY -- reported so the inflation it would have bought is visible.
  bar_voc    phase-vocoder time-compression of the WHOLE clip to B (pitch preserved, so arms differ
             in temporal detail alone; resample-speedup would shift pitch and confound). RISK: in
             SS.audio a x2 vocoder stretch alone cost ~4 points, so at x4-x6 some of any hi-bar gap
             would be vocoder ARTEFACT, not lost information -- SS81 in a new form.
  bar_latent native encoding, but only every k-th audio token VISIBLE (2D attention mask). Same
             B-token information budget with NO out-of-distribution input. No vocoder artefact.
  HEADROOM IS TAKEN AGAINST THE STRONGEST BAR: hi - max(bar_voc, bar_latent). Must be CI-clear >0.
  roundtrip  compress xr then stretch back x1/r -> native token count, vocoder damage only.
             hi - roundtrip isolates vocoder damage from token count.
Self-checks: ntok(hi) ~ 25*dur; ntok(bar) ~ B; hi must be CI-clear ABOVE CHANCE (headroom measured
on a floor is uninterpretable). Missing audio is a HARD ERROR, never a silent skip (the stale-JSON
lesson); every skip is counted and reported by reason.
NO per-layer masking here, so fast attention is fine; only the boundary re-measure needs eager.
"""
import os, sys, json, time, zipfile, io, collections, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.pop("HF_TOKEN", None)
import librosa, soundfile as sf, pandas as pd
from huggingface_hub import hf_hub_download

MID="Qwen/Qwen2.5-Omni-7B"; SR=16000; B=750
MAXDUR=int(os.environ.get("MAXDUR","180"))
CHANCE=None

def load_split():
    df=pd.read_parquet(hf_hub_download("gamma-lab-umd/MMAU-Pro","test.parquet",repo_type="dataset"))
    df=df[df['length_type'].isin(['long','ultra-long','ultra_long'])]      # the experimental CONDITION
    df=df[df['audio_path'].map(lambda a: a is not None and len(a)==1)]
    # MMAU-Pro is ORDERED BY CATEGORY: a prefix would be one task. This is the CV-Bench/MMAU trap
    # (ported the measurement but not the lesson, twice). Shuffle, as the prereg states.
    return df.sample(frac=1.0, random_state=246)

def main(n=0):
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration as M
    # hf_hub_download hung repeatedly on this 47.5GB blob (26 open sockets, zero growth) and
    # hf_transfer stalled outright, so the zip is fetched by scripts/dl_mmaupro_curl.sh instead.
    zp=os.environ.get("MMAUPRO_ZIP","/media/kavinder/hdd2/mmau_pro/data.zip")
    if not os.path.exists(zp):
        zp=hf_hub_download("gamma-lab-umd/MMAU-Pro","data.zip",repo_type="dataset")
    zf=zipfile.ZipFile(zp); names=set(zf.namelist())
    print(f"zip members: {len(names)}  sample: {list(sorted(names))[:3]}",flush=True)
    pr=AutoProcessor.from_pretrained(MID)
    # EAGER + layer-hook masking for bar_latent. The obvious trick -- zeroing entries in the 2D
    # attention_mask -- CORRUPTS mrope models: Qwen3-VL derives its 3D rope positions from
    # attention_mask, so dropping entries changed the position count and the forward died
    # ([3,955] vs [3,269]). Qwen2.5-Omni's thinker uses the same get_rope_index, so the audio run
    # would have hit the same fault -- or, worse, silently compacted positions.
    model=M.from_pretrained(MID, dtype=torch.bfloat16, device_map={"":0}, attn_implementation="eager").eval()
    tok=pr.tokenizer
    layers=model.model.layers if hasattr(model,"model") else model.layers
    st={"cols":None}
    def mk():
        def pre(mod,args,kwargs):
            if st["cols"] is None: return None
            am=kwargs.get("attention_mask")
            if am is None: am=args[1] if len(args)>1 else None
            if am is None or am.dtype==torch.bool:
                raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
            am=am.clone(); am[:,:,:,st["cols"]]=torch.finfo(am.dtype).min
            kwargs["attention_mask"]=am; return (args,kwargs)
        return pre
    for _l in layers: _l.self_attn.register_forward_pre_hook(mk(),with_kwargs=True)
    aid=getattr(model.config,"audio_token_index",None) or tok.convert_tokens_to_ids("<|AUDIO|>")

    def build(wav,q,ch):
        L="ABCDEFGHIJ"[:len(ch)]
        text=q+"\n"+"\n".join(f"{l}. {c}" for l,c in zip(L,ch))+"\nAnswer with the option's letter only."
        chat=pr.apply_chat_template([{"role":"user","content":[{"type":"audio","audio_url":"x"},
              {"type":"text","text":text}]}], tokenize=False, add_generation_prompt=True)
        for kw in ("audio","audios"):
            try: return pr(text=chat, **{kw:[wav]}, sampling_rate=SR, return_tensors="pt"), L
            except TypeError: continue
        raise RuntimeError("processor audio kwarg not found")

    def fwd(inp, L, keep_every=0):
        """keep_every>1 hides all but every k-th AUDIO token at every layer. The stride is the right
        construction here because audio is 1-D in time (unlike video, where a flat token stride
        silently becomes a SPATIAL bar); only the masking mechanism had to change."""
        ids=inp["input_ids"][0]; pos=(ids==aid).nonzero().flatten()
        nt=len(pos)
        inp={k:(v.to(model.device) if hasattr(v,"to") else v) for k,v in inp.items()}
        if keep_every>1:
            drop=[int(p) for i,p in enumerate(pos) if i%keep_every]
            st["cols"]=torch.tensor(drop,device=model.device); nt=nt-len(drop)
        else: st["cols"]=None
        try:
            with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        finally: st["cols"]=None
        ltr=[tok.encode(x,add_special_tokens=False)[0] for x in L]
        return torch.softmax(lg[ltr],-1).tolist(), nt

    df=load_split(); items=df
    skips=collections.Counter(); out=[]; t0=time.time(); checked=False
    fo=open("data/phase246_longaudio_g1.jsonl","w")
    for _,r in items.iterrows():
        if n and len(out)>=n: break          # n = items PROCESSED, not rows scanned
        ch=list(r['choices']) if r['choices'] is not None else []
        if len(ch)<2: skips["open_ended"]+=1; continue
        if r['answer'] not in ch: skips["answer_not_in_choices"]+=1; continue
        mem=r['audio_path'][0]
        if mem not in names:
            alt=[x for x in (mem, "data/"+os.path.basename(mem), os.path.basename(mem)) if x in names]
            if not alt: raise FileNotFoundError(f"audio member missing from zip: {mem}")
            mem=alt[0]
        wav,_=librosa.load(io.BytesIO(zf.read(mem)), sr=SR, mono=True)
        if len(wav)/SR < 35: skips["under_35s"]+=1; continue            # no knob below the window
        trunc=len(wav)/SR > MAXDUR
        wav=wav[:int(MAXDUR*SR)]; dur=len(wav)/SR
        gold=ch.index(r['answer'])
        rec={"id":str(r['id']),"cat":str(r['category']),"len":str(r['length_type']),"dur":dur,
             "nch":len(ch),"gold":gold,"maxdur_truncated":bool(trunc),"probs":{},"ntok":{}}
        inp,L=build(wav,r['question'],ch)
        p,nt=fwd(inp,L); rec["probs"]["hi"]=p; rec["ntok"]["hi"]=nt
        rate=max(1.0, nt/B); rec["rate"]=rate
        # bar_latent: same encode, only every k-th audio token visible
        k=max(1,int(round(nt/B)))
        p,n2=fwd(inp,L,keep_every=k); rec["probs"]["bar_latent"]=p; rec["ntok"]["bar_latent"]=n2
        # bar_voc: whole clip compressed to budget
        cw=librosa.effects.time_stretch(wav,rate=rate) if rate>1.01 else wav
        i2,_=build(cw,r['question'],ch); p,n3=fwd(i2,L); rec["probs"]["bar_voc"]=p; rec["ntok"]["bar_voc"]=n3
        # roundtrip: vocoder damage at NATIVE token count
        rw=librosa.effects.time_stretch(cw,rate=1.0/rate) if rate>1.01 else wav
        i3,_=build(rw[:len(wav)],r['question'],ch); p,n4=fwd(i3,L); rec["probs"]["roundtrip"]=p; rec["ntok"]["roundtrip"]=n4
        # bar_trunc: DIAGNOSTIC straw bar
        i4,_=build(wav[:30*SR],r['question'],ch); p,n5=fwd(i4,L); rec["probs"]["bar_trunc"]=p; rec["ntok"]["bar_trunc"]=n5
        if not checked:
            exp=25*dur
            print(f"  [selfcheck] dur={dur:.0f}s ntok hi={nt} (expect ~{exp:.0f}, {nt/exp:.2f}x) "
                  f"latent={n2} voc={n3} trunc={n5} (budget {B})",flush=True)
            assert 0.7<=nt/exp<=1.3, f"hi token count {nt} != ~25*dur {exp}"
            assert abs(n3-B)/B<=0.35, f"bar_voc {n3} far from budget {B}"
            checked=True
        out.append(rec); fo.write(json.dumps(rec)+"\n"); fo.flush()
        if len(out)%25==0:
            a=lambda A:100*np.mean([int(np.argmax(x["probs"][A]))==x["gold"] for x in out])
            print(f"  [{len(out)}] {(time.time()-t0)/len(out):.1f}s/it dur~{np.mean([x['dur'] for x in out]):.0f}s | "
                  f"hi={a('hi'):.1f} lat={a('bar_latent'):.1f} voc={a('bar_voc'):.1f} rt={a('roundtrip'):.1f} tr={a('bar_trunc'):.1f}",flush=True)
    fo.close(); json.dump(out,open("data/phase246_longaudio_g1.json","w"))
    report(out, skips)

def report(out, skips=None):
    rng=np.random.default_rng(0)
    cor=lambda r,a: float(int(np.argmax(r["probs"][a]))==r["gold"])
    def ci(a,b):
        d=np.array([cor(r,a)-cor(r,b) for r in out]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
        return d.mean()*100, float(np.percentile(m,2.5)), float(np.percentile(m,97.5))
    print(f"\n=== G1  n={len(out)}  mean dur {np.mean([r['dur'] for r in out]):.0f}s  "
          f"MAXDUR-truncated {100*np.mean([r['maxdur_truncated'] for r in out]):.0f}% ===")
    if skips: print("  skips:",dict(skips))
    print("  cats:",dict(collections.Counter(r['cat'] for r in out)))
    for a in ("hi","bar_latent","bar_voc","roundtrip","bar_trunc"):
        print(f"  {a:11s} acc {100*np.mean([cor(r,a) for r in out]):5.1f}   audio tokens {np.mean([r['ntok'][a] for r in out]):6.0f}")
    ch=100*np.mean([1.0/r['nch'] for r in out])
    hi=100*np.mean([cor(r,'hi') for r in out])
    d=np.array([cor(r,'hi')-1.0/r['nch'] for r in out]); m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
    print(f"\n  SANITY hi vs chance: {hi:.1f} vs {ch:.1f}  diff {d.mean()*100:+.1f} [{np.percentile(m,2.5):+.1f},{np.percentile(m,97.5):+.1f}]"
          f"  {'ok' if np.percentile(m,2.5)>0 else 'AT FLOOR -> headroom uninterpretable'}")
    accs={a:100*np.mean([cor(r,a) for r in out]) for a in ("bar_latent","bar_voc")}
    strongest=max(accs,key=accs.get)
    print(f"  strongest honest bar: {strongest} ({accs[strongest]:.1f})")
    m_,lo,hi_=ci("hi",strongest)
    print(f"\n  HEADROOM   hi - {strongest:10s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]   {'PASS -> G2' if lo>0 else 'FAIL -> stop (as on MMAU)'}")
    for a,b,lab in (("hi","roundtrip","vocoder damage only (hi - roundtrip)"),
                    ("bar_voc","bar_latent","vocoder vs latent bar"),
                    (strongest,"bar_trunc","honest bar - straw bar (inflation avoided)")):
        m_,lo,hi_=ci(a,b); print(f"  {lab:42s} = {m_:+.1f} [{lo:+.1f},{hi_:+.1f}]")
    print("\n  by length_type:")
    for lt in sorted({r['len'] for r in out}):
        s=[r for r in out if r['len']==lt]
        print(f"    {lt:11s} n={len(s):4d}  hi {100*np.mean([cor(r,'hi') for r in s]):5.1f}  {strongest} {100*np.mean([cor(r,strongest) for r in s]):5.1f}")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="report":
        report(json.load(open("data/phase246_longaudio_g1.json")))
    else: main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
