"""
Phase 245 G0 -- DOES ANY ALM NATIVELY ENCODE LONG AUDIO?
Route B gate 0. AVR-audio needs a budget knob that adds INFORMATION, not tokens. On 10s MMAU
clips there is none: time-stretch mints tokens without adding acoustic detail, headroom came out
NEGATIVE (hi@2.0-bar = -3.8) and the identity correctly predicted zero gain (SS.audio).
Long audio is the candidate knob: past the encoder window the model MUST compress, so encoding
more of the clip recovers information the compressed version destroyed -- the audio analogue of
image upscaling, not of image stretching.
THIS GATE is instrumentation only: tokens vs duration, processors only, no LM weights, no GPU.
    PLATEAU  (tokens flat past ~30s)  -> the model TRUNCATES. Out: a truncated bar is the SS81
                                        straw bar (pipeline fault), which is vetoed as a framing.
    LINEAR   (tokens ~ duration)      -> native long encoding. In.
Also records tokens/sec so the eager-attention feasibility cap (26GB gate) can be set before any
benchmark is chosen: 10min @ 25 tok/s = 15k audio tokens, which will not fit materialised.
CONTENT-INDEPENDENT: token count depends only on duration, so the probe signal is a synthetic
tone. This measures the tokenizer; it is not an evaluation and builds no dataset.
"""
import os, sys, json, numpy as np
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
os.environ.pop("HF_TOKEN", None)          # invalid+leaked; use ~/.cache/huggingface/token
from transformers import AutoProcessor, AutoConfig

CANDS = {
 "qwen2_audio": "Qwen/Qwen2-Audio-7B-Instruct",          # incumbent, known 25 tok/s, 30s window
 "qwen25_omni": "Qwen/Qwen2.5-Omni-7B",
 "phi4_mm":     "microsoft/Phi-4-multimodal-instruct",
 "af3":         "nvidia/audio-flamingo-3",
 "ultravox":    "fixie-ai/ultravox-v0_5-llama-3_1-8b",
 "minicpmo":    "openbmb/MiniCPM-o-2_6",
 "meralion":    "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION",
 "voxtral":     "mistralai/Voxtral-Mini-3B-2507",
 "omni3b":      "Qwen/Qwen2.5-Omni-3B",
}
DURS = [10, 30, 60, 120, 240]
SR   = 16000

def tone(sec):
    t = np.arange(int(sec*SR), dtype=np.float32)/SR
    return (0.1*np.sin(2*np.pi*440*t)).astype(np.float32)

def audio_token_id(pr, cfg):
    for k in ("audio_token_index","audio_token_id"):
        v = getattr(cfg, k, None)
        if v is None and hasattr(cfg,"sound_tower_cfg"):
            v = getattr(cfg.sound_tower_cfg, k, None)
        if isinstance(v,int): return v
    tk = getattr(pr,"tokenizer",None)
    for s in ("<|AUDIO|>","<|audio|>","<|audio_pad|>","<sound>","<|endoftext10|>"):
        if tk is None: break
        i = tk.convert_tokens_to_ids(s)
        if i is not None and i >= 0 and i != tk.unk_token_id: return i
    return None

def probe(tag, mid):
    out = {"model": mid, "counts": {}}
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        pr  = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    except Exception as e:
        print(f"{tag:12s} UNAVAILABLE  {type(e).__name__}: {str(e)[:110]}", flush=True)
        return {"model": mid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    aid = audio_token_id(pr, cfg)
    out["audio_token_id"] = aid
    if aid is None:
        print(f"{tag:12s} NO AUDIO TOKEN ID FOUND -> cannot count placeholders", flush=True)
        return out
    for d in DURS:
        try:
            msg = [{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":"x"}]}]
            try:    chat = pr.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception: chat = "<|audio_bos|><|AUDIO|><|audio_eos|> x"
            # processors disagree on the kwarg name (audio / audios / no sampling_rate)
            inp = None; last = None
            for kw in ("audio","audios"):
                for extra in ({"sampling_rate":SR}, {}):
                    try:
                        inp = pr(text=chat, **{kw:[tone(d)]}, **extra, return_tensors="pt"); break
                    except Exception as e: last = e
                if inp is not None: break
            if inp is None: raise last
            ids = inp["input_ids"][0]
            n   = int((ids == aid).sum())
            out["counts"][d] = n
            print(f"{tag:12s} {d:4d}s -> {n:6d} audio tokens  ({n/d:6.1f} tok/s)", flush=True)
        except Exception as e:
            out["counts"][d] = None
            print(f"{tag:12s} {d:4d}s -> ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
    c = {k:v for k,v in out["counts"].items() if v}
    # Judge on the TAIL (>=30s): Qwen2-Audio is linear to 30s then hard-clips, so a full-span
    # ratio calls a truncating model "PARTIAL". What matters for AVR is growth PAST the window.
    tail = {k:v for k,v in c.items() if k >= 30}
    if len(tail) >= 2:
        ks = sorted(tail); ratio_tok = tail[ks[-1]]/max(tail[ks[0]],1); ratio_dur = ks[-1]/ks[0]
        out["verdict"] = ("PLATEAU" if ratio_tok < 1.3 else
                          "LINEAR"  if ratio_tok >= 0.7*ratio_dur else "PARTIAL")
        out["tail_ratio"] = [ratio_tok, ratio_dur]
        print(f"{tag:12s} VERDICT {out['verdict']}  past 30s: tokens x{ratio_tok:.2f} over duration x{ratio_dur:.1f}", flush=True)
    return out

if __name__ == "__main__":
    sel = sys.argv[1:] or list(CANDS)
    res = {t: probe(t, CANDS[t]) for t in sel if t in CANDS}
    os.makedirs("data", exist_ok=True)
    json.dump(res, open("data/phase245_longaudio_g0.json","w"), indent=1)
    print("\n=== G0 SUMMARY (LINEAR = usable for AVR-long) ===")
    for t,r in res.items():
        print(f"  {t:12s} {r.get('verdict', r.get('error','no counts'))}")
