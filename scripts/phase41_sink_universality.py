"""
Phase 41: is the COLUMNAR sink universal across VLM architectures, or a Qwen3-VL quirk?

THE CLAIM BEING GENERALISED
---------------------------
§6A established, on Qwen3-VL-2B, that the "attention sink at the image corners" is really a COLUMN
effect: last column 5.1x enriched, first column 2.9x, and the top and bottom rows 0.9x / 0.8x --
i.e. not enriched at all. Transpose and interior-replay tests ruled out register tokens, and the
effect is present at L0. The reading is that it is a RASTER SERIALIZATION artifact: column gw-1 sits
before a row wrap and column 0 after one, while top and bottom rows are ordinary raster positions.

That reading makes a claim about how VLMs FLATTEN IMAGES INTO SEQUENCES, which is common to all of
them. So it should not be Qwen-specific. This tests it on four architectures spanning two families,
two vision stacks, two tokenisation schemes and a 3.5x parameter range.

THE SHARPER TEST THE LLaVA MODELS PROVIDE
-----------------------------------------
Qwen2-VL and Qwen3-VL flatten a grid with NO explicit row marker; the row boundary exists only
implicitly, in the position encoding. LLaVA-OneVision and LLaVA-NeXT instead splice a LEARNED
`image_newline` embedding in at the end of every row.

    serialization account  -> where an EXPLICIT row separator exists, the sink should sit ON IT.
    corner/2-D account     -> a newline token is not a corner and should attract nothing special.

These predictions differ, and only one of them can survive. Newline positions are found by matching
the merged input embeddings against the model's own `image_newline` parameter, so no unpad logic is
reimplemented and nothing depends on our own arithmetic.

MEASUREMENT
-----------
Last-token attention over image positions, per-layer L1-normalised then averaged over a RELATIVE
layer block [0.55L, 0.95L] -- the architecture-agnostic form of Qwen3-VL's L16-26 of 28. Reported as
ENRICHMENT (mass share / area share), so a value of 1.0 is "exactly its fair share" and the number
is comparable across models with different grids and token counts.

Images are fitted small so that full attention matrices fit; this reduces token counts but does not
change the geometry under test.
"""
import json
import os
import statistics as st
from collections import defaultdict

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor

OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase41_sink_universality.json"
N_ITEMS = 24
Image.MAX_IMAGE_PIXELS = None

MODELS = [
    ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct", "grid", 448),
    ("qwen2vl", "Qwen/Qwen2-VL-7B-Instruct", "grid", 448),
    ("onevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "newline", 336),
    ("llavanext", "llava-hf/llava-v1.6-vicuna-7b-hf", "newline", 336),
]


def load(mid):
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    m.eval()
    return m, AutoProcessor.from_pretrained(mid)


def img_token_id(model, pr):
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int):
            return v
    t = getattr(pr, "tokenizer", None)
    return t.convert_tokens_to_ids("<image>")


def newline_vec(model):
    for path in (("model", "image_newline"), ("image_newline",),
                 ("model", "model", "image_newline")):
        o = model
        ok = True
        for p in path:
            o = getattr(o, p, None)
            if o is None:
                ok = False
                break
        if ok and isinstance(o, torch.nn.Parameter):
            return o.data
    return None


def analyse(name, mid, mode, side, imgs_qs):
    print(f"\n{'='*70}\n{name}  ({mid})\n{'='*70}", flush=True)
    model, pr = load(mid)
    L = model.config.get_text_config().num_hidden_layers
    block = list(range(int(0.55 * L), int(0.95 * L) + 1))
    print(f"  {L} layers; relative block {block[0]}-{block[-1]}", flush=True)
    itid = img_token_id(model, pr)
    nl = newline_vec(model) if mode == "newline" else None
    if mode == "newline" and nl is None:
        print("  !! image_newline not found; cannot locate row separators. SKIPPING.")
        del model
        torch.cuda.empty_cache()
        return None
    acc = defaultdict(list)
    ok = 0
    for ip, q in imgs_qs:
        try:
            img = Image.open(ip).convert("RGB")
            w, h = img.size
            sc = side / max(w, h)
            img = img.resize((max(32, int(w * sc)), max(32, int(h * sc))), Image.BICUBIC)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = pr(images=img, text=chat, return_tensors="pt")
            pos = (inp["input_ids"][0] == itid).nonzero().flatten().tolist()
            if len(pos) < 40:
                continue
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True, output_hidden_states=True)
            # a bf16 checkpoint loaded in fp16 yields all-NaN attention; refuse it loudly
            if not torch.isfinite(out.attentions[block[0]]).all():
                raise RuntimeError("non-finite attention -- wrong dtype for this checkpoint")
            base, n_img = pos[0], len(pos)
            a = torch.zeros(n_img, dtype=torch.float32, device=model.device)
            for li in block:
                r = out.attentions[li][0, :, -1, base:base + n_img].float().mean(0)
                a += r / (r.sum() + 1e-12)
            a = (a / len(block)).cpu()

            if mode == "grid":
                g = inp["image_grid_thw"][0].tolist()
                ms = getattr(pr.image_processor, "merge_size", 2)
                gh, gw = g[1] // ms, g[2] // ms
                if gh * gw != n_img:
                    del out; continue
                idx = {"LAST COL": [], "FIRST COL": [], "TOP ROW": [], "BOTTOM ROW": [], "INTERIOR": []}
                for i in range(n_img):
                    rr, cc = i // gw, i % gw
                    k = ("LAST COL" if cc == gw - 1 else "FIRST COL" if cc == 0 else
                         "TOP ROW" if rr == 0 else "BOTTOM ROW" if rr == gh - 1 else "INTERIOR")
                    idx[k].append(i)
            else:
                emb = out.hidden_states[0][0, base:base + n_img].float()
                d = (emb - nl.float().to(emb.device)).abs().max(-1).values
                isnl = (d < 1e-2).cpu().tolist()
                nn = sum(isnl)
                if nn < 3:
                    del out; continue
                idx = {"NEWLINE (row sep)": [i for i in range(n_img) if isnl[i]],
                       "LAST BEFORE NEWLINE": [i for i in range(n_img)
                                               if i + 1 < n_img and isnl[i + 1] and not isnl[i]],
                       "FIRST AFTER NEWLINE": [i for i in range(n_img)
                                               if i > 0 and isnl[i - 1] and not isnl[i]],
                       "OTHER": [i for i in range(n_img)
                                 if not isnl[i] and not (i + 1 < n_img and isnl[i + 1])
                                 and not (i > 0 and isnl[i - 1])]}
            for k, ii in idx.items():
                if ii:
                    acc[k].append((float(a[ii].sum()), len(ii) / n_img))
            ok += 1
            del out
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"   skip: {type(e).__name__}: {str(e)[:70]}"); continue
    del model
    torch.cuda.empty_cache()
    if not ok:
        print("  no usable items")
        return None
    print(f"  n = {ok} images")
    print(f"  {'region':<24}{'mass':>8}{'area':>8}{'enrichment':>13}")
    res = {}
    for k, v in acc.items():
        m = st.mean(x[0] for x in v)
        ar = st.mean(x[1] for x in v)
        # keep PER-IMAGE values: a cross-architecture claim needs CIs, not four point estimates
        res[k] = {"mass": m, "area": ar, "enrichment": m / max(ar, 1e-9), "n": ok,
                  "per_image_enrichment": [x[0] / max(x[1], 1e-9) for x in v]}
        print(f"  {k:<24}{100*m:>7.1f}%{100*ar:>7.1f}%{m/max(ar,1e-9):>12.1f}x")
    return res


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    items = []
    for ex in ds:
        ip = os.path.join(root, ex["image"])
        if os.path.exists(ip):
            items.append((ip, ex["text"]))
        if len(items) >= N_ITEMS:
            break
    print(f"{len(items)} V*Bench images")

    allres = {}
    if os.path.exists(OUT):
        allres = json.load(open(OUT))
    for name, mid, mode, side in MODELS:
        if name in allres:
            print(f"skip {name} (done)")
            continue
        try:
            r = analyse(name, mid, mode, side, items)
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:150]}")
            r = None
        if r:
            allres[name] = {"mode": mode, "regions": r}
            json.dump(allres, open(OUT, "w"), indent=1)

    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE SUMMARY -- enrichment (mass share / area share)")
    print("=" * 70)
    for name, d in allres.items():
        print(f"\n  {name}  [{d['mode']}]")
        for k, v in d["regions"].items():
            pe = v.get("per_image_enrichment")
            ci = ""
            if pe:
                import random as _r
                rr = _r.Random(0); m_ = len(pe)
                bs = sorted(sum(pe[rr.randrange(m_)] for _ in range(m_)) / m_ for _ in range(4000))
                ci = f"   CI[{bs[100]:.1f},{bs[3900]:.1f}]"
            print(f"    {k:<24}{v['enrichment']:>7.1f}x{ci}")
    print("""
  GRID models: a large LAST COL / FIRST COL enrichment with TOP/BOTTOM ROW near 1.0x reproduces
    the columnar finding -- rows are ordinary, columns are not.
  NEWLINE models: enrichment ON the explicit row separator is the serialization account's
    strongest prediction; a 2-D corner account predicts nothing special there.""")


if __name__ == "__main__":
    main()
