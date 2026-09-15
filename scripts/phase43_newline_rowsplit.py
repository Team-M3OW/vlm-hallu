"""
Phase 43: split the LLaVA `OTHER` bucket by POSITION IN ROW, so all four architectures are scored
on the SAME five regions.

THE PROBLEM WITH PHASE 41's TABLE
---------------------------------
For the grid models (Qwen2/3-VL) Phase 41 compared four disjoint regions of image content: last
column, first column, top row, bottom row. For the newline models (LLaVA-OneVision, LLaVA-NeXT) it
compared the separator against a single `OTHER` bucket holding 95.4% of positions -- every real
image token, regardless of column.

So the LLaVA rows only established "the separator beats the average image token". The Qwen rows
established something stronger and more specific: "the TRAILING boundary beats the LEADING boundary,
and both beat ROWS". The discriminating prediction -- rows are ordinary, the trailing boundary is
not -- was never tested in the newline models. Two different claims were sharing one sentence.

THE FIX, FREE FROM THE SAME FORWARD PASSES
------------------------------------------
Newline positions give the row segmentation directly: the tokens between two consecutive separators
ARE a row. So non-separator tokens can be bucketed exactly as in the grid models -- first-in-row,
last-in-row, top row, bottom row, interior -- and the identical five-region table computed.

    LLaVA rows ~1.0x and the pre-separator token elevated  -> one claim across four models.
    LLaVA rows ELEVATED                                    -> two phenomena, and §4.1's headline
                                                              must say so.

Rows are segmented per contiguous image-token run, because these models concatenate a base image
with anyres tiles; a run is one image's raster and its rows are comparable within it.
"""
import json
import os
import statistics as st
from collections import Counter, defaultdict

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase43_newline_rowsplit.json"
N_ITEMS = 24
MODELS = [("onevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf", 336),
          ("llavanext", "llava-hf/llava-v1.6-vicuna-7b-hf", 336)]
Image.MAX_IMAGE_PIXELS = None


def newline_vec(model):
    for path in (("model", "image_newline"), ("image_newline",), ("model", "model", "image_newline")):
        o = model
        for p in path:
            o = getattr(o, p, None)
            if o is None:
                break
        if isinstance(o, torch.nn.Parameter):
            return o.data
    return None


def regions(isnl, n):
    """Bucket every image position using the separators as the row segmentation.

    ONLY MODAL-LENGTH SEGMENTS COUNT AS ROWS. These models prepend a BASE IMAGE whose tokens carry
    no separators at all, so a naive "segment between separators = row" reading swallows the entire
    base image as row 0. That is not a hypothetical: the first version of this function put 59.8% of
    all positions into TOP ROW on OneVision (one row of ~19 cannot exceed ~5%) and correspondingly
    mislabelled the final tile row as BOTTOM ROW, which then read as 3.0x "elevated". The whole
    rows-elevated verdict was an artifact of this bucketing.

    Genuine rows all have the same width, so the modal segment length identifies them; irregular
    segments (the base image, any padding) are excluded from the row buckets entirely rather than
    being forced into one.
    """
    segs, cur = [], []
    for i in range(n):
        if isnl[i]:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    out = defaultdict(list)
    out["NEWLINE (row sep)"] = [i for i in range(n) if isnl[i]]
    lens = Counter(len(s) for s in segs)
    if not lens:
        return out, 0
    modal = lens.most_common(1)[0][0]
    rows = [s for s in segs if len(s) == modal and modal >= 3]
    nr = len(rows)
    if nr < 3:
        return out, nr
    for ri, row in enumerate(rows):
        for ci, i in enumerate(row):
            if ci == len(row) - 1:
                out["LAST COL (pre-sep)"].append(i)
            elif ci == 0:
                out["FIRST COL"].append(i)
            elif ri == 0:
                out["TOP ROW"].append(i)
            elif ri == nr - 1:
                out["BOTTOM ROW"].append(i)
            else:
                out["INTERIOR"].append(i)
    out["EXCLUDED (base image / irregular)"] = [i for s in segs if len(s) != modal for i in s]
    return out, nr


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

    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, mid, side in MODELS:
        if name in allres:
            print(f"skip {name}")
            continue
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
        model.eval()
        pr = AutoProcessor.from_pretrained(mid)
        nl = newline_vec(model)
        L = model.config.get_text_config().num_hidden_layers
        block = list(range(int(0.55 * L), int(0.95 * L) + 1))
        itid = getattr(model.config, "image_token_id", None) or \
            getattr(model.config, "image_token_index", None)
        acc, ok, nrows = defaultdict(list), 0, []
        for ip, q in items:
            try:
                img = Image.open(ip).convert("RGB")
                w, h = img.size
                sc = side / max(w, h)
                img = img.resize((max(32, int(w * sc)), max(32, int(h * sc))), Image.BICUBIC)
                msgs = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": q}]}]
                chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = pr(images=img, text=chat, return_tensors="pt")
                pos = (inp["input_ids"][0] == itid).nonzero().flatten().tolist()
                if len(pos) < 40:
                    continue
                inp = inp.to(model.device)
                with torch.no_grad():
                    out = model(**inp, output_attentions=True, output_hidden_states=True)
                if not torch.isfinite(out.attentions[block[0]]).all():
                    raise RuntimeError("non-finite attention")
                base, n = pos[0], len(pos)
                a = torch.zeros(n, dtype=torch.float32, device=model.device)
                for li in block:
                    r = out.attentions[li][0, :, -1, base:base + n].float().mean(0)
                    a += r / (r.sum() + 1e-12)
                a = (a / len(block)).cpu()
                emb = out.hidden_states[0][0, base:base + n].float()
                d = (emb - nl.float().to(emb.device)).abs().max(-1).values
                isnl = (d < 1e-2).cpu().tolist()
                if sum(isnl) < 3:
                    del out; continue
                idx, nr = regions(isnl, n)
                nrows.append(nr)
                for k, ii in idx.items():
                    if ii:
                        acc[k].append((float(a[ii].sum()), len(ii) / n))
                ok += 1
                del out
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   skip: {type(e).__name__}: {str(e)[:60]}")
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        if not ok:
            continue
        print(f"  n={ok} images, median {st.median(nrows):.0f} rows/image")
        print(f"  {'region':<22}{'mass':>8}{'area':>8}{'enrichment':>13}{'95% CI':>16}")
        res = {}
        import random
        for k in ["NEWLINE (row sep)", "LAST COL (pre-sep)", "FIRST COL",
                  "TOP ROW", "BOTTOM ROW", "INTERIOR",
                  "EXCLUDED (base image / irregular)"]:
            v = acc.get(k)
            if not v:
                continue
            m = st.mean(x[0] for x in v); ar = st.mean(x[1] for x in v)
            pe = [x[0] / max(x[1], 1e-9) for x in v]
            rr = random.Random(0); mm = len(pe)
            bs = sorted(sum(pe[rr.randrange(mm)] for _ in range(mm)) / mm for _ in range(4000))
            res[k] = {"enrichment": m / max(ar, 1e-9), "ci": [bs[100], bs[3900]], "n": ok}
            print(f"  {k:<22}{100*m:>7.1f}%{100*ar:>7.1f}%{m/max(ar,1e-9):>12.1f}x"
                  f"   [{bs[100]:.1f},{bs[3900]:.1f}]")
        allres[name] = res
        json.dump(allres, open(OUT, "w"), indent=1)

    print("\n" + "=" * 70)
    print("DOES THE SAME FIVE-REGION PATTERN HOLD IN THE NEWLINE MODELS?")
    print("=" * 70)
    for name, r in allres.items():
        sep = r.get("NEWLINE (row sep)", {}).get("enrichment", float("nan"))
        tr = max(r.get("TOP ROW", {}).get("enrichment", 0),
                 r.get("BOTTOM ROW", {}).get("enrichment", 0))
        print(f"  {name:<12} separator {sep:.1f}x   max(top,bottom row) {tr:.1f}x   "
              f"{'ROWS ORDINARY -> one claim' if tr < 1.5 else 'ROWS ELEVATED -> two phenomena'}")


if __name__ == "__main__":
    main()
