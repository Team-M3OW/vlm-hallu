"""
Phase 48: WHICH QUERY TOKEN'S attention should the localizer read? Free -- same forward pass.

THE GAP THIS TARGETS
--------------------
Oracle placement scores 92.7% in ONE pass at 300 tokens, against uniform's 56.5%. So the ceiling at
1x cost is enormous and the entire gap is PROPOSAL QUALITY: the ring-masked attention peak lands
inside the GT box on only 15.7% of items, with median miss 0.170 of image width. Phase 47 showed
that buying proposals with extra passes is a losing trade -- the uniform curve rises 56.5 -> 63.9 ->
71.7% at 1x/2x/4x, so a 4-pass re-ranker must clear 71.7%. The only place left to win is a BETTER
PROPOSAL AT THE SAME COST.

WHAT HAS NEVER BEEN VARIED
--------------------------
Every phase so far read attention from the LAST prompt token. For a 4-way MCQ that token sits after
the option text and the instruction "Answer with the option's letter...", i.e. several dozen tokens
downstream of the words naming the thing to find. Attention from the QUESTION's content words lives
in the SAME attention tensor -- a different set of query rows -- so reading it costs nothing.

If a content word localises better than the last token, the localizer improves for free, and every
downstream number (coverage, the gate, the adaptive policy, the prior-art table) improves with it.

READ-OUTS, all on the same forward pass and the same ring-masked block mean L16-26:
    last          the deployed choice: final prompt token
    last_k        mean over the final k prompt tokens
    question      mean over the question's own token span (before the options)
    content       mean over the question's content words only (stopwords removed)
    noun          the single highest-attention content token
    max_pool      per-image-cell max over all question tokens
    mean_all      mean over every prompt token

PRIMARY METRIC: gt_pct, rank of the GT-centre cell as a fraction of n_img, chance 0.500 EXACTLY.
SECONDARY: P(peak inside GT) and mean coverage of a W=0.15 window -- the quantity §3 says decides
the sign of the allocation effect.
"""
import json
import os
import re
import statistics as st
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase48_query_token.jsonl"
BLOCK = list(range(16, 27))
B0 = 300
STOP = set("""a an the is are was were be been being of in on at to for with from by and or but if
then than that this these those it its as what which who whom whose where when how why do does did
you your i we they he she them his her their there here can could would should will shall may might
must not no nor so such only own same too very just about into over under above below picture image
photo answer option letter given choices directly please question following""".split())
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            W_, H_ = img.size
            gx0 = min(b[0] for b in ann["bbox"]) / W_; gy0 = min(b[1] for b in ann["bbox"]) / H_
            gx1 = max(b[0] + b[2] for b in ann["bbox"]) / W_
            gy1 = max(b[1] + b[3] for b in ann["bbox"]) / H_
            text = ex["text"]
            small = fit(img, B0)
            inp = build(small, text)
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            n_img = gh * gw
            ids = inp["input_ids"][0].tolist()
            pos = [i for i, v in enumerate(ids) if v == itid]
            base, end = pos[0], pos[-1] + 1
            T = len(ids)
            # the question span: prompt text after the image, before the option block
            q_first = end
            qline = text.split("\n")[0]
            q_ids = tok(qline, add_special_tokens=False)["input_ids"]
            q_last = min(T - 1, q_first + len(q_ids))
            # content-word positions inside the question span
            content = []
            for i in range(q_first, q_last):
                w = tok.decode([ids[i]]).strip().lower()
                w = re.sub(r"[^a-z]", "", w)
                if len(w) >= 3 and w not in STOP:
                    content.append(i)
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            # A[q, c]: attention from prompt position q to image cell c, block-averaged
            A = torch.zeros(T, n_img, dtype=torch.float32, device=model.device)
            for L in BLOCK:
                a = out.attentions[L][0, :, :, base:base + n_img].float().mean(0)
                A += a / (a.sum(-1, keepdim=True) + 1e-12)
            A = (A / len(BLOCK)).cpu()
            del out
            torch.cuda.empty_cache()

            def ring(v):
                m = v.reshape(gh, gw).clone()
                o = torch.full_like(m, -1.0)
                if gh > 2 and gw > 2:
                    o[1:-1, 1:-1] = m[1:-1, 1:-1]
                else:
                    o = m.clone()
                return o.flatten()

            READ = {"last": A[-1],
                    "last_k4": A[-4:].mean(0),
                    "question": A[q_first:max(q_first + 1, q_last)].mean(0),
                    "content": A[content].mean(0) if content else A[-1],
                    "noun": A[content][A[content].max(-1).values.argmax()] if content else A[-1],
                    "max_pool": A[q_first:max(q_first + 1, q_last)].max(0).values,
                    "mean_all": A[end:].mean(0)}
            rec = {"question_id_full": qid, "category": ex["category"], "grid": [gh, gw],
                   "gt_box_frac": [gx0, gy0, gx1, gy1], "n_content": len(content),
                   "readouts": {}}
            gr = min(gh - 1, int(((gy0 + gy1) / 2) * gh))
            gc = min(gw - 1, int(((gx0 + gx1) / 2) * gw))
            gi = gr * gw + gc
            for nm, v in READ.items():
                f = ring(v)
                j = int(f.argmax().item())
                rec["readouts"][nm] = {
                    "gt_pct": float((f > f[gi]).sum()) / n_img,
                    "peak": [((j % gw) + .5) / gw, ((j // gw) + .5) / gh],
                    "peak_val": float(f.max())}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
