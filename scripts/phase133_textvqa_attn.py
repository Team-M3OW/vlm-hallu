"""
Phase 133: MORE BOXED TRAINING DATA -- TextVQA via Visual-CoT (public, 18.5k boxed items).

Four rounds of architecture search (phases 45, 102, 112, 130) and every training change (118, 122)
say the head is limited by 191 boxed items, not by model class. This is the one lever that attacks
that: dump Qwen3-VL attention maps for N TextVQA items whose evidence boxes come from Visual-CoT
(deepcs233/Visual-CoT, metadata/textvqa_cot_train.jsonl), images from lmms-lab/textvqa train.
TextVQA is the closest public source to V*Bench's regime: 1024px natural photos, presupposing
questions, median target 3.1 merged tokens at B=300, 23.6% below 1 token.

Mirrors phase 30c column-for-column (grid, n_img_tokens, gt_box_frac, attn L0..L27 as last-token
attention over image tokens, head-averaged, sum-normalised later by the head builder).

LOCALISATION PROMPT ends at the answer-emission point (SS16A): question + the TextVQA answer
instruction. No options exist for TextVQA, and 121b showed options contribute nothing.

PRE-REGISTERED (phase 134 evaluates):
    P1  head trained on TextVQA only, applied ZERO-SHOT to V*Bench  vs  V*Bench OOF head (63.4 @0.25)
        CI clear of zero -> "data was the limit" holds and the method gains, fully zero-shot
        parity            -> no gain but the OOF-on-eval-benchmark objection is removed
        loss              -> the 191-item head is at the ceiling for this signal; data route closed
    P2  same for the readable head (log-linear + per-layer nb)
    P3  TextVQA + V*Bench(OOF) combined
"""
import io, json, os, random, sys, time, glob
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT = f"{D}/phase133_textvqa_attn_qwen3.jsonl"
B0, N = 300, 3000
INSTR = "Answer the question using a single word or phrase."
Image.MAX_IMAGE_PIXELS = None

def main():
    meta = [json.loads(l) for l in open(hf_hub_download("deepcs233/Visual-CoT", "metadata/textvqa_cot_train.jsonl", repo_type="dataset"))]
    by_img = {}
    for r in meta:
        by_img.setdefault(os.path.splitext(r["image"])[0], []).append(r)
    files = sorted(glob.glob(f"{os.environ['HF_HUB_CACHE']}/datasets--lmms-lab--textvqa/snapshots/*/data/train-*.parquet"))
    print(f"Visual-CoT textvqa boxes: {len(meta)} rows / {len(by_img)} images; parquets: {len(files)}", flush=True)
    # collect candidate (image_id, row) pairs across parquets without decoding images
    cands = []
    for f in files:
        t = pq.read_table(f, columns=["image_id", "question_id", "question"])
        for i in range(t.num_rows):
            iid = t.column("image_id")[i].as_py()
            if iid in by_img:
                q = t.column("question")[i].as_py()
                for r in by_img[iid]:
                    if r["question"].strip().lower() == q.strip().lower():
                        cands.append((f, i, iid, r)); break
    print(f"joined question-level: {len(cands)}", flush=True)
    rng = random.Random(133); rng.shuffle(cands); sel = cands[:N]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager"); model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); itid = model.config.image_token_id
    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False, add_generation_prompt=True), return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size; sc = (target/max(measure(img), 1))**0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC); rz = measure(cur)
            if best is None or abs(rz-target) < abs(best[1]-target): best = (cur, rz)
            if rz == 0 or abs(rz-target)/target <= tol: break
            sc *= (target/rz)**0.5
        return best
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["question_id_full"] for l in open(OUT)}
    cache = {}; t0, n = time.time(), 0
    with open(OUT, "a") as fout:
        for f, i, iid, r in sel:
            qid = f"textvqa/{iid}/{i}"
            if qid in done: continue
            if f not in cache: cache = {f: pq.read_table(f, columns=["image"])}
            raw = cache[f].column("image")[i].as_py(); raw = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(raw)).convert("RGB"); W, H = img.size
            bb = json.loads(r["bboxs"]) if isinstance(r["bboxs"], str) else r["bboxs"]
            gx0 = min(b[0] for b in bb)/float(r["width"]); gy0 = min(b[1] for b in bb)/float(r["height"])
            gx1 = max(b[2] for b in bb)/float(r["width"]); gy1 = max(b[3] for b in bb)/float(r["height"])
            small, realized = fit(img, B0)
            inp = build(small, r["question"].strip() + "\n" + INSTR)
            g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1]//2, g[2]//2; n_img = gh*gw
            pos = (inp["input_ids"][0] == itid).nonzero().flatten(); base = int(pos[0].item())
            inp = inp.to(model.device)
            with torch.no_grad(): out = model(**inp, output_attentions=True)
            rec = {"question_id_full": qid, "category": "textvqa", "label": r["answer"], "img_wh": [W, H],
                   "realized_tokens": realized, "grid": [gh, gw], "n_img_tokens": n_img,
                   "gt_box_frac": [gx0, gy0, gx1, gy1], "gt_area_frac": (gx1-gx0)*(gy1-gy0),
                   "gt_tokens": (gx1-gx0)*gw*(gy1-gy0)*gh, "attn": {}}
            if len(pos) != n_img: rec["note"] = f"token count mismatch {len(pos)} vs {n_img}"
            for L in range(len(out.attentions)):
                a = out.attentions[L][0, :, -1, base:base+n_img].float().mean(0)
                rec["attn"][f"L{L}"] = [round(float(v), 8) for v in a.tolist()]
            del out
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n += 1
            if n % 50 == 0:
                el = time.time()-t0; print(f"  [{n}/{len(sel)}] {n/el:.2f} it/s eta={(len(sel)-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    print(f"Done -> {OUT}", flush=True)

main()
