"""
Phase 121: the LOCALISATION PROMPT as a lever.

Phase 72b's bug -- localising on the bare question with the options stripped -- cost 19.3pp of
crop-arm accuracy on HR-Bench, the largest single effect found in the project. Only bare-vs-options
has ever been compared. This sweeps the prompt used for the LOCALISATION pass only; the answer pass
is untouched (coverage is the metric here; end-task follows only if coverage moves).

PRIOR ART (checked first): ViCrop (2502.17422) localises with a "locate first" instruction:
"Your task is to answer the question by first locating the relevant region in the image. Question:
{q}". LookWise (2603.00171) extracts key nouns and uses them as cross-attention queries -- phase 48
found noun-token attention is at chance here, so that route is not repeated. ZoomEye prompts for
object presence at each tree node (a different mechanism).

VARIANTS (localisation prompt)
  V0 full        question + options + answer instruction      <- CURRENT (phases 71/97/116)
  V1 vicrop      ViCrop's locate-first prefix + V0            <- prior art
  V2 noinstr     question + options, instruction line removed
  V3 search      question + options + "Which region of the image contains the evidence needed to
                 answer this?"
  V4 bare        question only                                <- 72b's bug, as the reference

PRE-REGISTERED: primary = OOF learned-head top-1 coverage at W=0.25 per variant, both models,
against V0. Secondary = deployed block-mean argmax coverage. A variant is adopted only if it beats V0
on BOTH models with a bootstrap CI clear of zero and by more than the ~1pp noise floor (SS14Z).
Otherwise V0 stands and this is recorded as "the options are what matters; wording beyond that is
noise."
Output mirrors phase30c/phase74 so phase 70's feature builder consumes it unchanged.
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
from transformers import AutoProcessor, AutoModelForImageTextToText

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
OUT = f"{D}/phase121_prompts_{WHICH}.jsonl"
B0 = 300
Image.MAX_IMAGE_PIXELS = None
VICROP = "Your task is to answer the question by first locating the relevant region in the image.\n"
SEARCH = "\nWhich region of the image contains the evidence needed to answer this?"

def variants(text):
    lines = text.strip().split("\n")
    q = lines[0]
    has_instr = lines[-1].lower().startswith("answer with")
    body = "\n".join(lines[:-1]) if has_instr else text.strip()
    return {"V0_full": text.strip(), "V1_vicrop": VICROP + text.strip(), "V2_noinstr": body,
            "V3_search": body + SEARCH, "V4_bare": q}

def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
                                                        attn_implementation="eager"); model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); itid = model.config.image_token_id
    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False, add_generation_prompt=True), return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size; sc = (target/max(measure(img), 1))**0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r)**0.5
        return best
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)["question_id_full"])
    t0, n = time.time(), 0
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}" if "question_id" in ex else ex.get("question_id_full")
            if qid not in props or qid in done: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img, _ = fit(Image.open(ip).convert("RGB"), B0)
            rec = {"question_id_full": qid, "category": ex["category"], "gt_box_frac": props[qid]["gt_box_frac"], "variants": {}}
            for vn, vt in variants(ex["text"]).items():
                inp = build(img, vt)
                g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1]//2, g[2]//2; n_img = gh*gw
                base = int((inp["input_ids"][0] == itid).nonzero().flatten()[0].item())
                inp = inp.to(model.device)
                with torch.no_grad(): out = model(**inp, output_attentions=True)
                A = np.stack([out.attentions[L][0, :, -1, base:base+n_img].float().mean(0).cpu().numpy()
                              for L in range(len(out.attentions))])
                del out
                rec["variants"][vn] = {"grid": [gh, gw], "n_img_tokens": n_img,
                                       "attn": {f"L{i}": np.round(A[i], 7).tolist() for i in range(A.shape[0])}}
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
            if n % 20 == 0:
                el = time.time()-t0; print(f"  [{n}] {n/el:.2f} it/s eta={(191-len(done)-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    print(f"Done -> {OUT}", flush=True)

main()
