"""
Phase 1, second architecture: forced single-token P(yes)/P(no) eval of Qwen3-VL-2B-Instruct
(4-bit, on GPU0) on the same POPE(coco) positives+negatives as phase1_eval.py. Cross-architecture
check per advisor review: a curve that reproduces across LLaVA-1.5-7B AND Qwen3-VL-2B is a much
stronger result than either alone.

NOTE on x-axis: Qwen3-VL uses dynamic per-image resolution/token count (confirmed empirically in
phase0c_verify_qwen.py -- 640x425 -> 260 image tokens, 1024x768 -> 768 tokens, via smart-resize to
a patch*merge_size grid). Replicating LLaVA's patch_token_frac geometry for Qwen's dynamic grid is
substantial extra engineering for a secondary check, so this script uses pixel_area_frac (already
computed, model-agnostic) as the x-axis instead. patch_token_frac remains the primary/headline
metric for the LLaVA run.

GPU note (2026-09-02): originally scoped for GPU1 (RTX A400, 3.68GiB usable), which turned out
infeasible -- Qwen's dynamic per-image token count meant larger images OOM'd mid-run even though
the model loaded with ~50MB to spare, and the resulting CUDA errors cascaded through the rest of
the batch. Deferred at the time; now unblocked by running on GPU0 instead (LLaVA's run finished,
freeing ~8.5GB there against the ~40GB co-tenant job) -- comfortably fits a 2B model in 4-bit plus
activation memory for capped-resolution images. `max_pixels` is now capped via
`image_processor.size = {"longest_edge": ..., "shortest_edge": ...}` (confirmed empirically that
setting the plain `max_pixels` attribute alone is silently ignored by the fast image processor;
the `size` dict is what the resize logic actually reads) -- this bounds worst-case per-item
memory AND makes token count comparable across items, both needed now that OOM is a live risk
again on a smaller card than LLaVA's GPU0 budget would suggest.
"""
import json
import time
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

import sys
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase1_results_qwen.jsonl"
# Run this script with CUDA_VISIBLE_DEVICES=0 (physical GPU0, RTX 6000 Ada -- ~8.5GB free now that
# the LLaVA run has finished). No longer targeting GPU1 -- see module docstring.
DEVICE = "cuda:0"
MAX_PIXELS = 451584  # ~576*28*28, matches LLaVA's 576-token budget in spirit; caps worst-case
                      # per-item activation memory and bounds Qwen's dynamic token count.
MIN_PIXELS = 3136     # Qwen2VL default floor (56*56), keeps tiny images from degenerating further.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = build_items(limit=args.limit)
    print(f"Total items to evaluate: {len(items)} "
          f"(pos={sum(1 for i in items if i['label']=='yes')}, "
          f"neg={sum(1 for i in items if i['label']=='no')})")

    done_uids = set()
    if args.resume:
        try:
            with open(OUT_PATH) as f:
                for line in f:
                    done_uids.add(json.loads(line)["uid"])
            print(f"Resuming: {len(done_uids)} already done")
        except FileNotFoundError:
            pass

    print("Loading Qwen3-VL-2B in 4-bit on", DEVICE, "...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Cap max/min pixels via the `size` dict -- confirmed empirically that setting the plain
    # `max_pixels` attribute alone does not affect the actual resize (module docstring).
    processor.image_processor.size = {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
    import gc
    model = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID, quantization_config=bnb_config, device_map={"": 0},
            )
            break
        except torch.OutOfMemoryError:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info(0)[0] / 1e6
            print(f"OOM on load attempt {attempt+1}/{max_attempts} (GPU0 free={free_mb:.0f}MB), "
                  f"retrying in 15s")
            time.sleep(15)
    if model is None:
        raise RuntimeError(f"Could not load Qwen3-VL-2B after {max_attempts} attempts, GPU too busy.")
    model.eval()
    print("device_map:", getattr(model, "hf_device_map", None))

    tok = processor.tokenizer
    # Qwen3-VL's instruct outputs spread probability mass across case/spacing variants
    # (yes/Yes/ yes/ Yes) -- confirmed empirically via top-5 next-token dump on smoke items,
    # where lowercase 'yes'/'no' outrank capitalized 'Yes'/'No'. Pool all single-token variants
    # via logsumexp per class instead of picking one spelling.
    yes_variants = ["yes", "Yes", " yes", " Yes"]
    no_variants = ["no", "No", " no", " No"]
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in yes_variants})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in no_variants})
    print(f"yes_ids={yes_ids} ({[tok.decode([i]) for i in yes_ids]}), "
          f"no_ids={no_ids} ({[tok.decode([i]) for i in no_ids]})")

    black_img_cache = {}

    def get_black_image(size):
        if size not in black_img_cache:
            black_img_cache[size] = Image.new("RGB", size, (0, 0, 0))
        return black_img_cache[size]

    def build_prompt(question):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question} Please answer this question with yes or no."},
            ]},
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def p_yes(image, question, debug_argmax=False):
        prompt = build_prompt(question)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        if debug_argmax:
            top5 = torch.topk(logits, 5)
            print("  top5 next-token:", [(tok.decode([i]), round(v.item(), 2))
                                          for v, i in zip(top5.values, top5.indices)])
        yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
        no_logit = torch.logsumexp(logits[no_ids], dim=0)
        two = torch.stack([yes_logit, no_logit])
        probs = torch.softmax(two, dim=0)
        return probs[0].item()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for item in items:
            if item["uid"] in done_uids:
                continue
            img = item["image"].convert("RGB")
            t_item = time.time()
            try:
                py_real = p_yes(img, item["question"], debug_argmax=(n_done < 5))
                py_blank = p_yes(get_black_image(img.size), item["question"])
            except Exception as e:
                print(f"ERROR on {item['uid']}: {e}")
                torch.cuda.empty_cache()
                continue
            if n_done < 5:
                print(f"  item={item['uid']} label={item['label']} p_yes_real={py_real:.3f} "
                      f"p_yes_blank={py_blank:.3f} took={time.time()-t_item:.2f}s")
            rec = {k: v for k, v in item.items() if k != "image"}
            rec["p_yes_real"] = py_real
            rec["p_yes_blank"] = py_blank
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = len(items) - len(done_uids) - n_done
                eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                print(f"[{n_done}/{len(items)-len(done_uids)}] rate={rate:.2f}/s ETA={eta_min:.1f}min "
                      f"GPU_mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print(f"Done. Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
