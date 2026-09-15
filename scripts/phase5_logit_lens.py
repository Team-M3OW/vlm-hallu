"""
Phase 5: layer-wise logit-lens trajectory analysis (per user request, 2026-09-03) -- find the
CULPRIT layer(s) where confident-denial hallucinations are introduced, rather than just measuring
another correlate of the effect.

For each item, run one forward pass with output_hidden_states=True. This returns hidden_states, a
tuple of length num_layers+1 (embedding output, then each of the 28 decoder layers' output). At
each layer l, apply the model's OWN final norm + lm_head to the hidden state at the answer-token
position -- this is the standard "logit lens" technique (Nostalgebraist 2020): it approximates
"what would the model have answered if it stopped thinking at layer l," using the real trained
unembedding matrix, not a separately trained probe. Compute P(yes) at each layer via the same
logsumexp yes/no-token pooling used throughout this project.

Three groups compared:
  - confident_denial: true positives, p_yes_real < 0.01 (model confidently says NO to something
    present) -- the hallucination cases we want to explain.
  - confident_correct_small: true positives in the SAME (smallest) area bin, p_yes_real > 0.95
    (model confidently and CORRECTLY says YES) -- the fairest comparison group, matched on
    difficulty/area so any trajectory difference isn't just "small vs large objects."
  - blank: the SAME images' blank/black-image control already computed in phase1 (constant
    reference for "no real evidence" baseline).

If confident_denial trajectories track toward "yes" in early/mid layers and only flip to "no" in
specific late layers (while confident_correct_small stays "yes" throughout), that pinpoints the
culprit layer(s): evidence is present and initially favored, then overridden.
"""
import json
import sys
import time
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase5_logit_lens_results.jsonl"
N_PER_GROUP = 80


def main():
    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs = [json.loads(l) for l in f]
    positives = [r for r in recs if r["label"] == "yes" and r.get("pixel_area_frac") is not None]

    # The cached p_yes_real values were computed with the 4-BIT QUANTIZED model
    # (phase1_eval_qwen.py). Logit-lens work needs clean fp16 hidden states -- quantization noise
    # at every layer would pollute the trajectory. Verified empirically before trusting this: for
    # one candidate item, cached (4-bit) p_yes=0.0067 vs fp16-recomputed=0.169 -- NOT the same
    # item under this project's own "confident denial" (<0.01) criterion once precision changes.
    # So: recompute p_yes_real fresh in fp16 for the candidate pool, and select cohorts from THAT,
    # not from the 4-bit cached values, to keep the item selection internally consistent with the
    # precision actually used for the trajectory analysis.
    candidate_denials_4bit = [r for r in positives if r["p_yes_real"] < 0.05]  # generous net
    small_bin = [r for r in positives if r["pixel_area_frac"] < 0.02]
    candidate_correct_small_4bit = [r for r in small_bin if r["p_yes_real"] > 0.90]

    candidate_uids = {r["uid"] for r in candidate_denials_4bit} | {r["uid"] for r in candidate_correct_small_4bit}
    print(f"candidate pool (4-bit prefilter, will be re-verified in fp16): "
          f"{len(candidate_denials_4bit)} denial candidates, "
          f"{len(candidate_correct_small_4bit)} correct-small candidates, "
          f"{len(candidate_uids)} unique uids")

    print("Loading items...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in candidate_uids}
    print(f"  resolved {len(items_by_uid)}/{len(candidate_uids)}")

    print("Loading Qwen3-VL-2B (fp16, not 4-bit -- clean precision for interpretability)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    tok = processor.tokenizer
    yes_variants = ["yes", "Yes", " yes", " Yes"]
    no_variants = ["no", "No", " no", " No"]
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in yes_variants})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in no_variants})

    def build_prompt_local(question):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question} Please answer this question with yes or no."},
        ]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def p_yes_fp16_final(image, question):
        prompt = build_prompt_local(question)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        yl = torch.logsumexp(logits[yes_ids], dim=0)
        nl = torch.logsumexp(logits[no_ids], dim=0)
        return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

    print("Re-verifying candidates' P(yes) under fp16 (single forward pass each)...")
    fp16_p_yes = {}
    for i, r in enumerate(candidate_denials_4bit + candidate_correct_small_4bit):
        if r["uid"] in fp16_p_yes:
            continue
        item = items_by_uid.get(r["uid"])
        if item is None:
            continue
        fp16_p_yes[r["uid"]] = p_yes_fp16_final(item["image"].convert("RGB"), item["question"])
        if (i + 1) % 100 == 0:
            print(f"  reverified {i+1}/{len(candidate_uids)}")

    confident_denial = [r for r in candidate_denials_4bit
                         if fp16_p_yes.get(r["uid"], 1.0) < 0.01][:N_PER_GROUP]
    confident_correct_small = [r for r in candidate_correct_small_4bit
                                if fp16_p_yes.get(r["uid"], 0.0) > 0.95][:N_PER_GROUP]
    print(f"\nAFTER fp16 re-verification: confident_denial n={len(confident_denial)}, "
          f"confident_correct_small n={len(confident_correct_small)}")

    targets = [(r, "confident_denial") for r in confident_denial] + \
              [(r, "confident_correct_small") for r in confident_correct_small]

    lm_head = model.lm_head
    final_norm = model.model.language_model.norm

    def p_yes_from_hidden(h):
        # h: [hidden_dim] at the last token position for one layer. Apply the model's OWN final
        # norm + lm_head -- the logit-lens approximation -- then pool yes/no logits identically to
        # every other forced-choice measurement in this project.
        with torch.no_grad():
            normed = final_norm(h.unsqueeze(0))
            logits = lm_head(normed)[0]
        yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
        no_logit = torch.logsumexp(logits[no_ids], dim=0)
        probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
        return probs[0].item()

    def trajectory(image, question):
        prompt = build_prompt_local(question)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (num_layers+1) tensors, each [batch, seq, hidden]
        traj = []
        for h in out.hidden_states:
            last_tok_h = h[0, -1]
            traj.append(p_yes_from_hidden(last_tok_h))
        return traj

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            item = items_by_uid.get(r["uid"])
            if item is None:
                continue
            img = item["image"].convert("RGB")
            traj = trajectory(img, item["question"])
            rec = {"uid": r["uid"], "group": group, "category": r["category"],
                   "pixel_area_frac": r["pixel_area_frac"],
                   "p_yes_real_4bit_cached": r["p_yes_real"],
                   "p_yes_real_fp16": fp16_p_yes[r["uid"]],
                   "trajectory": traj}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}/{len(targets)}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
