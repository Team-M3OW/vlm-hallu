"""
Phase 6e: attention-patching intervention -- the causal test Phase 6c's correlational finding was
missing. Phase 6c showed that for confident_denial items, attention to the true object's own image
tokens (at the answer position, last few layers) is statistically indistinguishable from attention
to an object nobody asked about, while the SAME mechanism is demonstrably query-target-conditioned
on items the model gets right. That is a correlational signature. This asks the causal question:
if we MECHANICALLY force attention onto the object's own tokens at the answer position -- to a
level matching or exceeding what correctly-answered items show -- does P(yes) recover?
  - If yes: attention allocation is a real causal bottleneck, and this is a concrete, no-finetuning
    fix -- the strongest possible result for this project.
  - If no (P(yes) stays low even with attention forced onto the right tokens): attention isn't the
    bottleneck; whatever fails happens downstream (value content, MLP integration, or the final
    unembedding), and Phase 6c's signature, while real, isn't the culprit -- a different but still
    informative negative result.

Method: monkeypatch `Qwen3VLTextAttention.forward` (transformers' Qwen3-VL text-decoder attention)
to reproduce the library's exact eager-attention computation, but for the LAST query position only
(where yes/no is about to be generated -- causally sufficient and self-contained under causal
masking: modifying only the last position's attention row cannot leak into any other position's
computation, since no earlier position can attend to a later one) and only within a chosen set of
layers, redistribute the post-softmax attention distribution so the queried object's own token set
receives a TARGET total share F of the attention mass (rescaling the rest proportionally to
preserve their relative pattern and keep the row summing to 1). Two intervention targets per item,
compute-matched:
  - object patch: redistribute onto the item's actual queried-object token set.
  - random patch: redistribute onto a random SAME-SIZE set of non-object image tokens -- controls
    for "any attention redistribution helps somehow" (the filler-token confound that sank Phase 6d)
    vs. "specifically boosting the TRUE object helps."
Two target shares F=0.30 (moderate) and F=0.60 (aggressive) as a dose-response check, applied to
the last N_LAST_LAYERS=4 decoder layers (matching where Phase 6/6c measured enrichment).
"""
import json
import sys
import time
import random
import torch
import torch.nn as nn
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl_mod

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import Ctx, grid_from_image, object_token_indices, load_bbox_lookup

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase6e_attention_patch_results.jsonl"
TOTAL_LAYERS = 28
# Patch ONLY the final layer. An earlier 4-layer version (patching layers 24-27) produced a
# symmetric object-vs-random null (advisor review, 2026-09-07): layer 27's query reads keys built
# from layer 26's OUTPUT, which was itself already patched, so the "realized share" measured at
# layer 27 reflected four stacked renormalizations, not a clean single intervention -- the result
# was uninterpretable, not evidence against attention-causality. Single-layer patching avoids this.
PATCH_LAYERS = {TOTAL_LAYERS - 1}
# F=0.30/0.60 (the original targets) were 3-6x higher than what confident_correct_small items
# NATURALLY show: median object-token attention SHARE (obj_attn_mass/total_img_attn_mass) in
# data/phase6_attention_results.jsonl is 0.094 for confident_correct_small vs 0.012 for
# confident_denial -- i.e. single-digit percent, not 30-60%. That made the original F values an
# intervention far outside the range any real item occupies, indistinguishable from generically
# shoving the model off-manifold (advisor review). Use realistic targets instead.
TARGET_SHARES = [0.094, 0.20]

_PATCH = {"active": False, "positions": None, "share": 0.0}


def set_patch(positions, share):
    _PATCH["active"] = True
    _PATCH["positions"] = positions
    _PATCH["share"] = share


def clear_patch():
    _PATCH["active"] = False
    _PATCH["positions"] = None


def _patched_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen3_vl_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_rep = qwen3_vl_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_rep = qwen3_vl_mod.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if _PATCH["active"] and self.layer_idx in PATCH_LAYERS:
        pos = _PATCH["positions"]
        share = _PATCH["share"]
        # Do the share-rescaling arithmetic in float32 -- individual object-token attention
        # probabilities can be small enough (~1e-5 to 1e-7) to underflow fp16 subnormal range
        # (min positive ~5.96e-8) when summed/divided, which produced NaN via inf = share/~0
        # in the first version of this patch (caught by smoke-testing before the full run).
        last = attn_weights[:, :, -1, :].clone().float()  # [bsz, n_heads, kv_len]
        s_obj = last[:, :, pos].sum(dim=-1, keepdim=True)  # [bsz, n_heads, 1]
        s_obj_clamped = torch.clamp(s_obj, min=1e-6)
        rest_mask = torch.ones(last.shape[-1], dtype=torch.bool, device=last.device)
        rest_mask[pos] = False
        s_rest = last[:, :, rest_mask].sum(dim=-1, keepdim=True)
        s_rest_clamped = torch.clamp(s_rest, min=1e-6)
        # only boost where current share is below target -- never REDUCE an already-high share
        do_boost = (s_obj < share).squeeze(-1)  # [bsz, n_heads]
        obj_scale = (share / s_obj_clamped)
        rest_scale = ((1 - share) / s_rest_clamped)
        boosted = last.clone()
        boosted[:, :, pos] = last[:, :, pos] * obj_scale
        boosted[:, :, rest_mask] = last[:, :, rest_mask] * rest_scale
        new_last = torch.where(do_boost.unsqueeze(-1), boosted, last).to(attn_weights.dtype)
        attn_weights = attn_weights.clone()
        attn_weights[:, :, -1, :] = new_last

    attn_output = torch.matmul(attn_weights, value_rep)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def install_patch():
    qwen3_vl_mod.Qwen3VLTextAttention.forward = _patched_forward


def p_yes_and_share(ctx, image, question, obj_positions=None, patch_positions=None, share=0.0):
    """One forward pass. If patch_positions is given, the module-level patch is armed for the
    duration of this call (last-position-only, PATCH_LAYERS only). Returns (p_yes, obj_share_last_layer)
    -- the realized share attention.attention gives obj_positions at the LAST patched layer, read
    straight from output_attentions, as a check that the patch actually took effect."""
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{question} Please answer this question with yes or no."},
    ]}]
    text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx.processor(images=image, text=text, return_tensors="pt").to(ctx.model.device)

    if patch_positions is not None:
        set_patch(patch_positions, share)
    try:
        with torch.no_grad():
            out = ctx.model(**inputs, output_attentions=True)
    finally:
        clear_patch()

    logits = out.logits[0, -1]
    yl = torch.logsumexp(logits[ctx.yes_ids], dim=0)
    nl = torch.logsumexp(logits[ctx.no_ids], dim=0)
    p_yes = torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

    realized_share = None
    if obj_positions is not None:
        last_layer_attn = out.attentions[-1][0, :, -1, :].mean(dim=0)  # mean over heads
        realized_share = last_layer_attn[obj_positions].sum().item()
    return p_yes, realized_share


def main():
    install_patch()

    with open(f"{DATA}/phase6_attention_results.jsonl") as f:
        phase6_recs = [json.loads(l) for l in f]
    denials = [r for r in phase6_recs if r["group"] == "confident_denial"]
    print(f"confident_denial cohort: {len(denials)}")

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    target_uids = {r["uid"] for r in denials}
    print("Loading items...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    done_uids = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done_uids)} already done")
    except FileNotFoundError:
        pass

    print("Loading Qwen3-VL-2B (fp16, eager attention, PATCHED forward installed)...")
    ctx = Ctx()

    rng = random.Random(42)
    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r in denials:
            uid = r["uid"]
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                continue
            rfull = recs_by_uid.get(uid)
            key = (rfull["split"], str(rfull["question_id"]))
            bboxes = bbox_by_key.get(key)
            imgsize = imgsize_by_key.get(key)
            if not bboxes or imgsize is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = imgsize
            grid_h, grid_w, resized_h, resized_w = grid_from_image(ctx, img)
            obj_tok_idx = object_token_indices(bboxes, img_w, img_h, resized_w, resized_h, grid_h, grid_w)
            if not obj_tok_idx:
                continue

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{item['question']} Please answer this question with yes or no."},
            ]}]
            probe_text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            probe_inputs = ctx.processor(images=img, text=probe_text, return_tensors="pt")
            image_mask = (probe_inputs["input_ids"][0] == ctx.image_token_id)
            img_positions = image_mask.nonzero().flatten()
            n_img = len(img_positions)
            obj_seq_positions = torch.tensor(
                sorted(img_positions[i].item() for i in obj_tok_idx if i < n_img), dtype=torch.long)
            if len(obj_seq_positions) == 0:
                continue

            non_obj_pool = [p.item() for p in img_positions if p.item() not in set(obj_seq_positions.tolist())]
            n_rand = min(len(obj_seq_positions), len(non_obj_pool))
            random_seq_positions = torch.tensor(
                sorted(rng.sample(non_obj_pool, n_rand)), dtype=torch.long)

            base_py, base_share = p_yes_and_share(ctx, img, item["question"], obj_seq_positions)

            rec = {"uid": uid, "category": r["category"], "pixel_area_frac": r["pixel_area_frac"],
                   "n_obj_tokens": len(obj_seq_positions), "n_image_tokens": n_img,
                   "baseline_p_yes": base_py, "baseline_obj_share": base_share}

            for share in TARGET_SHARES:
                obj_py, obj_share = p_yes_and_share(ctx, img, item["question"], obj_seq_positions,
                                                     obj_seq_positions, share)
                rand_py, rand_share_on_obj = p_yes_and_share(ctx, img, item["question"], obj_seq_positions,
                                                              random_seq_positions, share)
                rec[f"obj_patch_F{share}_p_yes"] = obj_py
                rec[f"obj_patch_F{share}_realized_share"] = obj_share
                rec[f"random_patch_F{share}_p_yes"] = rand_py
                rec[f"random_patch_F{share}_obj_share"] = rand_share_on_obj

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
