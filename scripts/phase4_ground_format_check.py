"""
Determine Qwen3-VL-2B-Instruct's actual grounding/bbox output format empirically, before building
the real localize-vs-answer dissociation test (per advisor, 2026-09-03). Uses a few real POPE
positive items where the model answered "yes" confidently (should have no trouble localizing an
object it agrees is present) to see what text format the model naturally produces for a grounding
prompt.
"""
import json
import sys
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

with open("/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase1_results_qwen_dedup.jsonl") as f:
    recs = [json.loads(l) for l in f]
confident_yes = [r for r in recs if r["label"] == "yes" and r["p_yes_real"] > 0.95][:5]
uids = {r["uid"] for r in confident_yes}

print("Loading items...")
all_items = build_items()
items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in uids}

print("Loading Qwen3-VL-2B...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
)
model.eval()

PROMPTS = [
    "Locate the {obj} in the image and output its bounding box coordinates.",
    "Where is the {obj} in this image? Output the bounding box as (x1,y1,x2,y2).",
]

for r in confident_yes:
    item = items_by_uid.get(r["uid"])
    if item is None:
        continue
    img = item["image"].convert("RGB")
    print(f"\n=== {r['uid']} category={r['category']} img_size={img.size} p_yes={r['p_yes_real']:.3f} ===")
    for prompt_template in PROMPTS:
        prompt = prompt_template.format(obj=r["category"])
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"  prompt: {prompt}")
        print(f"  output: {gen!r}")
