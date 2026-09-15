"""
Phase 0c: verify Qwen3-VL-2B-Instruct's image-token handling (id, count, whether it's fixed or
dynamic-per-image) before using it as the second-architecture check. Qwen-family VL models use
dynamic resolution (smart-resize to a patch/merge-size multiple), so the token count is NOT fixed
like LLaVA's 576 -- must be read off each item's actual processor output, not hardcoded.
"""
from transformers import AutoProcessor
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

proc = AutoProcessor.from_pretrained(MODEL_ID)
tok = proc.tokenizer

img = Image.new("RGB", (640, 425), (128, 128, 128))
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Is there a snowboard in the image? Please answer this question with yes or no."},
    ]},
]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("PROMPT TEXT:\n", text)

inputs = proc(images=img, text=text, return_tensors="pt")
print("input keys:", list(inputs.keys()))
print("input_ids shape:", inputs["input_ids"].shape)
if "image_grid_thw" in inputs:
    print("image_grid_thw:", inputs["image_grid_thw"])
image_token = getattr(proc, "image_token", "<|image_pad|>")
image_token_id = tok.convert_tokens_to_ids(image_token) if isinstance(image_token, str) else image_token
print("image_token:", image_token, "id:", image_token_id)
n_img_tok = (inputs["input_ids"][0] == image_token_id).sum().item()
print("n image tokens for this 640x425 image:", n_img_tok)

# try a second, differently-sized image to confirm token count is dynamic (not fixed)
img2 = Image.new("RGB", (1024, 768), (64, 64, 64))
inputs2 = proc(images=img2, text=text, return_tensors="pt")
n_img_tok2 = (inputs2["input_ids"][0] == image_token_id).sum().item()
print("n image tokens for 1024x768 image:", n_img_tok2, "(should differ from above if dynamic)")
if "image_grid_thw" in inputs2:
    print("image_grid_thw (2):", inputs2["image_grid_thw"])
