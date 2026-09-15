"""
Phase 0b: empirically verify (don't inherit from llava-lens) the actual image-token id, image-token
count, and CLIP preprocessing geometry (resize+crop vs resize+pad) used by the currently-installed
transformers (5.3.0) for llava-hf/llava-1.5-7b-hf. This gates whether phase0_area_join.py's analytic
resize-crop assumption is correct.
"""
from transformers import AutoProcessor
from PIL import Image
import numpy as np

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

proc = AutoProcessor.from_pretrained(MODEL_ID)
print("image_token:", getattr(proc, "image_token", None))
print("patch_size:", getattr(proc, "patch_size", None))
print("vision_feature_select_strategy:", getattr(proc, "vision_feature_select_strategy", None))
tok = proc.tokenizer
img_tok = getattr(proc, "image_token", "<image>")
img_tok_id = tok.convert_tokens_to_ids(img_tok)
print("image_token_id:", img_tok_id)

ip = proc.image_processor
print("image_processor config:", ip.size, getattr(ip, "crop_size", None), getattr(ip, "do_center_crop", None), getattr(ip, "resample", None))

# build a synthetic image with a distinct-colored marker rectangle at a known location, run it
# through the processor, and inspect the resulting pixel_values shape + do a manual resize-crop
# check by comparing marker location before/after.
w, h = 640, 425  # a typical COCO-ish non-square size
img = Image.new("RGB", (w, h), (0, 0, 0))
arr = np.array(img)
# marker box in original coords
mx0, my0, mx1, my1 = 500, 50, 600, 150
arr[my0:my1, mx0:mx1] = (255, 0, 0)
img = Image.fromarray(arr)

out = proc(images=img, text="<image>\nUSER: test ASSISTANT:", return_tensors="pt")
pv = out["pixel_values"]
print("pixel_values shape:", pv.shape)
input_ids = out["input_ids"][0]
n_image_tokens = (input_ids == img_tok_id).sum().item()
print("n_image_tokens in input_ids:", n_image_tokens)
positions = (input_ids == img_tok_id).nonzero().flatten().tolist()
contiguous = positions == list(range(positions[0], positions[0] + len(positions))) if positions else False
print("image token positions contiguous:", contiguous, "first/last:", positions[0] if positions else None, positions[-1] if positions else None)

# locate marker in processed pixel_values to confirm resize-crop (not resize-pad) geometry
px = pv[0].permute(1, 2, 0).numpy()  # H,W,C after unnormalize approx (channels are normalized, just look at relative signal)
# red channel high, others low/negative relatively -> find bounding box of "marker-like" pixels
r, g, b = px[..., 0], px[..., 1], px[..., 2]
marker_mask = (r > r.mean() + 1.5 * r.std()) & (g < g.mean())
ys, xs = np.where(marker_mask)
if len(xs) > 0:
    print("marker found in processed image at x:[{},{}] y:[{},{}] out of {}".format(xs.min(), xs.max(), ys.min(), ys.max(), px.shape[:2]))
else:
    print("marker not clearly detected (normalization may obscure it) -- inspect manually if needed")
