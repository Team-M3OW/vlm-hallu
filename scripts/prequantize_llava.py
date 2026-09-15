"""
One-off: quantize LLaVA-1.5-7B to 4-bit and save the pre-quantized checkpoint to disk, so the
main eval loop loads already-4-bit shards directly (peak memory ~= resident ~4.5GB) instead of
materializing full fp16 shards on GPU0 before quantizing (the transient spike that was OOM'ing
against the other ~39GB job sharing this GPU).
"""
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUT_DIR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/models/llava-1.5-7b-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

try:
    print("Attempting CPU-side quantization (avoids the GPU load-time spike entirely) ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cpu",
    )
except Exception as e:
    print(f"CPU quantization failed ({e}); falling back to GPU0 with tight cap + disk offload "
          f"for any overflow shard.")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        device_map={"": 0}, max_memory={0: "6GiB"},
        offload_folder="/home/kavinder/ARNABI_ARSH/vlm-hallu/models/_offload_tmp",
    )
print("Loaded. Saving pre-quantized checkpoint to", OUT_DIR)
model.save_pretrained(OUT_DIR)
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.save_pretrained(OUT_DIR)
print("Done.")
