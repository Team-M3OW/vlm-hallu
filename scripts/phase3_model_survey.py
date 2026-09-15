"""
Prevalence survey (2026-09-03): how widespread is center-crop preprocessing across popular
open-weight VLMs? Fetches ONLY preprocessor_config.json (a few KB, no model weights) from the HF
Hub for a spread of VLM families/eras, and inspects it for a center-crop step vs an
aspect-ratio-preserving alternative (padding, dynamic/native resolution, resize-only).

Cheap and general: turns "LLaVA-1.5 has a preprocessing bug" into "how many widely-used VLMs
inherit this same risk."
"""
import json
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoProcessor

MODELS = [
    ("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5 (2023)"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", "LLaVA-NeXT/1.6 (2024)"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "LLaVA-OneVision (2024)"),
    ("Salesforce/instructblip-vicuna-7b", "InstructBLIP (2023)"),
    ("Salesforce/blip2-opt-2.7b", "BLIP-2 (2023)"),
    ("HuggingFaceM4/idefics2-8b", "Idefics2 (2024)"),
    ("HuggingFaceM4/Idefics3-8B-Llama3", "Idefics3 (2024)"),
    ("Qwen/Qwen-VL-Chat", "Qwen-VL (2023)"),
    ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL (2024)"),
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL (2025, already tested)"),
    ("OpenGVLab/InternVL2-8B", "InternVL2 (2024)"),
    ("openbmb/MiniCPM-V-2_6", "MiniCPM-V-2.6 (2024)"),
    ("google/paligemma-3b-mix-224", "PaliGemma (2024)"),
    ("adept/fuyu-8b", "Fuyu-8B (2023)"),
    ("microsoft/kosmos-2-patch14-224", "Kosmos-2 (2023)"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", "Llama-3.2-Vision (2024)"),
]


def resolve_via_processor_class(model_id):
    """Instantiate the actual processor class to read its RESOLVED do_center_crop default
    (transformers processor classes often bake in a class-level default that is absent from the
    saved preprocessor_config.json, so raw-JSON field presence alone is not reliable for a
    definitive verdict -- this is the more trustworthy check)."""
    for loader in (AutoImageProcessor, AutoProcessor):
        try:
            proc = loader.from_pretrained(model_id, trust_remote_code=True)
            ip = getattr(proc, "image_processor", proc)
            do_crop = getattr(ip, "do_center_crop", None)
            crop_size = getattr(ip, "crop_size", None)
            if do_crop is True:
                return f"CROPS (resolved do_center_crop=True, crop_size={crop_size})"
            elif do_crop is False:
                return "no crop (resolved do_center_crop=False)"
            else:
                return f"UNRESOLVED even via class instantiation (no do_center_crop attr on {type(ip).__name__})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
    return f"could not instantiate: {last_err}"


def main():
    results = []
    for model_id, label in MODELS:
        try:
            path = hf_hub_download(model_id, "preprocessor_config.json")
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            results.append((model_id, label, "ERROR",
                             f"ERROR (config fetch): {type(e).__name__}: {str(e)[:80]}"))
            continue

        do_crop = cfg.get("do_center_crop", None)
        crop_size = cfg.get("crop_size", None)
        do_resize = cfg.get("do_resize", None)
        size = cfg.get("size", None)
        proc_type = cfg.get("image_processor_type", cfg.get("processor_class", "?"))

        if do_crop is True:
            bucket = "CROPS"
            verdict = "CROPS (center-crop confirmed via raw config)"
        elif do_crop is False:
            bucket = "NOCROP"
            verdict = "no crop (do_center_crop=False in raw config)"
        elif "min_pixels" in cfg or "max_pixels" in cfg:
            bucket = "NOCROP"
            verdict = "dynamic resolution, no fixed crop (Qwen2VL-style min/max_pixels)"
        else:
            # Ambiguous from raw JSON alone -- resolve via actual processor class defaults.
            resolved = resolve_via_processor_class(model_id)
            if resolved.startswith("CROPS"):
                bucket = "CROPS"
            elif resolved.startswith("no crop"):
                bucket = "NOCROP"
            else:
                bucket = "UNCLEAR"
            verdict = f"[raw config ambiguous] -> class-resolved: {resolved}"

        results.append((model_id, label, bucket,
                         f"{verdict} | processor_type={proc_type} | "
                         f"do_resize={do_resize} size={size} crop_size={crop_size}"))

    print(f"{'Model':<45s} {'Era/label':<32s} {'Bucket':<8s} Verdict")
    print("-" * 150)
    for model_id, label, bucket, verdict in results:
        print(f"{model_id:<45s} {label:<32s} {bucket:<8s} {verdict}")

    n_crop = sum(1 for _, _, b, _ in results if b == "CROPS")
    n_nocrop = sum(1 for _, _, b, _ in results if b == "NOCROP")
    n_unclear = sum(1 for _, _, b, _ in results if b == "UNCLEAR")
    n_error = sum(1 for _, _, b, _ in results if b == "ERROR")
    print(f"\nSummary (honest buckets, no substring-matching bugs): {n_crop} CONFIRMED center-crop, "
          f"{n_nocrop} CONFIRMED no-crop/dynamic, {n_unclear} genuinely unclear, "
          f"{n_error} unresolvable (gated/missing), out of {len(results)} surveyed")


if __name__ == "__main__":
    main()
