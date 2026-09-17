"""
Phase 173 -- LASER (arXiv 2602.04304) ported: per-sample layer selection by Visual Activation by Query.
    A^con_{l,h} = ReLU( attn_with_query - attn_without_query ),  VAQ_{l,h} = ||A^con_{l,h}||_2,
    layer VAQ = mean of top-K heads; l* = argmax_l VAQ_l per sample; localise on A^con_{l*}.
We need the WITHOUT-query attention maps, which the stored dumps lack. Same items, same 300-token fit,
same read-out position (last prompt token) as phase 30c/74. Three conditions per item:
    q      the full V0 prompt (question + options + answer-instruction)          [= stored maps]
    noq    LASER's ablation: query removed, instruction kept ("Answer with the option's letter ...")
    bare   image + generation prompt only
Saves head-mean maps in jsonl (compatible with every analyser) and per-head maps in a float16 npz.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
MODELS = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}
B0 = 300; Image.MAX_IMAGE_PIXELS = None
INSTR = "Answer with the option's letter from the given choices directly."

def main(tag, dtype="bf16", suffix=""):
    DT={"bf16":torch.bfloat16,"fp16":torch.float16}[dtype]; suf=("" if dtype=="bf16" else f"_{dtype}")+suffix
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    mid = MODELS[tag]; out_path = f"data/phase173_laser_{tag}{suf}.jsonl"; npz_path = f"data/phase173_laser_{tag}{suf}_heads.npz"
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset"); ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=DT, device_map={"": 0}, attn_implementation="eager").eval()
    pr = AutoProcessor.from_pretrained(mid); itid = model.config.image_token_id
    MS = getattr(pr.image_processor, "merge_size", 1); nL = model.config.text_config.num_hidden_layers
    def build(img, text):
        content = [{"type": "image"}] + ([{"type": "text", "text": text}] if text else [])
        chat = pr.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")
    def measure(img):
        i = build(img, "x"); return int(sum(g[1]*g[2]//(MS*MS) for g in i["image_grid_thw"].tolist()))
    def fit(img, target, refine=6, tol=0.08):
        W_, H_ = img.size; sc = (target/max(measure(img),1))**0.5; best=None
        for _ in range(refine):
            cur = img.resize((max(28,int(W_*sc)), max(28,int(H_*sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best=(cur,r)
            if r==0 or abs(r-target)/target<=tol: break
            sc *= (target/r)**0.5
        return best[0]
    def maps(img, text):
        inp = build(img, text); pos = (inp["input_ids"][0]==itid).nonzero().flatten()
        base, ntok = int(pos[0].item()), int(len(pos))
        with torch.no_grad(): o = model(**inp.to(model.device), output_attentions=True)
        H = np.stack([o.attentions[L][0, :, -1, base:base+ntok].float().cpu().numpy() for L in range(nL)])   # L x heads x ntok
        del o; torch.cuda.empty_cache(); return H, ntok
    heads = {}; n=0; t0=time.time()
    with open(out_path, "w") as fout:
        for ex in ds:
            ap = os.path.splitext(os.path.join(root, ex["image"]))[0]+".json"
            if not os.path.exists(ap): continue
            ann = json.load(open(ap))
            if not ann.get("bbox"): continue
            img = Image.open(os.path.join(root, ex["image"])).convert("RGB"); IW, IH = img.size
            gt = [min(b[0] for b in ann["bbox"])/IW, min(b[1] for b in ann["bbox"])/IH,
                  max(b[0]+b[2] for b in ann["bbox"])/IW, max(b[1]+b[3] for b in ann["bbox"])/IH]
            small = fit(img, B0)
            conds = {"q": ex["text"], "noq": INSTR, "bare": ""}
            rec = {"question_id_full": f"{ex['category']}/{ex['question_id']}", "category": ex["category"], "gt_box_frac": gt, "attn": {}}
            ok = True; ntoks=set()
            for c, text in conds.items():
                H, ntok = maps(small, text); ntoks.add(ntok)
                if not np.isfinite(H).all(): ok=False; break
                # DEPLOYED convention (phases 30c/74): average RAW heads, then normalise over image tokens.
                Hm = H.mean(1); Hm = Hm / np.maximum(Hm.sum(-1, keepdims=True), 1e-12)
                rec["attn"][c] = {f"L{i}": [round(float(v), 8) for v in Hm[i]] for i in range(nL)}
                heads[f"{rec['question_id_full']}|{c}"] = H.astype(np.float32)          # raw per-head, unnormalised
            if not ok or len(ntoks)!=1: continue
            inp = build(small, ex["text"]); g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1]//MS, g[2]//MS
            if gh*gw != ntoks.pop(): continue
            rec["grid"]=[gh,gw]; rec["n_img_tokens"]=gh*gw
            fout.write(json.dumps(rec)+"\n"); fout.flush(); n+=1
            if n%20==0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    np.savez_compressed(npz_path, **heads)
    print(f"wrote {n} items -> {out_path}, heads -> {npz_path}", flush=True)
if __name__ == "__main__": main(sys.argv[1], *(sys.argv[2:4]))
