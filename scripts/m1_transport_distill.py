"""
M1 (GPU): MECHANISM-GUIDED FINE-TUNING — move the localization signal into the causal channel.

Finding this is built on (§20B, §52, §61): the image reaches the answer at L0-16, but the attention map that
localizes the evidence lives at L17-21 and is causally inert. DWA works around this by reading the inert band.
This experiment instead TRAINS the model so that layers inside the transport window (L10-16) carry the
localization signal, with a LoRA and an attention-alignment loss to the GT-box mask, plus the task loss.

PRE-REGISTERED (written before the run)
  P1 (primary)  fixed-layer L14 read-out coverage on HELD-OUT items improves by >= 5 points over the
                pre-training model (paired, ring-masked argmax at W=0.25).
  P2            the causal profile shifts: masking all text rows' image attention for prefix L0..16 gives a
                LARGER output KL after training than before (the window now carries more evidence); the
                answer-row-only mask stays at the noise floor.
  P3            one-pass accuracy (no crop) improves by >= 3 points on held-out items.
  If P1 fails, the localization signal is not relocatable by this objective — reported as a negative.

Split: every 4th dataset item held out (48 test, 143 train), fixed before training.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
SMOKE = "--smoke" in sys.argv
CEONLY = "--ceonly" in sys.argv
SPLIT = 1 if "--split1" in sys.argv else 3
TAG = ("_ceonly" if CEONLY else "") + ("_split1" if SPLIT == 1 else "") + ("_smoke" if SMOKE else "")
LAM = 0.0 if CEONLY else 1.0
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = f"{D}/data/m1_transport_distill{TAG}.json"
ADAPTER = f"{D}/models/m1_lora_qwen3{TAG}"
B0, W = 300, 0.25; ALIGN_L = list(range(10, 17)); EPOCHS = 3; ACC = 4; LR = 1e-4
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
                                                        attn_implementation="eager")
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
    layers = model.model.language_model.layers
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]

    def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
    def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
    def measure(i): return int(sum(g[1] * g[2] // 4 for g in build(i, "x")["image_grid_thw"].tolist()))
    def fit(img, target=B0, refine=6, tol=0.08):
        W_, H_ = img.size; sc = (target / max(measure(img), 1)) ** 0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol: break
            sc *= (target / r) ** 0.5
        return best[0]

    # ---- precompute items ----
    items = []
    for idx, ex in enumerate(ds):
        ip = os.path.join(root, ex["image"])
        if not os.path.exists(ip): continue
        img = fit(Image.open(ip).convert("RGB"))
        inp = build(img, ex["text"])
        ids = inp["input_ids"][0]; pos = (ids == itid).nonzero().flatten()
        if len(pos) == 0: continue
        g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1] // 2, g[2] // 2
        j = os.path.splitext(ip)[0] + ".json"
        boxes = json.load(open(j)).get("bbox") if os.path.exists(j) else None
        if not boxes: continue
        iw, ih = Image.open(ip).size
        gt = (min(b[0] for b in boxes) / iw, min(b[1] for b in boxes) / ih,
              max(b[0] + b[2] for b in boxes) / iw, max(b[1] + b[3] for b in boxes) / ih)
        yy, xx = np.mgrid[0:gh, 0:gw]
        x0c, x1c = xx / gw, (xx + 1) / gw; y0c, y1c = yy / gh, (yy + 1) / gh
        ox = np.maximum(0, np.minimum(x1c, gt[2]) - np.maximum(x0c, gt[0]))
        oy = np.maximum(0, np.minimum(y1c, gt[3]) - np.maximum(y0c, gt[1]))
        mask = (ox * oy * gh * gw).ravel().astype(np.float32)   # cell coverage
        mask = mask / max(mask.sum(), 1e-9)
        lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
        items.append({"qid": f"{ex['category']}/{ex['question_id']}", "inp": inp, "imgpos": (int(pos[0]), int(len(pos))),
                      "mask": mask, "label": lab, "gh": gh, "gw": gw, "test": idx % 4 == SPLIT})
    print(f"{len(items)} items ({sum(i['test'] for i in items)} test)", flush=True)

    # ---- LoRA ----
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg); model.print_trainable_parameters()

    caps = {}
    hooks = []
    def mk(l):
        def fn(mod, args, output):
            caps[l] = output[1] if isinstance(output, tuple) else None
        return fn
    for l in ALIGN_L:
        hooks.append(layers[l].self_attn.register_forward_hook(mk(l)))

    def probs_letters(lg):
        return torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)

    def forward_item(it, want_attn=True):
        inp = {k: v.to(model.device) for k, v in it["inp"].items()}
        caps.clear()
        out = model(**inp)
        lg = out.logits[0, -1].float()
        p = probs_letters(lg)
        a = None
        if want_attn and all(l in caps and caps[l] is not None for l in ALIGN_L):
            base, nt = it["imgpos"]
            a = torch.stack([caps[l][0, :, -1, base:base + nt].float().mean(0) for l in ALIGN_L])  # (Layers, cells)
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        return p, a

    def loss_item(it):
        inp = {k: v.to(model.device) for k, v in it["inp"].items()}
        caps.clear()
        out = model(**inp)
        lg = out.logits[0, -1].float()
        ce = torch.nn.functional.cross_entropy(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]).unsqueeze(0),
                                               torch.tensor([it["label"]], device=lg.device))
        base, nt = it["imgpos"]
        tgt = torch.tensor(it["mask"], device=lg.device)
        tgt = tgt / tgt.sum().clamp_min(1e-9)
        la = 0.0
        for l in ALIGN_L:
            if caps[l] is None: continue
            av = caps[l][0, :, -1, base:base + nt].float().mean(0)
            av = av / av.sum().clamp_min(1e-9)
            la = la + (1.0 - (av * tgt).sum() / (av.norm() * tgt.norm()).clamp_min(1e-9))
        return ce + LAM * la / len(ALIGN_L), float(ce), float(la) / len(ALIGN_L)

    def evaluate(tag):
        model.eval()
        res = []
        with torch.no_grad():
            for it in items:
                if not it["test"]: continue
                if SMOKE and len(res) >= 4: break
                p, a = forward_item(it)
                rec = {"qid": it["qid"], "label": it["label"], "probs": [round(float(v), 6) for v in p.tolist()]}
                if a is not None:
                    gh, gw = it["gh"], it["gw"]
                    rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
                    for li, l in enumerate(ALIGN_L):
                        av = a[li].cpu().numpy(); j = int(np.argmax(np.where(rm, av, -1e9)))
                        cx, cy = (j % gw + .5) / gw, (j // gw + .5) / gh
                        g = it["mask"]
                        # coverage of the W window at (cx,cy) against the soft mask
                        x0, x1 = max(0., cx - W / 2), min(1., cx + W / 2); y0, y1 = max(0., cy - W / 2), min(1., cy + W / 2)
                        rec[f"cov_L{l}"] = float(sum(g[k] for k in range(len(g))
                                                     if x0 <= (k % gw + .5) / gw <= x1 and y0 <= (k // gw + .5) / gh <= y1))
                res.append(rec)
        acc = 100 * np.mean([int(np.argmax(r["probs"]) == r["label"]) for r in res])
        cov = {f"L{l}": float(np.mean([r.get(f"cov_L{l}", 0) for r in res])) for l in ALIGN_L}
        print(f"  [{tag}] held-out n={len(res)} acc {acc:.1f}%  cov " +
              " ".join(f"L{l}:{cov[f'L{l}']:.3f}" for l in ALIGN_L), flush=True)
        return {"acc": acc, "cov": cov, "items": res}

    before = evaluate("before")
    # ---- train ----
    model.train()
    tr = [it for it in items if not it["test"]]
    if SMOKE: tr = tr[:3]
    params = [p for p in model.parameters() if p.requires_grad]
    optz = torch.optim.AdamW(params, lr=LR)
    step = 0; t0 = time.time()
    for ep in range(EPOCHS):
        rng = np.random.default_rng(2100 + ep); order = rng.permutation(len(tr))
        acc_g = 0.0
        for k, i in enumerate(order):
            it = tr[i]
            loss, ce, la = loss_item(it)
            (loss / ACC).backward()
            if SMOKE:
                gn = float(sum(p.grad.norm() ** 2 for p in params if p.grad is not None) ** 0.5)
                print(f"  smoke step {k}: loss {float(loss):.4f} ce {ce:.4f} att {la:.4f} gradnorm {gn:.4f}", flush=True)
            acc_g += float(ce)
            if (k + 1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optz.step(); optz.zero_grad(set_to_none=True); step += 1
            if (k + 1) % 20 == 0:
                print(f"  ep{ep} {k+1}/{len(tr)} loss {(float(loss)):.3f} ce {acc_g/20:.3f}  {time.time()-t0:.0f}s", flush=True)
                acc_g = 0.0
    after = evaluate("after")
    model.save_pretrained(ADAPTER)
    json.dump({"before": {k: v for k, v in before.items() if k != "items"},
               "after": {k: v for k, v in after.items() if k != "items"},
               "items_before": before["items"], "items_after": after["items"]}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}; adapter {ADAPTER}", flush=True)


main()
