"""
Phase 104: THE LLM INTERPRETABILITY LITERATURE SAYS OUR READ-OUT IS THE WRONG QUANTITY.

Every locator in this project -- and in the VLM allocation literature -- reads RAW attention and
averages it over a block of layers. Two well-cited results say both halves of that are naive:

  Abnar & Zuidema (ACL'20), "Quantifying Attention Flow". Information from different tokens gets
  progressively MIXED across layers, so per-layer attention is not a faithful attribution and
  averaging layers treats them as independent when they are not. Attention ROLLOUT instead
  multiplies the per-layer matrices with a residual identity, renormalised.

  Kobayashi et al. (EMNLP'20), "Attention is Not Only a Weight". The attention weight is only half
  of what determines the output; the other half is the norm of the transformed value vector. Their
  headline case is BERT appearing to attend to special tokens -- high alpha, low ||v||. That is our
  serialization sink, in the text domain, five years earlier. Our head currently has to LEARN the
  sink from explicit last-column/last-row flags; norm weighting may remove it for free.

  Chefer et al. (CVPR'21) argue attention alone is unfaithful and use gradient-weighted relevance.
  Included here in its cheap form, grad x attention, clamped at zero.

ARMS (all locators, all scored on the deployed metric: top-1 coverage with the ring mask)
    block        mean of BLOCK layers, raw attention                    <- THE INCUMBENT
    block_all    mean of ALL layers, raw attention                      <- the known-bad control
    normw        mean of BLOCK layers of (attention x ||v||)            <- Kobayashi
    normw_all    mean of ALL layers of (attention x ||v||)
    rollout      Abnar & Zuidema rollout, last row over image columns   <- principled aggregation
    rollout_nw   rollout with each layer right-weighted by ||v||        <- both fixes
    gradattn     mean over layers of relu(grad x attention)             <- Chefer-lite

PRIOR ART, checked first this time: value/key norms ARE already used for VLM token PRUNING
(2406.12335 and successors), and one of them states outright that sink tokens show "a striking
contrast between their attention scores and value vector norms". The increment here is applying it
to PROPOSAL GENERATION and scoring it on evidence coverage, not on pruned-model accuracy. Rollout
for VLM crop placement appears open but must be re-checked before any claim.

WHAT WOULD CHANGE THE PAPER
    a training-free locator beating the learned head -> the method gets simpler AND more principled
    none of them beating it                          -> "text-attribution fixes do not transfer to
                                                        visual localisation", which is its own result
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", "data/phase71a_head_proposals.json", (16, 27)),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", "data/phase80a_qwen2vl_proposals.json", (15, 27))}
MODEL_ID, PROP, BLK = CFG[WHICH]
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
OUT = f"{D}/data/phase104_locators_{WHICH}.jsonl"
B0, W = 300, 0.25
Image.MAX_IMAGE_PIXELS = None


def coverage(cx, cy, gt, w=W):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2
    if x0 < 0: x0, x1 = 0.0, w
    if y0 < 0: y0, y1 = 0.0, w
    if x1 > 1: x0, x1 = 1 - w, 1.0
    if y1 > 1: y0, y1 = 1 - w, 1.0
    gx0, gy0, gx1, gy1 = gt
    inter = max(0.0, min(gx1, x1) - max(gx0, x0)) * max(0.0, min(gy1, y1) - max(gy0, y0))
    return inter / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(f"{D}/{PROP}"))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    img_tok_id = model.config.image_token_id

    layers = (model.model.language_model.layers if hasattr(model.model, "language_model")
              else model.language_model.model.layers)
    NL = len(layers)
    vstore = {}
    def mk_hook(i):
        def h(mod, inp, out):
            vstore[i] = out.detach().float()[0]      # (n, kv_dim)
        return h
    for i, L in enumerate(layers):
        L.self_attn.v_proj.register_forward_hook(mk_hook(i))

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best

    def ring(gh, gw):
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        return m.ravel()

    t0, n = time.time(), 0
    with open(OUT, "w") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}" if "question_id" in ex else None
            qid = qid or ex.get("question_id_full")
            if qid not in props: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            gt = props[qid]["gt_box_frac"]
            img, _ = fit(Image.open(ip).convert("RGB"), B0)
            inp = build(img, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            n_img = gh * gw
            pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
            base = int(pos[0].item())
            inp = inp.to(model.device)
            vstore.clear()
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            A = torch.stack([a[0].float().mean(0) for a in out.attentions])     # (NL, n, n)
            ntok = A.shape[-1]
            last = ntok - 1
            vn = torch.stack([vstore[i].norm(dim=-1) for i in range(NL)])       # (NL, n)
            vn = vn / vn.mean(-1, keepdim=True).clamp_min(1e-9)

            raw = A[:, last, base:base + n_img]                                  # (NL, n_img)
            raw = raw / raw.sum(-1, keepdim=True).clamp_min(1e-12)
            nw = A[:, last, :] * vn                                              # (NL, n)
            nw = nw[:, base:base + n_img]
            nw = nw / nw.sum(-1, keepdim=True).clamp_min(1e-12)

            eye = torch.eye(ntok, device=A.device)
            def roll(weight=None):
                R = eye.clone()
                for l in range(NL):
                    M = A[l] if weight is None else A[l] * weight[l].unsqueeze(0)
                    M = 0.5 * M + 0.5 * eye
                    M = M / M.sum(-1, keepdim=True).clamp_min(1e-12)
                    R = M @ R
                v = R[last, base:base + n_img]
                return (v / v.sum().clamp_min(1e-12)).detach().cpu().numpy()
            ro, ro_nw = roll(), roll(vn)

            del out
            # grad x attention: one backward pass from the top logit at the last position
            inp2 = {k: v for k, v in inp.items()}
            o2 = model(**inp2, output_attentions=True)
            lg = o2.logits[0, -1]
            att = o2.attentions
            gr = torch.autograd.grad(lg.max(), att, retain_graph=False, allow_unused=True)
            ga = []
            for a, gd in zip(att, gr):
                if gd is None: ga.append(np.zeros(n_img)); continue
                r_ = (gd[0].float() * a[0].float()).clamp_min(0).mean(0)[last, base:base + n_img]
                ga.append((r_ / r_.sum().clamp_min(1e-12)).detach().cpu().numpy())
            ga = np.stack(ga).mean(0)
            del o2, att, gr

            b0, b1 = BLK
            maps = {
                "block":      raw[b0:b1].mean(0).detach().cpu().numpy(),
                "block_all":  raw.mean(0).detach().cpu().numpy(),
                "normw":      nw[b0:b1].mean(0).detach().cpu().numpy(),
                "normw_all":  nw.mean(0).detach().cpu().numpy(),
                "rollout":    ro,
                "rollout_nw": ro_nw,
                "gradattn":   ga,
            }
            rm = ring(gh, gw)
            yy, xx = np.mgrid[0:gh, 0:gw]
            fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
            cov = np.array([coverage(float(fx[i]), float(fy[i]), gt) for i in range(n_img)])
            rec = {"question_id_full": qid, "category": ex["category"], "grid": [gh, gw],
                   "n_img": n_img, "cov_max": float(cov.max()),
                   "pick": {}, "cov": {}, "gt_pct": {}}
            for k, m in maps.items():
                s = np.where(rm, m, -1e9)
                j = int(np.argmax(s))
                rec["pick"][k] = j
                rec["cov"][k] = float(cov[j])
                gtc = cov >= 0.5
                rec["gt_pct"][k] = float(np.mean(m[gtc].mean() < m)) if gtc.any() else None
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    print(f"Done. {n} rows -> {OUT}", flush=True)


main()
