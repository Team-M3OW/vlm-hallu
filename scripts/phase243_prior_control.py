"""Phase 243 (old-4): is the below-cliff model answer the language prior's answer?

Text-only control: run the V*Bench question (no image) through the same checkpoint, score the option
letters, and compare with the image-conditioned answer on below-cliff vs above-cliff items (cliff =
target extent < 0.25 merged tokens at B=300, boxes from phase30c). If below the cliff the model's
answer agrees with the text-only prior more than above it, the below-chance failure is the language
prior winning over an unresolved image.

Usage: phase243_prior_control.py <tag>    tag in {qwen3_2b, qwen2_7b}
"""
import json, os, sys, time, numpy as np, torch
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"qwen3_2b": ("Qwen/Qwen3-VL-2B-Instruct", "data/phase78_w_sweep.jsonl"),
          "qwen2_7b": ("Qwen/Qwen2-VL-7B-Instruct", "data/phase97m_merged_qwen2vl.jsonl")}
TAG = sys.argv[1]; MID, GRID = MODELS[TAG]
OUT = f"{D}/data/phase243_prior_{TAG}.jsonl"
rng = np.random.default_rng(243)

model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map={"": 0}).eval()
pr = AutoProcessor.from_pretrained(MID); tok = pr.tokenizer
letters = B.letter_ids(tok, 4)

items = B.load("vstar", None)
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
t0 = time.time(); n = 0
with open(OUT, "a") as f:
    for qid, img, q, gold, stratum, kind in items:
        if qid in done or not kind.startswith("mcq"): continue
        chat = pr.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": q}]}],
                                      tokenize=False, add_generation_prompt=True)
        inp = pr(text=[chat], return_tensors="pt").to(model.device)
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in letters]), 0)
        rec = {"qid": qid, "stratum": stratum, "gold": gold, "probs": [round(float(v), 6) for v in p.tolist()]}
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 50 == 0: print(f"  [{n}] {(time.time()-t0)/n:.2f}s/item", flush=True)
print(f"Done -> {OUT} ({n} new)", flush=True)

prior = {json.loads(l)["qid"]: json.loads(l) for l in open(OUT)}
grid = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/{GRID}")}
boxes = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl")}
q = [x for x in grid if x in prior]
lab = np.array([grid[x]["category"] == "direct_attributes" for x in q])
ext = np.array([(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(boxes[x]["gt_box_frac"]) * 300 for x in q])
below = ext < 0.25
model_pick = np.array([int(np.argmax(grid[x]["probs"]["uniform@300"])) for x in q])
prior_pick = np.array([int(np.argmax(prior[x]["probs"])) for x in q])
gold = np.array([grid[x]["label"] for x in q])
agree = (model_pick == prior_pick).astype(float)
print(f"\n{TAG}: n={len(q)}  prior acc {np.mean(prior_pick==gold)*100:.1f}%  model acc {np.mean(model_pick==gold)*100:.1f}%")
def ci(x):
    b = x[rng.integers(0, len(x), (6000, len(x)))].mean(1)
    return x.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100
for nm, m in (("all", np.ones(len(q), bool)), ("below cliff", below), ("above cliff", ~below)):
    if m.sum() < 5: continue
    a, l, h = ci(agree[m])
    print(f"  {nm:12s} n={m.sum():3d}  P(model pick = prior pick) {a:5.1f}% [{l:.1f},{h:.1f}]  |  model acc {np.mean(model_pick[m]==gold[m])*100:5.1f}%")
wrong = (model_pick != gold) & below
if wrong.sum() >= 5:
    a, l, h = ci(agree[wrong])
    print(f"  below-cliff WRONG (n={wrong.sum()}): pick = prior pick {a:.1f}% [{l:.1f},{h:.1f}]  (prior itself wrong on {(prior_pick[wrong]!=gold[wrong]).mean()*100:.0f}% of these)")
