"""Shared benchmark loaders + scoring spec, so AVR (phase213) and DWA (phase224) score IDENTICAL
item sets. NOTHING IS FILTERED: the full public split is used, and the scorer adapts to the item.

Each loader yields (qid, image, prompt, gold, stratum, kind) where
  kind = "mcq<N>"  -> gold is the index of the correct option; score by logits over letters A..(N-1)
  kind = "open"    -> gold is the reference string; score by short generation + normalised exact match

Filtering to only the 4-way items would report on a non-standard subset (493 of 2638 on CV-Bench),
so the scorer generalises instead. POPE is deliberately excluded: its "no" items contain no object
to crop toward, and cropping biases a yes/no existence question toward "no".

cvbench      nyu-visionx/CV-Bench (2638). Native task labels are the strata; `Relation` is the
             cross-instance analogue of V*'s relative_position.
realworldqa  xai-org/RealworldQA (765). Mixed letter-MCQ and free-form.
"""
import os, re, string
for _v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(_v,None)
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache")
LET="ABCDEFGH"
MCQ_SUFFIX="\nAnswer with the option's letter from the given choices directly."

def cvbench(limit=None):
    from datasets import load_dataset
    ds=load_dataset("nyu-visionx/CV-Bench")["test"]; n=0
    for e in ds:
        ch=list(e["choices"]); k=len(ch)
        m=re.search(r"\(([A-H])\)", str(e["answer"]))
        if not m or k<2 or k>len(LET): continue
        gi=LET.index(m.group(1))
        if gi>=k: continue
        q=e["question"]+"\n"+"\n".join(f"({LET[i]}) {c}" for i,c in enumerate(ch))+MCQ_SUFFIX
        yield (f"cvbench/{e['idx']}", e["image"], q, gi, str(e["task"]), f"mcq{k}")
        n+=1
        if limit and n>=limit: break

def realworldqa(limit=None):
    from datasets import load_dataset
    ds=load_dataset("xai-org/RealworldQA")["test"]; n=0
    for i,e in enumerate(ds):
        a=str(e["answer"]).strip(); q=str(e["question"]).rstrip()
        opts=re.findall(r"^\s*([A-H])\.", q, re.M)
        if len(a)==1 and a in LET and len(opts)>=2:
            k=len(opts)
            if not q.endswith("directly."): q=q+MCQ_SUFFIX
            yield (f"rwqa/{i}", e["image"], q, LET.index(a), "mcq", f"mcq{k}")
        else:
            yield (f"rwqa/{i}", e["image"], q, a, "open", "open")
        n+=1
        if limit and n>=limit: break

def textvqa(limit=None):
    """lmms-lab/textvqa validation (5000). Open-ended; gold is the 10 human answers and an item counts
    as correct on a normalised match to ANY of them (the harness's approximation of VQA accuracy).
    Single stratum: these are OCR/attribute questions about one text region."""
    from datasets import load_dataset
    ds=load_dataset("lmms-lab/textvqa")["validation"]; n=0  # test annotations are hidden; validation is the eval split
    for e in ds:
        ans=[str(a).strip() for a in e["answers"]]
        ans=[a for a in ans if a]
        if not ans: continue
        q=str(e["question"]).strip()+"\nAnswer with a single word or short phrase."
        yield (f"textvqa/{e['question_id']}", e["image"], q, ans, "ocr", "open")
        n+=1
        if limit and n>=limit: break

_ART=re.compile(r"\b(a|an|the)\b")
def norm(s):
    s=str(s).lower().strip()
    s=s.translate(str.maketrans("","",string.punctuation))
    s=_ART.sub(" ",s)
    return " ".join(s.split())
def open_match(pred,gold): return norm(pred)==norm(gold)

def _sample(items,limit,seed=225):
    """Deterministic shuffle-then-take. A PREFIX would be biased: CV-Bench is ordered by task, so
    the first N items are all `Count`. Shuffling preserves the kind/stratum mixture; nothing is
    excluded by type."""
    if not limit or limit>=len(items): return items
    import random
    r=random.Random(seed); idx=list(range(len(items))); r.shuffle(idx)
    return [items[i] for i in sorted(idx[:limit])]

def load(name,limit=None):
    items=list(LOADERS[name]())
    return _sample(items,limit)


def vstar(limit=None):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from PIL import Image as _I
    root=snapshot_download("craigwu/vstar_bench",repo_type="dataset")
    ds=load_dataset("craigwu/vstar_bench")["test"]; n=0
    for e in ds:
        lab=LET.index(e["label"]) if isinstance(e["label"],str) else int(e["label"])
        img=_I.open(os.path.join(root,e["image"]))
        yield (f"{e['category']}/{e['question_id']}", img, e["text"], lab, e["category"], "mcq4")
        n+=1
        if limit and n>=limit: break

def _as_img(v):
    """HR-Bench stores images as base64 JPEG STRINGS. Decode here so every loader yields PIL."""
    from PIL import Image as _I
    import io as _io, base64 as _b64
    if isinstance(v,_I.Image): return v
    if isinstance(v,dict) and "bytes" in v: return _I.open(_io.BytesIO(v["bytes"]))
    if isinstance(v,(bytes,bytearray)): return _I.open(_io.BytesIO(v))
    if isinstance(v,str): return _I.open(_io.BytesIO(_b64.b64decode(v)))
    raise TypeError(f"unhandled image type {type(v)}")

def _hrbench(cfg,limit=None):
    from datasets import load_dataset
    ds=load_dataset("DreamMr/hr-bench","hrbench_version_split")[cfg]; n=0
    for e in ds:
        q=e["question"]+"\n"+"\n".join(f"({c}) {e[c]}" for c in "ABCD" if c in e)+MCQ_SUFFIX
        i=int(e.get("index",n))
        yield (f"{cfg}/{i}", _as_img(e["image"]), q, LET.index(str(e["answer"]).strip()[0]),
               e.get("category","all"), "mcq4")
        n+=1
        if limit and n>=limit: break

def hr4k(limit=None): return _hrbench("hrbench_4k",limit)
def hr8k(limit=None): return _hrbench("hrbench_8k",limit)

LOADERS={"cvbench":cvbench,"realworldqa":realworldqa,"textvqa":textvqa,
        "vstar":vstar,"hr4k":hr4k,"hr8k":hr8k}
if __name__=="__main__":
    import collections
    for k,fn in LOADERS.items():
        items=list(fn())
        print(f"{k}: {len(items)} items (UNFILTERED)")
        print("   kinds:",dict(collections.Counter(x[5] for x in items)))
        print("   strata:",dict(collections.Counter(x[4] for x in items)))
        sz=[items[i][1].size for i in range(0,min(len(items),400),80)]
        print("   sample sizes:",sz)

# ---------------------------------------------------------------------------
# Adaptive scoring. Both methods must score an item the SAME way, so it lives here.
def letter_ids(tok, n):
    """token-id variants for the first token of each option letter A..(n-1)"""
    out=[]
    for c in LET[:n]:
        ids=set()
        for form in (c, f" {c}", f"({c}"):
            t=tok(form, add_special_tokens=False)["input_ids"]
            if t: ids.add(t[-1] if form!=f"({c}" else t[-1])
        out.append(sorted(ids))
    return out

def score_item(model, pr, tok, inp, kind, max_new=8):
    """Returns (probs_or_None, pred_string_or_None). mcq<N> -> letter logits; open -> short greedy."""
    import torch
    if kind.startswith("mcq"):
        n=int(kind[3:])
        ids=letter_ids(tok,n)
        with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
        p=torch.softmax(torch.stack([torch.logsumexp(lg[i],0) for i in ids]),0)
        return [round(float(v),6) for v in p.tolist()], None
    with torch.no_grad():
        out=model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=getattr(tok,"pad_token_id",None) or getattr(tok,"eos_token_id",None))
    gen=out[0][inp["input_ids"].shape[1]:]
    return None, tok.decode(gen, skip_special_tokens=True).strip()

def item_correct(rec, arm):
    """rec carries kind+gold; arm is a key in rec['probs'] (mcq) or rec['preds'] (open)."""
    if rec["kind"]=="open":
        gold=rec["gold"]; pred=rec["preds"][arm]
        if isinstance(gold,(list,tuple)):
            return 1.0 if any(open_match(pred,g) for g in gold) else 0.0
        return 1.0 if open_match(pred, gold) else 0.0
    import numpy as np
    return 1.0 if int(np.argmax(rec["probs"][arm]))==int(rec["gold"]) else 0.0
