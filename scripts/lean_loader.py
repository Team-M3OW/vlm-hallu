"""
Memory-lean replacement for `phase1_eval.build_items()` when you only need SOME items.

WHY THIS EXISTS
---------------
`build_items()` stores `row["image"]` for all 5553 POPE rows, which materializes every decoded PIL
image at once: measured **peak RSS 17.0 GB** (837 MB after imports, 1.8 GB after the COCO bbox
lookup, 17.0 GB after build_items). Earlier phases got away with it; with other jobs on the box
holding ~19 GB of 30 GB, it now OOMs and the process dies with no traceback.

This indexes the HF POPE dataset by (category, question_id) and decodes images ONE AT A TIME on
demand, so peak memory is one image rather than 5553.

uid convention is identical to build_items(): `pos_{split}_{question_id}` / `neg_{split}_{qid}`,
so uids already recorded in earlier phases' outputs resolve correctly. Note build_items() also
SUBSAMPLES negatives (500/split via the global RNG); we do not re-do that sampling -- we resolve
whatever uids the caller asks for, which are already fixed in the upstream result files.
"""
import re


class LeanItems:
    def __init__(self):
        from datasets import load_dataset
        self.ds = load_dataset("lmms-lab/pope", split="test")
        self.index = {}
        for i, row in enumerate(self.ds):
            self.index[(row["category"], str(row["question_id"]))] = i

    @staticmethod
    def parse(uid):
        m = re.match(r"^(pos|neg)_(random|popular|adversarial)_(\d+)$", uid)
        if not m:
            return None
        return m.group(2), m.group(3)

    def has(self, uid):
        k = self.parse(uid)
        return k is not None and k in self.index

    def get(self, uid):
        """-> {'image': PIL.Image, 'question': str} or None. Decodes one image."""
        k = self.parse(uid)
        if k is None or k not in self.index:
            return None
        row = self.ds[self.index[k]]
        return {"image": row["image"], "question": row["question"]}
