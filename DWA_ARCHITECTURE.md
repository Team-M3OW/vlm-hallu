# DWA ridge — architecture

Rendered figure: `paper/figs/fig_dwa_arch.pdf` / `.png`.
Mermaid source: `paper/figs/mermaid/dwa_ridge.mmd`.

```mermaid
flowchart LR
  subgraph TRAIN["TRAINING — once, ~50 boxed examples"]
    BOX["boxed examples<br/>V*Bench GT boxes"] --> LAB["label per cell <i>c</i>:<br/><b>y(c)</b> = fraction of the GT box<br/>inside the W=0.25 window at c"]
    LAB --> FIT["<b>one closed-form ridge</b><br/>standardise each training fold<br/><b>w = (XᵀX + αI)⁻¹Xᵀy</b>, α=1<br/>intercept unpenalised<br/>OOF GroupKFold(5) × 3 seeds,<br/>grouped by item"]
    FIT --> W["<b>weights w</b> — 64 coefficients<br/>(logA 28 + rank 28 + geo 7 + intercept)<br/><i>w<sub>ℓ</sub> is signed: the fitted weights ARE the depth filter</i>"]
  end

  subgraph INF["INFERENCE — per question"]
    IMG["image W₀×H₀"] --> P1["<b>PASS 1 — localise</b><br/>encode @300 visual tokens<br/>full prompt ending at the<br/>answer-emission position"]
    P1 --> ATT["per-layer attention<br/>final prompt token → image cells<br/>head-mean, row-normalised per layer<br/><b>A<sub>ℓ</sub>(c)</b> — 28 layers × C cells"]
    ATT --> PHI["<b>per-cell feature vector φ(c)</b><br/>log A<sub>ℓ</sub> (28) | within-layer rank (28)<br/>head stats log-max, std (56)* | geometry (7)<br/>log 3×3 nb · centre · edge · sink flags"]
    PHI --> S["<b>score map</b><br/>s(c) = wᵀφ(c)"]
    W --> S
    S --> C["placement<br/><b>ring-masked arg max</b><br/>c* = arg max<sub>c</sub> s(c)"]
    C --> CR["crop W=0.25 window at c*<br/>from the <u>original</u> image<br/>re-encode @300 tokens"]
    CR --> ANS["<b>PASS 2 — answer</b><br/>arg max over option-letter<br/>log-probabilities"]
  end

  ANS --> COST["<b>cost: 300 + 300 = 600 visual tokens = the equal-compute bar</b>"]
  W -. "at deployment the same ~50 boxes are reused;<br/>no refit per benchmark" .-> S
  PHI -. "*head-augmented variant = 91 features (phase 231–233)" .-> S

  classDef train fill:#dcfce7,stroke:#1a7f37,stroke-width:1.6px,color:#052e16
  classDef infer fill:#dbeafe,stroke:#0969da,stroke-width:1.6px,color:#0b2545
  classDef key fill:#fff8c5,stroke:#9a6700,stroke-width:2.2px,color:#3d2c00
  classDef cost fill:#f6f8fa,stroke:#57606a,stroke-width:1.4px,color:#24292f
  class BOX,LAB,FIT train
  class IMG,P1,ATT,PHI,C,CR,ANS infer
  class W,S key
  class COST cost
```

## Components (as implemented)

| block | detail | code |
|---|---|---|
| Localise pass | 300 visual tokens, full prompt ending at the answer-emission position; attention read from the final prompt token to image cells, head-mean, row-normalised per layer | `phase30c_dump_attn.py`, `phase225_newbench.py::localise` |
| Features (deployed, 63) | 28 log A_l, 28 within-layer ranks, 7 geometry (log 3x3 nb of the block-mean map, centre/edge, fx/fy, last-col/row sink flags) | `phase202_feature_ablation.py::feats_from` |
| Features (head variant, 91) | 28 log A_l, 28 log head-max, 28 head-std (per-cell head distribution), 7 geometry | `phase232_fit_headridge.py` |
| Ridge | `w = (XᵀX + αI)⁻¹Xᵀy`, α=1, unpenalised intercept, standardisation per training fold, OOF GroupKFold(5) × 3 seeds grouped by item | `phase184_allarms.py`, `phase202_*`, `phase232_*` |
| Label | fraction of the GT box inside the W=0.25 window centred at the cell | `phase70_rerank_head.coverage` |
| Placement | ring-masked arg max of `s(c) = wᵀφ(c)` | `fig_ridge_scores.py`, `phase225_*` |
| Answer | crop W=0.25 at the cell from the original image, re-encode @300 tokens, arg max over option-letter log-probs | `phase184_allarms.py::answer` |
| Cost | 300 + 300 = 600 visual tokens = equal-compute bar | — |
