# Zero-shot transfer table + Table 6 audit (strictly from `FINDINGS.md`)

Date: 2026-09-26. Sources: `FINDINGS.md` (§71, §74, §14M), `data/phase225_*.jsonl`,
`data/phase224_dwat_*.jsonl`. No number below is estimated; missing entries are reported as missing.

## Transfer protocol (corrected: V* training and HR fitting)

The original DWA is **fitted (trained) on V*Bench** --- out-of-fold, against GT-box coverage labels
(`fig_ridge_w_*.npy`) --- and is then **applied to HR-Bench with NO refit**. The only target-side
computation is the **label-free feature standardisation** (mean/std recomputed over the unlabelled
target set; within each image for the extended benchmark protocol). A ridge refit on HR-Bench is
impossible in principle, because **HR-Bench ships no bounding boxes**, hence no coverage labels.

> "Weights fitted OOF on V* (`fig_ridge_w_*.npy`), applied to HR-Bench with **NO refit**; only the
> feature standardisation is recomputed on the target (label-free)." --- FINDINGS §71 (line 8679)

> "trained on V*Bench, 191 items, GT-box coverage labels (phase72a, frozen); applied to HR-Bench 4k,
> 200 instances x 4 permutations, 4032x4032 images; **refitted: nothing.** Not W, not the layer
> block, not the ring mask, not B0." --- `scripts/phase190_hr4k_qwen3_ridge.py` (Phase 72c header)

> "V*Bench-transferred 40% firing rate, **not refit**. No oracle arm --- HR-Bench ships no boxes."
> --- FINDINGS line 2647; see also lines 3830 and 3851 ("with zero refitting: +4.9pp per-row and
> +6.5pp on HR-Bench's own strict...").

**So the correct phrasing is: "DWA is trained on V*Bench and transferred to HR-Bench with no refit;
only the label-free feature standardisation is recomputed on the target."** The phrasing "trained on
V*, then fitted on HR" is incorrect and must not be used.

## AVR accuracy across all four benchmarks and models (4x4)

| Model | Benchmark | n | Reference (%) | AVR (%) | AVR - reference |
|---|---|---|---|---|---|
| Qwen3-VL-2B | V* | 191 | 62.3 | 67.5 | +5.2 (CI-clear) |
| Qwen3-VL-2B | HR-4K | 800 | 59.0 | 59.9 | +0.9 |
| Qwen3-VL-2B | CV-Bench | 2,638 | 79.9 | 80.4 | +0.5 |
| Qwen3-VL-2B | RealworldQA | 765 | 62.6 | 63.3 | +0.7 |
| Qwen2-VL-7B | V* | 191 | 59.2 | 64.9 | +5.8 (CI-clear) |
| Qwen2-VL-7B | HR-4K | 800 | 57.0 | 57.1 | +0.1 |
| Qwen2-VL-7B | CV-Bench | 2,638 | 75.0 | 71.5 | -3.5 (CI-clear loss) |
| Qwen2-VL-7B | RealworldQA | 765 | 64.7 | 62.6 | -2.1 |
| LLaVA-OV-7B | V* | 191 | 51.8 | 58.6 | +6.8 (CI-clear) |
| LLaVA-OV-7B | HR-4K | 800 | 48.2 | 46.2 | -2.0 |
| LLaVA-OV-7B | CV-Bench | 2,638 | 76.6 | 76.0 | -0.6 |
| LLaVA-OV-7B | RealworldQA | 765 | 61.6 | 60.0 | -1.6 |
| InternVL3-8B | (all) | --- | --- | n/a | n/a (no resolution ladder) |

Source: `data/phase225_{model}_{bench}.jsonl` (`avr` arm, paired vs `uniform@lo`); InternVL3-8B
cannot express AVR (token count is size-invariant; E_lo = E_hi), reported as infeasible.

## Zero-shot vs fitted (in-domain) DWA

"Train DWA for the specific benchmark" is only possible where the benchmark ships boxes. Beyond V*,
that is **CV-Bench Depth+Distance** (the only CV-Bench tasks with bounding boxes, n=1,200 of 2,638);
HR-Bench and RealworldQA ship **no boxes at all**, so an in-domain fit is impossible there in
principle. On V*, DWA is already V*-fitted, so its "zero-shot" and "fitted" columns coincide.

**4x4 zero-shot DWA** (reference / DWA / delta) is Table 6 above. Where an in-domain fit exists:

| CV-Bench Depth+Distance, n=400 | Reference | AVR | DWA zero-shot | DWA fitted (in-domain) | Fitted - zero-shot |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 87.5 | 86.8 | 75.5 | 75.8 | +0.3 |
| Qwen2-VL-7B | 78.5 | 77.8 | 76.5 | **78.0** | **+1.5** |
| LLaVA-OV-7B | -- | -- | -- | not run | -- |
| InternVL3-8B | -- | -- | -- | not run | -- |

Finding: fitting on the benchmark's own boxes does **not** rescue the scene-level tasks. Qwen3 is
unchanged (75.8 vs 75.5); Qwen2 recovers +1.5 (not CI-clear) and lands within 0.5 of its reference.
The transfer loss is model-dependent; the scene-level failure is not. AVR is the only arm that matches
the reference on both checkpoints. Logged as FINDINGS §88; scripts
`scripts/phase250_indomain_dwa.py`, data `data/phase250_indomain_{qwen3_2b,qwen2_7b}.jsonl`.

```latex
\begin{table}[htb]
    \centering
    \caption{\textbf{Zero-shot vs fitted DWA.} DWA is fitted on V*Bench and either transferred to the target (zero-shot) or refitted out-of-fold on the target's own boxes (fitted), where the benchmark ships boxes. HR-Bench and RealworldQA ship none, so no in-domain fit exists there. On V* the two coincide by construction.}
    \label{tab:dwa_fitted}
    \begin{tabular}{llccccc}
    \toprule
    \textbf{Model} & \textbf{Benchmark} & \textbf{Reference (\%)} & \textbf{Zero-shot (\%)} & \textbf{Fitted (\%)} & $\mathbf{\Delta}$ \\
    \midrule
    \multirow{4}{*}{Qwen3-VL-2B} & V* & 62.3 & 72.8 & 72.8 & $0.0$ \\
    & HR-4K & 59.0 & 62.5 & -- (no boxes) & -- \\
    & CV-Bench (Depth+Distance, n=400) & 87.5 & 75.5 & 75.8 & $+0.3$ \\
    & RealworldQA & 62.6 & 56.9 & -- (no boxes) & -- \\
    \midrule
    \multirow{4}{*}{Qwen2-VL-7B} & V* & 59.2 & 68.1 & 68.1 & $0.0$ \\
    & HR-4K & 57.0 & 59.1 & -- (no boxes) & -- \\
    & CV-Bench (Depth+Distance, n=400) & 78.5 & 76.5 & 78.0 & $+1.5$ \\
    & RealworldQA & 64.7 & 57.5 & -- (no boxes) & -- \\
    \bottomrule
    \end{tabular}
\end{table}
```

## Figures

- `paper/figs/fig_vlm_dwa_tikz.pdf` (also `.png`, source `paper/figs/fig_vlm_dwa_tikz.tex`):
  VLM internals + DWA --- layer bar with transport window (L0--15), boundary (L16), read-out band
  (L17--21) and answer formation (L22+); the ridge chain from per-layer attention to the crop and
  pass 2. Caption in `main.tex` (`fig:vlm_dwa`).
- `paper/figs/fig_avr_tikz.pdf` (also `.png`, source `paper/figs/fig_avr_tikz.tex`): AVR --- 900-token
  encoding, attention ranking at L12--16, top-10% retention (90 tokens), restriction from L17, and
  the 16,290 token-layer allowance (~97% of the 16,800 reference); no crop. Caption in `main.tex`
  (`fig:avr`).
- `paper/figs/fig_dwa_qualitative.pdf` (source `scripts/fig_dwa_qualitative.py`): block-mean edge
  artifact vs DWA on `direct_attributes/32` (water bottle, red). Caption at the end of this file.

## Task 1 — Zero-shot transfer table

**MISSING ZERO-SHOT DATA:** Source Qwen2-VL-7B (V*Bench) → Target LLaVA-OV-7B (V*Bench) — no
cross-model DWA weight transfer exists in `findings.md`. (§14M's cross-model weight load is a
*pruning* experiment — Qwen3 weights on Qwen2 attention at 10% keep, 50.3% — not the DWA crop
transfer.)

**MISSING ZERO-SHOT DATA:** Source Qwen2-VL-7B (V*Bench) → Target Qwen2-VL-7B (HR-4K) — the Baseline
and Zero-Shot accuracies are not stated in `findings.md`. §71 reports only the delta
`dwa_t − block` = **+14.5 [+7.5,+21.8]** (pooled, n=400 items / 100 question groups), and lists
`dwa_t − bar` pooled = −2.2 among four cells; no absolute accuracies.

**MISSING ZERO-SHOT DATA:** Source Qwen3-VL-2B (V*Bench) → Target Qwen2-VL-7B (V*Bench) — not in
`findings.md`.

For reference, what FINDINGS *does* contain is same-model cross-benchmark transfer (§71:
Qwen2→HR-4k/8k, Qwen3→HR-4k/8k; §74: LLaVA-OV→HR-8k with absolutes bar 50.7 / block 51.7 /
dwa_t 53.5).

```latex
\begin{table}[htb]
    \centering
    \caption{\textbf{Zero-shot transfer of DWA spatial weights.} Regression weights learned on the source model and dataset are applied directly to the target without retraining. \textbf{Entries marked ``--'' are not present in our experiment log; they are not reported rather than estimated.}}
    \label{tab:dwa_zeroshot}
    \begin{tabular}{llccc}
    \toprule
    \textbf{Source (Trained Weights)} & \textbf{Target (Evaluated On)} & \textbf{Baseline (\%)} & \textbf{Zero-Shot (\%)} & $\mathbf{\Delta}$ \\
    \midrule
    Qwen2-VL-7B (V*Bench) & LLaVA-OV-7B (V*Bench) & -- & -- & -- \\
    Qwen2-VL-7B (V*Bench) & Qwen2-VL-7B (HR-4K)   & -- & -- & -- \\
    Qwen3-VL-2B (V*Bench) & Qwen2-VL-7B (V*Bench) & -- & -- & -- \\
    \bottomrule
    \end{tabular}
\end{table}
```

## Task 2 — Table 6 verification

Cross-checked every cell against `findings.md` contexts and re-verified against the phase225 dumps
(the table's source). **No corrections needed** — all values are confirmed. Two notes:

- Three values are not stated anywhere in `findings.md` prose — Qwen3 RealworldQA DWA **56.9**,
  LLaVA-OV RealworldQA reference **61.6**, InternVL3 CV-Bench reference **81.7** — so they were
  re-verified directly from `phase225_{model}_{bench}.jsonl` (56.9 / 61.6 / 81.7 ✓). Strict
  FINDINGS-only provenance would mark these three as data-derived.
- Historical note: an older table draft listed InternVL3 CV-Bench Δ as −13.8; the current dumps give
  **−13.1** (n=2638), which is what the draft already has — keep it.

```latex
\begin{table}[htb]
    \centering
    \caption{\textbf{DWA accuracy across all four benchmarks and models.} The reference is the uncropped image at the equal-compute budget (600 visual tokens). Differences are paired and in percentage points; $\checkmark$ denotes the interval excludes zero (significant gain), and $\times$ denotes a significant loss.}
    \label{tab:dwa_all_benchmarks}
    \resizebox{\linewidth}{!}{
    \begin{tabular}{llccccc}
    \toprule
    \textbf{Model} & \textbf{Benchmark} & $\mathbf{n}$ & \textbf{Reference (\%)} & \textbf{DWA (\%)} & $\mathbf{\Delta}$ & \textbf{Sig.} \\
    \midrule
    \multirow{4}{*}{Qwen3-VL-2B} & V* & 191 & 62.3 & 72.8 & $+10.5$ & $\checkmark$ \\
    & HR-4K & 800 & 59.0 & 62.5 & $+3.5$ & \\
    & CV-Bench & 2,638 & 79.9 & 65.8 & $-14.1$ & $\times$ \\
    & RealworldQA & 765 & 62.6 & 56.9 & $-5.8$ & $\times$ \\
    \midrule
    \multirow{4}{*}{Qwen2-VL-7B} & V* & 191 & 59.2 & 68.1 & $+8.9$ & $\checkmark$ \\
    & HR-4K & 800 & 57.0 & 59.1 & $+2.1$ & \\
    & CV-Bench & 2,638 & 75.0 & 66.0 & $-9.0$ & $\times$ \\
    & RealworldQA & 765 & 64.7 & 57.5 & $-7.2$ & $\times$ \\
    \midrule
    \multirow{4}{*}{LLaVA-OV-7B} & V* & 191 & 51.8 & 65.4 & $+13.6$ & $\checkmark$ \\
    & HR-4K & 800 & 48.2 & 56.5 & $+8.2$ & $\checkmark$ \\
    & CV-Bench & 2,638 & 76.6 & 65.9 & $-10.7$ & $\times$ \\
    & RealworldQA & 765 & 61.6 & 59.2 & $-2.4$ & \\
    \midrule
    \multirow{4}{*}{InternVL3-8B} & V* & 191 & 74.9 & 75.9 & $+1.0$ & \\
    & HR-4K & 800 & 70.5 & 65.6 & $-4.9$ & $\times$ \\
    & CV-Bench & 2,638 & 81.7 & 68.6 & $-13.1$ & $\times$ \\
    & RealworldQA & 765 & 67.6 & 59.9 & $-7.7$ & $\times$ \\
    \bottomrule
    \end{tabular}
    }
\end{table}
```

## Figure

Generated: `paper/figs/fig_dwa_qualitative.pdf` (220 dpi, 1×3 tight layout; script
`scripts/fig_dwa_qualitative.py`). Example: `direct_attributes/32` — "What is the color of the water
bottle?" — target 26×68 px at the left edge (0.25 merged tokens). Block-mean arg-max at the last
column (20,11) → crop answers A (black) ×, p=0.42; DWA arg-max on the bottle (2,7) → crop answers B
(red) ✓, p=1.00.

```latex
\begin{figure}[htb]
    \centering
    \includegraphics[width=\linewidth]{Figures/fig_dwa_qualitative.pdf}
    \caption{\textbf{Crop placement and artifact mitigation.} (Left) The input image and ground-truth bounding box for a sub-token target (26$\times$68 px at the image's left edge, 0.25 merged tokens). (Middle) Standard block-mean attention (Layers 16--26) collapses toward the grid edge; its crop misses the target and answers \textit{black} ($p{=}0.42$). (Right) DWA integrates early-layer signals, places a clean hot-spot on the target, and its crop answers \textit{red} correctly ($p{=}1.00$).}
    \label{fig:dwa_qualitative}
\end{figure}
```
