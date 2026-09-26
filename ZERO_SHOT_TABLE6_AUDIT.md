# Zero-shot transfer table + Table 6 audit (strictly from `FINDINGS.md`)

Date: 2026-09-26. Sources: `FINDINGS.md` (§71, §74, §14M), `data/phase225_*.jsonl`,
`data/phase224_dwat_*.jsonl`. No number below is estimated; missing entries are reported as missing.

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
