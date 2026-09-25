================================================================================================================
PANEL A -- OUR MODELS UNDER THE LITERATURE PROTOCOL (equal answer tokens, no localise charge)
================================================================================================================
model           benchmark     n  no-crop crop(W=.25) crop(W=.5)          delta best          tokens
Qwen3-VL-2B     V*          191     53.9        72.8       62.3 +18.8 [+11.0,+26.7]      295 vs 293
Qwen3-VL-2B     TextVQA     500     76.8        52.6       70.8  -6.0 [-10.2, -1.8]      296 vs 297
Qwen3-VL-2B     DocVQA      527     63.9        36.6       52.8 -11.2 [-15.9, -6.5]      299 vs 299
Qwen3-VL-2B     GQA          --  running
Qwen2-VL-7B     V*          120     48.3        70.0       62.5 +21.7 [+12.5,+30.8]      295 vs 307
Qwen2-VL-7B     TextVQA      --  running
Qwen2-VL-7B     DocVQA       19     57.9        36.8       63.2  +5.3 [-26.3,+36.8]      298 vs 300
Qwen2-VL-7B     GQA          --  running
LLaVA-OV-7B     V*          191     51.8        62.3       65.4 +13.6 [ +6.3,+20.9]    1246 vs 1215
LLaVA-OV-7B     TextVQA    5000     67.9        62.8       63.8  -4.2 [ -5.7, -2.6]    1286 vs 1292
LLaVA-OV-7B     DocVQA     1000     32.7        42.1       40.4  +9.4 [ +5.8,+13.0]    1303 vs 1303
LLaVA-OV-7B     GQA         248     61.3        50.8       51.6  -9.7 [-15.3, -4.0]    1283 vs 1284

================================================================================================================
PANEL B -- VICROP'S PUBLISHED TABLE 2 (reference; fixed-resolution baselines, no localise charge)
================================================================================================================
model              TextVQA        V*      POPE    DocVQA    AOKVQA       GQA     VQAv2
LLaVA-1.5 no-crop     47.80     42.41     85.27     15.97     59.01     60.48     75.57
LLaVA-1.5 rel-att     55.17     62.30     87.25     19.63     60.66     60.97     76.51
InstructBLIP no-crop     33.48     35.60     84.89      9.20     60.06     49.41     76.25
InstructBLIP rel-att     45.44     42.41     86.64      9.95     61.28     49.75     76.84

READINGS
--------
1. V*Bench: cropping wins for every model under the literature's own rules --
   Qwen3 +18.8 [+11.0,+26.7], Qwen2 +21.7 [+12.5,+30.8], LLaVA-OV +13.6 [+6.3,+20.9] --
   the same direction as ViCrop's published LLaVA-1.5 +19.9 / InstructBLIP +6.8.
2. DocVQA: the sign is set by the no-crop baseline's resolution regime, not the benchmark.
   LLaVA-OV at its ladder minimum (1303 tokens, no-crop 32.7) gains +9.4 from cropping,
   matching ViCrop's direction (+1.6 to +3.9 against a 336px baseline); Qwen3 at 299 vs 299
   tokens (no-crop 63.9) loses -11.2. Same operation, same benchmark, opposite sign.
3. TextVQA is the controlled demonstration: LLaVA-1.5 at fixed 336px gains +7.4 (ViCrop);
   LLaVA-OneVision, same family, anyres at 1286 tokens, loses -4.2. Only the baseline's
   resolution regime changed -- which is exactly the crossover this project measures.
4. GQA: cropping loses for LLaVA-OV (-9.7, n=248) where ViCrop reports it flat (+0.5);
   GQA is relation/scene-heavy, consistent with the scope law (crop helps local properties,
   not scene-level ones).
5. This is a RECONSTRUCTION of the literature protocol, not this project's headline
   evaluation. The paper's matched-compute grid (bar = uniform@600, crop = localise@300 +
   crop@300) is the honest method comparison; Panel A is for like-for-like reading against
   published tables.

CAVEATS
-------
* Open-ended items are scored by normalised any-of-N answer match, not the official
  VQA-accuracy / ANLS. Deltas are comparable; absolute numbers are not directly comparable
  to published values. On TextVQA the sign was verified under strict VQA accuracy too
  (bar 79.3 vs crop 54.6 under the matched-compute protocol).
* The localise pass is NOT charged in either panel (the literature's rule). Panel A crop
  deltas are therefore optimistic by one 300-token pass relative to no-crop.
* LLaVA-OneVision's no-crop arms run at its anyres ladder minimum (~1.25-1.30k tokens);
  its grid cannot reach 600. That is the correct native-ish baseline for this comparison,
  but it means LLaVA rows are not budget-matched with the Qwen rows.
* Partial cells: Qwen2 V* n=120/191; Qwen2 DocVQA n=19 (excluded from the reading);
  LLaVA GQA n=248; Qwen3/Qwen2 GQA and Qwen2 TextVQA still running.
* ViCrop numbers quoted from the table supplied by the user (Table 2 of the ViCrop paper).
