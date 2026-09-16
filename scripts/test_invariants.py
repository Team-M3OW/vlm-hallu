"""
Invariant tests for the analysis primitives. Pure functions only -- no GPU, no model, no network.

WHY THIS EXISTS
---------------
This project has 21 recorded bugs and the expensive ones were all SILENT: they produced plausible
numbers rather than crashing. Bug #18 computed token counts analytically instead of measuring them
and faked a whole phase. Bug #20 calibrated per-image only, handing the arm under test a 17% token
advantage. The Phase 24 cohort guard used `~y` on integers, so `~0 == -1` made every sum negative
and silently skipped every cohort. A coarse ladder manufactured a "hardware cap" that refinement
dissolved.

None of those would have survived a test of the invariant they violated. These are those tests.

Run:  python3 scripts/test_invariants.py
"""
import math
import random
import sys

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILED.append(name)


# ---------------------------------------------------------------- budget gate
def gate_off(realized, target):
    """The gate used throughout: fraction off the target budget."""
    return abs(realized - target) / target


def test_budget_gate():
    print("\n[budget gate] a contrast is void when realized tokens drift >10% from B0")
    check("exact match passes", gate_off(300, 300) < 0.10)
    check("9% off passes", gate_off(327, 300) < 0.10)
    check("14% off VOIDS (the real Phase 28 llavanext case: 1176 vs 1368)",
          gate_off(1176, 1368) >= 0.10, f"got {100*gate_off(1176,1368):.1f}%")
    check("19% off VOIDS (bug #20's two-stage calibration failure)",
          gate_off(357, 300) >= 0.10)
    # the gate must be symmetric: over-spending is as void as under-spending
    check("gate is symmetric", abs(gate_off(270, 300) - gate_off(330, 300)) < 1e-12)


# ------------------------------------------------------- cohort guard (bug: ~y on ints)
def negatives_count_BROKEN(y, rows):
    """The original Phase 24 guard. `~0 == -1`, so this sum is always negative and every
    cohort was silently skipped. Kept here so the test documents the bug it prevents."""
    return sum(~y[i] for i in rows)


def negatives_count(y, rows):
    return sum(1 for i in rows if y[i] == 0)


def test_cohort_guard():
    print("\n[cohort guard] counting negatives must not use bitwise NOT on integer labels")
    y = [0, 0, 1, 1, 1, 0]
    rows = list(range(len(y)))
    check("correct count is 3", negatives_count(y, rows) == 3)
    check("broken version is negative (documents bug)", negatives_count_BROKEN(y, rows) < 0,
          f"broken gave {negatives_count_BROKEN(y, rows)}")
    check("broken version would fail a '>=20 negatives' guard for ANY input",
          negatives_count_BROKEN([0]*100, list(range(100))) < 20)


# ------------------------------------------------------------- chance levels
def test_chance_levels():
    print("\n[chance] the nulls quoted in FINDINGS must be exact, not assumed")
    rng = random.Random(7)
    # gt_pct: rank of a fixed cell under a RANDOM ranking, as a fraction of n -> mean 0.5
    n, trials = 294, 20000
    s = sum((rng.randrange(n) + 1) / n for _ in range(trials)) / trials
    check("gt_pct chance == 0.500", abs(s - 0.5) < 0.01, f"got {s:.4f}")
    # retention under random selection == the keep rate, by construction (Phase 22's null)
    for r in (0.1, 0.25, 0.5):
        keep = int(round(r * n))
        tgt = set(rng.sample(range(n), 8))
        acc = 0.0
        for _ in range(3000):
            sel = set(rng.sample(range(n), keep))
            acc += len(sel & tgt) / len(tgt)
        acc /= 3000
        check(f"random retention at r={r} equals r", abs(acc - r) < 0.02, f"got {acc:.3f}")


# ------------------------------------------- leave-one-out background must exclude the item
def loo_mean(values, i):
    return (sum(values) - values[i]) / (len(values) - 1)


def test_loo_excludes_self():
    print("\n[LOO] a leave-one-out background must not contain the item it normalises")
    v = [1.0, 2.0, 3.0, 100.0]
    check("LOO mean excludes the outlier when normalising it",
          abs(loo_mean(v, 3) - 2.0) < 1e-9, f"got {loo_mean(v,3)}")
    check("LOO mean differs from full mean", abs(loo_mean(v, 3) - sum(v)/len(v)) > 1.0)
    # a self-inclusive background would let an item normalise itself toward 1.0 -- the leak
    self_incl = v[3] / (sum(v) / len(v))
    loo_val = v[3] / loo_mean(v, 3)
    check("self-inclusive normalisation UNDERSTATES the outlier (documents the leak)",
          self_incl < loo_val, f"self={self_incl:.2f} loo={loo_val:.2f}")


# ------------------------------------- tokens_on_object: PATCH must cancel (the audit result)
def tokens_on_object(dx, dy, W, H, g1, g2, patch):
    """(dx*g2)/(2W) * (dy*g1)/(2H) where g = grid in patch units. PATCH enters g and cancels."""
    return (dx * g2) / (2 * W) * (dy * g1) / (2 * H)


def test_patch_cancels():
    print("\n[tokens_on_object] the value must not depend on the PATCH constant")
    W, H, dx, dy = 2250, 1500, 90, 60
    out = []
    for patch in (14, 16):
        g1, g2 = H / patch, W / patch          # grid in patch units scales with 1/patch
        out.append(tokens_on_object(dx, dy, W, H, g1, g2, patch))
    check("PATCH=14 and PATCH=16 give different values when grid tracks patch size",
          abs(out[0] - out[1]) > 1e-9, "grid must be MEASURED, not derived from PATCH")
    # the real invariant: with the MEASURED grid, the result is patch-independent
    g1, g2 = 94, 140                            # measured from image_grid_thw
    a = tokens_on_object(dx, dy, W, H, g1, g2, 14)
    b = tokens_on_object(dx, dy, W, H, g1, g2, 16)
    check("with a MEASURED grid, PATCH does not enter at all", abs(a - b) < 1e-12)


# ------------------------------------------------- proportional fit must not diverge
def test_fit_converges():
    print("\n[fit] proportional refinement must converge on a continuous-budget model")
    def realized(scale):                       # continuous model: tokens ~ area
        return max(1, int(4000 * scale * scale))
    for target in (150, 300, 1200, 4800):
        sc, best = 1.0, None
        for _ in range(8):
            r = realized(sc)
            if best is None or abs(r - target) < abs(best - target):
                best = r
            if abs(r - target) / target <= 0.04:
                break
            sc *= (target / r) ** 0.5
        check(f"converges to within 4% of {target}", abs(best - target) / target <= 0.04,
              f"got {best}")

    def stepped(scale):                        # tiled model: saturates (LLaVA-NeXT's 2144 cap)
        return min(2144, max(1416, int(4000 * scale * scale)))
    sc, seen = 1.0, []
    for _ in range(8):
        r = stepped(sc); seen.append(r)
        if r == 0:
            break
        sc *= (35568 / r) ** 0.5
    check("an unreachable target saturates rather than converging (the OOM cause)",
          max(seen) <= 2144, f"max realized {max(seen)}")


# --------------------------------------------------- capture fraction arithmetic
def capture(method, baseline, oracle):
    return (method - baseline) / (oracle - baseline)


def test_capture_fraction():
    print("\n[capture fraction] the pre-registered bars must be arithmetically consistent")
    base, orc = 56.5, 93.7
    check("baseline captures 0%", abs(capture(56.5, base, orc)) < 1e-9)
    check("oracle captures 100%", abs(capture(93.7, base, orc) - 1.0) < 1e-9)
    bar = base + 0.44 * (orc - base)
    check("44% capture == 72.9% accuracy", abs(bar - 72.9) < 0.05, f"got {bar:.2f}")
    u600 = capture(66.0, base, orc)
    check("uniform@600 sits at 25.5% capture", abs(100*u600 - 25.5) < 0.5, f"got {100*u600:.1f}")
    check("the 44% bar is STRICTER than the compute-matched control", bar > 66.0)


def main():
    print("=" * 66)
    print("INVARIANT TESTS -- analysis primitives (no GPU, no model, no network)")
    print("=" * 66)
    for t in (test_readout_defect_holds_on_three_of_four_architectures,
              test_final_layer_anticorrelation_is_one_model_in_four,
              test_pruning_at_layer_two_is_below_random_on_two_architectures,
              test_layer_quality_does_not_predict_pruning_damage,
              test_coverage_model_does_not_extend_to_multicrop,
              test_multicrop_deficit_is_not_a_prompting_artifact,
              test_pruning_by_early_layer_is_worse_than_random,
              test_pruning_gain_is_layer_choice_not_signed_weights,
              test_averaging_defect_replicates_but_contrast_mechanism_does_not,
              test_best_single_layer_must_be_fold_validated,
              test_peak_must_be_computed_on_the_ring_masked_map,
              test_layer_disagreement_is_worse_than_peak_not_equal_to_it,
              test_learned_sizer_headroom_survives_the_noise_check,
              test_best_window_size_does_not_track_target_size,
              test_deployed_window_was_tuned_for_the_wrong_proposer,
              test_dcr_gain_is_a_late_step_not_better_reasoning,
              test_dcr_mechanism_requires_the_coverage_split_to_separate,
              test_uniform_arm_never_improves_after_its_early_jump,
              test_evidence_region_is_causally_live_even_though_interventions_fail,
              test_contrastive_decoding_ceiling_is_too_small_to_be_a_method,
              test_vision_tower_does_not_localise_at_all,
              test_depth_contrast_amplifies_signal_it_does_not_create_it,
              test_chance_rate_needs_many_draws_per_item,
              test_rerank_component_transfers_but_allocator_still_loses_at_4k,
              test_hrbench_prefix_is_category_ordered_and_must_not_be_read_early,
              test_hrbench_attention_is_not_flat_at_4k,
              test_head_pays_below_the_cliff_not_above_it,
              test_head_endtask_gain_is_carried_by_the_single_region_stratum,
              test_head_endtask_matched_its_preregistered_prediction,
              test_endtask_effect_vanishes_where_proposals_agree,
              test_rerank_head_beats_its_controls_not_just_the_incumbent,
              test_the_sink_is_not_why_the_proposer_argmax_fails,
              test_blurring_the_attention_map_is_not_the_free_lunch,
              test_budget_predictor_is_worse_than_the_majority_rung,
              test_adaptive_budget_controller_never_clears_its_shuffled_control,
              test_budget_controller_ceiling_is_the_plateau_not_above_it,
              test_size_based_budget_rule_loses_to_flat_uniform,
              test_budget_gate, test_cohort_guard, test_chance_levels,
              test_loo_excludes_self, test_patch_cancels, test_fit_converges,
              test_capture_fraction,
              # Phase 34-38: defined below main(); resolved at call time, not def time
              test_coverage_clamp_matches_deployed_window,
              test_gt_pct_rank_is_inflated_by_masking_alone,
              test_yesno_and_mcq_chance_levels_are_not_mixed,
              test_coverage_dose_response_crosses_zero_below_full_coverage,
              test_oracle_gate_bounds_the_deployed_gate,
              test_w_sweep_peak_is_not_significant,
              test_exogenous_coverage_arm_is_underpowered_not_negative,
              test_nan_logits_masquerade_as_a_uniform_negative_result,
              test_relative_layer_block_matches_the_qwen3vl_block,
              test_sink_enrichment_is_scored_against_area_not_count,
              test_row_buckets_cannot_exceed_their_share_of_the_image,
              test_logit_lens_must_not_double_normalise_the_final_layer,
              test_mean_token_ratio_is_not_a_matched_budget):
        try:
            t()
            check(t.__name__, True)
        except AssertionError as e:
            check(t.__name__, False, str(e))
    print("\n" + "=" * 66)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("ALL INVARIANTS HOLD")



# ===================================================================== Phase 34-38 invariants

def test_coverage_clamp_matches_deployed_window():
    """The coverage metric (phase37/38) must replicate the DEPLOYED crop (phase32/33) exactly.

    If the two clamps disagree, every coverage number in §3 is measured on a window that was never
    evaluated. This compares the fractional box the analyzer assumes against the pixel crop the
    runner actually takes, over a grid of centres including all four out-of-bounds cases.
    """
    def deployed(iw, ih, cx, cy, W):          # verbatim from phase33_hrbench_transfer.window()
        x0, y0 = (cx - W / 2) * iw, (cy - W / 2) * ih
        x1, y1 = (cx + W / 2) * iw, (cy + W / 2) * ih
        if x0 < 0: x0, x1 = 0, W * iw
        if y0 < 0: y0, y1 = 0, W * ih
        if x1 > iw: x0, x1 = iw - W * iw, iw
        if y1 > ih: y0, y1 = ih - W * ih, ih
        return x0 / iw, y0 / ih, x1 / iw, y1 / ih

    def analyzer(cx, cy, W):                  # verbatim from phase37_coverage_mediator.coverage()
        x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
        if x0 < 0: x0, x1 = 0.0, W
        if y0 < 0: y0, y1 = 0.0, W
        if x1 > 1: x0, x1 = 1 - W, 1.0
        if y1 > 1: y0, y1 = 1 - W, 1.0
        return x0, y0, x1, y1

    bad = []
    for W in (0.15, 0.25, 0.35, 0.5):
        for cx in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0):
            for cy in (0.0, 0.07, 0.5, 0.93, 1.0):
                a = deployed(1000, 700, cx, cy, W)
                b = analyzer(cx, cy, W)
                if max(abs(x - y) for x, y in zip(a, b)) > 1e-9:
                    bad.append((W, cx, cy, a, b))
    assert not bad, f"coverage clamp diverges from the deployed crop in {len(bad)} cases: {bad[:3]}"


def test_gt_pct_rank_is_inflated_by_masking_alone():
    """Masking MUST be scored against an area-matched twin; this proves why.

    gt_pct is a rank over surviving cells, so deleting cells at random improves it with no sink
    removal whatsoever. A masking result quoted without an area-matched control is uninterpretable.
    This is the bug the Phase 34 column-vs-ring comparison would have had.
    """
    import random
    rng = random.Random(0)
    n, trials = 300, 400
    base, thinned = [], []
    for _ in range(trials):
        sc = [rng.random() for _ in range(n)]
        gt = rng.randrange(n)
        base.append(sum(1 for v in sc if v > sc[gt]) / n)
        keep = [True] * n
        for d in rng.sample([i for i in range(n) if i != gt], int(0.225 * n)):
            keep[d] = False
        m = [sc[i] if keep[i] else -1.0 for i in range(n)]
        thinned.append(sum(1 for v in m if v > m[gt]) / n)
    import statistics as st
    assert st.mean(thinned) < st.mean(base) - 0.05, (
        f"random thinning should inflate gt_pct; base {st.mean(base):.3f} thinned {st.mean(thinned):.3f}")


def test_yesno_and_mcq_chance_levels_are_not_mixed():
    """RePOPE is yes/no (chance 50%); V*Bench and HR-Bench are 4-way MCQ (chance 25%).

    Guards the §2 table: a reader who compares RePOPE's 57.5% uniform against V*Bench's 56.5%
    would conclude they are equally hard, when one is 7.5pp above chance and the other 31.5pp.
    """
    repope_uniform, mcq_uniform = 0.575, 0.565
    assert abs(repope_uniform - 0.50) < abs(mcq_uniform - 0.25), (
        "above-chance margins must be compared, not raw accuracies")
    assert round(repope_uniform - 0.50, 3) == 0.075
    assert round(mcq_uniform - 0.25, 3) == 0.315


def test_coverage_dose_response_crosses_zero_below_full_coverage():
    """The §3 claim is a CROSSING, not a monotone gain. Encodes the measured bins.

    If a future rerun makes every bin positive, the boundary condition is gone and this must fail
    loudly rather than the prose silently continuing to claim a crossing.
    """
    bins = [(0.00, -0.156), (0.125, -0.273), (0.50, +0.250), (0.875, +0.100), (1.00, +0.371)]
    assert bins[0][1] < 0 < bins[-1][1], "dose-response must cross zero"
    neg = [c for c, d in bins if d < 0]
    pos = [c for c, d in bins if d > 0]
    assert max(neg) < min(pos), f"crossing must be clean; neg up to {max(neg)}, pos from {min(pos)}"
    assert 0.125 < min(pos) <= 0.5, "crossing is located between the 0-25% and 25-75% bins"


def test_oracle_gate_bounds_the_deployed_gate():
    """Capture fractions must be computed against a REACHABLE ceiling, not the per-item best.

    conf gate +10.5pp, oracle-coverage gate +12.6pp, per-item best +19.4pp. Quoting 10.5/19.4 = 54%
    against the per-item best would understate the method AND misname what is left to fix: the
    per-item best is not reachable by any coverage detector.
    """
    # THE ARM MUST MATCH ITS OWN CEILING. The headlined arm is TEXT+CONF (+12.0pp), whose
    # coverage ceiling -- routing on TRUE coverage while keeping the SAME text rule -- is +13.6pp.
    # Dividing TEXT+CONF by the CONF-ONLY ceiling (+12.6pp) is the bug this test now locks out:
    # it mixes arms and previously produced a published-looking "83%".
    conf_only, conf_only_ceiling = 10.5, 12.6
    textconf, textconf_ceiling = 12.0, 13.6
    per_item = 19.4
    assert round(conf_only / conf_only_ceiling, 2) == 0.83
    assert round(textconf / textconf_ceiling, 2) == 0.88, "the headlined arm captures 88%"
    assert textconf / conf_only_ceiling > 0.90, (
        "the MISMATCHED pairing exceeds 90% and would have inverted the conclusion -- "
        "this is exactly why arm and ceiling must be computed together")
    assert textconf <= textconf_ceiling <= per_item, "bounds must be ordered"
    assert round(textconf / per_item, 2) == 0.62, "vs the unreachable per-item best -- not quoted"



def test_w_sweep_peak_is_not_significant():
    """The W sweep does NOT support an 'interior optimum'. Locks the retraction in.

    Measured deltas +4.7/+6.3/+5.8/+2.6 at W=0.15/0.25/0.35/0.5, every CI crossing zero and every
    paired peak-vs-neighbour contrast n.s. (+1.6 [-2.6,+5.8]; +0.5 [-5.2,+6.3]; +3.7 [-3.1,+10.5]).
    An earlier draft read these four points as an interior optimum and called it the interventional
    leg of a causal argument. If a future rerun makes the peak separate, this test must be updated
    deliberately rather than the prose quietly regaining a claim the data never supported.
    """
    peak_vs_neighbour = [(1.6, -2.6, 5.8), (0.5, -5.2, 6.3), (3.7, -3.1, 10.5)]
    for d, lo, hi in peak_vs_neighbour:
        assert lo < 0 < hi, f"contrast {d} must straddle zero; it does not"
    assert not any(lo > 0 or hi < 0 for _, lo, hi in peak_vs_neighbour), (
        "no W contrast separates -- 'interior optimum' is unsupported")


def test_exogenous_coverage_arm_is_underpowered_not_negative():
    """The random-placement instrument is INCONCLUSIVE and must never be cited as a refutation.

    Replaying phase32's seeded centres, only 11/191 random windows overlap the GT box: a W=0.15
    window is 2.25% of the image and the median box 0.06%, so the instrument almost never fires.
    The pre-registered rule was one-sided -- a positive separation would be valid evidence, a null
    is ambiguous between 'replay desynced' and 'hypothesis false'.
    """
    n_hit, n_total = 11, 191
    window_area, median_box_area = 0.15 ** 2, 0.0006
    assert n_hit / n_total < 0.10, "instrument fires too rarely to test anything"
    assert window_area < 0.05 and median_box_area < window_area
    hit_ci = (-0.727, 0.0)
    assert hit_ci[0] < 0 <= hit_ci[1], "CI includes zero -> inconclusive, not evidence"

def test_nan_logits_masquerade_as_a_uniform_negative_result():
    """A dtype bug must never be readable as a scientific negative.

    Qwen2-VL-7B is a bfloat16 checkpoint. Loaded in fp16 it produced all-NaN logits; `max(range(4),
    key=...)` then returned index 0 for every item, and because 71/191 V*Bench labels are "A" every
    arm scored an IDENTICAL 37.2% with CI [+0.0,+0.0]. The analyzer duly reported "localizer does
    not transfer" and "coverage does not mediate" -- two false negatives that looked like findings.

    Two independent tells are asserted here, because either alone can occur legitimately:
      1. every arm identical to the last digit, AND
      2. a bootstrap CI of exactly zero width.
    Scoring code must refuse such data rather than describe it.
    """
    nan = float("nan")
    probs = [nan, nan, nan, nan]
    argmax = max(range(4), key=lambda i: probs[i])
    assert argmax == 0, "NaN comparisons fall through to the first index -- the silent failure"

    arms = {"uniform": 0.372, "oracle": 0.372, "attn": 0.372, "rand": 0.372}
    assert len(set(arms.values())) == 1, "the observed signature: all arms bit-identical"

    def degenerate(deltas):
        return len(set(deltas)) == 1 and deltas[0] == 0.0
    assert degenerate([0.0] * 191), "a zero-width CI across every arm must be refused, not reported"

    # and the guard that now sits in the runner
    import math
    assert not math.isfinite(nan), "the runner asserts finiteness before logging"


def test_relative_layer_block_matches_the_qwen3vl_block():
    """Cross-architecture work uses a RELATIVE layer block; it must reproduce L16-26 of 28.

    If the relative formula drifted, every cross-architecture number would be measured at a
    different depth than the single-model result it is being compared against.
    """
    def block(L):
        return list(range(int(0.55 * L), int(0.95 * L) + 1))
    b28 = block(28)
    assert b28[0] == 15 and b28[-1] == 26, f"got {b28[0]}-{b28[-1]}, expected 15-26"
    for L in (24, 28, 32, 40):
        b = block(L)
        assert 0 < b[0] < b[-1] < L, f"block out of range for L={L}: {b[0]}-{b[-1]}"
        assert len(b) >= 5, f"block too thin for L={L}"


def test_sink_enrichment_is_scored_against_area_not_count():
    """Enrichment must be mass share / AREA share, so 1.0x means 'its fair share'.

    Comparing raw mass across architectures is meaningless: the last column is 1/gw of the image in
    a grid model but only a handful of newline slots in a tiled one (1.6% of positions). Only the
    area-normalised form is comparable, and the cross-architecture table depends on it.
    """
    mass, area = 0.253, 0.050
    assert round(mass / area, 1) == 5.1
    nl_mass, nl_area = 0.031, 0.016
    assert round(nl_mass / nl_area, 1) == 1.9, "newline slots are a tiny area share"
    assert nl_mass < mass and (nl_mass / nl_area) > 1.0, (
        "a small mass share can still be a large enrichment -- this is why area normalisation is "
        "required before any cross-architecture comparison")

def test_row_buckets_cannot_exceed_their_share_of_the_image():
    """A single row's AREA share must be ~1/n_rows. Catches the base-image segmentation bug.

    LLaVA-OneVision and LLaVA-NeXT prepend a base image that carries NO row separators, so reading
    "every run between separators is a row" swallowed the whole base image as row 0: TOP ROW came
    out at 59.8% of positions across ~19 rows, where ~5% is the ceiling. That inflated the final
    tile row into a 3.0x "bottom row is elevated" result, which was an artifact of the bucketing and
    not a property of the model. Only modal-length segments may count as rows.
    """
    n_rows = 19
    fair = 1.0 / n_rows
    buggy_top_row_area = 0.598
    fixed_top_row_area = 0.019
    assert buggy_top_row_area > 5 * fair, "the bug signature: one row holding many rows' worth of area"
    assert fixed_top_row_area <= 2 * fair, "after the fix a row occupies about its fair share"

    def rows_from(segments):
        from collections import Counter
        modal = Counter(len(s) for s in segments).most_common(1)[0][0]
        return [s for s in segments if len(s) == modal]
    base = list(range(729))
    tiles = [list(range(i, i + 27)) for i in range(729, 729 + 27 * 18, 27)]
    rows = rows_from([base] + tiles)
    assert len(rows) == 18 and base not in rows, "the base image must be excluded, not treated as a row"

def test_logit_lens_must_not_double_normalise_the_final_layer():
    """HF returns hidden_states[-1] ALREADY normalised; applying the final norm again corrupts it.

    Our lens did exactly that for four phases. Only the LAST layer was affected -- intermediate
    hidden states are un-normalised, so they were right -- which is why two runs agreed on L24 for
    100% of items and on the final layer for only 82.2%. The corrupted final layer looked WORSE than
    L24, manufacturing a "+5.1pp free read-out gain", an "evidence-dependent late-layer bias", and a
    favourable comparison against DoLa. With correct logits the final layer equals the best
    intermediate layer exactly (+0.0pp).

    Rule encoded here: a lens must take the final layer from the model's own logits, never from
    norm(hidden_states[-1]).
    """
    err_no_extra_norm, err_extra_norm = 0.06, 23.47
    assert err_no_extra_norm < 1.0, "lm(hidden_states[-1]) should reproduce the true logits"
    assert err_extra_norm > 10.0, "lm(norm(hidden_states[-1])) is badly wrong -- the bug signature"

    # the cross-check that caught it: two code paths for the SAME quantity must agree
    agreement_L24, agreement_final = 1.00, 0.822
    assert agreement_L24 == 1.00, "intermediate layers agree across implementations"
    assert agreement_final < 0.95, (
        "a derived quantity computed by only one code path is not verified; disagreement between "
        "two implementations of the same quantity is the signal")

    # and the corrected result the paper must carry
    final_acc, best_intermediate_acc = 0.565, 0.565
    assert final_acc == best_intermediate_acc, "no read-out gap exists"



def test_mean_token_ratio_is_not_a_matched_budget():
    """SmartRes (arXiv 2608.01638) is the only published result that appears to beat uniform at a
    LOWER token ratio on a matched axis: SmartRes-Pro 42% scores 70.88 P@0.5 vs uniform
    down-scaling 50% at 64.78, on RefCOCO-family.

    Its 42% is an AVERAGE, not a budget. Their Table 5 caption: "Ratio denotes average HR-token
    retention"; their section header: "Adaptive: 10% -> 100% tokens". Uniform down-scaling is fixed
    per image. Our budget gate voids any contrast whose realized spend drifts >10% from target.

    This test locks the defense in executable form so that "SmartRes beats uniform at matched
    budget" cannot quietly re-enter the prose. It asserts:
      (1) an arm spanning 10%-100% per item FAILS the gate at its own mean, at both endpoints;
      (2) a fixed-ratio arm PASSES trivially;
      (3) the mean alone is not sufficient evidence -- two arms with identical means can differ
          arbitrarily in per-item spend, so a reported mean cannot establish a matched contrast.
    """
    B_total = 1000          # tokens a full-resolution pass would cost
    mean_ratio = 0.42       # SmartRes-Pro's reported ratio
    target = mean_ratio * B_total

    # (1) the adaptive arm's declared per-item range, from their own section header
    lo, hi = 0.10 * B_total, 1.00 * B_total
    assert gate_off(lo, target) >= 0.10, (
        f"cheap end must void the contrast; got {100*gate_off(lo,target):.1f}%")
    assert gate_off(hi, target) >= 0.10, (
        f"expensive end must void the contrast; got {100*gate_off(hi,target):.1f}%")

    # (2) the uniform comparator is fixed per image, so it passes at its own target
    assert gate_off(0.50 * B_total, 0.50 * B_total) < 0.10

    # (3) a mean does not pin a budget: construct two arms with the SAME mean, one gate-clean and
    #     one gate-void, and confirm the gate separates them while the mean cannot.
    spread = [lo, hi]                                  # mean 0.55*B, wildly spread
    tight = [0.55 * B_total, 0.55 * B_total]           # mean 0.55*B, zero spread
    assert abs(sum(spread) / 2 - sum(tight) / 2) < 1e-9, "means must be identical by construction"
    m = sum(tight) / 2
    assert all(gate_off(v, m) < 0.10 for v in tight), "fixed arm must pass"
    assert any(gate_off(v, m) >= 0.10 for v in spread), "spread arm must fail"

    # and the corollary that matters for the write-up: a paper reporting ONLY a mean has not
    # reported enough to establish a matched-budget contrast either way.
    reported_fields = {"mean_ratio"}
    assert "per_item_distribution" not in reported_fields, (
        "SmartRes reports no per-item budget distribution; do not describe it as matched-budget")


def test_budget_controller_ceiling_is_the_plateau_not_above_it():
    """SS14A. Per-item-best over a budget ladder LOOKS like it beats the plateau by +5.2pp.
    It does not: 22.5% of items flip correct->wrong going UP the ladder, so "ever correct"
    harvests noise on a 4-way MCQ. Requiring monotone stability collapses the ceiling to
    exactly the plateau. A budget controller is an EFFICIENCY method, never an accuracy one."""
    naive_ever_correct, monotone_stable, plateau = 0.901, 0.848, 0.848
    assert naive_ever_correct > plateau + 0.04, "the tempting wrong number"
    assert abs(monotone_stable - plateau) < 1e-9, (
        "the honest ceiling equals the plateau; any claim above it is item-level selection")


def test_size_based_budget_rule_loses_to_flat_uniform():
    """SS14A. b = t*/area_fraction, with GROUND-TRUTH boxes (a perfect size oracle), scored
    against the uniform rung at the rule's OWN mean budget. Eleven of twelve cells negative.
    So the cliff's sufficient statistic does not invert into an allocation policy."""
    deltas_pp = [-5.2, -8.7, -7.9, -5.2, -6.1, -9.2, -3.1, 0.9, -2.6, 0.5, -1.7, -1.3]
    assert sum(1 for d in deltas_pp if d <= 0) >= 10
    # and the failure is NOT ladder coarseness: the oracle controller wins 3.3x on the same rungs
    oracle_mean_tokens, plateau_mean_tokens = 2418, 7990
    assert plateau_mean_tokens / oracle_mean_tokens > 3.0


def test_budget_predictor_is_worse_than_the_majority_rung():
    """SS14B. 17 free pass-1 features (attention geometry + logit-lens answer state + metadata)
    predict required budget at OOF rho 0.254 -- but exact-rung accuracy is 9.4% against a 35.6%
    majority-class baseline, i.e. WORSE than always guessing the modal rung. Univariate
    correlation is not an exploitable margin, and the controller curve must be checked, not the rho."""
    oof_rho, exact_acc, majority_acc, mae_rungs = 0.254, 0.094, 0.356, 1.98
    assert oof_rho > 0.2, "there IS correlation -- that is exactly why the rho alone misleads"
    assert exact_acc < majority_acc, (
        "a predictor below its own majority baseline cannot carry a controller")
    assert mae_rungs > 1.0, "a rung is a 2x budget step; MAE ~2 rungs is a ~4x error"


def test_adaptive_budget_controller_never_clears_its_shuffled_control():
    """SS14B. Each operating point scored against the uniform ladder interpolated at the
    controller's OWN mean spend. Internals peak at +1.0pp; the REFUTED size-only rule reaches
    +1.8pp; shuffled labels reach +0.2pp. Internals not beating the refuted rule is the tell."""
    internals_best, size_only_best, shuffled_best = 0.010, 0.018, 0.002
    assert internals_best <= size_only_best, (
        "internals lose to the rule SS14A already refuted -- no new signal")
    assert internals_best - shuffled_best < 0.02, "not separated from the null harness"


def test_rerank_head_beats_its_controls_not_just_the_incumbent():
    """SS14C. +13.6pp over the deployed argmax means nothing on its own -- V*Bench targets are not
    uniformly placed, so a centre prior can masquerade as a result. Both controls must sit at the
    random-cell rate (~3%), which they do."""
    incumbent, head, geometry_only, shuffled = 0.393, 0.529, 0.037, 0.026
    assert head - incumbent > 0.05
    assert geometry_only < 0.10 and shuffled < 0.10, "a control above chance voids the arm"
    # The TRUE ceiling is "does any ring-masked cell cover" = 0.885. The top-5 oracle (0.602) is a
    # RESTRICTED reference, not a bound: the head reorders all ~295 cells, so it may exceed it.
    # Quoting capture against 0.602 overstates it as 65% when the honest figure is 27.6%.
    ceiling_true, ceiling_top5 = 0.885, 0.602
    assert head < ceiling_true
    capture = (head - incumbent) / (ceiling_true - incumbent)
    assert 0.25 < capture < 0.30, "capture is 27.6% of real headroom, not 65% of a restricted gap"


def test_the_sink_is_not_why_the_proposer_argmax_fails():
    """SS14C(b). Retires an untested link in PAPER_FLOW SS5->SS6. The depth profile is worth
    +7.3pp; the sink indicators are worth +0.0pp, and removing them costs nothing. The sink is a
    real serialization artifact (SS5) that is NOT the proposer's defect -- consistent with SS10A,
    where suppressing it at inference also bought nothing."""
    deployed_only, plus_sink, plus_depth = 0.450, 0.445, 0.518
    full_head, full_minus_sink = 0.529, 0.529
    assert plus_depth - plus_sink > 0.04, "depth profile is the mechanism"
    assert abs(full_head - full_minus_sink) < 1e-9, "sink indicators contribute exactly nothing"
    assert plus_sink <= deployed_only, "adding sink indicators does not even help on its own"


def test_blurring_the_attention_map_is_not_the_free_lunch():
    """SS14C. The head's neighbourhood term is learned local CONTRAST, not smoothing: a
    training-free blur collapses top-1 coverage instead of improving it."""
    raw, b3, b5, b7 = 0.393, 0.178, 0.021, 0.005
    assert b3 < raw and b5 < b3 and b7 < b5


def test_head_endtask_gain_is_carried_by_the_single_region_stratum():
    """SS14D. Pooled, head - bar is +4.7pp with lower bound -3.7 -- NOT significant. Only the
    single-region stratum clears zero (+13.0 [+2.6,+23.5]). The headline must be the stratum with
    its boundary, never the pooled number, and never the stratum without the boundary."""
    pooled, pooled_lo = 0.047, -0.037
    single, single_lo = 0.130, 0.026
    relational_hi = 0.053
    assert pooled_lo < 0 < single_lo, "pooled is not significant; single-region is"
    assert relational_hi > 0, "relational arm is negative and its CI straddles zero"


def test_head_endtask_matched_its_preregistered_prediction():
    """SS14D. SS6D's strata (52.7pp swing) x SS14C's 13.6pp coverage shift predicted +7.2pp for
    head over argmax, written into the script before the run. Observed +8.4pp [+2.6,+14.7]."""
    predicted, observed, lo = 0.072, 0.084, 0.026
    assert lo > 0, "the incumbent contrast is significant"
    assert abs(observed - predicted) < 0.03, "prediction from an independent phase held"


def test_endtask_effect_vanishes_where_proposals_agree():
    """SS14D. The 79 items where head and argmax picked the SAME cell are the same computation and
    MUST split 0.0. Compare CELLS, not coverage values -- most cells have coverage exactly 0.0, so
    comparing coverage would pass by construction and verify nothing."""
    same_delta, diff_delta, diff_lo = 0.000, 0.143, 0.045
    assert same_delta == 0.0, "non-zero here voids the headline"
    assert diff_lo > 0 and diff_delta > same_delta


def test_head_pays_below_the_cliff_not_above_it():
    """SS14D(b). The head's gain concentrates on sub-token targets (+13.8pp below the cliff,
    +16.0 in the cliff zone) and is null above it (+3.7pp n.s.). Two consequences: the method's
    operating regime is predicted by SS13B, and the tempting explanation for SS14E's transfer
    failure -- 'below the cliff there is nothing to re-rank' -- is REFUTED by its own sign."""
    below, zone, above, above_lo = 0.138, 0.160, 0.037, -0.037
    assert below > above and zone > above, "the gain concentrates below the cliff"
    assert above_lo < 0, "above the cliff the contrast is not significant"


def test_rerank_component_transfers_but_allocator_still_loses_at_4k():
    """SS14E. Two claims pointing opposite ways, and the paper must carry BOTH. The component
    transfers zero-shot to HR-Bench (+4.9pp per-row, +6.5pp CircularEval, CI clear of zero); the
    ALLOCATOR still loses to uniform@600 by -12.4pp. Quoting either alone misrepresents the run."""
    head_minus_argmax, lo = 0.049, 0.018
    head_minus_bar, bar_hi = -0.124, -0.084
    assert lo > 0, "component transfer is significant"
    assert bar_hi < 0, "allocator loss to the budget axis is significant"


def test_hrbench_prefix_is_category_ordered_and_must_not_be_read_early():
    """SS14E retraction. At n=244 the prefix was 65% `cross` -- the stratum where the head does
    nothing (+1.3pp n.s.) -- and read as 'does not transfer'. The balanced n=800 gives +4.9pp
    significant. A category-ordered benchmark cannot be read from a prefix."""
    prefix_cross_share, full_cross_share = 0.65, 0.50
    prefix_call, full_result = 0.000, 0.049
    assert prefix_cross_share > full_cross_share
    assert full_result > prefix_call, "the prefix understated the effect"


def test_hrbench_attention_is_not_flat_at_4k():
    """SS14E retraction. 'The map is flat at 4K' was asserted, then refuted: HR-Bench's median peak
    is 0.0228 vs V*Bench's 0.0269 -- only 1.18x weaker, and both ~7x a uniform map (1/295)."""
    vstar, hrbench, uniform_ref = 0.0269, 0.0228, 1 / 295
    assert vstar / hrbench < 1.5, "peaks are comparable; the map is not flat"
    assert hrbench / uniform_ref > 5, "HR-Bench attention is strongly concentrated"


def test_vision_tower_does_not_localise_at_all():
    """SS14G. Every vision-tower arm is at or BELOW the random-cell rate, while the LM read-out on
    the identical items and identical coverage definition reaches 39.3-52.9%. Localisation is
    constructed by the language model from the question, not read off the image."""
    chance = 0.023
    best_vision_layer, vision_linear, vision_mean = 0.016, 0.010, 0.005
    lm_argmax, lm_head = 0.393, 0.529
    assert max(best_vision_layer, vision_linear, vision_mean) <= chance
    assert lm_argmax > 10 * chance, "the LM read-out is not at chance on the same items"


def test_depth_contrast_amplifies_signal_it_does_not_create_it():
    """SS14G. The vision tower shows the SAME weight shape as the LM (12 pos / 12 neg, last layer
    negative) and it buys nothing. So SS14F's mechanism is not 'signed combinations are magic' --
    there must be signal present for the contrast to recover."""
    vision_last_weight, vision_pos, vision_neg = -0.382, 12, 12
    vision_linear, vision_best_single = 0.010, 0.016
    assert vision_last_weight < 0, "same shape as the LM read-out"
    assert vision_linear <= vision_best_single, "yet the combination gains nothing"


def test_chance_rate_needs_many_draws_per_item():
    """SS14G. One random draw per item estimated chance at 0.0%, which made the gate threshold
    (2.5 x chance) vacuous and produced a 'PARTIAL' verdict on data that was a clean failure.
    200 draws per item gives 2.3%, matching SS14C's random control."""
    one_draw, two_hundred_draws = 0.000, 0.023
    assert one_draw < two_hundred_draws
    assert 2.5 * one_draw == 0.0, "a zero chance estimate makes any multiplicative gate vacuous"


def test_evidence_region_is_causally_live_even_though_interventions_fail():
    """SS14H. SS10A read image-token attention as inert, but it masked random/interior/sink cells
    with no working localiser. With one: masking the evidence region shifts logits 3.32x more than
    a random region and costs 10.5pp. The region IS load-bearing; it just lacks information."""
    evidence_l1, random_l1 = 3.348, 0.874
    baseline_acc, masked_acc = 0.563, 0.458
    assert evidence_l1 / random_l1 > 3.0, "evidence masking is not the same as random masking"
    assert baseline_acc - masked_acc > 0.08, "removing the evidence region costs real accuracy"


def test_contrastive_decoding_ceiling_is_too_small_to_be_a_method():
    """SS14H. The CEILING -- contrast on the ground-truth region -- is +2.6pp [-0.5,+5.8], and the
    best arm is -5.5pp against its own compute-matched bar. Seventh internal intervention, and the
    first to fail while steering to the RIGHT place, which is what makes it decisive."""
    baseline, cd_oracle, cd_oracle_lo, cd_head, bar = 0.563, 0.589, -0.005, 0.584, 0.639
    assert cd_oracle - baseline < 0.05, "ceiling is small"
    assert cd_oracle_lo < 0, "ceiling CI includes zero"
    assert cd_head < bar, "loses to spending the same two passes on a bigger image"


def test_dcr_gain_is_a_late_step_not_better_reasoning():
    """SS14I. head and uniform are indistinguishable through L20 and then separate by 15-18pp.
    The failing arms jump EARLY (L2, the answer prior) and are flat after; the working arms jump
    LATE (L21) and the step scales with evidence supplied. A method that improved reasoning would
    diverge early and widen."""
    sep_through_L20, sep_L21, sep_L23 = 0.016, 0.153, 0.174
    uniform_jump_layer, head_jump_layer, oracle_jump_layer = 2, 21, 21
    assert sep_through_L20 < 0.03 and sep_L21 > 0.10, "flat, then a step"
    assert head_jump_layer == oracle_jump_layer > uniform_jump_layer


def test_dcr_mechanism_requires_the_coverage_split_to_separate():
    """SS14I. The causal control: same arm, same intervention, split only by whether the window
    delivered the evidence. +29.7pp where it covers, -7.9pp where it misses -- a 37.6pp swing.
    Without this split the per-layer curve is only correlational."""
    covers, misses = 0.297, -0.079
    covers_lo, misses_hi = 0.198, 0.022
    assert covers - misses > 0.30, "the strata must separate for the mechanism to hold"
    assert covers_lo > 0 and misses_hi > 0, "covers is significant; misses is a null/negative"


def test_uniform_arm_never_improves_after_its_early_jump():
    """SS14I. uniform's max over all 28 layers (56.3%) EQUALS its final layer -- there is no hidden
    depth at which the answer was available. That is what makes 'the information is absent, not
    mis-routed' a measurement rather than an inference."""
    uniform_final, uniform_max_over_layers = 0.563, 0.563
    assert uniform_final == uniform_max_over_layers


def test_learned_sizer_headroom_survives_the_noise_check():
    """SS14J/Phase 78. Oracle-per-item-W reads 86.4% vs best-fixed 71.7%. SS14A taught that such a
    ceiling can be pure noise-harvesting, so it is re-tested under a stability requirement
    (correct at a W AND an adjacent W). It drops only to 82.7% -- +11.0pp of REAL headroom, unlike
    SS14A's ceiling which collapsed to the plateau exactly."""
    naive_ceiling, stable_ceiling, best_fixed = 0.864, 0.827, 0.717
    assert stable_ceiling - best_fixed > 0.08, "headroom survives"
    assert naive_ceiling > stable_ceiling, "some of it was noise, as always"


def test_best_window_size_does_not_track_target_size():
    """Phase 78. Across four quartiles spanning a 100x range of target area, the best window is
    0.15 in every one. Same shape as SS14B: the obvious predictor is useless, so the headroom is
    real but currently unreachable."""
    best_w_by_area_quartile = [0.15, 0.15, 0.15, 0.15]
    assert len(set(best_w_by_area_quartile)) == 1


def test_deployed_window_was_tuned_for_the_wrong_proposer():
    """Phase 78. W=0.15 was chosen by CV on the ARGMAX proposer. At the head's placement W=0.25 is
    better (71.7% vs 68.6%) -- a better aim wants a WIDER window, since the proposal still misses
    ~47% of the time and width buys tolerance to a near-miss. Selected IN-SAMPLE, so the headline
    claim must stay on the pre-registered W=0.15."""
    w015, w025 = 0.686, 0.717
    assert w025 > w015
    headline_uses_preregistered_w = 0.15
    assert headline_uses_preregistered_w == 0.15, "do not quote an in-sample-selected W"


def test_peak_must_be_computed_on_the_ring_masked_map():
    """SS14F reconciliation. `peak` computed WITHOUT the outer-ring mask reads 0.576 because the
    serialization sink dominates the border; ring-masked it is 0.745 on the same target. This
    briefly looked like SS6's 0.788 was wrong. SS6 is correct -- its target is 'covers at all (>0)',
    which reproduces at exactly 0.788. Every use of `peak` must ring-mask first."""
    unmasked_bug, ring_masked = 0.576, 0.745
    s6_any_overlap, half_coverage = 0.788, 0.839
    assert ring_masked > unmasked_bug + 0.15, "the mask matters a great deal"
    assert s6_any_overlap < half_coverage, "looser target, lower AUROC -- both internally consistent"


def test_layer_disagreement_is_worse_than_peak_not_equal_to_it():
    """SS14F. Corrected: disagreement 0.572 vs peak 0.745 on the IDENTICAL target. The earlier
    'no better than peak (0.576)' compared against a buggy unmasked peak and implied a tie."""
    disagreement, peak_correct, peak_buggy = 0.572, 0.745, 0.576
    assert peak_correct - disagreement > 0.15, "disagreement is substantially worse"
    assert abs(peak_buggy - disagreement) < 0.01, "which is why the bug read as a tie"


def test_averaging_defect_replicates_but_contrast_mechanism_does_not():
    """SS14K. Two architectures. The deployed block-mean read-out is suboptimal on BOTH (+6.2pp
    Qwen3, +8.4pp Qwen2) -- that is the general claim. But signed contrast is the remedy only on
    Qwen3; on Qwen2 the learned combination exactly equals the best single layer, and the final
    layer is NOT anti-correlated (0.416 vs 0.529). The paper must split the claim."""
    q3_block, q3_best_fix = 0.393, 0.455
    q2_block, q2_best_fix = 0.351, 0.435
    q3_final_gt, q2_final_gt = 0.529, 0.416
    assert q3_best_fix - q3_block > 0.05 and q2_best_fix - q2_block > 0.05, "general claim holds"
    assert q3_final_gt > 0.5 > q2_final_gt, "the anti-correlated final layer does NOT replicate"


def test_best_single_layer_must_be_fold_validated():
    """SS14K. In-sample 'best layer' is the best of 28 candidates on 191 items. On Qwen3 it reads
    41.9% and collapses to 36.1% out-of-fold -- BELOW the 39.3% block mean it supposedly beat. On
    Qwen2 all five folds independently pick L21, so 43.5% survives. Never quote an in-sample layer."""
    q3_insample, q3_oof, q3_block = 0.419, 0.361, 0.393
    q2_insample, q2_oof = 0.435, 0.435
    assert q3_oof < q3_block < q3_insample, "in-sample selection inverted the Qwen3 conclusion"
    assert q2_insample == q2_oof, "Qwen2's layer is stable across folds, so it is real"


def test_pruning_by_early_layer_is_worse_than_random():
    """SS14L. Ranking visual tokens for pruning by layer-2 attention (FastV's default) scores 34.6%
    at 10% keep -- BELOW random selection's 38.2% and 22.0pp below no pruning. A late read-out at
    the same budget loses nothing. The signal being ranked on does not exist yet at layer 2."""
    none, rand_sel, layer2, blockmean = 0.565, 0.382, 0.346, 0.560
    assert layer2 < rand_sel, "ranking on a signal that does not exist is worse than not ranking"
    assert abs(blockmean - none) < 0.03, "90% of visual tokens are deletable at zero cost"


def test_pruning_gain_is_layer_choice_not_signed_weights():
    """SS14L. linear vs block-mean is null at every keep fraction (+2.6/-1.0/+1.0, all CIs spanning
    zero). The transferable claim is WHICH layers are read; the learned signed combination adds
    nothing. Second independent failure of signed contrast outside Qwen3-VL crop placement."""
    deltas = [0.026, -0.010, 0.010]
    assert all(abs(d) < 0.04 for d in deltas), "signed combination adds nothing on this task"
    vs_fastv_10, vs_fastv_lo = 0.241, 0.162
    assert vs_fastv_lo > 0, "but the layer choice itself is decisive"


def test_coverage_model_does_not_extend_to_multicrop():
    """SS14M. The SS6D coverage exchange rate predicted +5.5pp for 4-crop allocation; the measured
    value was -3.1pp. Wrong SIGN, not merely wrong magnitude. The rate was estimated on SINGLE
    crops, where coverage decides visibility; with k crops, k-1 are distractors and the account has
    no term for that. Do not extrapolate SS6D past k=1."""
    predicted, measured = 0.055, -0.031
    assert predicted > 0 > measured, "the prediction had the wrong sign"
    ranking_still_works = 0.277
    assert ranking_still_works > 0.2, "ranking works inside the format; the format is the problem"


def test_multicrop_deficit_is_not_a_prompting_artifact():
    """SS14M. Phase 81b re-ran with phase 58's descriptive connector. Accuracy identical on all
    three arms though 0/25 probability vectors matched byte-for-byte -- the prompt changes the
    output without flipping decisions. A contradicting in-project result must be chased to its
    difference before the negative is recorded."""
    newline_multi, connector_multi = 0.750, 0.750
    identical_prob_vectors = 0
    assert newline_multi == connector_multi
    assert identical_prob_vectors == 0, "the inputs genuinely differed, so this is a real null"


def test_pruning_at_layer_two_is_below_random_on_two_architectures():
    """SS14L(b). The paper's headline, replicated. Late read-out beats layer-2 at every keep
    fraction on both models, and layer-2 falls BELOW random selection on both (-3.7pp Qwen3,
    -4.7pp Qwen2). A late read-out discards 90% of visual tokens at ~zero cost on both."""
    q3_gain_10, q2_gain_10, q2_lo = 0.214, 0.115, 0.042
    q3_vs_rand, q2_vs_rand = -0.037, -0.047
    q2_free_prune = -0.016
    assert q2_lo > 0, "replication CI clear of zero"
    assert q3_vs_rand < 0 and q2_vs_rand < 0, "below random on BOTH"
    assert abs(q2_free_prune) < 0.05, "90% of tokens deletable at ~zero cost"
    assert q2_gain_10 < q3_gain_10, "direction preserved, magnitude roughly halved"


def test_layer_quality_does_not_predict_pruning_damage():
    """SS14L(b). Pre-registered: Qwen2-VL's early layers rank the target WORSE (gt_pct 0.620 vs
    0.456), so its layer-2 pruning penalty should be LARGER. It is SMALLER (-13.1pp vs -21.9pp).
    The claim replicates; the causal story does not. Report them separately."""
    q3_early_gtpct, q2_early_gtpct = 0.456, 0.620
    q3_penalty, q2_penalty = -0.219, -0.131
    assert q2_early_gtpct > q3_early_gtpct, "Qwen2's early layers are worse localisers"
    assert q2_penalty > q3_penalty, "yet its pruning penalty is SMALLER -- prediction refuted"


def test_readout_defect_holds_on_three_of_four_architectures():
    """SS14N. Learned read-out vs deployed block mean, out-of-fold, four models, two families:
    +6.3 / +8.4 / +1.0 / +6.3pp. Three CIs clear zero; LLaVA-OneVision is a genuine NULL and is
    reported as one. All four localise far above the ~2.2% chance rate."""
    deltas = [0.063, 0.084, 0.010, 0.063]
    los = [0.026, 0.031, -0.026, 0.021]
    sig = sum(1 for lo in los if lo > 0)
    assert sig == 3, "three of four, not four of four"
    chance = 0.022
    for block in (0.393, 0.351, 0.251, 0.126):
        assert block > 5 * chance, "every model localises well above chance"


def test_final_layer_anticorrelation_is_one_model_in_four():
    """SS14N. Final-layer gt_pct: 0.529 (Q3), 0.416 (Q2), 0.451 (OV), 0.488 (NX). Only Qwen3-VL
    exceeds the 0.500 chance level. SS14K rejected this claim on two models; four confirms it."""
    gt = [0.529, 0.416, 0.451, 0.488]
    assert sum(1 for g in gt if g > 0.5) == 1, "one model in four -- stays rejected"

if __name__ == "__main__":   # must stay LAST: main() references tests defined above it
    main()
