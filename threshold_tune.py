"""
threshold_tune.py — sweep decision thresholds on the LightGBM validation
probabilities and pick the one that maximises macro-averaged F0.5.

Pipeline position: runs AFTER train.py, BEFORE inference.

Why reconstruct ground_truth from the label column rather than reloading
the raw TSV: val_predictions.tsv already contains the true label for every
blocked candidate pair in the val fold.  Entities that are true singletons
AND produced zero blocking candidates are absent from val_predictions.tsv
entirely -- they score 1.0 at every threshold (predict-empty-correctly) so
their inclusion would shift the absolute F0.5 level upward uniformly without
changing WHICH threshold is optimal.  The argmax is identical either way;
omitting them keeps this script self-contained with no dependency on the raw
dataset files.

Usage:
    python threshold_tune.py
    python threshold_tune.py --val-preds output/model/val_predictions.tsv \\
                             --output    output/model/threshold.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import f0_5_macro  # noqa: E402  — use the repo's exact F0.5 implementation

SWEEP_LO = 0.30
SWEEP_HI = 0.90
SWEEP_STEP = 0.01


def build_ground_truth(df: pd.DataFrame) -> dict:
    """
    Derive ground_truth dict from the label column in val_predictions.tsv.
    For each source1_entity_id, the true match set is every candidate_entity_id
    where label == 1.  Entities with all-zero labels get an empty frozenset,
    which scoring.py's score_entity() treats as a true singleton (scores 1.0
    iff predicted set is also empty).
    """
    gt = {}
    for s1_id, grp in df.groupby("source1_entity_id", sort=False):
        gt[s1_id] = frozenset(grp.loc[grp["label"] == 1, "candidate_entity_id"])
    return gt


def predictions_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """
    For each source1_entity_id, collect candidates whose pred_proba >= threshold.
    Returns {s1_id: frozenset(candidate_ids)} — empty frozenset if none pass.
    All s1 IDs that appear in df are present in the output so that singletons
    (all-zero label, no candidates passing threshold) are explicitly represented
    as empty-prediction rather than missing from the dict.
    """
    passed = df[df["pred_proba"] >= threshold]
    preds = {s1_id: frozenset() for s1_id in df["source1_entity_id"].unique()}
    for s1_id, grp in passed.groupby("source1_entity_id", sort=False):
        preds[s1_id] = frozenset(grp["candidate_entity_id"])
    return preds


def run(val_preds_path: str, output_path: str):
    print(f"Loading val predictions: {val_preds_path}", flush=True)
    df = pd.read_csv(val_preds_path, sep="\t", dtype={"source1_entity_id": str,
                                                       "candidate_entity_id": str})
    df.columns = [c.strip() for c in df.columns]

    required = {"source1_entity_id", "candidate_entity_id", "label", "pred_proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"val_predictions.tsv is missing columns: {sorted(missing)}")

    print(f"  {len(df):,} candidate pairs, "
          f"{df['source1_entity_id'].nunique():,} unique S1 entities", flush=True)

    ground_truth = build_ground_truth(df)
    n_singletons = sum(1 for v in ground_truth.values() if not v)
    print(f"  {n_singletons:,} true singletons (no GT matches) in val set\n", flush=True)

    thresholds = np.round(np.arange(SWEEP_LO, SWEEP_HI + SWEEP_STEP / 2, SWEEP_STEP), 4)
    results = []

    for t in thresholds:
        preds = predictions_at_threshold(df, float(t))
        r = f0_5_macro(preds, ground_truth)
        results.append((float(t), r))

    # -----------------------------------------------------------------------
    # Full-resolution table: print every row but annotate the best
    # -----------------------------------------------------------------------
    best_t, best_r = max(results, key=lambda x: x[1]["f0_5_macro"])

    print(f"{'Threshold':>10}  {'F0.5_macro':>10}  {'singleton_acc':>13}  "
          f"{'singleton_fp_rate':>17}  {'nonsingleton_f0.5':>17}")
    print("-" * 75)

    # Print full resolution near the peak (±0.10), coarser elsewhere.
    peak_lo = round(best_t - 0.10, 4)
    peak_hi = round(best_t + 0.10, 4)

    for t, r in results:
        in_peak_region = peak_lo <= t <= peak_hi
        # Outside peak region show every 0.05 step only.
        if not in_peak_region and round(t * 100) % 5 != 0:
            continue
        marker = " <-- BEST" if t == best_t else ""
        ns_f05 = r["nonsingleton_f0_5"]
        ns_str = f"{ns_f05:.4f}" if not (isinstance(ns_f05, float) and ns_f05 != ns_f05) else "  n/a"
        sing_fp = r["singleton_fp_rate"]
        sing_fp_str = f"{sing_fp:.4f}" if not (isinstance(sing_fp, float) and sing_fp != sing_fp) else "   n/a"
        print(f"  {t:8.2f}  {r['f0_5_macro']:10.4f}  "
              f"{r['singleton_accuracy']:13.4f}  {sing_fp_str:>17}  {ns_str:>17}{marker}")

    print("-" * 75)
    print(f"\nBEST THRESHOLD : {best_t:.2f}")
    print(f"  F0.5 macro          : {best_r['f0_5_macro']:.6f}")
    print(f"  Singleton accuracy  : {best_r['singleton_accuracy']:.4f}")
    print(f"  Singleton FP rate   : {best_r['singleton_fp_rate']:.4f}")
    print(f"  Non-singleton F0.5  : {best_r['nonsingleton_f0_5']:.4f}")
    print(f"  Entities evaluated  : {best_r['n_entities']:,} "
          f"({best_r['n_singletons']:,} singletons)")

    # -----------------------------------------------------------------------
    # Edge-of-range warning: if the best threshold is at the boundary of the
    # swept range, the true optimum may lie outside it — the sweep itself is
    # wrong, not the threshold.
    # -----------------------------------------------------------------------
    if best_t <= SWEEP_LO:
        print(f"\n[WARNING] Best threshold ({best_t}) is at the LOWER edge of the "
              f"sweep range ({SWEEP_LO}–{SWEEP_HI}). The true optimum may be below "
              f"{SWEEP_LO}. Re-run with a lower --sweep-lo before trusting this value.")
    if best_t >= SWEEP_HI:
        print(f"\n[WARNING] Best threshold ({best_t}) is at the UPPER edge of the "
              f"sweep range ({SWEEP_LO}–{SWEEP_HI}). The true optimum may be above "
              f"{SWEEP_HI}. Re-run with a higher --sweep-hi before trusting this value.")

    # -----------------------------------------------------------------------
    # Save chosen threshold so inference can load it without hardcoding.
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "threshold": best_t,
        "f0_5_macro": round(best_r["f0_5_macro"], 8),
        "sweep_lo": SWEEP_LO,
        "sweep_hi": SWEEP_HI,
        "sweep_step": SWEEP_STEP,
        "val_preds_path": val_preds_path,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nThreshold saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep decision thresholds on LightGBM val probabilities to maximise F0.5."
    )
    parser.add_argument(
        "--val-preds",
        default="output/model/val_predictions.tsv",
        help="Path to val_predictions.tsv written by train.py.",
    )
    parser.add_argument(
        "--output",
        default="output/model/threshold.json",
        help="Where to save the chosen threshold (JSON). Loaded by the inference script.",
    )
    args = parser.parse_args()

    run(args.val_preds, args.output)


if __name__ == "__main__":
    main()
