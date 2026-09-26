"""
train.py — LightGBM binary classifier for business-entity matching.

Pipeline position: comes after feature extraction (blocking + similarity
scoring) and before threshold tuning.  This script:
  1. Builds full training candidate pairs from the train fold (same blocking
     logic as baseline_pipeline.py) and labels them against ground truth.
  2. Splits at the SOURCE-1 ENTITY level, not row level, to prevent leakage:
     every candidate pair for a given S1 entity must land in the same fold,
     otherwise the model can memorise entity-level blocking artefacts.
  3. Trains a LightGBM binary classifier with a fixed config (no search).
  4. Saves the model + a validation-probability TSV for threshold tuning.

Usage:
    python train.py --data-root dataset --output-dir output/model
    python train.py --data-root dataset --output-dir output/model --val-frac 0.12
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_pipeline import (  # noqa: E402
    build_candidate_pairs,
    add_similarity_features,
    load_source,
    load_ground_truth,
)
from scoring import deterministic_split  # noqa: E402

# ---------------------------------------------------------------------------
# Feature columns used for training.  ID and text columns are excluded.
# Exactly the numeric/float columns produced by add_similarity_features().
# ---------------------------------------------------------------------------
FEATURE_COLS = ["name_jaccard", "addr_jaccard"]

# LightGBM fixed config — no hyperparameter search.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


def build_labeled_pairs(data_root: str, s1_entity_ids) -> pd.DataFrame:
    """
    Build candidate pairs for a given set of S1 entity IDs, add similarity
    features, then label each pair using ground truth.

    We filter S1 to only the requested entity IDs but always search against
    the FULL S2/S3 pool — matching test-time behaviour so pair-count
    statistics and feature distributions are realistic.
    """
    train_dir = os.path.join(data_root, "train")

    print("  Loading source files...", flush=True)
    s1_full = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # Restrict S1 to the requested entity IDs.
    s1 = s1_full[s1_full["entity_id"].isin(s1_entity_ids)].reset_index(drop=True)
    print(f"  S1 rows in split: {len(s1):,} (of {len(s1_full):,})", flush=True)

    print("  Building candidate pairs (blocking)...", flush=True)
    pairs = build_candidate_pairs(s1, s2, s3)
    print(f"  {len(pairs):,} raw candidate pairs", flush=True)

    print("  Computing similarity features...", flush=True)
    pairs = add_similarity_features(pairs)

    # Label: positive if candidate_entity_id is in the S1 entity's GT match set.
    def is_match(row):
        return int(row["candidate_entity_id"] in gt.get(row["source1_entity_id"], frozenset()))

    pairs["label"] = pairs.apply(is_match, axis=1)
    return pairs[["source1_entity_id", "candidate_entity_id"] + FEATURE_COLS + ["label"]]


def run(data_root: str, output_dir: str, val_frac: float, seed: int):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(data_root, "train")

    # Load S1 just for entity IDs so we can do the entity-level split.
    print("Loading S1 entity IDs for split...", flush=True)
    s1_ids = load_source(os.path.join(train_dir, "train_source1.tsv"))["entity_id"]

    # Entity-level split: all candidate pairs for an S1 entity land in the same
    # fold.  Row-level splits would leak because pairs from the same entity share
    # blocking-key membership, name strings, and address tokens — features the
    # model would memorise rather than generalise.
    train_entity_ids, val_entity_ids = deterministic_split(s1_ids, val_frac=val_frac, seed=seed)
    print(
        f"Entity split — train: {len(train_entity_ids):,}  val: {len(val_entity_ids):,}",
        flush=True,
    )

    print("\n[Building TRAIN pairs]", flush=True)
    train_df = build_labeled_pairs(data_root, train_entity_ids)

    print("\n[Building VAL pairs]", flush=True)
    val_df = build_labeled_pairs(data_root, val_entity_ids)

    n_train_pos = int(train_df["label"].sum())
    n_train_neg = len(train_df) - n_train_pos
    n_val_pos = int(val_df["label"].sum())
    n_val_neg = len(val_df) - n_val_pos

    # Compute scale_pos_weight from the ACTUAL measured ratio in training data,
    # not a hardcoded constant.  The ratio changes as blocking evolves (tighter
    # blocking = fewer negatives per positive), so hardcoding would silently
    # become wrong with any blocking change.
    spw = n_train_neg / n_train_pos if n_train_pos > 0 else 1.0

    print("\n" + "=" * 60)
    print("CLASS BALANCE")
    print(f"  Train: {n_train_pos:,} pos / {n_train_neg:,} neg  ({n_train_pos/len(train_df):.2%} positive)")
    print(f"  Val:   {n_val_pos:,} pos / {n_val_neg:,} neg  ({n_val_pos/len(val_df):.2%} positive)")
    print(f"  scale_pos_weight (neg/pos in train): {spw:.2f}")
    print("=" * 60, flush=True)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["label"].values

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, feature_name=FEATURE_COLS)

    params = {**LGBM_PARAMS, "scale_pos_weight": spw}
    # Pop sklearn-style keys not accepted by lgb.train().
    n_estimators = params.pop("n_estimators")
    params.pop("random_state", None)
    params.pop("n_jobs", None)

    print("\nTraining LightGBM...", flush=True)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iter = booster.best_iteration
    val_proba = booster.predict(X_val, num_iteration=best_iter)

    val_auc = roc_auc_score(y_val, val_proba)
    val_logloss = log_loss(y_val, val_proba)

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  Best iteration:  {best_iter}")
    print(f"  Val AUC:         {val_auc:.6f}")
    print(f"  Val logloss:     {val_logloss:.6f}")
    print(f"  Train rows:      {len(train_df):,}")
    print(f"  Val rows:        {len(val_df):,}")
    print("=" * 60, flush=True)

    # Save model in LightGBM native format.
    model_path = os.path.join(output_dir, "lgbm_model.txt")
    booster.save_model(model_path)
    print(f"\nModel saved: {model_path}")

    # Save validation probabilities + IDs + true labels.
    # Required by the next stage (threshold tuning) which needs calibrated
    # probabilities, not binary predictions, to sweep decision thresholds.
    val_out = val_df[["source1_entity_id", "candidate_entity_id", "label"]].copy()
    val_out["pred_proba"] = val_proba
    val_proba_path = os.path.join(output_dir, "val_predictions.tsv")
    val_out.to_csv(val_proba_path, sep="\t", index=False)
    print(f"Val predictions saved: {val_proba_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM matcher on blocking-generated candidate pairs."
    )
    parser.add_argument("--data-root", default="dataset",
                        help="Root directory containing train/ and test/ subdirs.")
    parser.add_argument("--output-dir", default="output/model",
                        help="Where to save the model and val_predictions.tsv.")
    parser.add_argument("--val-frac", type=float, default=0.12,
                        help="Fraction of S1 entities held out for validation (default 0.12).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Hash seed for deterministic_split (must match other scripts).")
    args = parser.parse_args()

    run(args.data_root, args.output_dir, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
