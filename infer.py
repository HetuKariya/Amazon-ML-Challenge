"""
infer.py — Run the trained LightGBM matcher on the test set and write the
two required submission files:
  output/matching_results.tsv  — final predicted matches, one row per S1 entity
  output/candidate_pairs.tsv   — blocking candidates (superset of matches)

Pipeline position: runs AFTER train.py + threshold_tune.py.

Output format (enforced by utils/validate_submission.py):
  - TAB-separated, UTF-8, no BOM
  - matching_results.tsv header : source1_entity_id\tmatched_entity_ids
  - candidate_pairs.tsv header  : source1_entity_id\tcandidate_entity_ids
  - Every test_source1.tsv entity_id gets exactly one row
  - ID lists: comma-separated S2-/S3- IDs, no quoting, no duplicates per row
  - Empty string (not whitespace, not "None") for singletons
  - Every matched ID must appear in the candidate list for that entity

Usage:
    python infer.py
    python infer.py --data-root dataset --model-dir output/model --output-dir output
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_pipeline import (  # noqa: E402
    build_candidate_pairs,
    add_similarity_features,
    load_source,
)

# Features must match exactly what train.py trained on.
FEATURE_COLS = ["name_jaccard", "addr_jaccard"]

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


def load_threshold(model_dir: str) -> float:
    threshold_path = os.path.join(model_dir, "threshold.json")
    if not os.path.isfile(threshold_path):
        raise FileNotFoundError(
            f"threshold.json not found at {threshold_path}. "
            "Run threshold_tune.py first."
        )
    with open(threshold_path, encoding="utf-8") as fh:
        data = json.load(fh)
    t = float(data["threshold"])
    print(f"Loaded threshold: {t}  (F0.5 on val: {data.get('f0_5_macro', 'n/a')})")
    return t


def load_model(model_dir: str) -> lgb.Booster:
    model_path = os.path.join(model_dir, "lgbm_model.txt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"lgbm_model.txt not found at {model_path}. "
            "Run train.py first."
        )
    booster = lgb.Booster(model_file=model_path)
    print(f"Loaded model: {model_path}")
    return booster


def verify_and_write(path: str, header: str, rows: dict, required_ids: set, col_label: str):
    """
    Write a submission TSV, asserting exactly the right set of S1 entity rows.

    Raises a clear RuntimeError (not a silent bad write) if the row set
    doesn't match test_source1.tsv exactly — catches pipeline bugs before
    they produce a silently invalid submission file.
    """
    written = set(rows.keys())
    missing = required_ids - written
    extra = written - required_ids
    if missing or extra:
        raise RuntimeError(
            f"[{col_label}] Row-set mismatch before writing {os.path.basename(path)}:\n"
            + (f"  Missing {len(missing)} S1 entities, e.g.: {sorted(missing)[:5]}\n" if missing else "")
            + (f"  Extra   {len(extra)} S1 entities, e.g.: {sorted(extra)[:5]}\n" if extra else "")
            + "This is a pipeline bug — the output must contain every test_source1 entity exactly once."
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        for s1_id in sorted(rows):          # sort for reproducible diffs
            id_list = rows[s1_id]
            # Validator requires: no duplicates within a list; only S2-/S3- prefixes.
            deduped = sorted(set(id_list))  # sorted for determinism
            fh.write(f"{s1_id}\t{','.join(deduped)}\n")

    n_non_empty = sum(1 for v in rows.values() if v)
    print(f"Wrote {path}  ({len(rows):,} rows, {n_non_empty:,} non-empty)")


def run(data_root: str, model_dir: str, output_dir: str):
    test_dir = os.path.join(data_root, "test")

    # ------------------------------------------------------------------
    # 1. Load model and threshold
    # ------------------------------------------------------------------
    booster = load_model(model_dir)
    threshold = load_threshold(model_dir)

    # ------------------------------------------------------------------
    # 2. Load test source files
    # ------------------------------------------------------------------
    print("\nLoading test source files...", flush=True)
    t1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    t2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    t3 = load_source(os.path.join(test_dir, "test_source3.tsv"))

    all_test_s1_ids: set = set(t1["entity_id"].tolist())
    print(f"  test_source1: {len(t1):,} entities")
    print(f"  test_source2: {len(t2):,} entities")
    print(f"  test_source3: {len(t3):,} entities")

    # ------------------------------------------------------------------
    # 3. Build candidate pairs (blocking)
    # ------------------------------------------------------------------
    print("\nBuilding candidate pairs (blocking)...", flush=True)
    pairs = build_candidate_pairs(t1, t2, t3)
    print(f"  {len(pairs):,} raw blocked candidate pairs")

    # ------------------------------------------------------------------
    # 4. Add similarity features and score
    # ------------------------------------------------------------------
    print("\nComputing similarity features...", flush=True)
    pairs = add_similarity_features(pairs)

    print("Scoring candidate pairs with LightGBM...", flush=True)
    X = pairs[FEATURE_COLS].values
    pairs["pred_proba"] = booster.predict(X)

    # ------------------------------------------------------------------
    # 5. Build candidate_pairs.tsv dict (ALL blocked candidates, pre-threshold)
    #    Every S1 entity must appear, even those with zero blocking candidates.
    # ------------------------------------------------------------------
    candidate_rows: dict = {eid: [] for eid in all_test_s1_ids}
    for s1_id, grp in pairs.groupby("source1_entity_id", sort=False):
        candidate_rows[s1_id] = grp["candidate_entity_id"].tolist()

    # ------------------------------------------------------------------
    # 6. Build matching_results.tsv dict (only candidates above threshold)
    #    Subset constraint: matched ⊆ candidates per entity — guaranteed by
    #    construction since we filter the same pairs DataFrame.
    # ------------------------------------------------------------------
    matched_rows: dict = {eid: [] for eid in all_test_s1_ids}
    passed = pairs[pairs["pred_proba"] >= threshold]
    for s1_id, grp in passed.groupby("source1_entity_id", sort=False):
        matched_rows[s1_id] = grp["candidate_entity_id"].tolist()

    # ------------------------------------------------------------------
    # 7. Assert completeness BEFORE writing (fail loudly, not silently)
    # ------------------------------------------------------------------
    print("\nVerifying output row sets...", flush=True)
    # These assert calls raise RuntimeError with a clear message on mismatch.
    verify_and_write(
        os.path.join(output_dir, "candidate_pairs.tsv"),
        CANDIDATE_HEADER,
        candidate_rows,
        all_test_s1_ids,
        "candidate_pairs",
    )
    verify_and_write(
        os.path.join(output_dir, "matching_results.tsv"),
        MATCHING_HEADER,
        matched_rows,
        all_test_s1_ids,
        "matching_results",
    )

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    n_entities = len(all_test_s1_ids)
    n_predicted_match = sum(1 for v in matched_rows.values() if v)
    n_singleton = n_entities - n_predicted_match
    singleton_rate = n_singleton / n_entities if n_entities else 0.0

    match_counts = [len(v) for v in matched_rows.values() if v]
    mean_matches = sum(match_counts) / len(match_counts) if match_counts else 0.0

    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print(f"  Test S1 entities        : {n_entities:,}")
    print(f"  Predicted singletons    : {n_singleton:,}  ({singleton_rate:.1%})")
    print(f"  Predicted non-singletons: {n_predicted_match:,}")
    print(f"  Mean matches per non-singleton entity: {mean_matches:.2f}")
    print(f"  Decision threshold used : {threshold}")
    print("=" * 60)
    print("\nNext step — validate before submitting:")
    print(
        f"  python utils/validate_submission.py "
        f"--matching {os.path.join(output_dir, 'matching_results.tsv')} "
        f"--candidate {os.path.join(output_dir, 'candidate_pairs.tsv')} "
        f"--test-dir {test_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score test candidate pairs with the trained LightGBM model and write submission files."
    )
    parser.add_argument(
        "--data-root", default="dataset",
        help="Root directory containing train/ and test/ subdirs (default: dataset).",
    )
    parser.add_argument(
        "--model-dir", default="output/model",
        help="Directory with lgbm_model.txt and threshold.json from train.py/threshold_tune.py "
             "(default: output/model).",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Where to write matching_results.tsv and candidate_pairs.tsv (default: output).",
    )
    args = parser.parse_args()
    run(args.data_root, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
