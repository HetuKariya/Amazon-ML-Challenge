"""
Baseline pipeline for the Business Entity Resolution Challenge.

Purpose: a fast, sanity-check baseline -- NOT the final model. It uses
exact-match blocking on (normalized_name, country) as both the
candidate generator AND the final matcher (no separate classifier
yet). This validates the whole plumbing -- file I/O, TSV formatting,
the F_0.5 scorer, submission writing -- and gives you a first real
number before you invest in fuzzy blocking + a trained classifier
(steps 4-5).

Usage:
    # Local validation using only train/ files (no test files touched)
    python3 baseline_pipeline.py --data-root dataset --mode cv

    # Generate the actual submission files from test/
    python3 baseline_pipeline.py --data-root dataset --mode submit --output-dir output

Design notes for the scale involved (millions of rows per file):
    - All blocking is done via pandas merge (hash join), which is
      O(n + m), never O(n * m). Nothing in this script does a python-
      level double loop over records.
    - Very short normalized names (<2 chars after suffix-stripping,
      e.g. a business named just "Inc") are EXCLUDED from blocking
      entirely rather than left as an empty-string key -- an empty key
      would silently bucket every such business together into one
      giant block and either blow up memory or produce mass false
      merges. These entities fall back to "no candidates found" here;
      tightening this is exactly the kind of thing step 4 (feature
      engineering) should improve on.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, address_tokens, jaccard  # noqa: E402
from scoring import f0_5_macro, deterministic_split  # noqa: E402

MIN_KEY_LEN = 2  # normalized names shorter than this are excluded from blocking


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def load_ground_truth(path: str) -> dict:
    """Returns {source1_entity_id: frozenset(matched_ids)}."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        ids = frozenset(x for x in row.matched_entity_ids.split(",") if x)
        gt[row.source1_entity_id] = ids
    return gt


def add_blocking_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'block_key' column: normalized_name + '||' + country.
    Rows whose normalized name is too short to be a safe blocking key
    get block_key = None (excluded from any merge).
    """
    df = df.copy()
    norm = df["business_name"].map(normalize_name)
    too_short = norm.str.len() < MIN_KEY_LEN
    df["block_key"] = norm + "||" + df["country"]
    df.loc[too_short, "block_key"] = None
    return df


def _blocked_pairs(s1_valid, other_b, source_label):
    """
    Exact-key merge, keeping address text on both sides so a similarity
    score can be computed afterward without a second lookup pass.
    Returns columns: source1_entity_id, candidate_entity_id, source,
    s1_address, cand_address.
    """
    m = s1_valid.merge(other_b, on="block_key", suffixes=("_s1", "_cand"))
    m = m.rename(columns={
        "entity_id_s1": "source1_entity_id",
        "entity_id_cand": "candidate_entity_id",
        "business_address_s1": "s1_address",
        "business_address_cand": "cand_address",
    })
    m["source"] = source_label
    return m[["source1_entity_id", "candidate_entity_id", "source", "s1_address", "cand_address"]]


def build_candidate_pairs(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """
    Exact-key blocking join. Returns a long DataFrame, one row per
    (source1_entity_id, candidate_entity_id) blocked pair, with both
    sides' raw address text attached for downstream similarity scoring.
    Entities with zero candidates simply have no rows here -- callers
    must fill those back in against the full s1 id list.
    """
    cols = ["entity_id", "block_key", "business_address"]
    s1b = add_blocking_key(s1)[cols]
    s2b = add_blocking_key(s2)[cols].dropna(subset=["block_key"])
    s3b = add_blocking_key(s3)[cols].dropna(subset=["block_key"])
    s1_valid = s1b.dropna(subset=["block_key"])

    p2 = _blocked_pairs(s1_valid, s2b, "S2")
    p3 = _blocked_pairs(s1_valid, s3b, "S3")
    pairs = pd.concat([p2, p3], ignore_index=True)

    # Diagnostics: block-size sanity check. A handful of huge blocks
    # (generic names) is expected and fine; thousands of huge blocks
    # means the blocking key needs tightening before this scales.
    pair_counts_per_key = pairs.groupby("source1_entity_id").size()
    if len(pair_counts_per_key) and pair_counts_per_key.max() > 5000:
        worst = pair_counts_per_key.idxmax()
        print(f"  [warning] entity {worst} has {pair_counts_per_key.max():,} candidate pairs "
              f"-- likely a generic-name block; similarity filtering below should clean this up")

    return pairs


def add_address_similarity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Adds an 'addr_jaccard' column: token-set Jaccard similarity of the
    two sides' business_address.

    Normalizes each UNIQUE address string exactly once and looks the
    result up for every row, instead of re-running the regex-based
    tokenizer per pair. This matters a lot here: a single generic-name
    block (e.g. every "Meridian" collision) repeats the same address
    strings across thousands of pair rows, so naive per-row tokenization
    redoes the same regex work thousands of times over."""
    pairs = pairs.copy()
    n_pairs = len(pairs)
    unique_addrs = pd.unique(
        pd.concat([pairs["s1_address"], pairs["cand_address"]], ignore_index=True)
    )
    print(f"  tokenizing {len(unique_addrs):,} unique addresses "
          f"(from {n_pairs:,} candidate pairs)...")
    tok_map = {addr: address_tokens(addr) for addr in unique_addrs}
    s1_tok = pairs["s1_address"].map(tok_map)
    cand_tok = pairs["cand_address"].map(tok_map)
    print("  scoring pairs...")
    pairs["addr_jaccard"] = [jaccard(a, b) for a, b in zip(s1_tok, cand_tok)]
    return pairs


def candidates_dict_from_pairs(pairs: pd.DataFrame, all_s1_ids) -> dict:
    """Every id in all_s1_ids appears, empty set if it had no rows in `pairs`."""
    result = {eid: set() for eid in all_s1_ids}
    for eid, group in pairs.groupby("source1_entity_id")["candidate_entity_id"]:
        result[eid].update(group)
    return {eid: frozenset(ids) for eid, ids in result.items()}


def build_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """Back-compat wrapper: candidate id sets with no similarity filtering
    applied (this is what the original baseline used directly as its
    prediction -- kept here only for candidate_pairs.tsv, which is
    supposed to be the pre-filter candidate set)."""
    pairs = build_candidate_pairs(s1, s2, s3)
    return candidates_dict_from_pairs(pairs, s1["entity_id"])


def tune_threshold(pairs: pd.DataFrame, ground_truth: dict, all_s1_ids,
                    thresholds=None):
    """Grid-searches the addr_jaccard threshold that maximizes local
    F_0.5 macro on the given (validation) ground truth. Returns
    (best_threshold, best_result_dict, full_grid_results)."""
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pairs = add_address_similarity(pairs)
    grid = []
    best = (None, None)
    for t in thresholds:
        kept = pairs[pairs["addr_jaccard"] >= t]
        preds = candidates_dict_from_pairs(kept, all_s1_ids)
        result = f0_5_macro(preds, ground_truth)
        grid.append((t, result))
        if best[1] is None or result["f0_5_macro"] > best[1]["f0_5_macro"]:
            best = (t, result)
    return best[0], best[1], grid


def _write_tsv(path: str, id_col: str, list_col: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for eid, ids in data.items():
            # Sort for determinism / reproducible diffs across runs.
            f.write(f"{eid}\t{','.join(sorted(ids))}\n")


def write_matching_results(path: str, predictions: dict):
    _write_tsv(path, "source1_entity_id", "matched_entity_ids", predictions)


def write_candidate_pairs(path: str, candidates: dict):
    _write_tsv(path, "source1_entity_id", "candidate_entity_ids", candidates)


def run_cv(data_root: str, val_frac: float, seed: int):
    train_dir = os.path.join(data_root, "train")
    print("Loading train files...")
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    train_ids, val_ids = deterministic_split(s1["entity_id"], val_frac=val_frac, seed=seed)
    print(f"train fold: {len(train_ids):,} entities | val fold: {len(val_ids):,} entities")

    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    val_gt = {eid: gt[eid] for eid in val_ids}
    all_val_ids = s1_val["entity_id"].tolist()

    print("Building candidate pairs for the validation fold "
          "(searched against the FULL train_source2/3 pool, same as test-time)...")
    pairs = build_candidate_pairs(s1_val, s2, s3)
    print(f"  {len(pairs):,} total blocked candidate pairs generated")

    candidates = candidates_dict_from_pairs(pairs, all_val_ids)
    unfiltered_result = f0_5_macro(candidates, val_gt)
    print("\n=== UNFILTERED (raw block = match, old baseline) ===")
    for k, v in unfiltered_result.items():
        print(f"  {k}: {v}")

    print("\nTuning address-similarity threshold on the validation fold...")
    best_t, best_result, grid = tune_threshold(pairs, val_gt, all_val_ids)
    print("\n=== THRESHOLD GRID ===")
    for t, r in grid:
        print(f"  threshold={t:.1f}  f0_5_macro={r['f0_5_macro']:.4f}  "
              f"singleton_acc={r['singleton_accuracy']:.4f}  "
              f"precision={r['mean_precision_nonsingleton']:.4f}  "
              f"recall={r['mean_recall_nonsingleton']:.4f}")

    print(f"\n=== BEST: threshold={best_t} ===")
    for k, v in best_result.items():
        print(f"  {k}: {v}")

    n_with_candidates = sum(1 for v in candidates.values() if v)
    print(f"\n  entities with >=1 candidate (pre-filter): {n_with_candidates:,} / "
          f"{len(candidates):,} ({n_with_candidates/len(candidates):.1%})")
    print(f"  (this is the RECALL CEILING -- no threshold can recover entities missing here;"
          f" that requires better blocking, not better filtering)")
    return best_t, best_result


def run_submit(data_root: str, output_dir: str, threshold: float):
    test_dir = os.path.join(data_root, "test")

    print("Loading test files...")
    t1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    t2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    t3 = load_source(os.path.join(test_dir, "test_source3.tsv"))

    print(f"test_source1: {len(t1):,} rows | test_source2: {len(t2):,} rows | "
          f"test_source3: {len(t3):,} rows")

    print("Building candidate pairs...")
    all_ids = t1["entity_id"].tolist()
    pairs = build_candidate_pairs(t1, t2, t3)
    candidates = candidates_dict_from_pairs(pairs, all_ids)  # unfiltered -> candidate_pairs.tsv

    print(f"Applying tuned address-similarity threshold ({threshold}) for final matches...")
    pairs = add_address_similarity(pairs)
    kept = pairs[pairs["addr_jaccard"] >= threshold]
    predictions = candidates_dict_from_pairs(kept, all_ids)  # filtered -> matching_results.tsv

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidates_path = os.path.join(output_dir, "candidate_pairs.tsv")
    write_matching_results(matching_path, predictions)
    write_candidate_pairs(candidates_path, candidates)

    n_with_matches = sum(1 for v in predictions.values() if v)
    print(f"\nWrote {matching_path}")
    print(f"Wrote {candidates_path}")
    print(f"entities with >=1 predicted match: {n_with_matches:,} / {len(predictions):,} "
          f"({n_with_matches/len(predictions):.1%})")
    print("\nRun utils/validate_submission.py on these before uploading.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--mode", choices=["cv", "submit"], required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None,
                         help="addr_jaccard threshold for --mode submit; "
                              "get this from the 'BEST' line printed by --mode cv")
    args = parser.parse_args()

    if args.mode == "cv":
        run_cv(args.data_root, args.val_frac, args.seed)
    else:
        if args.threshold is None:
            parser.error("--mode submit requires --threshold <value from cv mode's BEST result>")
        run_submit(args.data_root, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()