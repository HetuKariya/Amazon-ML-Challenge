"""
Baseline pipeline for the Business Entity Resolution Challenge.

Purpose: a fast, sanity-check baseline -- NOT the final model. Candidate
generation uses two blocking passes unioned together:
  - exact key: normalized_name + country (order-sensitive, tight)
  - loose key: first-two-normalized-name-tokens + country (catches
    trailing descriptor/typo differences past the first two words)
Final matches are these candidates filtered by a combined name+address
token-Jaccard similarity score, threshold-tuned on a validation split.
This is still a heuristic, not a trained classifier (that's steps 4-5) --
but it validates the whole plumbing (file I/O, TSV formatting, the F_0.5
scorer, submission writing) and gives a real number to improve on.

Usage:
    # Local validation using only train/ files (no test files touched)
    python3 baseline_pipeline.py --data-root dataset --mode cv

    # Generate the actual submission files from test/
    python3 baseline_pipeline.py --data-root dataset --mode submit --output-dir output --threshold <BEST_FROM_CV>

Design notes for the scale involved (millions of rows per file):
    - All blocking is done via pandas merge (hash join), which is
      O(n + m), never O(n * m). Nothing in this script does a python-
      level double loop over records.
    - Very short normalized names (<2 chars after suffix-stripping,
      e.g. a business named just "Inc") are EXCLUDED from blocking
      entirely rather than left as an empty-string key -- an empty key
      would silently bucket every such business together into one
      giant block and either blow up memory or produce mass false
      merges.
    - The loose key is capped at MAX_LOOSE_BLOCK_SIZE candidates per
      key: a first-two-tokens key shared by more than that many records
      is almost certainly a generic prefix, not a real collision, and
      isn't worth the compute.
    - Name/address tokenization is done once per UNIQUE string, not
      once per candidate pair -- a single generic-name block repeats
      the same strings across thousands of rows, so naive per-row
      tokenization redoes the same regex work thousands of times over.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, name_tokens, address_signal_tokens, jaccard  # noqa: E402
from scoring import f0_5_macro, deterministic_split  # noqa: E402

MIN_KEY_LEN = 2  # normalized names shorter than this are excluded from blocking
MAX_LOOSE_BLOCK_SIZE = 20000  # safety cap: a looser key producing a block bigger
                               # than this is almost certainly a generic prefix
                               # (e.g. "global" or "prime") -- skip it rather than
                               # pay the compute cost for what's overwhelmingly noise


EXPECTED_SOURCE_COLS = {"entity_id", "business_name", "business_address", "country"}
EXPECTED_GT_COLS = {"source1_entity_id", "matched_entity_ids"}


def _check_columns(df: pd.DataFrame, expected: set, path: str) -> pd.DataFrame:
    """Strips stray whitespace from column names and fails loudly with a
    clear message (instead of a cryptic KeyError deep in pandas) if the
    expected columns aren't present -- most commonly caused by a UTF-8
    BOM or odd whitespace in the header row."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: expected columns {sorted(expected)} but found {list(df.columns)!r} "
            f"(missing: {sorted(missing)}). This usually means a UTF-8 BOM or stray "
            f"whitespace/control character in the file's header row -- check the file "
            f"wasn't re-saved by a text editor that adds a BOM."
        )
    return df


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return _check_columns(df, EXPECTED_SOURCE_COLS, path)


def load_ground_truth(path: str) -> dict:
    """Returns {source1_entity_id: frozenset(matched_ids)}.

    Uses direct column access (not itertuples()) deliberately: itertuples()
    builds attribute names from column headers, and silently falls back to
    a positional name like '_1' for any column whose header string isn't a
    valid Python identifier for whatever reason (stray invisible character,
    encoding artifact from how a zip was extracted, etc.) -- causing an
    AttributeError even though plain df["matched_entity_ids"] access works
    fine on the exact same file."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = _check_columns(df, EXPECTED_GT_COLS, path)
    ids = df["source1_entity_id"].to_numpy()
    matches = df["matched_entity_ids"].to_numpy()
    return {s1_id: frozenset(x for x in m.split(",") if x) for s1_id, m in zip(ids, matches)}


def add_blocking_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two blocking-key columns:
      - 'block_key': normalized_name + '||' + country (exact, order-sensitive)
      - 'block_key_loose': first two normalized-name tokens + '||' + country
        (catches trailing descriptor differences, minor typos in later
        tokens, DBA-style additions -- anything after the first two words)

    Rows whose normalized name is too short to be a safe blocking key
    get both columns set to None (excluded from any merge).
    """
    df = df.copy()
    norm = df["business_name"].map(normalize_name)
    too_short = norm.str.len() < MIN_KEY_LEN
    tokens = norm.str.split()
    loose_name = tokens.map(lambda toks: " ".join(toks[:2]) if toks else "")
    df["block_key"] = norm + "||" + df["country"]
    df["block_key_loose"] = loose_name + "||" + df["country"]
    df.loc[too_short, ["block_key", "block_key_loose"]] = None
    return df


def _key_subset(df_with_keys: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Extracts (entity_id, key, business_name, business_address) for one
    blocking-key column, generically renamed to 'key' so the same merge
    code works regardless of which key produced it."""
    sub = df_with_keys[["entity_id", key_col, "business_name", "business_address"]].copy()
    sub = sub.rename(columns={key_col: "key"})
    return sub.dropna(subset=["key"])


def _blocked_pairs(s1_sub: pd.DataFrame, other_sub: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Merge on the generic 'key' column, keeping both name and address text
    on each side so similarity features can be computed afterward without
    a second lookup pass. Returns columns: source1_entity_id,
    candidate_entity_id, source, s1_name, cand_name, s1_address, cand_address.
    """
    m = s1_sub.merge(other_sub, on="key", suffixes=("_s1", "_cand"))
    m = m.rename(columns={
        "entity_id_s1": "source1_entity_id",
        "entity_id_cand": "candidate_entity_id",
        "business_name_s1": "s1_name",
        "business_name_cand": "cand_name",
        "business_address_s1": "s1_address",
        "business_address_cand": "cand_address",
    })
    m["source"] = source_label
    return m[["source1_entity_id", "candidate_entity_id", "source",
              "s1_name", "cand_name", "s1_address", "cand_address"]]


def build_candidate_pairs(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass blocking join (exact full-name key, then a looser first-two-
    tokens key), unioned together. Returns a long DataFrame, one row per
    unique (source1_entity_id, candidate_entity_id) blocked pair, with
    both sides' raw name/address text attached for downstream similarity
    scoring. Entities with zero candidates simply have no rows here --
    callers must fill those back in against the full s1 id list.
    """
    s1k = add_blocking_key(s1)
    s2k = add_blocking_key(s2)
    s3k = add_blocking_key(s3)

    frames = []
    for key_col in ("block_key", "block_key_loose"):
        s1_sub = _key_subset(s1k, key_col)
        s2_sub = _key_subset(s2k, key_col)
        s3_sub = _key_subset(s3k, key_col)

        if key_col == "block_key_loose":
            # Safety cap: a loose key this common is almost certainly a
            # generic prefix, not a real business name collision -- skip
            # it rather than pay for a mostly-noise merge.
            sizes2 = s2_sub.groupby("key").size()
            bad2 = set(sizes2[sizes2 > MAX_LOOSE_BLOCK_SIZE].index)
            if bad2:
                print(f"  [loose-key cap] excluding {len(bad2)} overly generic loose keys "
                      f"from S2 (> {MAX_LOOSE_BLOCK_SIZE:,} candidates each)")
                s2_sub = s2_sub[~s2_sub["key"].isin(bad2)]
            sizes3 = s3_sub.groupby("key").size()
            bad3 = set(sizes3[sizes3 > MAX_LOOSE_BLOCK_SIZE].index)
            if bad3:
                print(f"  [loose-key cap] excluding {len(bad3)} overly generic loose keys "
                      f"from S3 (> {MAX_LOOSE_BLOCK_SIZE:,} candidates each)")
                s3_sub = s3_sub[~s3_sub["key"].isin(bad3)]

        frames.append(_blocked_pairs(s1_sub, s2_sub, "S2"))
        frames.append(_blocked_pairs(s1_sub, s3_sub, "S3"))

    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

    # Diagnostics: per-entity pair-count sanity check. A handful of huge
    # blocks (generic names) is expected and fine; thousands of huge
    # blocks means the blocking keys need tightening before this scales.
    pair_counts_per_key = pairs.groupby("source1_entity_id").size()
    if len(pair_counts_per_key) and pair_counts_per_key.max() > 5000:
        worst = pair_counts_per_key.idxmax()
        print(f"  [warning] entity {worst} has {pair_counts_per_key.max():,} candidate pairs "
              f"-- likely a generic-name block; similarity filtering below should clean this up")

    return pairs


def add_similarity_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Adds 'addr_jaccard' and 'name_jaccard' columns (kept as separate
    signals -- see tune_threshold for why they're NOT collapsed into one
    combined scalar).

    Normalizes each UNIQUE name/address string exactly once and looks
    the result up for every row, instead of re-running the regex-based
    tokenizer per pair -- a single generic-name block repeats the same
    strings across thousands of pair rows, so naive per-row tokenization
    redoes the same regex work thousands of times over.

    Address similarity uses address_signal_tokens (generic words like
    'st'/'ave'/'suite' stripped out) rather than raw address_tokens --
    otherwise two unrelated addresses that happen to both contain "St"
    and "Suite" register as partially similar for no meaningful reason."""
    pairs = pairs.copy()
    n_pairs = len(pairs)

    unique_addrs = pd.unique(
        pd.concat([pairs["s1_address"], pairs["cand_address"]], ignore_index=True)
    )
    print(f"  tokenizing {len(unique_addrs):,} unique addresses "
          f"(from {n_pairs:,} candidate pairs)...")
    addr_tok_map = {a: address_signal_tokens(a) for a in unique_addrs}
    s1_atok = pairs["s1_address"].map(addr_tok_map)
    cand_atok = pairs["cand_address"].map(addr_tok_map)

    unique_names = pd.unique(
        pd.concat([pairs["s1_name"], pairs["cand_name"]], ignore_index=True)
    )
    print(f"  tokenizing {len(unique_names):,} unique names...")
    name_tok_map = {n: name_tokens(n) for n in unique_names}
    s1_ntok = pairs["s1_name"].map(name_tok_map)
    cand_ntok = pairs["cand_name"].map(name_tok_map)

    print("  scoring pairs...")
    pairs["addr_jaccard"] = [jaccard(a, b) for a, b in zip(s1_atok, cand_atok)]
    pairs["name_jaccard"] = [jaccard(a, b) for a, b in zip(s1_ntok, cand_ntok)]
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
                    name_thresholds=None, addr_thresholds=None):
    """Grid-searches OVER BOTH name_jaccard and addr_jaccard thresholds
    independently (not collapsed into one combined scalar) to maximize
    local F_0.5 macro on the given (validation) ground truth.

    Why two separate thresholds instead of one: name and address carry
    different, non-interchangeable evidence. A single combined score
    (whether average or min) forces a fixed tradeoff between them; a 2D
    grid lets the data tell us, e.g., "require decent address overlap
    regardless of name" if that turns out to separate true/false matches
    better than any symmetric combination would.

    Returns (best_name_threshold, best_addr_threshold, best_result_dict,
    full_grid_results) where full_grid_results is a list of
    ((name_t, addr_t), result_dict) sorted by f0_5_macro descending.
    """
    if name_thresholds is None:
        name_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
    if addr_thresholds is None:
        addr_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    pairs = add_similarity_features(pairs)
    grid = []
    for tn in name_thresholds:
        for ta in addr_thresholds:
            kept = pairs[(pairs["name_jaccard"] >= tn) & (pairs["addr_jaccard"] >= ta)]
            preds = candidates_dict_from_pairs(kept, all_s1_ids)
            result = f0_5_macro(preds, ground_truth)
            grid.append(((tn, ta), result))
    grid.sort(key=lambda x: x[1]["f0_5_macro"], reverse=True)
    best_thresholds, best_result = grid[0]
    return best_thresholds[0], best_thresholds[1], best_result, grid


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


def run_cv(data_root: str, val_frac: float, seed: int, sample_s1: int = None):
    train_dir = os.path.join(data_root, "train")
    print("Loading train files...")
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    train_ids, val_ids = deterministic_split(s1["entity_id"], val_frac=val_frac, seed=seed)
    print(f"train fold: {len(train_ids):,} entities | val fold: {len(val_ids):,} entities")

    if sample_s1 is not None and sample_s1 < len(val_ids):
        # Deterministic subsample for fast iteration -- still searched
        # against the FULL s2/s3 pool, so blocking behavior (incl. any
        # generic-key explosions) is realistic; only the number of S1
        # entities scored is reduced.
        val_ids = set(sorted(val_ids)[:sample_s1])
        print(f"  --sample-s1 given: using {len(val_ids):,} of the val fold for a quick run")

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

    print("\nTuning name/address similarity thresholds (2D grid) on the validation fold...")
    best_tn, best_ta, best_result, grid = tune_threshold(pairs, val_gt, all_val_ids)
    print("\n=== TOP 15 THRESHOLD COMBINATIONS (by f0_5_macro) ===")
    for (tn, ta), r in grid[:15]:
        print(f"  name>={tn:.1f}  addr>={ta:.1f}  f0_5_macro={r['f0_5_macro']:.4f}  "
              f"singleton_acc={r['singleton_accuracy']:.4f}  "
              f"precision={r['mean_precision_nonsingleton']:.4f}  "
              f"recall={r['mean_recall_nonsingleton']:.4f}")

    print(f"\n=== BEST: name_threshold={best_tn}  addr_threshold={best_ta} ===")
    for k, v in best_result.items():
        print(f"  {k}: {v}")

    n_with_candidates = sum(1 for v in candidates.values() if v)
    print(f"\n  entities with >=1 candidate (pre-filter): {n_with_candidates:,} / "
          f"{len(candidates):,} ({n_with_candidates/len(candidates):.1%})")
    print(f"  (this is the RECALL CEILING -- no threshold can recover entities missing here;"
          f" that requires better blocking, not better filtering)")
    return best_tn, best_ta, best_result


def run_submit(data_root: str, output_dir: str, name_threshold: float, addr_threshold: float):
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

    print(f"Applying tuned thresholds (name>={name_threshold}, addr>={addr_threshold}) "
          f"for final matches...")
    pairs = add_similarity_features(pairs)
    kept = pairs[(pairs["name_jaccard"] >= name_threshold) & (pairs["addr_jaccard"] >= addr_threshold)]
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
    parser.add_argument("--name-threshold", type=float, default=None,
                         help="name_jaccard threshold for --mode submit; "
                              "get this from the 'BEST' line printed by --mode cv")
    parser.add_argument("--addr-threshold", type=float, default=None,
                         help="addr_jaccard threshold for --mode submit; "
                              "get this from the 'BEST' line printed by --mode cv")
    parser.add_argument("--sample-s1", type=int, default=None,
                         help="(cv mode only) use only this many S1 entities from the "
                              "validation fold, for fast iteration -- still matched "
                              "against the full S2/S3 pool")
    args = parser.parse_args()

    if args.mode == "cv":
        run_cv(args.data_root, args.val_frac, args.seed, sample_s1=args.sample_s1)
    else:
        if args.name_threshold is None or args.addr_threshold is None:
            parser.error("--mode submit requires --name-threshold and --addr-threshold "
                         "(from the 'BEST' line printed by --mode cv)")
        run_submit(args.data_root, args.output_dir, args.name_threshold, args.addr_threshold)


if __name__ == "__main__":
    main()