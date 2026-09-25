"""
EDA for the Business Entity Resolution Challenge (Amazon ML Challenge 2026).

Run from the directory that contains `dataset/`:
    python3 eda.py --data-root dataset

Designed for multi-million-row files: reads with dtype=str (no silent
type coercion on IDs), streams referential-integrity checks in chunks
instead of loading everything into memory twice, and never does an
O(n^2) operation.

Outputs a text report to stdout. Read the "WHAT TO LOOK FOR" section
at the end for the interpretation checklist.
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def load_source(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing_cols = set(SOURCE_COLS) - set(df.columns)
    if missing_cols:
        print(f"  [WARNING] {path} missing expected columns: {missing_cols}")
    return df


def load_ground_truth(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def prefix_of(entity_id_series):
    return entity_id_series.str.split("-").str[0]


def report_source_file(name, df):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    if df is None:
        return
    n = len(df)
    print(f"rows: {n:,}")

    # Duplicate entity_id check (fatal if not unique)
    dup_ids = n - df["entity_id"].nunique()
    print(f"duplicate entity_id rows: {dup_ids:,}")

    # Prefix consistency: e.g. train_source2.tsv should be 100% 'S2'
    prefixes = prefix_of(df["entity_id"]).value_counts()
    print(f"entity_id prefixes present: {dict(prefixes)}")

    # Empty-string "missing" values (files use keep_default_na=False, so
    # true blanks show up as '' rather than NaN -- check both)
    for col in SOURCE_COLS:
        if col not in df.columns:
            continue
        empty = (df[col].astype(str).str.strip() == "").sum()
        if empty:
            print(f"  '{col}': {empty:,} empty-string values ({empty/n:.2%})")

    # Country distribution
    print(f"country distribution:\n{df['country'].value_counts()}")

    # Text length stats -- these drive blocking-key design and model
    # feature scaling, and outliers often reveal parsing bugs
    name_len = df["business_name"].str.len()
    addr_len = df["business_address"].str.len()
    print(f"business_name length: {name_len.describe().to_string()}")
    print(f"business_address length: {addr_len.describe().to_string()}")

    # Token count (rough proxy for name/address complexity, relevant to
    # TF-IDF / Jaccard blocking key design)
    name_tokens = df["business_name"].str.split().str.len()
    print(f"business_name token count: mean={name_tokens.mean():.2f}, "
          f"max={name_tokens.max()}")

    # Very short names/addresses are often data-quality issues or
    # aggressive truncation -- worth eyeballing a sample
    short_names = df[name_len <= 3]
    if len(short_names):
        print(f"\n{len(short_names)} names with <=3 chars, sample:")
        print(short_names[["entity_id", "business_name"]].head(5).to_string(index=False))

    # Duplicate (name, address, country) rows within the same source file
    # -- indicates the source itself isn't fully deduplicated
    dup_records = df.duplicated(subset=["business_name", "business_address", "country"]).sum()
    print(f"exact duplicate (name, address, country) rows within file: {dup_records:,} "
          f"({dup_records/n:.2%})")


def report_ground_truth(gt, s1_ids, s2_ids, s3_ids):
    print(f"\n{'=' * 70}\ntrain_ground_truth.tsv\n{'=' * 70}")
    if gt is None:
        return
    n = len(gt)
    print(f"rows: {n:,}")

    dup_s1 = n - gt["source1_entity_id"].nunique()
    print(f"duplicate source1_entity_id rows: {dup_s1:,}")

    # Referential integrity: every source1_entity_id should exist in
    # train_source1.tsv
    if s1_ids is not None:
        missing_s1 = (~gt["source1_entity_id"].isin(s1_ids)).sum()
        print(f"source1_entity_id not found in train_source1.tsv: {missing_s1:,}")

    # Parse match lists
    match_lists = gt["matched_entity_ids"].apply(
        lambda s: [x for x in s.split(",") if x] if s else []
    )
    match_counts = match_lists.apply(len)

    n_singletons = (match_counts == 0).sum()
    print(f"\n--- TARGET DISTRIBUTION (this is what your model predicts) ---")
    print(f"singletons (no match): {n_singletons:,} ({n_singletons/n:.2%})")
    print(f"match count distribution:\n{match_counts.value_counts().sort_index().head(20)}")
    print(f"max matches for a single entity: {match_counts.max()}")
    print(f"mean matches (incl. singletons): {match_counts.mean():.3f}")
    print(f"mean matches (excl. singletons): "
          f"{match_counts[match_counts > 0].mean():.3f}")

    # Split matches by source (S2 vs S3) -- imbalance here matters for
    # blocking strategy (e.g. if S3 rarely matches, your blocking key
    # design for S3 needs to be looser, not tighter)
    all_matches = [m for sub in match_lists for m in sub]
    match_prefix_counts = Counter(m.split("-")[0] for m in all_matches)
    print(f"\nmatched IDs by source: {dict(match_prefix_counts)}")

    # Referential integrity for matched IDs against source2/source3
    if s2_ids is not None and s3_ids is not None:
        flat = pd.Series(all_matches)
        s2_matches = flat[flat.str.startswith("S2-")]
        s3_matches = flat[flat.str.startswith("S3-")]
        bad_s2 = (~s2_matches.isin(s2_ids)).sum()
        bad_s3 = (~s3_matches.isin(s3_ids)).sum()
        print(f"matched S2 ids not found in train_source2.tsv: {bad_s2:,}")
        print(f"matched S3 ids not found in train_source3.tsv: {bad_s3:,}")

    # Duplicate IDs *within* a single match list (should be 0 per the spec)
    dup_within = match_lists.apply(lambda lst: len(lst) != len(set(lst)))
    print(f"rows with duplicate IDs inside their own match list: {dup_within.sum():,}")


def check_train_test_country_overlap(train_dfs, test_dfs):
    print(f"\n{'=' * 70}\nCOUNTRY: train vs test\n{'=' * 70}")
    train_countries = set()
    for df in train_dfs:
        if df is not None:
            train_countries |= set(df["country"].unique())
    test_countries = set()
    for df in test_dfs:
        if df is not None:
            test_countries |= set(df["country"].unique())
    print(f"countries in train: {sorted(train_countries)}")
    print(f"countries in test:  {sorted(test_countries)}")
    unseen = test_countries - train_countries
    print(f"countries in test but NOT in train (must still be handled!): {sorted(unseen)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset",
                         help="path to the dataset/ folder (contains train/ and test/)")
    args = parser.parse_args()

    root = args.data_root
    print(f"Loading from: {os.path.abspath(root)}")

    train_s1 = load_source(os.path.join(root, "train", "train_source1.tsv"))
    train_s2 = load_source(os.path.join(root, "train", "train_source2.tsv"))
    train_s3 = load_source(os.path.join(root, "train", "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(root, "train", "train_ground_truth.tsv"))

    test_s1 = load_source(os.path.join(root, "test", "test_source1.tsv"))
    test_s2 = load_source(os.path.join(root, "test", "test_source2.tsv"))
    test_s3 = load_source(os.path.join(root, "test", "test_source3.tsv"))

    report_source_file("train_source1.tsv", train_s1)
    report_source_file("train_source2.tsv", train_s2)
    report_source_file("train_source3.tsv", train_s3)
    report_source_file("test_source1.tsv", test_s1)
    report_source_file("test_source2.tsv", test_s2)
    report_source_file("test_source3.tsv", test_s3)

    report_ground_truth(
        gt,
        s1_ids=set(train_s1["entity_id"]) if train_s1 is not None else None,
        s2_ids=set(train_s2["entity_id"]) if train_s2 is not None else None,
        s3_ids=set(train_s3["entity_id"]) if train_s3 is not None else None,
    )

    check_train_test_country_overlap(
        [train_s1, train_s2, train_s3], [test_s1, test_s2, test_s3]
    )

    print(f"""
{'=' * 70}
WHAT TO LOOK FOR IN THE OUTPUT ABOVE
{'=' * 70}
- duplicate entity_id rows should be exactly 0 in every source file and
  in train_ground_truth.tsv -- if not, that's a data bug or a schema
  misread (check sep="\\t" actually took effect).
- entity_id prefixes should be 100% pure per file (train_source2.tsv
  should be 100% 'S2', etc.) -- any stray prefix means row corruption
  or a multi-line address that broke the TSV parse.
- singleton rate from train_ground_truth.tsv is your single most
  important number: it sets the baseline for "predict nothing" and
  tells you how aggressively your blocking/matching should lean toward
  precision. A high singleton rate (common in ER) means most of your
  F_0.5 is won or lost on correctly leaving entities unmatched.
- match count distribution: if most non-singleton entities match
  exactly 1-2 records, your model is mostly a binary "is this the
  same business" classifier per candidate pair, not a multi-way
  clustering problem -- simplifies your architecture choice.
- matched IDs by source (S2 vs S3 split): a large imbalance suggests
  one source is noisier/sparser and may need a different blocking
  threshold per source rather than one global threshold.
- referential integrity checks (missing/bad ids) should all be 0.
  Any nonzero count here means you cannot trust naive joins later --
  build defensive validation into your pipeline from day one.
- country distribution + the train/test overlap check: confirm France
  only appears in test. This is the single biggest generalization trap
  in this competition -- any feature that hard-codes US/India address
  formats (e.g. regex for US ZIP or Indian PIN code) will silently
  degrade on French addresses. Design blocking keys and features to be
  country-agnostic (token overlap, character n-grams) rather than
  country-specific parsing, or explicitly branch logic by country with
  a generic fallback for unseen ones.
- business_name / business_address length distributions: the long tail
  on address length (you already see addresses up to 222 chars in the
  sample) suggests some addresses embed extra descriptive text
  (landmarks, unit numbers) -- your string-similarity features should
  be robust to length mismatches (e.g. token-set ratio, not raw edit
  distance, which penalizes length differences directly).
- exact duplicate (name, address, country) rows within one source file:
  if nonzero, that source isn't fully deduplicated even before
  cross-source matching, which matters for how you interpret "Source 1
  is the deduplicated reference."
""")


if __name__ == "__main__":
    main()
