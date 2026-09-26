"""
Blocking recall evaluator for Amazon ML Challenge 2026.

Purpose
-------
Evaluate candidate-blocking keys against the REAL ground truth without
materializing the huge S1 x S2/S3 candidate-pair table.

This is deliberately different from blocking_v2.py:
- It reads the ground-truth matched IDs.
- It fetches only the S2/S3 rows that are actually true matches.
- It computes blocking keys on S1 validation rows and true matched rows.
- It reports how many true matches each name, address, or hybrid blocking key would recover.

No giant Cartesian merge is created.

Run from student_resource/:
    python blocking_recall_eval.py --data-root dataset

Expected output includes:
    - mean true-match recall per validation entity
    - global true-match recall
    - fraction of non-singleton entities with at least one recovered match
    - union recall for combinations of keys
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address, address_signal_tokens
from scoring import deterministic_split


CHUNK_SIZE = 250_000
MIN_TOKEN_LEN = 3


def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_ground_truth(path):
    # Category 1: switched from itertuples() to direct column access.
    # itertuples() builds attribute names from column headers and silently
    # renames any column whose header isn't a valid Python identifier to '_1',
    # '_2', etc. -- causing an AttributeError on row.matched_entity_ids even
    # though df["matched_entity_ids"] works fine. baseline_pipeline.py already
    # documents this exact failure mode and avoids it the same way.
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]  # strip BOM/stray whitespace
    gt = {}
    for s1_id, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1_id] = [x for x in matched.split(",") if x]
    return gt


def make_name_features(series):
    """
    Compute several country-agnostic name blocking signatures.
    normalize_name() itself remains the canonical implementation.

    Category 3 note: MIN_TOKEN_LEN = 3 filters sub-3-char tokens when
    computing first_token/last_token here. The pipeline's add_blocking_key()
    uses a different criterion (MIN_KEY_LEN = 3 on the full normalized name,
    not per-token). This means eval recall numbers for 'first_token' reflect
    a slightly stricter definition than what the pipeline indexes -- actual
    pipeline recall may be marginally higher for names whose first meaningful
    token is 2 chars (e.g. "AB Solutions" -> first_token="ab" in pipeline
    but "solutions" here). Keep MIN_TOKEN_LEN=3 as a conservative floor;
    do not change it to match the pipeline without re-running the eval.
    """
    # Category 2: name_norm uses vectorized Series.map instead of a Python
    # loop -- normalize_name() uses compiled regexes so the per-call cost is
    # low, but avoiding the list accumulation overhead matters at 100K+ rows.
    # sorted_name and boundary retain the loop because their conditional logic
    # (sort+dedup, compact[:4]+compact[-4:]) doesn't vectorize cleanly without
    # .apply(), which is equally slow.
    norm_series = series.astype(str).map(normalize_name)
    values = norm_series.tolist()

    name_norm = norm_series.tolist()
    sorted_name = []
    first_token = []
    last_token = []
    first_last = []
    boundary = []

    for n in values:
        toks = [t for t in n.split() if len(t) >= MIN_TOKEN_LEN]

        if toks:
            first_token.append(toks[0])
            last_token.append(toks[-1])
        else:
            first_token.append("")
            last_token.append("")

        if len(toks) >= 2:
            first_last.append(toks[0] + "|" + toks[-1])
        else:
            first_last.append("")

        if len(toks) >= 2:
            sorted_name.append(" ".join(sorted(set(toks))))
        else:
            sorted_name.append("")

        compact = re.sub(r"[^a-z0-9]+", "", n)
        if len(compact) >= 8:
            boundary.append(compact[:4] + "|" + compact[-4:])
        else:
            boundary.append("")

    return {
        "name_norm": name_norm,
        "sorted_name": sorted_name,
        "first_token": first_token,
        "last_token": last_token,
        "first_last": first_last,
        "boundary": boundary,
    }


def make_address_features(series):
    """
    Compute country-agnostic address blocking signatures.

    These are deliberately conservative:
      - address_norm: normalized full address
      - address_sorted_signal: sorted unique informative address tokens
      - address_first3_signal: first 3 informative address tokens
      - address_digits: digit sequences from the address
      - address_boundary: first 4 + last 4 alphanumeric characters

    The evaluator adds country to every blocking key during comparison.
    Empty/low-information keys are returned as "" and therefore cannot
    produce a blocking hit.
    """
    norm_series = series.astype(str).map(normalize_address)
    values = norm_series.tolist()

    address_norm = norm_series.tolist()
    address_sorted_signal = []
    address_first3_signal = []
    address_digits = []
    address_boundary = []

    for raw, n in zip(values, values):
        sig = sorted(set(address_signal_tokens(raw)))

        if len(sig) >= 2:
            address_sorted_signal.append(" ".join(sig))
        else:
            address_sorted_signal.append("")

        if sig:
            address_first3_signal.append(" ".join(sig[:3]))
        else:
            address_first3_signal.append("")

        nums = re.findall(r"\d+", n)
        if nums:
            address_digits.append("|".join(nums))
        else:
            address_digits.append("")

        compact = re.sub(r"[^a-z0-9]+", "", n)
        if len(compact) >= 8:
            address_boundary.append(compact[:4] + "|" + compact[-4:])
        else:
            address_boundary.append("")

    return {
        "address_norm": address_norm,
        "address_sorted_signal": address_sorted_signal,
        "address_first3_signal": address_first3_signal,
        "address_digits": address_digits,
        "address_boundary": address_boundary,
    }


def fetch_true_matches(source_path, needed_ids):
    """
    Read the large source file in chunks and retain only records whose
    entity_id is present in the ground-truth match lists.
    """
    if not needed_ids:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "business_address", "country"]
        )

    print(
        f"  scanning {os.path.basename(source_path)} in chunks "
        f"for {len(needed_ids):,} true-match IDs...",
        flush=True,
    )

    pieces = []

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        if not hit.empty:
            pieces.append(
                hit[["entity_id", "business_name", "business_address", "country"]].copy()
            )

    if not pieces:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "country"]
        )

    out = pd.concat(pieces, ignore_index=True)

    required_cols = {"entity_id", "business_name", "business_address", "country"}
    missing = required_cols - set(out.columns)
    if missing:
        raise RuntimeError(
            f"{os.path.basename(source_path)} is missing required columns after fetching true matches: {sorted(missing)}"
        )

    # Defensive check: every fetched ID should be unique.
    if out["entity_id"].duplicated().any():
        dup = int(out["entity_id"].duplicated().sum())
        raise RuntimeError(
            f"Duplicate entity IDs found while fetching true matches: {dup}"
        )

    print(f"  fetched {len(out):,} matched rows", flush=True)
    return out


def prepare_match_info(df):
    name_feats = make_name_features(df["business_name"])
    addr_feats = make_address_features(df["business_address"])

    out = df[["entity_id", "country"]].copy()

    for k, v in name_feats.items():
        out[k] = v

    for k, v in addr_feats.items():
        out[k] = v

    # Composite keys intentionally require both a useful name signal and a
    # useful address signal. This catches cases where either name or address
    # alone is too noisy while remaining much cheaper than fuzzy search.
    out["first_token_addr_digits"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_token"], out["address_digits"])
    ]
    out["first_token_addr_first3"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_token"], out["address_first3_signal"])
    ]
    out["first_token_addr_boundary"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_token"], out["address_boundary"])
    ]

    return out


def evaluate_key(s1_info, match_rows, gt, key_name):
    """
    Evaluate one blocking key.

    For every actual GT match, test whether the S1 entity and matched S2/S3
    record have the same key. Then aggregate:
      - pair/global recall
      - mean per-entity recall
      - entity coverage (at least one true match recoverable)

    Category 2: replaced itertuples() row loop with a vectorized merge.
    The original loop did 382K Python iterations per key (× 6 keys = 2.3M
    total) and was the dominant runtime cost (~3+ minutes). The merge is
    O(n) and runs in pandas C code.
    """
    feature_cols = ["entity_id", "country", key_name]
    s1_keys = s1_info[feature_cols].copy()
    s1_keys["s1_composite"] = s1_keys[key_name].astype(str) + "||" + s1_keys["country"].astype(str)

    # Category 1 fix: the original guard was `s1_key != ""`, but the
    # composite key is never literally "" -- an empty feature produces
    # "||US" which would spuriously match any other blank-named entity in
    # that country. Check the feature part alone is non-empty.
    s1_keys["s1_valid"] = s1_keys[key_name].astype(str).str.strip() != ""

    match_cols = ["source1_entity_id", "matched_entity_id", "country", key_name]
    mr = match_rows[match_cols].copy()
    mr["m_composite"] = mr[key_name].astype(str) + "||" + mr["country"].astype(str)

    merged = mr.merge(
        s1_keys[["entity_id", "s1_composite", "s1_valid"]],
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    )

    merged["hit"] = (
        merged["s1_valid"].fillna(False)
        & (merged["s1_composite"] == merged["m_composite"])
    )

    total_true = len(merged)
    total_recovered = int(merged["hit"].sum())

    per_entity = merged.groupby("source1_entity_id").agg(
        covered=("hit", "sum"),
        total=("hit", "count"),
    )
    per_entity["recall"] = per_entity["covered"] / per_entity["total"]
    entity_with_any = int((per_entity["covered"] > 0).sum())
    n_entities = len(per_entity)

    return {
        "mean_entity_recall": float(per_entity["recall"].mean()) if n_entities else 0.0,
        "global_pair_recall": total_recovered / total_true if total_true else 0.0,
        "entities_with_any_true_match_recovered": entity_with_any,
        "n_non_singleton_entities": n_entities,
        "entity_coverage": entity_with_any / n_entities if n_entities else 0.0,
        "total_true_pairs": total_true,
        "recovered_true_pairs": total_recovered,
    }


def evaluate_union(s1_info, match_rows, key_names):
    """
    Category 2: replaced itertuples() loop with a vectorized approach.
    For each key, compute a composite key for S1 and the matched row, then
    mark a pair as recovered if ANY key matches. Uses pandas merge per key
    then takes the union via boolean OR -- same semantics as the original
    loop but without Python-level row iteration.
    """
    # Build S1 composite keys for all requested key_names at once.
    s1_sub = s1_info[["entity_id", "country"] + list(key_names)].copy()
    for kn in key_names:
        col = f"s1_{kn}_composite"
        s1_sub[col] = s1_sub[kn].astype(str) + "||" + s1_sub["country"].astype(str)
        # Category 1: guard against empty feature part (see evaluate_key fix).
        s1_sub[f"s1_{kn}_valid"] = s1_sub[kn].astype(str).str.strip() != ""

    mr = match_rows[["source1_entity_id", "matched_entity_id", "country"] + list(key_names)].copy()
    for kn in key_names:
        mr[f"m_{kn}_composite"] = mr[kn].astype(str) + "||" + mr["country"].astype(str)

    merged = mr.merge(
        s1_sub[["entity_id"] + [f"s1_{kn}_composite" for kn in key_names]
                              + [f"s1_{kn}_valid" for kn in key_names]],
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    )

    # A pair is recovered if ANY key matches (union semantics).
    recovered = pd.Series(False, index=merged.index)
    for kn in key_names:
        recovered |= (
            merged[f"s1_{kn}_valid"].fillna(False)
            & (merged[f"s1_{kn}_composite"] == merged[f"m_{kn}_composite"])
        )
    merged["hit"] = recovered

    total_true = len(merged)
    total_recovered = int(merged["hit"].sum())

    per_entity = merged.groupby("source1_entity_id").agg(
        covered=("hit", "sum"),
        total=("hit", "count"),
    )
    per_entity["recall"] = per_entity["covered"] / per_entity["total"]
    covered_entities = int((per_entity["covered"] > 0).sum())
    n_entities = len(per_entity)

    return {
        "mean_entity_recall": float(per_entity["recall"].mean()) if n_entities else 0.0,
        "global_pair_recall": total_recovered / total_true if total_true else 0.0,
        "entity_coverage": covered_entities / n_entities if n_entities else 0.0,
        "covered_entities": covered_entities,
        "n_non_singleton_entities": n_entities,
        "total_true_pairs": total_true,
        "recovered_true_pairs": total_recovered,
    }


def build_match_rows(gt, s1_val_ids, source_prefix):
    """
    Explode only validation ground-truth matches for one source.
    """
    rows = []

    for eid in s1_val_ids:
        for mid in gt[eid]:
            if mid.startswith(source_prefix + "-"):
                rows.append(
                    {
                        "source1_entity_id": eid,
                        "matched_entity_id": mid,
                    }
                )

    return pd.DataFrame(rows)


def run(data_root, val_frac, seed, sample_frac=None, sample_s1=None):
    train_dir = os.path.join(data_root, "train")

    print("Loading S1 + ground truth...", flush=True)
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    gt = load_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )

    _, val_ids = deterministic_split(
        s1["entity_id"], val_frac=val_frac, seed=seed
    )

    # Optional deterministic sample of the validation fold. This is the
    # intended first-pass mode for a multi-million-row dataset: it preserves
    # the exact same validation split semantics while reducing only the S1
    # entities whose true-match rows are inspected.
    if sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("--sample-frac must be in (0, 1]")
        # Hash-based sampling avoids bias from lexical ordering of entity IDs
        # while staying perfectly reproducible across runs/processes.
        import hashlib
        sample_seed = seed + 100003
        sample_threshold = int(sample_frac * (2 ** 32))
        val_ids = {
            eid for eid in val_ids
            if int(hashlib.md5(f"{sample_seed}-{eid}".encode()).hexdigest()[:8], 16) < sample_threshold
        }
        if not val_ids:
            raise RuntimeError("Sample produced zero validation entities; increase --sample-frac")
        print(
            f"sampled validation fold: {len(val_ids):,} entities "
            f"({sample_frac:.1%} of validation fold)",
            flush=True,
        )
    elif sample_s1 is not None:
        if sample_s1 <= 0:
            raise ValueError("--sample-s1 must be > 0")
        import hashlib
        ranked = sorted(
            val_ids,
            key=lambda eid: hashlib.md5(f"{seed + 100003}-{eid}".encode()).hexdigest(),
        )
        val_ids = set(ranked[:min(sample_s1, len(ranked))])
        print(
            f"sampled validation fold: {len(val_ids):,} entities "
            f"(--sample-s1)",
            flush=True,
        )

    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"validation S1 entities: {len(s1_val):,}", flush=True)
    print("Preparing S1 name + address signatures...", flush=True)

    # IMPORTANT: address-based keys are now part of the recall experiment,
    # so S1 must be prepared with both name and address features as well.
    s1_info = prepare_match_info(s1_val)

    s2_match_pairs = build_match_rows(gt, val_ids, "S2")
    s3_match_pairs = build_match_rows(gt, val_ids, "S3")

    print(
        f"true S2 validation matches: {len(s2_match_pairs):,}",
        flush=True,
    )
    print(
        f"true S3 validation matches: {len(s3_match_pairs):,}",
        flush=True,
    )

    needed_s2 = set(s2_match_pairs["matched_entity_id"])
    needed_s3 = set(s3_match_pairs["matched_entity_id"])

    s2_true = fetch_true_matches(
        os.path.join(train_dir, "train_source2.tsv"),
        needed_s2,
    )
    s3_true = fetch_true_matches(
        os.path.join(train_dir, "train_source3.tsv"),
        needed_s3,
    )

    def attach_pairs(pair_df, true_df):
        feature_cols = [
            "source1_entity_id",
            "matched_entity_id",
            "country",
            "name_norm",
            "sorted_name",
            "first_token",
            "last_token",
            "first_last",
            "boundary",
            "address_norm",
            "address_sorted_signal",
            "address_first3_signal",
            "address_digits",
            "address_boundary",
            "first_token_addr_digits",
            "first_token_addr_first3",
            "first_token_addr_boundary",
        ]
        if pair_df.empty:
            return pd.DataFrame(columns=feature_cols)

        m = pair_df.merge(
            true_df,
            left_on="matched_entity_id",
            right_on="entity_id",
            how="left",
            suffixes=("", "_matched"),
            validate="one_to_one",
        )

        m = m.rename(columns={"country": "country"})

        # Category 1: fill NaN feature columns that arise when a matched
        # entity_id is absent from the source file (data gap). Without this,
        # those rows produce "nan||nan" composite keys in evaluate_key/union
        # which silently inflate miss counts rather than failing loudly.
        for col in [
            "name_norm", "sorted_name", "first_token", "last_token",
            "first_last", "boundary",
            "address_norm", "address_sorted_signal", "address_first3_signal",
            "address_digits", "address_boundary",
            "first_token_addr_digits", "first_token_addr_first3",
            "first_token_addr_boundary", "country",
        ]:
            if col in m.columns:
                m[col] = m[col].fillna("")

        # true_df feature columns are already named correctly.
        return m[feature_cols]

    s2_eval = attach_pairs(
        s2_match_pairs, prepare_match_info(s2_true)
    )
    s3_eval = attach_pairs(
        s3_match_pairs, prepare_match_info(s3_true)
    )

    all_eval = pd.concat([s2_eval, s3_eval], ignore_index=True)

    keys = [
        # Existing name-only keys
        "name_norm",
        "sorted_name",
        "first_token",
        "last_token",
        "first_last",
        "boundary",

        # New address-only keys
        "address_norm",
        "address_sorted_signal",
        "address_first3_signal",
        "address_digits",
        "address_boundary",

        # Conservative hybrid keys
        "first_token_addr_digits",
        "first_token_addr_first3",
        "first_token_addr_boundary",
    ]

    print("\n" + "=" * 78)
    print("INDIVIDUAL BLOCKING-KEY RECALL")
    print("=" * 78)

    for key in keys:
        r = evaluate_key(s1_info, all_eval, gt, key)

        print(f"\n[{key}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
            f"  ({r['entities_with_any_true_match_recovered']:,}/"
            f"{r['n_non_singleton_entities']:,})"
        )

    unions = [
        ("name: first_token only",
         ["first_token"]),

        ("name: all name keys",
         ["name_norm", "sorted_name", "first_token",
          "last_token", "first_last", "boundary"]),

        ("address: digits only",
         ["address_digits"]),

        ("address: norm + digits",
         ["address_norm", "address_digits"]),

        ("hybrid: first_token + address_digits",
         ["first_token", "first_token_addr_digits"]),

        ("hybrid: first_token + address_first3",
         ["first_token", "first_token_addr_first3"]),

        ("name + address_digits",
         ["name_norm", "sorted_name", "first_token",
          "last_token", "first_last", "boundary",
          "address_digits"]),

        ("name + address_norm + digits",
         ["name_norm", "sorted_name", "first_token",
          "last_token", "first_last", "boundary",
          "address_norm", "address_digits"]),

        ("name + all address keys",
         ["name_norm", "sorted_name", "first_token",
          "last_token", "first_last", "boundary",
          "address_norm", "address_sorted_signal",
          "address_first3_signal", "address_digits",
          "address_boundary"]),

        ("all name + all address + hybrids",
         keys),
    ]

    print("\n" + "=" * 78)
    print("UNION BLOCKING-KEY RECALL")
    print("=" * 78)

    for label, key_names in unions:
        r = evaluate_union(s1_info, all_eval, key_names)

        print(f"\n[{label}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
            f"  ({r['covered_entities']:,}/"
            f"{r['n_non_singleton_entities']:,})"
        )

    print("\nNOTE:")
    print(
        "These numbers measure whether a TRUE match shares a blocking key "
        "with its S1 entity. They do not yet include generic-block caps or "
        "candidate-pair counts. Address keys are diagnostic at this stage; "
        "before full blocking we must also measure the number of candidates "
        "each selected key would generate."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="deterministically evaluate only this fraction of the validation fold, e.g. 0.10"
    )
    parser.add_argument(
        "--sample-s1", type=int, default=None,
        help="alternative deterministic cap on validation S1 entities"
    )
    args = parser.parse_args()

    if args.sample_frac is not None and args.sample_s1 is not None:
        parser.error("use only one of --sample-frac or --sample-s1")

    run(
        args.data_root,
        args.val_frac,
        args.seed,
        sample_frac=args.sample_frac,
        sample_s1=args.sample_s1,
    )