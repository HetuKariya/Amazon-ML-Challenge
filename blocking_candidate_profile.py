"""
Candidate-volume profiler for Amazon ML Challenge 2026 blocking.

Purpose
-------
Measure how many S2/S3 candidate pairs each blocking key would generate
for the SAME deterministic validation sample used by blocking_recall_eval.py.

This does NOT materialize candidate pairs. It:
  1) samples the validation S1 entities,
  2) scans full S2/S3 in chunks,
  3) builds frequency maps for each blocking key,
  4) looks up those frequencies for the sampled S1 keys, and
  5) reports candidate-pair volume statistics.

Use this before full-scale blocking so we can choose a high-recall,
manageable set of blocking keys.
"""

import argparse
import os
import time
from collections import defaultdict

import pandas as pd

from blocking_recall_eval import (
    load_source,
    load_ground_truth,
    make_name_features,
    make_address_features,
)
from scoring import deterministic_split


KEYS = [
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
]

DISPLAY_NAMES = {
    "name_norm": "name_norm",
    "sorted_name": "sorted_name",
    "first_token": "first_token",
    "last_token": "last_token",
    "first_last": "first_last",
    "boundary": "boundary",
    "address_norm": "address_norm",
    "address_sorted_signal": "address_sorted_signal",
    "address_first3_signal": "address_first3_signal",
    "address_digits": "address_digits",
    "address_boundary": "address_boundary",
}


def prepare_s1(s1_val: pd.DataFrame) -> pd.DataFrame:
    name_feats = make_name_features(s1_val["business_name"])
    addr_feats = make_address_features(s1_val["business_address"])

    out = s1_val[["entity_id", "country"]].copy()
    for k, v in name_feats.items():
        out[k] = v
    for k, v in addr_feats.items():
        out[k] = v

    return out


def build_frequency_maps(
    source_path: str,
    keys,
    chunksize: int,
) -> dict:
    """
    Returns {key: {country||key_value: frequency}} for one source.
    Empty feature values are excluded.
    """
    freq = {k: defaultdict(int) for k in keys}

    print(f"\nScanning {os.path.basename(source_path)}...", flush=True)
    t0 = time.time()
    rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            chunksize=chunksize,
        ),
        start=1,
    ):
        rows += len(chunk)

        name_feats = make_name_features(chunk["business_name"])
        addr_feats = make_address_features(chunk["business_address"])

        # Work in a temporary frame so all keys are computed once per row.
        feat = pd.DataFrame(
            {
                **name_feats,
                **addr_feats,
                "country": chunk["country"].astype(str).tolist(),
            }
        )

        for key in keys:
            vals = feat[key].astype(str)
            valid = vals.str.strip() != ""
            if not valid.any():
                continue

            comps = vals[valid] + "||" + feat.loc[valid, "country"].astype(str)
            counts = comps.value_counts(sort=False)

            target = freq[key]
            for comp, count in counts.items():
                target[comp] += int(count)

        if chunk_no % 4 == 0:
            elapsed = time.time() - t0
            print(
                f"  processed {rows:,} rows "
                f"({rows / max(elapsed, 1e-9):,.0f} rows/s)",
                flush=True,
            )

    print(
        f"  finished {rows:,} rows in {time.time() - t0:.1f}s",
        flush=True,
    )

    return freq


def profile_source_key_usage(
    s1_info: pd.DataFrame,
    freq2: dict,
    freq3: dict,
    keys,
    block_cap: int,
) -> pd.DataFrame:
    records = []

    for key in keys:
        s2map = freq2[key]
        s3map = freq3[key]

        vals = s1_info[key].astype(str)
        countries = s1_info["country"].astype(str)

        counts = []
        for value, country in zip(vals, countries):
            if not value.strip():
                counts.append(0)
                continue
            comp = value + "||" + country
            counts.append(s2map.get(comp, 0) + s3map.get(comp, 0))

        ser = pd.Series(counts, dtype="int64")
        nonzero = ser[ser > 0]

        records.append(
            {
                "key": DISPLAY_NAMES[key],
                "sample_s1": len(ser),
                "s1_with_candidates": int((ser > 0).sum()),
                "coverage": float((ser > 0).mean()) if len(ser) else 0.0,
                "total_candidate_pairs": int(ser.sum()),
                "mean_candidates_per_s1": float(ser.mean()) if len(ser) else 0.0,
                "median_candidates_nonzero": (
                    float(nonzero.median()) if len(nonzero) else 0.0
                ),
                "p90_candidates_nonzero": (
                    float(nonzero.quantile(0.90)) if len(nonzero) else 0.0
                ),
                "p95_candidates_nonzero": (
                    float(nonzero.quantile(0.95)) if len(nonzero) else 0.0
                ),
                "max_candidates_per_s1": int(ser.max()) if len(ser) else 0,
                "s1_over_cap": int((ser > block_cap).sum()),
                "max_s2_block": max(s2map.values()) if s2map else 0,
                "max_s3_block": max(s3map.values()) if s3map else 0,
            }
        )

    return pd.DataFrame(records).sort_values(
        by=["total_candidate_pairs", "p95_candidates_nonzero"],
        ascending=[True, True],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--block-cap",
        type=int,
        default=20_000,
        help="diagnostic only: counts S1 entities above this candidate count",
    )
    parser.add_argument(
        "--output",
        default="blocking_candidate_profile.csv",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")

    print("Loading S1 + ground truth...", flush=True)
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    _, val_ids = deterministic_split(
        s1["entity_id"],
        val_frac=args.val_frac,
        seed=args.seed,
    )

    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1]")

    import hashlib

    sample_seed = args.seed + 100003
    threshold = int(args.sample_frac * (2**32))
    val_ids = {
        eid
        for eid in val_ids
        if int(
            hashlib.md5(f"{sample_seed}-{eid}".encode()).hexdigest()[:8],
            16,
        )
        < threshold
    }

    if not val_ids:
        raise RuntimeError("Sample produced zero validation entities.")

    print(
        f"sampled validation fold: {len(val_ids):,} entities "
        f"({args.sample_frac:.1%} of validation fold)",
        flush=True,
    )

    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    print("Preparing S1 name + address signatures...", flush=True)
    s1_info = prepare_s1(s1_val)

    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    freq2 = build_frequency_maps(s2_path, KEYS, args.chunk_size)
    freq3 = build_frequency_maps(s3_path, KEYS, args.chunk_size)

    report = profile_source_key_usage(
        s1_info,
        freq2,
        freq3,
        KEYS,
        args.block_cap,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 40)

    print("\n" + "=" * 120)
    print("CANDIDATE-VOLUME PROFILE (SAMPLED VALIDATION S1)")
    print("=" * 120)
    print(
        report.to_string(
            index=False,
            formatters={
                "coverage": "{:.4f}".format,
                "mean_candidates_per_s1": "{:.1f}".format,
                "median_candidates_nonzero": "{:.1f}".format,
                "p90_candidates_nonzero": "{:.1f}".format,
                "p95_candidates_nonzero": "{:.1f}".format,
            },
        )
    )

    report.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}", flush=True)
    print(
        "\nInterpretation: this is the raw per-key candidate volume before "
        "pair deduplication, classifier filtering, or any production block cap.",
        flush=True,
    )


if __name__ == "__main__":
    main()
