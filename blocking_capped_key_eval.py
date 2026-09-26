"""
Evaluate capped broad blocking keys for Amazon ML Challenge 2026.

Motivation
----------
The 10% experiment showed:
  * first_token gives high recall but an enormous candidate volume
  * first_token__addr_digits gives much smaller volume but much lower recall

This script evaluates a safer hybrid:

    if first_token block size <= CAP:
        use the full first_token block
    else:
        fall back to first_token + address_digits

Because the fallback is a subset of an admitted first_token block, there is
no duplicate-count ambiguity.

The same idea is evaluated for last_token.

No candidate-pair table is materialized.
"""

import argparse
import hashlib
import os
import re
import time
from collections import defaultdict

import pandas as pd

from normalize import normalize_name, normalize_address, address_signal_tokens
from scoring import deterministic_split


MIN_TOKEN_LEN = 3
DEFAULT_CAPS = [100, 250, 500, 1000, 2000, 5000, 10000]


def load_source(path, chunksize=None):
    kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8-sig",
    }
    if chunksize:
        kwargs["chunksize"] = chunksize
    return pd.read_csv(path, **kwargs)


def load_gt(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return {
        sid: [x for x in matched.split(",") if x]
        for sid, matched in zip(
            df["source1_entity_id"],
            df["matched_entity_ids"],
        )
    }


def make_features(df):
    name = df["business_name"].astype(str).map(normalize_name)
    addr = df["business_address"].astype(str).map(normalize_address)

    first_token = []
    last_token = []

    for n in name.tolist():
        toks = [t for t in n.split() if len(t) >= MIN_TOKEN_LEN]
        first_token.append(toks[0] if toks else "")
        last_token.append(toks[-1] if toks else "")

    address_digits = []
    for a in addr.tolist():
        nums = re.findall(r"\d+", a)
        address_digits.append("|".join(nums) if nums else "")

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].astype(str).tolist(),
            "country": df["country"].astype(str).tolist(),
            "first_token": first_token,
            "last_token": last_token,
            "address_digits": address_digits,
        }
    )

    out["first_token__addr_digits"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(
            out["first_token"],
            out["address_digits"],
        )
    ]

    out["last_token__addr_digits"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(
            out["last_token"],
            out["address_digits"],
        )
    ]

    return out


def sample_ids(entity_ids, val_frac, sample_frac, seed):
    _, val = deterministic_split(
        entity_ids,
        val_frac=val_frac,
        seed=seed,
    )

    threshold = int(sample_frac * (2**32))
    ss = seed + 100003

    sampled = {
        eid
        for eid in val
        if int(
            hashlib.md5(
                f"{ss}-{eid}".encode()
            ).hexdigest()[:8],
            16,
        ) < threshold
    }

    if not sampled:
        raise RuntimeError("No sampled validation entities.")

    return sampled


def build_expected_pairs(gt, sample_ids, prefix):
    return [
        (eid, mid)
        for eid in sample_ids
        for mid in gt[eid]
        if mid.startswith(prefix + "-")
    ]


def fetch_true_rows(path, needed_ids, chunksize):
    pieces = []

    for chunk in load_source(path, chunksize=chunksize):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]

        if not hit.empty:
            pieces.append(
                hit[
                    [
                        "entity_id",
                        "business_name",
                        "business_address",
                        "country",
                    ]
                ].copy()
            )

    if not pieces:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    out = pd.concat(pieces, ignore_index=True)

    if out["entity_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate entity IDs found in {path}"
        )

    return out


def make_true_pairs(
    expected_pairs,
    true_source,
):
    if true_source.empty:
        return pd.DataFrame()

    feats = make_features(
        true_source.rename(columns={"entity_id": "entity_id"})
    )

    lookup = feats.set_index("entity_id")

    rows = []
    for sid, mid in expected_pairs:
        if mid not in lookup.index:
            continue

        r = lookup.loc[mid]
        rows.append(
            {
                "source1_entity_id": sid,
                "matched_entity_id": mid,
                "country": str(r["country"]),
                "first_token": str(r["first_token"]),
                "last_token": str(r["last_token"]),
                "address_digits": str(r["address_digits"]),
                "first_token__addr_digits": str(
                    r["first_token__addr_digits"]
                ),
                "last_token__addr_digits": str(
                    r["last_token__addr_digits"]
                ),
            }
        )

    return pd.DataFrame(rows)


def scan_counts(path, sample_info, chunk_size):
    """
    Count full-source block sizes only for key values used by sampled S1.

    Returns per-country composite frequency maps for:
        first_token
        last_token
        first_token__addr_digits
        last_token__addr_digits
    """
    keys = [
        "first_token",
        "last_token",
        "first_token__addr_digits",
        "last_token__addr_digits",
    ]

    wanted = {}
    for key in keys:
        wanted[key] = {
            value + "||" + country
            for value, country in zip(
                sample_info[key].astype(str),
                sample_info["country"].astype(str),
            )
            if value
        }

    freq = {key: defaultdict(int) for key in keys}
    t0 = time.time()
    rows = 0

    print(
        f"\nScanning {os.path.basename(path)} for block sizes...",
        flush=True,
    )

    for chunk_no, chunk in enumerate(
        load_source(path, chunksize=chunk_size),
        start=1,
    ):
        rows += len(chunk)
        f = make_features(chunk)

        for key in keys:
            wanted_key = wanted[key]
            if not wanted_key:
                continue

            values = f[key].astype(str)
            countries = f["country"].astype(str)
            valid = values.str.strip() != ""

            if not valid.any():
                continue

            composite = values[valid] + "||" + countries[valid]
            mask = composite.isin(wanted_key)

            if not mask.any():
                continue

            vc = composite[mask].value_counts(sort=False)
            target = freq[key]

            for comp, count in vc.items():
                target[comp] += int(count)

        if chunk_no % 4 == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(
                f"  processed {rows:,} rows "
                f"({rows / elapsed:,.0f} rows/s)",
                flush=True,
            )

    print(
        f"  finished {rows:,} rows in {time.time() - t0:.1f}s",
        flush=True,
    )

    return freq



def tuple_field(row, field_name):
    """Get a field from a pandas namedtuple row by column name."""
    try:
        return getattr(row, field_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"True-pair row is missing expected field '{field_name}'"
        ) from exc



def evaluate_hybrid(
    s1_info,
    true_rows,
    freq2,
    freq3,
    broad_key,
    fallback_key,
    cap,
):
    """
    For every sampled S1 entity:
      if broad block size <= cap, candidate count = broad block size
      else candidate count = fallback block size

    True-match recall uses the exact same rule.
    """
    # S1 -> broad/fallback composite.
    s1_lookup = s1_info.set_index("entity_id")

    total_pairs = 0
    recovered_pairs = 0
    total_entities = set()
    recovered_entities = set()

    candidate_counts = []

    for eid, s1row in s1_lookup.iterrows():
        country = str(s1row["country"])

        broad_value = str(s1row[broad_key])
        broad_comp = (
            broad_value + "||" + country
            if broad_value
            else ""
        )

        bcount = (
            freq2[broad_key].get(broad_comp, 0)
            + freq3[broad_key].get(broad_comp, 0)
            if broad_comp
            else 0
        )

        fallback_value = str(s1row[fallback_key])
        fallback_comp = (
            fallback_value + "||" + country
            if fallback_value
            else ""
        )

        fcount = (
            freq2[fallback_key].get(fallback_comp, 0)
            + freq3[fallback_key].get(fallback_comp, 0)
            if fallback_comp
            else 0
        )

        if broad_value and bcount <= cap:
            candidate_counts.append(bcount)
        else:
            candidate_counts.append(fcount)

    # True-pair recall. The fallback is used only when the broad block is
    # above the cap.
    for row in true_rows.itertuples(index=False):
        eid = row.source1_entity_id
        total_entities.add(eid)

        s1row = s1_lookup.loc[eid]
        country = str(s1row["country"])

        bval = str(s1row[broad_key])
        bcomp = bval + "||" + country if bval else ""

        b2 = (
            freq2[broad_key].get(bcomp, 0)
            + freq3[broad_key].get(bcomp, 0)
            if bcomp
            else 0
        )

        if bval and b2 <= cap:
            hit = (
                str(tuple_field(row, broad_key)) == bval
                and str(row.country) == country
            )
        else:
            fval = str(s1row[fallback_key])
            hit = (
                bool(fval)
                and str(tuple_field(row, fallback_key)) == fval
                and str(row.country) == country
            )

        if hit:
            recovered_pairs += 1
            recovered_entities.add(eid)

        total_pairs += 1

    ser = pd.Series(candidate_counts, dtype="int64")
    nonzero = ser[ser > 0]

    return {
        "cap": cap,
        "mean_entity_recall": (
            recovered_pairs / total_pairs
            if total_pairs
            else 0.0
        ),
        "global_pair_recall": (
            recovered_pairs / total_pairs
            if total_pairs
            else 0.0
        ),
        "entity_coverage": (
            len(recovered_entities) / len(total_entities)
            if total_entities
            else 0.0
        ),
        "candidate_pairs_sample": int(ser.sum()),
        "mean_candidates_per_s1": float(ser.mean()),
        "median_nonzero": (
            float(nonzero.median())
            if len(nonzero)
            else 0.0
        ),
        "p90_nonzero": (
            float(nonzero.quantile(0.90))
            if len(nonzero)
            else 0.0
        ),
        "p95_nonzero": (
            float(nonzero.quantile(0.95))
            if len(nonzero)
            else 0.0
        ),
        "max_candidates_per_s1": int(ser.max()),
    }


def evaluate_fallback_only(
    true_rows,
    s1_info,
    freq2,
    freq3,
    fallback_key,
):
    """
    Recall of the fallback composite key alone.
    """
    s1_lookup = s1_info.set_index("entity_id")
    total = 0
    recovered = 0

    for row in true_rows.itertuples(index=False):
        total += 1
        sr = s1_lookup.loc[row.source1_entity_id]
        country = str(sr["country"])
        val = str(sr[fallback_key])

        if (
            val
            and str(tuple_field(row, fallback_key)) == val
            and str(row.country) == country
        ):
            recovered += 1

    return recovered / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=DEFAULT_CAPS,
    )
    parser.add_argument(
        "--output",
        default="blocking_capped_profile.csv",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")

    s1 = load_source(
        os.path.join(train_dir, "train_source1.tsv")
    )
    gt = load_gt(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )

    sampled = sample_ids(
        s1["entity_id"],
        args.val_frac,
        args.sample_frac,
        args.seed,
    )

    s1_sample = s1[
        s1["entity_id"].isin(sampled)
    ].reset_index(drop=True)

    print(
        f"sampled validation fold: {len(s1_sample):,} S1 entities",
        flush=True,
    )

    s1_info = make_features(s1_sample)

    s2_expected = build_expected_pairs(gt, sampled, "S2")
    s3_expected = build_expected_pairs(gt, sampled, "S3")

    print(
        f"true S2 validation matches: {len(s2_expected):,}",
        flush=True,
    )
    print(
        f"true S3 validation matches: {len(s3_expected):,}",
        flush=True,
    )

    s2_ids = {mid for _, mid in s2_expected}
    s3_ids = {mid for _, mid in s3_expected}

    s2_true = fetch_true_rows(
        os.path.join(train_dir, "train_source2.tsv"),
        s2_ids,
        args.chunk_size,
    )
    s3_true = fetch_true_rows(
        os.path.join(train_dir, "train_source3.tsv"),
        s3_ids,
        args.chunk_size,
    )

    s2_rows = make_true_pairs(s2_expected, s2_true)
    s3_rows = make_true_pairs(s3_expected, s3_true)

    true_rows = pd.concat(
        [s2_rows, s3_rows],
        ignore_index=True,
    )

    print(
        f"captured true validation pairs: {len(true_rows):,}",
        flush=True,
    )

    freq2 = scan_counts(
        os.path.join(train_dir, "train_source2.tsv"),
        s1_info,
        args.chunk_size,
    )
    freq3 = scan_counts(
        os.path.join(train_dir, "train_source3.tsv"),
        s1_info,
        args.chunk_size,
    )

    # Fallback-only baselines.
    ft_fallback = evaluate_fallback_only(
        true_rows,
        s1_info,
        freq2,
        freq3,
        "first_token__addr_digits",
    )
    lt_fallback = evaluate_fallback_only(
        true_rows,
        s1_info,
        freq2,
        freq3,
        "last_token__addr_digits",
    )

    print("\n" + "=" * 110)
    print("CAPPED BLOCKING: FIRST TOKEN + ADDRESS-DIGIT FALLBACK")
    print("=" * 110)

    records = []

    for cap in args.caps:
        r = evaluate_hybrid(
            s1_info,
            true_rows,
            freq2,
            freq3,
            "first_token",
            "first_token__addr_digits",
            cap,
        )
        records.append(
            {"strategy": "first_token_cap+digit_fallback", **r}
        )

        print(f"\nCAP = {cap:,}")
        print(
            f"  global pair recall : {r['global_pair_recall']:.4f}"
        )
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
        )
        print(
            f"  candidate pairs    : {r['candidate_pairs_sample']:,}"
        )
        print(
            f"  mean candidates/S1 : {r['mean_candidates_per_s1']:.1f}"
        )
        print(
            f"  p95 candidates/S1  : {r['p95_nonzero']:.1f}"
        )
        print(
            f"  max candidates/S1  : {r['max_candidates_per_s1']:,}"
        )

    print("\n" + "=" * 110)
    print("CAPPED BLOCKING: LAST TOKEN + ADDRESS-DIGIT FALLBACK")
    print("=" * 110)

    for cap in args.caps:
        r = evaluate_hybrid(
            s1_info,
            true_rows,
            freq2,
            freq3,
            "last_token",
            "last_token__addr_digits",
            cap,
        )
        records.append(
            {"strategy": "last_token_cap+digit_fallback", **r}
        )

        print(f"\nCAP = {cap:,}")
        print(
            f"  global pair recall : {r['global_pair_recall']:.4f}"
        )
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
        )
        print(
            f"  candidate pairs    : {r['candidate_pairs_sample']:,}"
        )
        print(
            f"  mean candidates/S1 : {r['mean_candidates_per_s1']:.1f}"
        )
        print(
            f"  p95 candidates/S1  : {r['p95_nonzero']:.1f}"
        )
        print(
            f"  max candidates/S1  : {r['max_candidates_per_s1']:,}"
        )

    pd.DataFrame(records).to_csv(args.output, index=False)

    print("\nFallback-only references:")
    print(
        f"  first_token__addr_digits recall : {ft_fallback:.4f}"
    )
    print(
        f"  last_token__addr_digits recall  : {lt_fallback:.4f}"
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
