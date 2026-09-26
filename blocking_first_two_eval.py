"""
Second-stage blocking selector for Amazon ML Challenge 2026.

Why this experiment exists
--------------------------
The first-token cap experiment still lost too much true-match recall.
The production baseline already has a more selective "loose name" idea:
the first TWO normalized name tokens + country. This script measures that
key directly against the real ground truth and measures its candidate volume.

It also tests:
  first_two
  first_two + address_digits
  first_two + address_first3
  first_two + address_norm
  capped first_two with address-digit fallback

The same deterministic 10% validation sample is used. No full candidate
pair table is materialized.
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


def load_ground_truth(path):
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


def sample_validation_ids(entity_ids, val_frac, sample_frac, seed):
    _, val_ids = deterministic_split(
        entity_ids,
        val_frac=val_frac,
        seed=seed,
    )

    threshold = int(sample_frac * (2**32))
    sample_seed = seed + 100003

    sampled = {
        eid
        for eid in val_ids
        if int(
            hashlib.md5(
                f"{sample_seed}-{eid}".encode()
            ).hexdigest()[:8],
            16,
        ) < threshold
    }

    if not sampled:
        raise RuntimeError("Sample produced zero validation entities.")

    return sampled


def make_features(df):
    name = df["business_name"].astype(str).map(normalize_name)
    addr = df["business_address"].astype(str).map(normalize_address)

    first_two = []
    first_token = []
    address_digits = []
    address_first3 = []

    for n in name.tolist():
        toks = [t for t in n.split() if len(t) >= MIN_TOKEN_LEN]
        first_token.append(toks[0] if toks else "")
        first_two.append(" ".join(toks[:2]) if len(toks) >= 2 else "")

    for a in addr.tolist():
        sig = sorted(set(address_signal_tokens(a)))
        nums = re.findall(r"\d+", a)
        address_digits.append("|".join(nums) if nums else "")
        address_first3.append(" ".join(sig[:3]) if sig else "")

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].astype(str).tolist(),
            "country": df["country"].astype(str).tolist(),
            "first_token": first_token,
            "first_two": first_two,
            "address_norm": addr.tolist(),
            "address_digits": address_digits,
            "address_first3": address_first3,
        }
    )

    out["first_two__addr_digits"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_two"], out["address_digits"])
    ]
    out["first_two__addr_first3"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_two"], out["address_first3"])
    ]
    out["first_two__addr_norm"] = [
        f"{n}||{a}" if n and a else ""
        for n, a in zip(out["first_two"], out["address_norm"])
    ]

    return out


def expected_pairs(gt, sample_ids, prefix):
    return [
        (eid, mid)
        for eid in sample_ids
        for mid in gt[eid]
        if mid.startswith(prefix + "-")
    ]


def fetch_true_rows(path, expected, chunk_size):
    needed = {mid for _, mid in expected}
    pieces = []

    for chunk in load_source(path, chunksize=chunk_size):
        hit = chunk[chunk["entity_id"].isin(needed)]
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
        raise RuntimeError(f"Duplicate entity IDs in {path}")

    return out


def true_rows_with_features(expected, source_rows):
    if source_rows.empty:
        return []

    feats = make_features(source_rows)
    lookup = feats.set_index("entity_id")

    rows = []
    for sid, mid in expected:
        if mid not in lookup.index:
            continue
        r = lookup.loc[mid]
        d = {
            "source1_entity_id": sid,
            "matched_entity_id": mid,
            "country": str(r["country"]),
        }
        for col in [
            "first_two",
            "address_norm",
            "address_digits",
            "address_first3",
            "first_two__addr_digits",
            "first_two__addr_first3",
            "first_two__addr_norm",
        ]:
            d[col] = str(r[col])
        rows.append(d)

    return rows


def recall_key(s1_info, true_rows, key):
    s1 = s1_info.set_index("entity_id")
    total = 0
    recovered = 0
    total_by_entity = defaultdict(int)
    recovered_by_entity = defaultdict(int)

    for row in true_rows:
        total += 1
        sid = row["source1_entity_id"]
        total_by_entity[sid] += 1

        sr = s1.loc[sid]
        if (
            str(sr[key])
            and str(row[key])
            and str(sr[key]) == str(row[key])
            and str(sr["country"]) == str(row["country"])
        ):
            recovered += 1
            recovered_by_entity[sid] += 1

    per_entity = [
        recovered_by_entity[sid] / n
        for sid, n in total_by_entity.items()
    ]
    covered = sum(v > 0 for v in recovered_by_entity.values())

    return {
        "mean_entity_recall": (
            sum(per_entity) / len(per_entity)
            if per_entity else 0.0
        ),
        "global_pair_recall": recovered / total if total else 0.0,
        "entity_coverage": (
            covered / len(total_by_entity)
            if total_by_entity else 0.0
        ),
    }


def union_recall(s1_info, true_rows, keys):
    s1 = s1_info.set_index("entity_id")

    total_by_entity = defaultdict(int)
    recovered_by_entity = defaultdict(int)
    total = 0
    recovered = 0

    for row in true_rows:
        total += 1
        sid = row["source1_entity_id"]
        total_by_entity[sid] += 1
        sr = s1.loc[sid]

        hit = False
        for key in keys:
            if (
                str(sr[key])
                and str(row[key])
                and str(sr[key]) == str(row[key])
                and str(sr["country"]) == str(row["country"])
            ):
                hit = True
                break

        if hit:
            recovered += 1
            recovered_by_entity[sid] += 1

    per_entity = [
        recovered_by_entity[sid] / n
        for sid, n in total_by_entity.items()
    ]
    covered = sum(v > 0 for v in recovered_by_entity.values())

    return {
        "mean_entity_recall": sum(per_entity) / len(per_entity)
        if per_entity else 0.0,
        "global_pair_recall": recovered / total if total else 0.0,
        "entity_coverage": covered / len(total_by_entity)
        if total_by_entity else 0.0,
    }


def scan_frequency_maps(path, s1_info, keys, chunk_size):
    """
    Count candidate frequencies only for composite values actually occurring
    in the sampled S1. This avoids retaining millions of irrelevant keys.
    """
    wanted = {}
    for key in keys:
        wanted[key] = {
            value + "||" + country
            for value, country in zip(
                s1_info[key].astype(str),
                s1_info["country"].astype(str),
            )
            if value.strip()
        }

    freq = {key: defaultdict(int) for key in keys}

    t0 = time.time()
    rows = 0
    print(f"\nScanning {os.path.basename(path)}...", flush=True)

    for chunk_no, chunk in enumerate(
        load_source(path, chunksize=chunk_size),
        start=1,
    ):
        rows += len(chunk)
        feats = make_features(chunk)

        for key in keys:
            wanted_key = wanted[key]
            if not wanted_key:
                continue

            vals = feats[key].astype(str)
            countries = feats["country"].astype(str)
            valid = vals.str.strip() != ""

            if not valid.any():
                continue

            composite = vals[valid] + "||" + countries[valid]
            mask = composite.isin(wanted_key)
            if not mask.any():
                continue

            counts = composite[mask].value_counts(sort=False)
            for comp, count in counts.items():
                freq[key][comp] += int(count)

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


def volume_key(s1_info, freq2, freq3, key):
    counts = []

    for value, country in zip(
        s1_info[key].astype(str),
        s1_info["country"].astype(str),
    ):
        if not value:
            counts.append(0)
            continue

        comp = value + "||" + country
        counts.append(
            freq2[key].get(comp, 0)
            + freq3[key].get(comp, 0)
        )

    ser = pd.Series(counts, dtype="int64")
    nonzero = ser[ser > 0]

    return {
        "candidate_pairs_sample": int(ser.sum()),
        "mean_candidates_per_s1": float(ser.mean()),
        "median_nonzero": float(nonzero.median())
        if len(nonzero) else 0.0,
        "p95_nonzero": float(nonzero.quantile(0.95))
        if len(nonzero) else 0.0,
        "max_candidates_per_s1": int(ser.max())
        if len(ser) else 0,
    }


def capped_first_two_volume_and_recall(
    s1_info,
    true_rows,
    freq2,
    freq3,
    cap,
):
    broad = "first_two"
    fallback = "first_two__addr_digits"
    s1 = s1_info.set_index("entity_id")

    candidate_counts = []
    total = 0
    recovered = 0
    total_by_entity = defaultdict(int)
    recovered_by_entity = defaultdict(int)

    for _, sr in s1.iterrows():
        country = str(sr["country"])

        bv = str(sr[broad])
        bc = bv + "||" + country if bv else ""
        bcount = (
            freq2[broad].get(bc, 0)
            + freq3[broad].get(bc, 0)
            if bc else 0
        )

        fv = str(sr[fallback])
        fc = fv + "||" + country if fv else ""
        fcount = (
            freq2[fallback].get(fc, 0)
            + freq3[fallback].get(fc, 0)
            if fc else 0
        )

        if bv and bcount <= cap:
            candidate_counts.append(bcount)
        else:
            candidate_counts.append(fcount)

    for row in true_rows:
        sid = row["source1_entity_id"]
        total += 1
        total_by_entity[sid] += 1
        sr = s1.loc[sid]
        country = str(sr["country"])

        bv = str(sr[broad])
        bc = bv + "||" + country if bv else ""
        bcount = (
            freq2[broad].get(bc, 0)
            + freq3[broad].get(bc, 0)
            if bc else 0
        )

        if bv and bcount <= cap:
            hit = (
                str(row[broad]) == bv
                and str(row["country"]) == country
            )
        else:
            fv = str(sr[fallback])
            hit = (
                bool(fv)
                and str(row[fallback]) == fv
                and str(row["country"]) == country
            )

        if hit:
            recovered += 1
            recovered_by_entity[sid] += 1

    ser = pd.Series(candidate_counts, dtype="int64")
    nonzero = ser[ser > 0]
    covered = sum(v > 0 for v in recovered_by_entity.values())
    per_entity = [
        recovered_by_entity[sid] / n
        for sid, n in total_by_entity.items()
    ]

    return {
        "cap": cap,
        "mean_entity_recall": sum(per_entity) / len(per_entity)
        if per_entity else 0.0,
        "global_pair_recall": recovered / total if total else 0.0,
        "entity_coverage": covered / len(total_by_entity)
        if total_by_entity else 0.0,
        "candidate_pairs_sample": int(ser.sum()),
        "mean_candidates_per_s1": float(ser.mean()),
        "median_nonzero": float(nonzero.median())
        if len(nonzero) else 0.0,
        "p95_nonzero": float(nonzero.quantile(0.95))
        if len(nonzero) else 0.0,
        "max_candidates_per_s1": int(ser.max())
        if len(ser) else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--caps", type=int, nargs="+",
                        default=[100, 250, 500, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--output-prefix", default="blocking_first_two")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")

    print("Loading S1 + ground truth...", flush=True)
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    gt = load_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )

    sampled = sample_validation_ids(
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

    print("Preparing S1 first-two + address signatures...", flush=True)
    s1_info = make_features(s1_sample)

    s2_expected = expected_pairs(gt, sampled, "S2")
    s3_expected = expected_pairs(gt, sampled, "S3")

    print(f"true S2 validation matches: {len(s2_expected):,}")
    print(f"true S3 validation matches: {len(s3_expected):,}")

    s2_true = fetch_true_rows(
        os.path.join(train_dir, "train_source2.tsv"),
        s2_expected,
        args.chunk_size,
    )
    s3_true = fetch_true_rows(
        os.path.join(train_dir, "train_source3.tsv"),
        s3_expected,
        args.chunk_size,
    )

    true_rows = (
        true_rows_with_features(s2_expected, s2_true)
        + true_rows_with_features(s3_expected, s3_true)
    )

    print(
        f"captured true validation pairs: {len(true_rows):,}",
        flush=True,
    )

    keys = [
        "first_two",
        "first_two__addr_digits",
        "first_two__addr_first3",
        "first_two__addr_norm",
    ]

    freq2 = scan_frequency_maps(
        os.path.join(train_dir, "train_source2.tsv"),
        s1_info,
        keys + ["first_token"],
        args.chunk_size,
    )
    freq3 = scan_frequency_maps(
        os.path.join(train_dir, "train_source3.tsv"),
        s1_info,
        keys + ["first_token"],
        args.chunk_size,
    )

    print("\n" + "=" * 110)
    print("FIRST-TWO-TOKEN / COMPOSITE TRUE-MATCH RECALL")
    print("=" * 110)

    for key in keys:
        r = recall_key(s1_info, true_rows, key)
        print(f"\n[{key}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(f"  entity coverage    : {r['entity_coverage']:.4f}")

    unions = [
        (
            "first_two + address_norm + address_digits",
            [
                "first_two",
                "first_two__addr_norm",
                "first_two__addr_digits",
            ],
        ),
        (
            "first_two + first3 + digits",
            [
                "first_two",
                "first_two__addr_first3",
                "first_two__addr_digits",
            ],
        ),
        (
            "all first_two composites",
            keys,
        ),
    ]

    print("\n" + "=" * 110)
    print("FIRST-TWO-TOKEN UNION RECALL")
    print("=" * 110)

    for label, ukeys in unions:
        r = union_recall(s1_info, true_rows, ukeys)
        print(f"\n[{label}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(f"  entity coverage    : {r['entity_coverage']:.4f}")

    print("\n" + "=" * 120)
    print("FIRST-TWO-TOKEN RAW CANDIDATE VOLUME")
    print("=" * 120)

    volume_records = []
    for key in keys:
        v = volume_key(s1_info, freq2, freq3, key)
        volume_records.append({"key": key, **v})
        print(f"\n[{key}]")
        print(f"  candidate pairs    : {v['candidate_pairs_sample']:,}")
        print(f"  mean candidates/S1 : {v['mean_candidates_per_s1']:.1f}")
        print(f"  median nonzero     : {v['median_nonzero']:.1f}")
        print(f"  p95 nonzero        : {v['p95_nonzero']:.1f}")
        print(f"  max/S1             : {v['max_candidates_per_s1']:,}")

    print("\n" + "=" * 110)
    print("CAPPED FIRST-TWO-TOKEN + ADDRESS-DIGIT FALLBACK")
    print("=" * 110)

    capped_records = []
    for cap in args.caps:
        r = capped_first_two_volume_and_recall(
            s1_info,
            true_rows,
            freq2,
            freq3,
            cap,
        )
        capped_records.append(r)

        print(f"\nCAP = {cap:,}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(f"  entity coverage    : {r['entity_coverage']:.4f}")
        print(f"  candidate pairs    : {r['candidate_pairs_sample']:,}")
        print(f"  mean candidates/S1 : {r['mean_candidates_per_s1']:.1f}")
        print(f"  p95 candidates/S1  : {r['p95_nonzero']:.1f}")
        print(f"  max candidates/S1  : {r['max_candidates_per_s1']:,}")

    pd.DataFrame(volume_records).to_csv(
        args.output_prefix + "_volume.csv",
        index=False,
    )
    pd.DataFrame(capped_records).to_csv(
        args.output_prefix + "_capped.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"  {args.output_prefix}_volume.csv")
    print(f"  {args.output_prefix}_capped.csv")


if __name__ == "__main__":
    main()
