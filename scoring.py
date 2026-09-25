"""
Local implementation of the official metric: macro-averaged F_0.5,
computed per Source-1 entity, singletons included.

This mirrors the spec exactly:
  - Fbeta = (1 + beta^2) * P * R / (beta^2 * P + R), beta = 0.5
  - A true singleton (empty ground truth) scores 1.0 if predicted
    empty, 0.0 if any match is predicted.
  - Macro average = mean over ALL ground-truth Source-1 entities
    (every entity must appear in `ground_truth`; entities missing from
    `predictions` are treated as an empty prediction, matching how a
    missing row would presumably be scored -- but note the real
    validator REJECTS submissions missing a Source-1 row entirely, so
    never rely on this fallback for your actual submission).
"""

BETA = 0.5
BETA_SQ = BETA * BETA


def f_beta(precision: float, recall: float, beta_sq: float = BETA_SQ) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def score_entity(pred_set: frozenset, true_set: frozenset) -> float:
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return f_beta(precision, recall)


def f0_5_macro(predictions: dict, ground_truth: dict) -> dict:
    """
    predictions, ground_truth: {source1_entity_id: frozenset(matched_ids)}

    Returns a dict with the macro score plus a few diagnostics useful
    for error analysis (mean precision/recall, singleton accuracy).
    """
    scores = []
    precisions = []
    recalls = []
    singleton_correct = 0
    n_singletons = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, frozenset())
        scores.append(score_entity(pred_set, true_set))

        if not true_set:
            n_singletons += 1
            if not pred_set:
                singleton_correct += 1
        else:
            tp = len(pred_set & true_set)
            precisions.append(tp / len(pred_set) if pred_set else 0.0)
            recalls.append(tp / len(true_set))

    n = len(scores)
    return {
        "f0_5_macro": sum(scores) / n if n else 0.0,
        "n_entities": n,
        "n_singletons": n_singletons,
        "singleton_accuracy": singleton_correct / n_singletons if n_singletons else float("nan"),
        "mean_precision_nonsingleton": sum(precisions) / len(precisions) if precisions else float("nan"),
        "mean_recall_nonsingleton": sum(recalls) / len(recalls) if recalls else float("nan"),
    }


def deterministic_split(entity_ids, val_frac: float = 0.2, seed: int = 42):
    """
    Deterministic, hash-based train/val split over Source-1 entity ids.
    Using a hash (not random.shuffle) means re-running this on a
    different subset of ids, or in a different process, still puts the
    same entity in the same fold -- important once you add anything
    that fits on 'train' and validates on 'val' across pipeline stages.
    """
    import hashlib

    val_ids = set()
    train_ids = set()
    threshold = int(val_frac * (2 ** 32))
    for eid in entity_ids:
        h = int(hashlib.md5(f"{seed}-{eid}".encode()).hexdigest()[:8], 16)
        if h < threshold:
            val_ids.add(eid)
        else:
            train_ids.add(eid)
    return train_ids, val_ids
