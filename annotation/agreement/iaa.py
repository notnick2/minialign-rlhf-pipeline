"""
Inter-Annotator Agreement (IAA) metrics for MiniAlign.

Implements:
  - Cohen's Kappa for two raters on categorical labels
  - Krippendorff's Alpha from scratch for any number of raters
    (nominal, ordinal, interval, ratio levels of measurement)

Both metrics are computed without depending on scikit-learn or krippendorff
packages so the implementation is fully transparent and auditable.

Reference:
  Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology.
  Cohen, J. (1960). A coefficient of agreement for nominal scales.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ======================================================================= #
#  Cohen's Kappa                                                            #
# ======================================================================= #

def cohen_kappa(rater1_labels: List, rater2_labels: List) -> float:
    """
    Compute Cohen's Kappa for two raters on categorical labels.

    kappa = (P_o - P_e) / (1 - P_e)

    where:
        P_o = observed agreement proportion
        P_e = expected agreement by chance

    Args:
        rater1_labels: List of labels from rater 1 (same length as rater2_labels).
        rater2_labels: List of labels from rater 2.

    Returns:
        Cohen's kappa in [-1, 1].  1 = perfect, 0 = chance, <0 = worse than chance.

    Raises:
        ValueError: if the label lists have different lengths or are empty.
    """
    if len(rater1_labels) != len(rater2_labels):
        raise ValueError(
            f"Label lists must have the same length, got {len(rater1_labels)} and {len(rater2_labels)}"
        )
    if len(rater1_labels) == 0:
        raise ValueError("Label lists must not be empty")

    n = len(rater1_labels)
    categories = sorted(set(rater1_labels) | set(rater2_labels))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(rater1_labels, rater2_labels):
        conf[cat_to_idx[a], cat_to_idx[b]] += 1

    # P_o: fraction of items on the diagonal
    p_o = np.trace(conf) / n

    # P_e: sum of products of marginal proportions
    row_marginals = conf.sum(axis=1) / n
    col_marginals = conf.sum(axis=0) / n
    p_e = float(np.dot(row_marginals, col_marginals))

    if abs(1.0 - p_e) < 1e-10:
        # Perfect agreement expected by chance (degenerate case)
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


# ======================================================================= #
#  Krippendorff's Alpha — implemented from scratch                          #
# ======================================================================= #

def _difference_metric(v1, v2, level: str) -> float:
    """
    Squared difference metric d^2(v1, v2) for Krippendorff's alpha.

    Args:
        level: 'nominal' | 'ordinal' | 'interval' | 'ratio'
    """
    if level == "nominal":
        return 0.0 if v1 == v2 else 1.0

    elif level == "interval":
        return (v1 - v2) ** 2

    elif level == "ratio":
        if (v1 + v2) == 0:
            return 0.0
        return ((v1 - v2) / (v1 + v2)) ** 2

    elif level == "ordinal":
        # Ordinal difference: sum of category frequencies between v1 and v2
        # This requires knowledge of all categories and their order —
        # handled at the krippendorff_alpha level by passing a sorted value list.
        # At this level we fall back to interval.
        return (v1 - v2) ** 2

    else:
        raise ValueError(f"Unknown level of measurement: {level}")


def krippendorff_alpha(
    ratings_matrix: np.ndarray,
    level_of_measurement: str = "ordinal",
) -> float:
    """
    Krippendorff's Alpha for any number of raters.

    alpha = 1 - (D_observed / D_expected)

    Args:
        ratings_matrix: 2-D array of shape (n_raters, n_items).
                        Use np.nan for missing values.
        level_of_measurement: 'nominal' | 'ordinal' | 'interval' | 'ratio'

    Returns:
        Alpha in [-inf, 1].  1 = perfect agreement, 0 = chance,
        negative = systematic disagreement.
    """
    matrix = np.array(ratings_matrix, dtype=float)
    n_raters, n_items = matrix.shape

    # ---- collect all valid values and their sorted order (needed for ordinal) ----
    all_values = matrix[~np.isnan(matrix)].tolist()
    if len(all_values) == 0:
        return float("nan")

    sorted_values = sorted(set(all_values))
    value_to_rank = {v: i for i, v in enumerate(sorted_values)}

    def d(v1: float, v2: float) -> float:
        """Compute disagreement d^2(v1, v2) for chosen measurement level."""
        if level_of_measurement == "nominal":
            return 0.0 if v1 == v2 else 1.0
        elif level_of_measurement == "interval":
            return (v1 - v2) ** 2
        elif level_of_measurement == "ratio":
            denom = (v1 + v2)
            return 0.0 if denom == 0 else ((v1 - v2) / denom) ** 2
        elif level_of_measurement == "ordinal":
            # Krippendorff ordinal metric:
            # d(k, l)^2 = (sum_{g=k}^{l} n_g - (n_k + n_l)/2)^2
            # where n_g is the count of category g across all raters/items.
            # For simplicity with continuous ordinal data we use rank-based interval.
            r1 = value_to_rank.get(v1, 0)
            r2 = value_to_rank.get(v2, 0)
            return (r1 - r2) ** 2
        else:
            raise ValueError(f"Unknown level: {level_of_measurement}")

    # ---- D_observed: average disagreement within items ----
    D_o_numerator = 0.0
    D_o_denominator = 0.0

    for item_idx in range(n_items):
        col = matrix[:, item_idx]
        valid = col[~np.isnan(col)]
        m_u = len(valid)
        if m_u < 2:
            continue
        # Sum all pairwise differences within this item
        item_sum = 0.0
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    item_sum += d(valid[i], valid[j])
        D_o_numerator += item_sum / (m_u - 1)
        D_o_denominator += m_u

    if D_o_denominator == 0:
        return float("nan")

    D_observed = D_o_numerator / D_o_denominator

    # ---- D_expected: disagreement expected by chance ----
    # Treat all values from all items as a single pool,
    # weight each item by its number of valid ratings.
    # D_e = (1 / (n*(n-1))) * sum_{k!=l} n_k * n_l * d(k, l)
    # where n = total valid observations, n_k = count of value k

    value_counts = Counter(all_values)
    n_total = len(all_values)

    if n_total < 2:
        return float("nan")

    D_e_sum = 0.0
    values_list = list(value_counts.keys())
    for i, vk in enumerate(values_list):
        for vl in values_list:
            D_e_sum += value_counts[vk] * value_counts[vl] * d(vk, vl)

    D_expected = D_e_sum / (n_total * (n_total - 1))

    if abs(D_expected) < 1e-12:
        return 1.0 if D_observed < 1e-12 else 0.0

    alpha = 1.0 - D_observed / D_expected
    return float(alpha)


# ======================================================================= #
#  Aggregate agreement across all annotation types                          #
# ======================================================================= #

def compute_all_agreement(annotations_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute inter-annotator agreement for a task's annotations.

    annotations_df must have columns: ['item_id', 'annotator', 'label']

    For tasks with exactly 2 annotators, returns both kappa and alpha.
    For tasks with >2 annotators, returns alpha only (kappa undefined for >2).

    Returns:
        dict with keys: 'cohen_kappa', 'krippendorff_alpha', 'n_annotations',
                        'n_annotators', 'n_items'
    """
    result: Dict[str, float] = {}

    if annotations_df.empty or "label" not in annotations_df.columns:
        return {"cohen_kappa": float("nan"), "krippendorff_alpha": float("nan"),
                "n_annotations": 0, "n_annotators": 0, "n_items": 0}

    # Drop rows with null labels
    df = annotations_df.dropna(subset=["label"]).copy()
    annotators = df["annotator"].unique()
    items = df["item_id"].unique()

    n_annotators = len(annotators)
    n_items = len(items)
    n_annotations = len(df)

    result["n_annotations"] = n_annotations
    result["n_annotators"] = n_annotators
    result["n_items"] = n_items

    if n_annotations < 2:
        result["cohen_kappa"] = float("nan")
        result["krippendorff_alpha"] = float("nan")
        return result

    # Build ratings matrix: rows=annotators, cols=items
    annotator_to_idx = {a: i for i, a in enumerate(annotators)}
    item_to_idx = {it: i for i, it in enumerate(items)}

    # Detect if labels are numeric (for ordinal/interval) or categorical (nominal)
    try:
        df["label_numeric"] = pd.to_numeric(df["label"])
        level = "ordinal"
        label_col = "label_numeric"
    except (ValueError, TypeError):
        # Encode categorical labels as integers for matrix
        label_set = sorted(df["label"].unique())
        label_enc = {lbl: i for i, lbl in enumerate(label_set)}
        df["label_numeric"] = df["label"].map(label_enc)
        level = "nominal"
        label_col = "label_numeric"

    matrix = np.full((n_annotators, n_items), fill_value=np.nan)
    for _, row in df.iterrows():
        a_idx = annotator_to_idx[row["annotator"]]
        i_idx = item_to_idx[row["item_id"]]
        matrix[a_idx, i_idx] = row[label_col]

    # Krippendorff's alpha (all annotators)
    alpha = krippendorff_alpha(matrix, level_of_measurement=level)
    result["krippendorff_alpha"] = round(alpha, 4) if not np.isnan(alpha) else float("nan")

    # Cohen's kappa (only when exactly 2 annotators share items)
    if n_annotators == 2:
        # Collect items annotated by both raters
        rater1 = df[df["annotator"] == annotators[0]].set_index("item_id")[label_col]
        rater2 = df[df["annotator"] == annotators[1]].set_index("item_id")[label_col]
        shared_items = rater1.index.intersection(rater2.index)
        if len(shared_items) >= 2:
            r1_labels = rater1.loc[shared_items].tolist()
            r2_labels = rater2.loc[shared_items].tolist()
            kappa = cohen_kappa(r1_labels, r2_labels)
            result["cohen_kappa"] = round(kappa, 4)
        else:
            result["cohen_kappa"] = float("nan")
    else:
        result["cohen_kappa"] = float("nan")

    return result


# ======================================================================= #
#  Convenience: interpret agreement scores                                  #
# ======================================================================= #

def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) kappa interpretation."""
    if np.isnan(kappa):
        return "insufficient data"
    if kappa < 0:
        return "less than chance agreement"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost perfect"
