from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score


def classification_metrics(y_true: Sequence[int], probability: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    mask = np.isfinite(probability)
    y_true = y_true[mask]
    probability = probability[mask]
    if len(y_true) == 0:
        return {"n": 0, "auc": np.nan, "balanced_accuracy": np.nan, "accuracy": np.nan, "mcc": np.nan}
    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, probability)
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "mcc": float(matthews_corrcoef(y_true, predicted)) if len(np.unique(predicted)) > 1 and len(np.unique(y_true)) > 1 else 0.0,
    }
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    metrics.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    return metrics


def cohen_d_scores(matrix: np.ndarray, y: Sequence[int], feature_names: Sequence[str]) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=float)
    y = np.asarray(y, dtype=int)
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(feature_names):
        x0 = matrix[y == 0, index]
        x1 = matrix[y == 1, index]
        x0 = x0[np.isfinite(x0)]
        x1 = x1[np.isfinite(x1)]
        if len(x0) < 2 or len(x1) < 2:
            rows.append({"feature": name, "cohen_d": np.nan, "abs_cohen_d": np.nan, "welch_p": np.nan, "auc_proxy": np.nan})
            continue
        variance0 = np.var(x0, ddof=1)
        variance1 = np.var(x1, ddof=1)
        pooled = math.sqrt(((len(x0) - 1) * variance0 + (len(x1) - 1) * variance1) / max(1, len(x0) + len(x1) - 2))
        d_value = (np.mean(x1) - np.mean(x0)) / pooled if pooled > 0 else np.nan
        try:
            p_value = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p_value = np.nan
        try:
            u_value = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
            auc = u_value / (len(x1) * len(x0))
            auc_proxy = max(auc, 1 - auc)
        except Exception:
            auc_proxy = np.nan
        rows.append({
            "feature": name,
            "cohen_d": float(d_value) if np.isfinite(d_value) else np.nan,
            "abs_cohen_d": float(abs(d_value)) if np.isfinite(d_value) else np.nan,
            "welch_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "auc_proxy": float(auc_proxy) if np.isfinite(auc_proxy) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["abs_cohen_d", "auc_proxy"], ascending=False)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    group_columns = [column for column in ["validation", "feature_set", "selected_k"] if column in metrics.columns]
    metric_columns = ["auc", "balanced_accuracy", "accuracy", "mcc"]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_repeats_or_runs"] = len(group)
        row["n"] = int(group["n"].iloc[0]) if "n" in group.columns and pd.notna(group["n"].iloc[0]) else np.nan
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_sd"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_min"] = float(np.min(values))
                row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_p_value(observed: float, null_values: Sequence[float]) -> float:
    values = np.asarray([value for value in null_values if np.isfinite(value)], dtype=float)
    if len(values) == 0 or not np.isfinite(observed):
        return np.nan
    return float((np.sum(values >= observed) + 1) / (len(values) + 1))
