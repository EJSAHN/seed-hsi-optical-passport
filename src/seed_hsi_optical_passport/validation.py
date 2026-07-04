from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import classification_metrics, cohen_d_scores, summarize_metrics


def classifier_pipeline(seed: int = 1) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )),
    ])


def predict_probability(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(matrix)
        if probability.shape[1] == 2:
            return probability[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(matrix)
        return 1.0 / (1.0 + np.exp(-decision))
    return model.predict(matrix).astype(float)


def repeated_stratified_cv(
    matrix: np.ndarray,
    y: Sequence[int],
    sample_ids: Sequence[str],
    feature_set: str,
    repeats: int,
    seed: int,
    selected_k: int | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    matrix = np.asarray(matrix, dtype=float)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y)
    n_splits = max(2, min(5, int(np.min(counts))))
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_counts: Counter = Counter()
    feature_name_list = list(feature_names) if feature_names is not None else None

    for repeat in range(repeats):
        cross_validator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        probability = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1)
        selected_by_fold: dict[int, str] = {}
        for fold, (train_index, test_index) in enumerate(cross_validator.split(matrix, y)):
            x_train = matrix[train_index]
            x_test = matrix[test_index]
            if selected_k is not None:
                if feature_name_list is None:
                    raise ValueError("feature_names must be provided when selected_k is used")
                ranking = cohen_d_scores(x_train, y[train_index], feature_name_list)
                selected = ranking.head(selected_k)["feature"].tolist()
                selected_index = [feature_name_list.index(feature) for feature in selected]
                x_train = x_train[:, selected_index]
                x_test = x_test[:, selected_index]
                selected_by_fold[fold] = ";".join(selected)
                selection_counts.update(selected)
            model = classifier_pipeline(seed=seed + repeat)
            model.fit(x_train, y[train_index])
            probability[test_index] = predict_probability(model, x_test)
            fold_id[test_index] = fold
        row = classification_metrics(y, probability)
        row.update({
            "validation": "random_stratified_cv",
            "feature_set": feature_set,
            "repeat": repeat + 1,
            "selected_k": selected_k if selected_k is not None else np.nan,
            "n_splits": n_splits,
        })
        metric_rows.append(row)
        for i, sample_id in enumerate(sample_ids):
            prediction_rows.append({
                "validation": "random_stratified_cv",
                "feature_set": feature_set,
                "repeat": repeat + 1,
                "fold": int(fold_id[i]),
                "heldout_group": "",
                "sample_id": sample_id,
                "y_true": int(y[i]),
                "prob_senegal": float(probability[i]) if np.isfinite(probability[i]) else np.nan,
                "selected_k": selected_k if selected_k is not None else np.nan,
                "selected_features_in_fold": selected_by_fold.get(int(fold_id[i]), ""),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), selection_counts


def leave_group_cv(
    matrix: np.ndarray,
    y: Sequence[int],
    groups: Sequence[str],
    sample_ids: Sequence[str],
    feature_set: str,
    validation_name: str,
    selected_k: int | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    matrix = np.asarray(matrix, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    probability = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    heldout_group = np.array([""] * len(y), dtype=object)
    selection_counts: Counter = Counter()
    selected_by_group: dict[str, str] = {}
    feature_name_list = list(feature_names) if feature_names is not None else None

    fold = 0
    for train_index, test_index in LeaveOneGroupOut().split(matrix, y, groups):
        if len(np.unique(y[train_index])) < 2:
            continue
        x_train = matrix[train_index]
        x_test = matrix[test_index]
        group = str(groups[test_index][0])
        selected_text = ""
        if selected_k is not None:
            if feature_name_list is None:
                raise ValueError("feature_names must be provided when selected_k is used")
            ranking = cohen_d_scores(x_train, y[train_index], feature_name_list)
            selected = ranking.head(selected_k)["feature"].tolist()
            selected_index = [feature_name_list.index(feature) for feature in selected]
            x_train = x_train[:, selected_index]
            x_test = x_test[:, selected_index]
            selected_text = ";".join(selected)
            selection_counts.update(selected)
        model = classifier_pipeline(seed=fold + 1)
        model.fit(x_train, y[train_index])
        probability[test_index] = predict_probability(model, x_test)
        fold_id[test_index] = fold
        heldout_group[test_index] = group
        selected_by_group[group] = selected_text
        fold += 1

    row = classification_metrics(y, probability)
    row.update({
        "validation": validation_name,
        "feature_set": feature_set,
        "repeat": 1,
        "selected_k": selected_k if selected_k is not None else np.nan,
        "n_groups": int(len(np.unique(groups))),
        "n_groups_used": int(fold),
    })
    predictions = pd.DataFrame({
        "validation": validation_name,
        "feature_set": feature_set,
        "repeat": 1,
        "fold": fold_id.astype(int),
        "heldout_group": heldout_group,
        "sample_id": list(sample_ids),
        "y_true": y.astype(int),
        "prob_senegal": probability,
        "selected_k": selected_k if selected_k is not None else np.nan,
        "selected_features_in_fold": [selected_by_group.get(str(group), "") for group in heldout_group],
    })
    return pd.DataFrame([row]), predictions, selection_counts


def permutation_null(
    matrix: np.ndarray,
    y: Sequence[int],
    sample_ids: Sequence[str],
    feature_set: str,
    permutations: int,
    repeats_per_permutation: int,
    seed: int,
    selected_k: int | None = None,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if permutations <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for permutation in range(permutations):
        y_permuted = rng.permutation(y)
        metrics, _, _ = repeated_stratified_cv(
            matrix=matrix,
            y=y_permuted,
            sample_ids=sample_ids,
            feature_set=feature_set,
            repeats=repeats_per_permutation,
            seed=seed + permutation * 11,
            selected_k=selected_k,
            feature_names=feature_names,
        )
        summary = summarize_metrics(metrics)
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["permutation"] = permutation + 1
        row["observed_feature_set"] = feature_set
        rows.append(row)
    return pd.DataFrame(rows)
