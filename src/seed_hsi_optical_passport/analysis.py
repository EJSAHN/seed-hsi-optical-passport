from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .features import build_feature_blocks, build_metadata_control
from .io import write_csv
from .metrics import cohen_d_scores, empirical_p_value, summarize_metrics
from .validation import leave_group_cv, permutation_null, repeated_stratified_cv
from .workbook import write_workbook


def parse_band_counts(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return sorted(set(values))


def _merge_group_columns(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "source", "acquisition_order", "validation_block", "source_specific_validation_block"]
    return predictions.merge(manifest[columns], on="sample_id", how="left")


def _stability_selection(
    matrix: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    derivative_mapping: pd.DataFrame,
    bootstrap: int,
    seed: int,
    selection_counts: Counter,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = Counter(selection_counts)
    for _ in range(max(0, bootstrap)):
        indices: list[int] = []
        for class_value in [0, 1]:
            pool = np.where(y == class_value)[0]
            size = max(2, int(round(0.8 * len(pool))))
            indices.extend(rng.choice(pool, size=size, replace=False).tolist())
        indices = sorted(indices)
        ranking = cohen_d_scores(matrix[indices], y[indices], feature_names)
        counts.update(ranking.head(10)["feature"].tolist())
    stability = pd.DataFrame({"feature": list(counts.keys()), "selection_count": list(counts.values())})
    if stability.empty:
        return stability
    mapping = derivative_mapping.rename(columns={"derivative_feature": "feature"})
    return stability.sort_values("selection_count", ascending=False).merge(mapping, on="feature", how="left")


def _dataset_overview(manifest: pd.DataFrame, feature_blocks: dict[str, tuple[np.ndarray, list[str]]]) -> pd.DataFrame:
    rows = [
        {"item": "samples", "value": len(manifest)},
        {"item": "niger_samples", "value": int((manifest["source_binary"] == 0).sum())},
        {"item": "senegal_samples", "value": int((manifest["source_binary"] == 1).sum())},
        {"item": "validation_blocks", "value": int(manifest["validation_block"].nunique())},
        {"item": "source_specific_validation_blocks", "value": int(manifest["source_specific_validation_block"].nunique())},
    ]
    for name, (matrix, _) in feature_blocks.items():
        rows.append({"item": f"{name}_features", "value": int(matrix.shape[1])})
    return pd.DataFrame(rows)


def _write_summary(
    output_dir: Path,
    dataset_overview: pd.DataFrame,
    source_summary: pd.DataFrame,
    minimal_summary: pd.DataFrame,
    metadata_summary: pd.DataFrame,
    permutation_nulls: pd.DataFrame,
    stability: pd.DataFrame,
) -> Path:
    lines: list[str] = []
    lines.append("# Optical-passport validation summary")
    lines.append("")
    values = dict(zip(dataset_overview["item"], dataset_overview["value"]))
    lines.append(f"The analysis included {int(values.get('samples', 0))} samples ({int(values.get('niger_samples', 0))} Niger and {int(values.get('senegal_samples', 0))} Senegal).")
    lines.append("")

    def select(summary: pd.DataFrame, feature_set: str, validation: str) -> pd.Series | None:
        if summary.empty:
            return None
        subset = summary[(summary["feature_set"].astype(str) == feature_set) & (summary["validation"].astype(str) == validation)]
        if subset.empty:
            return None
        return subset.iloc[0]

    lines.append("## Reflectance-derivative classifier")
    lines.append("")
    for validation in ["random_stratified_cv", "acquisition_block_holdout", "source_specific_block_holdout"]:
        row = select(source_summary, "avg_R_deriv", validation)
        if row is not None:
            lines.append(f"- {validation}: AUC={row['auc_mean']:.3f}, balanced accuracy={row['balanced_accuracy_mean']:.3f}, MCC={row['mcc_mean']:.3f}.")
    lines.append("")

    lines.append("## Minimal-band reflectance-derivative classifier")
    lines.append("")
    if not minimal_summary.empty:
        for validation in ["acquisition_block_holdout", "source_specific_block_holdout"]:
            subset = minimal_summary[minimal_summary["validation"].astype(str) == validation]
            if subset.empty:
                continue
            best = subset.sort_values("auc_mean", ascending=False).iloc[0]
            lines.append(f"- Best {validation}: k={int(best['selected_k'])}, AUC={best['auc_mean']:.3f}, balanced accuracy={best['balanced_accuracy_mean']:.3f}, MCC={best['mcc_mean']:.3f}.")
    lines.append("")

    lines.append("## Metadata-only negative control")
    lines.append("")
    for validation in ["random_stratified_cv", "acquisition_block_holdout", "source_specific_block_holdout"]:
        row = select(metadata_summary, "metadata_only", validation)
        if row is not None:
            lines.append(f"- {validation}: AUC={row['auc_mean']:.3f}, balanced accuracy={row['balanced_accuracy_mean']:.3f}, MCC={row['mcc_mean']:.3f}.")
    lines.append("")

    lines.append("## Permutation null")
    lines.append("")
    observed = select(source_summary, "avg_R_deriv", "random_stratified_cv")
    if observed is not None and not permutation_nulls.empty:
        subset = permutation_nulls[permutation_nulls["observed_feature_set"].astype(str) == "avg_R_deriv"]
        p_value = empirical_p_value(observed["auc_mean"], subset["auc_mean"].values) if not subset.empty else np.nan
        lines.append(f"- Empirical p-value for random-CV avg_R_deriv AUC: {p_value:.4f}.")
    else:
        lines.append("- Permutation null results were not generated.")
    lines.append("")

    lines.append("## Most stable derivative bands")
    lines.append("")
    if stability.empty:
        lines.append("Stability-selection results were not generated.")
    else:
        top = stability[["feature", "selection_count", "left_wavelength_nm", "right_wavelength_nm", "midpoint_nm"]].head(10)
        lines.append(top.to_markdown(index=False))
    lines.append("")

    path = output_dir / "analysis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    repeats: int = 50,
    permutations: int = 200,
    permutation_repeats: int = 2,
    bootstrap: int = 300,
    band_counts: str = "1,3,5,10,20",
    seed: int = 1729,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, feature_blocks, derivative_mapping = build_feature_blocks(input_dir)
    sample_ids = manifest["sample_id"].tolist()
    y = manifest["source_binary"].to_numpy(dtype=int)

    derivative_mapping_path = write_csv(derivative_mapping, output_dir, "derivative_band_mapping.csv")
    dataset_overview = _dataset_overview(manifest, feature_blocks)
    dataset_overview_path = write_csv(dataset_overview, output_dir, "dataset_overview.csv")
    block_manifest_path = write_csv(
        manifest[["sample_id", "source", "acquisition_order", "validation_block", "source_specific_validation_block", "source_binary"]],
        output_dir,
        "sample_validation_manifest.csv",
    )
    index_counts = manifest.groupby(["validation_block", "source"]).size().reset_index(name="n")
    write_csv(index_counts, output_dir, "validation_block_source_counts.csv")

    source_metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for name, (matrix, feature_names) in feature_blocks.items():
        metrics, predictions, _ = repeated_stratified_cv(matrix, y, sample_ids, name, repeats=repeats, seed=seed)
        source_metrics_frames.append(metrics)
        prediction_frames.append(predictions)
        metrics, predictions, _ = leave_group_cv(matrix, y, manifest["validation_block"].values, sample_ids, name, "acquisition_block_holdout")
        source_metrics_frames.append(metrics)
        prediction_frames.append(predictions)
        metrics, predictions, _ = leave_group_cv(matrix, y, manifest["source_specific_validation_block"].values, sample_ids, name, "source_specific_block_holdout")
        source_metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    source_metrics = pd.concat(source_metrics_frames, ignore_index=True)
    source_summary = summarize_metrics(source_metrics)
    source_metrics_path = write_csv(source_metrics, output_dir, "source_classification_metrics.csv")
    source_summary_path = write_csv(source_summary, output_dir, "source_classification_summary.csv")

    x_metadata, metadata_feature_names = build_metadata_control(manifest, input_dir)
    metadata_metrics_frames: list[pd.DataFrame] = []
    for validation_name, groups in [
        ("random_stratified_cv", None),
        ("acquisition_block_holdout", manifest["validation_block"].values),
        ("source_specific_block_holdout", manifest["source_specific_validation_block"].values),
    ]:
        if groups is None:
            metrics, predictions, _ = repeated_stratified_cv(x_metadata, y, sample_ids, "metadata_only", repeats=repeats, seed=seed)
        else:
            metrics, predictions, _ = leave_group_cv(x_metadata, y, groups, sample_ids, "metadata_only", validation_name)
        metadata_metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    metadata_metrics = pd.concat(metadata_metrics_frames, ignore_index=True)
    metadata_summary = summarize_metrics(metadata_metrics)
    metadata_metrics_path = write_csv(metadata_metrics, output_dir, "metadata_negative_control_metrics.csv")
    metadata_summary_path = write_csv(metadata_summary, output_dir, "metadata_negative_control_summary.csv")

    x_derivative, derivative_feature_names = feature_blocks["avg_R_deriv"]
    selected_counts: Counter = Counter()
    minimal_metric_frames: list[pd.DataFrame] = []
    minimal_prediction_frames: list[pd.DataFrame] = []
    for k in parse_band_counts(band_counts):
        feature_set = f"avg_R_deriv_minimal_k{k}"
        metrics, predictions, counts = repeated_stratified_cv(
            x_derivative, y, sample_ids, feature_set, repeats=repeats, seed=seed + 101,
            selected_k=k, feature_names=derivative_feature_names,
        )
        minimal_metric_frames.append(metrics)
        minimal_prediction_frames.append(predictions)
        selected_counts.update(counts)
        metrics, predictions, counts = leave_group_cv(
            x_derivative, y, manifest["validation_block"].values, sample_ids, feature_set,
            "acquisition_block_holdout", selected_k=k, feature_names=derivative_feature_names,
        )
        minimal_metric_frames.append(metrics)
        minimal_prediction_frames.append(predictions)
        selected_counts.update(counts)
        metrics, predictions, counts = leave_group_cv(
            x_derivative, y, manifest["source_specific_validation_block"].values, sample_ids, feature_set,
            "source_specific_block_holdout", selected_k=k, feature_names=derivative_feature_names,
        )
        minimal_metric_frames.append(metrics)
        minimal_prediction_frames.append(predictions)
        selected_counts.update(counts)

    minimal_metrics = pd.concat(minimal_metric_frames, ignore_index=True)
    minimal_summary = summarize_metrics(minimal_metrics)
    minimal_metrics_path = write_csv(minimal_metrics, output_dir, "minimal_band_metrics.csv")
    minimal_summary_path = write_csv(minimal_summary, output_dir, "minimal_band_summary.csv")
    prediction_frames.extend(minimal_prediction_frames)

    stability = _stability_selection(
        x_derivative, y, derivative_feature_names, derivative_mapping,
        bootstrap=bootstrap, seed=seed + 404, selection_counts=selected_counts,
    )
    stability_path = write_csv(stability, output_dir, "minimal_band_selection_stability.csv")

    effect_sizes = cohen_d_scores(x_derivative, y, derivative_feature_names).merge(
        derivative_mapping.rename(columns={"derivative_feature": "feature"}), on="feature", how="left"
    )
    effect_sizes_path = write_csv(effect_sizes, output_dir, "full_data_derivative_source_effect_sizes.csv")

    permutation_frames: list[pd.DataFrame] = []
    if permutations > 0:
        permutation_frames.append(permutation_null(
            x_derivative, y, sample_ids, "avg_R_deriv",
            permutations=permutations, repeats_per_permutation=permutation_repeats, seed=seed + 600,
        ))
        if 3 in parse_band_counts(band_counts):
            permutation_frames.append(permutation_null(
                x_derivative, y, sample_ids, "avg_R_deriv_minimal_k3",
                permutations=permutations, repeats_per_permutation=permutation_repeats, seed=seed + 900,
                selected_k=3, feature_names=derivative_feature_names,
            ))
    permutation_nulls = pd.concat(permutation_frames, ignore_index=True) if permutation_frames else pd.DataFrame()
    permutation_path = write_csv(permutation_nulls, output_dir, "permutation_null_metrics.csv")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = _merge_group_columns(predictions, manifest)
    prediction_path = write_csv(predictions, output_dir, "block_fold_predictions.csv")

    summary_path = _write_summary(
        output_dir=output_dir,
        dataset_overview=dataset_overview,
        source_summary=source_summary,
        minimal_summary=minimal_summary,
        metadata_summary=metadata_summary,
        permutation_nulls=permutation_nulls,
        stability=stability,
    )

    readme = pd.DataFrame({
        "field": ["Niger", "Senegal", "source label", "derivative feature ID", "midpoint wavelength"],
        "description": [
            "Encoded as class 0 in supervised classification.",
            "Encoded as class 1 in supervised classification.",
            "Accession-source label used for the optical-passport classifier.",
            "Analysis feature ID for an adjacent-band reflectance derivative feature.",
            "Wavelength midpoint used for article reporting of derivative bands.",
        ],
    })
    workbook_tables = {
        "README": readme,
        "Dataset_overview": dataset_overview,
        "Source_CV": source_summary,
        "Minimal_band_CV": minimal_summary,
        "Metadata_control": metadata_summary,
        "Permutation_null": permutation_nulls,
        "Derivative_mapping": derivative_mapping,
        "Stable_bands": stability,
        "Band_effects": effect_sizes.head(1000),
        "Predictions": predictions,
    }
    workbook_path = write_workbook(workbook_tables, output_dir / "Supplementary_Data_S1_optical_passport.xlsx")

    return {
        "dataset_overview": dataset_overview_path,
        "sample_validation_manifest": block_manifest_path,
        "source_metrics": source_metrics_path,
        "source_summary": source_summary_path,
        "metadata_metrics": metadata_metrics_path,
        "metadata_summary": metadata_summary_path,
        "minimal_metrics": minimal_metrics_path,
        "minimal_summary": minimal_summary_path,
        "stability": stability_path,
        "effect_sizes": effect_sizes_path,
        "derivative_mapping": derivative_mapping_path,
        "permutation_nulls": permutation_path,
        "predictions": prediction_path,
        "summary": summary_path,
        "workbook": workbook_path,
    }
