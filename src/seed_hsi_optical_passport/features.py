from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .io import clean_sample_id, source_to_binary, require_columns

META_COLUMNS = {
    "sample_id", "source", "source_code", "country", "country_code", "acquisition_order", "has_F", "has_R", "modality",
    "source_file", "file", "filename", "path", "status", "notes", "note", "index", "sample",
}


def parse_wavelength_from_column(column: object) -> float | None:
    text = str(column).strip()
    if text in META_COLUMNS or text.lower() in META_COLUMNS:
        return None
    try:
        value = float(text)
        if 300 <= value <= 2500:
            return value
    except Exception:
        pass
    matches = re.findall(r"(?<!\d)(\d{3,4}(?:[._]\d+)?)(?!\d)", text)
    values: list[float] = []
    for match in matches:
        try:
            value = float(match.replace("_", "."))
        except Exception:
            continue
        if 300 <= value <= 2500:
            values.append(value)
    return values[-1] if values else None


def wavelength_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    pairs: list[tuple[float, str]] = []
    for column in df.columns:
        if str(column) in META_COLUMNS or str(column).lower() in META_COLUMNS:
            continue
        wavelength = parse_wavelength_from_column(column)
        if wavelength is None:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() >= max(5, int(0.5 * len(df))):
            pairs.append((wavelength, column))
    if pairs:
        pairs = sorted(pairs, key=lambda item: item[0])
        return [column for _, column in pairs], np.array([w for w, _ in pairs], dtype=float)

    numeric_columns: list[str] = []
    for column in df.columns:
        if str(column) in META_COLUMNS or str(column).lower() in META_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() >= max(5, int(0.5 * len(df))):
            numeric_columns.append(column)
    return numeric_columns, np.arange(len(numeric_columns), dtype=float)


def align_wide_matrix(df: pd.DataFrame, manifest: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    require_columns(df, ["sample_id"], label)
    require_columns(manifest, ["sample_id"], "sample manifest")
    matrix = df.copy()
    matrix["sample_id"] = matrix["sample_id"].map(clean_sample_id)
    columns, wavelengths = wavelength_columns(matrix)
    if len(columns) < 5:
        raise ValueError(f"{label} contains too few spectral columns: {len(columns)}")
    order = manifest[["sample_id"]].copy()
    order["sample_id"] = order["sample_id"].map(clean_sample_id)
    aligned = order.merge(matrix[["sample_id"] + columns], on="sample_id", how="left")
    if aligned[columns].isna().all(axis=None):
        raise ValueError(f"{label} did not align with sample identifiers in the manifest")
    return aligned, columns, wavelengths


def rowwise_snv(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float, copy=True)
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    sd = np.nanstd(matrix, axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (matrix - mean) / sd


def derivative_features(matrix: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    if matrix.shape[1] < 2:
        raise ValueError("At least two wavelength bands are required for derivative features")
    spacing = np.diff(wavelengths)
    spacing = np.where(np.isfinite(spacing) & (np.abs(spacing) > 1e-12), spacing, 1.0)
    derivative = np.diff(matrix.astype(float), axis=1) / spacing.reshape(1, -1)
    midpoint = (wavelengths[:-1] + wavelengths[1:]) / 2.0
    names = [f"d{i + 1:03d}_{midpoint[i]:.3f}nm" for i in range(len(midpoint))]
    mapping = pd.DataFrame({
        "derivative_feature": names,
        "d_index_1based": np.arange(1, len(names) + 1),
        "left_wavelength_nm": wavelengths[:-1],
        "right_wavelength_nm": wavelengths[1:],
        "midpoint_nm": midpoint,
    })
    return derivative, names, mapping


def make_block_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    require_columns(manifest, ["sample_id"], "sample manifest")
    data = manifest.copy()
    data["sample_id"] = data["sample_id"].map(clean_sample_id)

    if "source" not in data.columns:
        if "country" in data.columns:
            data["source"] = data["country"]
        else:
            raise ValueError("sample manifest must contain a source column; country is accepted as a legacy alias")
    data["source"] = data["source"].astype(str).str.strip()

    if "acquisition_order" not in data.columns:
        if "hsi_index" in data.columns:
            data["acquisition_order"] = data["hsi_index"]
        else:
            data["acquisition_order"] = data["sample_id"].str.extract(r"(\d+)").astype(float)
    data["acquisition_order"] = pd.to_numeric(data["acquisition_order"], errors="coerce")
    if data["acquisition_order"].isna().any():
        raise ValueError("Some samples have missing acquisition-order values")

    data["source_binary"] = data["source"].map(source_to_binary)
    data["block_no5"] = np.ceil(data["acquisition_order"] / 5.0).astype(int)
    data["block_start"] = (data["block_no5"] - 1) * 5 + 1
    data["block_end"] = data["block_start"] + 4
    data["validation_block"] = [f"Block {chr(64 + int(block))}" for block in data["block_no5"]]
    data["source_prefix"] = data["source"].astype(str).str[0].str.upper()
    data["source_specific_validation_block"] = data["source_prefix"] + "_" + data["validation_block"].str.replace(" ", "", regex=False)
    return data


def build_metadata_control(manifest: pd.DataFrame, input_dir: Path) -> tuple[np.ndarray, list[str]]:
    meta = manifest[["sample_id", "acquisition_order", "block_no5"]].copy()
    block_dummies = pd.get_dummies(manifest["validation_block"], prefix="validation_block", dtype=float)
    meta = pd.concat([meta.reset_index(drop=True), block_dummies.reset_index(drop=True)], axis=1)

    optional_files = ["pixel_file_parse_report.csv", "pixel_feature_coverage.csv"]
    for filename in optional_files:
        matches = sorted(Path(input_dir).rglob(filename))
        if not matches:
            continue
        try:
            table = pd.read_csv(matches[0])
        except Exception:
            continue
        if "sample_id" not in table.columns:
            continue
        table = table.copy()
        table["sample_id"] = table["sample_id"].map(clean_sample_id)
        numeric_columns: list[str] = []
        for column in table.columns:
            label = str(column).lower()
            if column == "sample_id":
                continue
            if any(token in label for token in ["row", "pixel", "size", "bytes", "kb", "mb", "n_"]):
                numeric = pd.to_numeric(table[column], errors="coerce")
                if numeric.notna().sum() > 0:
                    numeric_columns.append(column)
        if not numeric_columns:
            continue
        if "modality" in table.columns:
            for modality in sorted(table["modality"].dropna().astype(str).unique()):
                subset = table[table["modality"].astype(str) == modality]
                aggregate = subset.groupby("sample_id")[numeric_columns].mean(numeric_only=True).reset_index()
                aggregate = aggregate.rename(columns={column: f"pixelmeta_{modality}_{column}" for column in numeric_columns})
                meta = meta.merge(aggregate, on="sample_id", how="left")
        else:
            aggregate = table.groupby("sample_id")[numeric_columns].mean(numeric_only=True).reset_index()
            aggregate = aggregate.rename(columns={column: f"pixelmeta_{Path(filename).stem}_{column}" for column in numeric_columns})
            meta = meta.merge(aggregate, on="sample_id", how="left")

    columns = [column for column in meta.columns if column != "sample_id"]
    matrix = meta[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return matrix, columns


def build_feature_blocks(input_dir: Path) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, list[str]]], pd.DataFrame]:
    from .io import read_table

    manifest = read_table(input_dir, "hsi_sample_manifest.csv", required=True)
    f_table = read_table(input_dir, "master_hsi_wide_F.csv", required=True)
    r_table = read_table(input_dir, "master_hsi_wide_R.csv", required=True)

    manifest = make_block_labels(manifest)
    f_aligned, f_columns, f_wavelengths = align_wide_matrix(f_table, manifest, "fluorescence HSI matrix")
    r_aligned, r_columns, r_wavelengths = align_wide_matrix(r_table, manifest, "reflectance HSI matrix")

    x_f = f_aligned[f_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_r = r_aligned[r_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_r_derivative, r_derivative_names, derivative_mapping = derivative_features(x_r, r_wavelengths)
    x_fr_snv = np.hstack([rowwise_snv(x_f), rowwise_snv(x_r)])
    fr_snv_names = [f"F_snv_{w:.3f}" for w in f_wavelengths] + [f"R_snv_{w:.3f}" for w in r_wavelengths]

    feature_blocks = {
        "avg_R_deriv": (x_r_derivative, r_derivative_names),
        "avg_R_raw": (x_r, [f"R_{w:.3f}" for w in r_wavelengths]),
        "avg_F_raw": (x_f, [f"F_{w:.3f}" for w in f_wavelengths]),
        "avg_FR_snv": (x_fr_snv, fr_snv_names),
    }
    return manifest, feature_blocks, derivative_mapping
