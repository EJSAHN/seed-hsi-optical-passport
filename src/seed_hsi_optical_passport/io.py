from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def find_table(input_dir: Path, filename: str, required: bool = True) -> Path | None:
    """Find a table by filename in an input directory or its subdirectories."""
    input_dir = Path(input_dir)
    direct = input_dir / filename
    if direct.exists():
        return direct
    matches = sorted(input_dir.rglob(filename))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Required input table not found: {filename}")
    return None


def read_table(input_dir: Path, filename: str, required: bool = True, **kwargs) -> pd.DataFrame | None:
    path = find_table(input_dir, filename, required=required)
    if path is None:
        return None
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def clean_sample_id(value: object) -> str:
    return str(value).strip().replace(" ", "")


def source_to_binary(source: object) -> int:
    """Encode Niger as 0 and Senegal as 1."""
    text = str(source).strip().lower()
    if text.startswith("sen") or text == "s":
        return 1
    if text.startswith("nig") or text == "n":
        return 0
    if text.startswith("s"):
        return 1
    if text.startswith("n"):
        return 0
    raise ValueError(f"Cannot map source label to binary: {source!r}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
