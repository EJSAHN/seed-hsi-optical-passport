from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def safe_sheet_name(name: str) -> str:
    clean = "".join(ch if ch not in r"[]:*?/\\" else "_" for ch in str(name))
    return clean[:31] if len(clean) > 31 else clean


def write_workbook(tables: Mapping[str, pd.DataFrame], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            if table is None:
                continue
            table.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
    return output_path
