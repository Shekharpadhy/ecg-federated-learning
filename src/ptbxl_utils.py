"""PTB-XL dataset loader.

Downloads: https://physionet.org/content/ptb-xl/1.0.3/
Place the dataset at data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
Returns empty arrays if dataset is not present — project works without it.
"""
import ast
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from .config import LEFT_WINDOW, PTBXL_DIR, RIGHT_WINDOW

WINDOW_SIZE = LEFT_WINDOW + RIGHT_WINDOW  # 200 samples

_SUPERCLASS_MAP = {
    "NORM": "NORM",
    "MI": "MI",
    "STTC": "STTC",
    "CD": "CD",
    "HYP": "HYP",
}


def _primary_superclass(scp_codes: dict) -> Optional[str]:
    for code in scp_codes:
        sc = _SUPERCLASS_MAP.get(code)
        if sc is not None:
            return sc
    return None


def load_ptbxl(max_records: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y_raw) windows from PTB-XL, or empty arrays if unavailable."""
    if PTBXL_DIR is None or not PTBXL_DIR.exists():
        return np.array([], dtype=np.float32), np.array([])

    csv_path = PTBXL_DIR / "ptbxl_database.csv"
    if not csv_path.exists():
        return np.array([], dtype=np.float32), np.array([])

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    X_all = []
    y_all = []

    for _, row in df.head(max_records).iterrows():
        label = _primary_superclass(row["scp_codes"])
        if label is None:
            continue

        record_path = str(PTBXL_DIR / row["filename_lr"])
        try:
            record = wfdb.rdrecord(record_path)
        except Exception:
            continue

        signal = record.p_signal[:, 0].astype(np.float32)

        for start in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = signal[start : start + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                X_all.append(window)
                y_all.append(label)

    if not X_all:
        return np.array([], dtype=np.float32), np.array([])

    return np.array(X_all, dtype=np.float32), np.array(y_all)
