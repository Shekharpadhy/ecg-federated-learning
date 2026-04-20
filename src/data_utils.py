import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    CACHE_FILE,
    DATA_DIR,
    LEFT_WINDOW,
    NUM_CLIENTS,
    RANDOM_STATE,
    RECORDS,
    RIGHT_WINDOW,
    TEST_SIZE,
)
from .download_data import download_mitbih
from .ptbxl_utils import load_ptbxl


def load_record(record_id: str):
    path = str(DATA_DIR / record_id)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    signal = record.p_signal[:, 0]
    return signal, annotation


def extract_beats_from_record(record_id: str):
    signal, annotation = load_record(record_id)
    X, y = [], []
    for idx, sym in zip(annotation.sample, annotation.symbol):
        start = idx - LEFT_WINDOW
        end = idx + RIGHT_WINDOW
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            if len(beat) == LEFT_WINDOW + RIGHT_WINDOW:
                X.append(beat)
                y.append(sym)
    return np.array(X, dtype=np.float32), np.array(y)


def load_dataset():
    # ── Disk cache: skip raw WFDB processing on repeat runs ──────────────────
    if CACHE_FILE.exists():
        cache = np.load(CACHE_FILE, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]
        print(f"[data] Loaded {len(X)} beats from cache.")
    else:
        download_mitbih(verbose=True)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        all_X, all_y = [], []
        for record_id in RECORDS:
            try:
                Xr, yr = extract_beats_from_record(record_id)
                if len(Xr) > 0:
                    all_X.append(Xr)
                    all_y.append(yr)
            except Exception:
                pass

        if not all_X:
            raise FileNotFoundError(
                f"No usable MIT-BIH records found in {DATA_DIR}. "
                "Use the Download button in the dashboard."
            )

        # Optional PTB-XL merge
        ptbxl_X, ptbxl_y = load_ptbxl()
        if len(ptbxl_X) > 0:
            all_X.append(ptbxl_X)
            all_y.append(ptbxl_y)
            print(f"[data] Merged {len(ptbxl_X)} PTB-XL windows.")

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        np.savez(CACHE_FILE, X=X, y=y)
        print(f"[data] Processed {len(X)} beats → cached.")

    # Drop classes with < 2 samples (stratified split needs ≥ 2)
    unique, counts = np.unique(y, return_counts=True)
    valid = set(unique[counts >= 2])
    mask = np.array([lbl in valid for lbl in y])
    X, y = X[mask], y[mask]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y, y_encoded, label_encoder


def get_train_test_data():
    X, y_raw, y_encoded, label_encoder = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    return X_train, X_test, y_train, y_test, y_raw, label_encoder


def split_into_clients(X, y, num_clients=NUM_CLIENTS):
    return list(zip(np.array_split(X, num_clients), np.array_split(y, num_clients)))
