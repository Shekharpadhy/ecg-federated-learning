"""Auto-download MIT-BIH records from PhysioNet using the wfdb library."""
import sys

import wfdb

from .config import DATA_DIR, RECORDS


def records_present() -> bool:
    """Return True if all configured records are already on disk."""
    for rec in RECORDS:
        if not (DATA_DIR / f"{rec}.hea").exists():
            return False
    return True


def download_mitbih(verbose: bool = True) -> None:
    """Download only the required MIT-BIH records from PhysioNet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if records_present():
        if verbose:
            print("MIT-BIH records already present — skipping download.")
        return

    if verbose:
        print(f"Downloading MIT-BIH records {RECORDS} from PhysioNet…")
        print("(this may take a minute on first run)")

    for rec in RECORDS:
        # Each record needs 3 files: .hea, .dat, .atr
        files = [f"{rec}.hea", f"{rec}.dat", f"{rec}.atr"]
        try:
            wfdb.dl_files("mitdb", str(DATA_DIR), files)
            if verbose:
                print(f"  ✓  record {rec}")
        except Exception as e:
            print(f"  ✗  record {rec} failed: {e}", file=sys.stderr)

    if verbose:
        present = [r for r in RECORDS if (DATA_DIR / f"{r}.hea").exists()]
        print(f"Downloaded {len(present)}/{len(RECORDS)} records → {DATA_DIR}")
