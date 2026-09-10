"""
Automated downloader for the synthetic Sparkov Credit Card benchmark dataset.

Dataset Provenance:
- Author: Brandon Harris / kartik2112
- Source: Kaggle Datasets (`kartik2112/fraud-detection`)
- Nature: Synthetic credit card transaction benchmark
- Destination: data/raw/benchmark/
"""

import os
import shutil
from pathlib import Path
from typing import List
import kagglehub

RAW_DIR = Path("data/raw/benchmark")
DATASET_HANDLE = "kartik2112/fraud-detection"


def acquire_benchmark_data() -> List[Path]:
    """
    Acquire the Sparkov benchmark dataset using kagglehub and copy raw CSVs into data/raw/benchmark/.
    
    Returns:
        List[Path]: Paths of acquired CSV files in data/raw/benchmark/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing_csvs = list(RAW_DIR.glob("*.csv"))

    if existing_csvs:
        print(f"[FOUND] {len(existing_csvs)} raw benchmark CSV(s) already exist in {RAW_DIR}:")
        for f in existing_csvs:
            print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        return existing_csvs

    print(f"[DOWNLOADING] Fetching dataset '{DATASET_HANDLE}' via kagglehub...")
    download_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"[CACHED] Dataset cached at: {download_dir}")

    downloaded_paths = []
    for csv_file in download_dir.glob("*.csv"):
        dest_file = RAW_DIR / csv_file.name
        print(f"[COPYING] Copying {csv_file.name} to {dest_file} ...")
        shutil.copy2(csv_file, dest_file)
        downloaded_paths.append(dest_file)

    print(f"[SUCCESS] Acquired {len(downloaded_paths)} CSV files into {RAW_DIR}.")
    return downloaded_paths


if __name__ == "__main__":
    acquire_benchmark_data()
