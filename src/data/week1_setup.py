# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/week1_setup.py -->
# File: code/src/data/week1_setup.py
# Week 1 Setup: folders, dataset configs, HDF5 detection, and initial inspection

from pathlib import Path
import gzip
import shutil
import pandas as pd

# -----------------------------
# Script & Data Paths
# -----------------------------
# script_dir: code/src/data
script_dir = Path(__file__).resolve().parent

# raw and processed data directories
raw_dir  = script_dir / "raw"
proc_dir = script_dir / "processed"

# config and notebooks at project root: code/
config_dir    = script_dir.parents[2] / "configs"
notebooks_dir = script_dir.parents[2] / "notebooks"

# features and models under src/
features_dir = script_dir.parent / "features"
models_dir   = script_dir.parent / "models"

print(f"Script dir: {script_dir}")
print(f"Raw data:   {raw_dir}")
print(f"Proc data:  {proc_dir}")
print(f"Configs:    {config_dir}")
print(f"Notebooks:  {notebooks_dir}")
print(f"Features:   {features_dir}")
print(f"Models:     {models_dir}")

# -----------------------------
# Folder Structure Setup
# -----------------------------
for d in [raw_dir, proc_dir, config_dir, notebooks_dir, features_dir, models_dir]:
    d.mkdir(parents=True, exist_ok=True)
for sub in ["NASDAQ_ITCH", "CRYPTO_TICK", "FI2010"]:
    (raw_dir / sub).mkdir(exist_ok=True)
    (proc_dir / sub).mkdir(exist_ok=True)
print("✅ Initialized full folder structure.")

# -----------------------------
# Detect or Instruct NASDAQ ITCH Source
# -----------------------------
itch_dir = raw_dir / "NASDAQ_ITCH"
h5_files = list(itch_dir.glob("*.h5"))
if h5_files:
    print(f"✔ Found ITCH store: {h5_files[0].name}")
else:
    (config_dir / "nasdaq_itch_download.txt").write_text(
        "1. Visit https://www.kaggle.com/datasets/eyalis/nasdaq-itch\n"
        "2. Download and unzip the .h5 files\n"
        "3. Place them into raw/NASDAQ_ITCH/\n"
    )
    print("⚠ NASDAQ ITCH instructions written.")

# -----------------------------
# Manual‐Download Configs for Crypto & FI2010
# -----------------------------
(config_dir / "kaggle_crypto_download.txt").write_text(
    "1. Visit https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data\n"
    "2. Download CSVs for BTC, ETH, ADA\n"
    "3. Place them into raw/CRYPTO_TICK/\n"
)
(config_dir / "fi2010_download.txt").write_text(
    "1. Visit https://www.kaggle.com/datasets/zcorr/fi2010\n"
    "2. Download FI2010_train.csv & FI2010_test.csv\n"
    "3. Place them into raw/FI2010/\n"
)
print("📑 Manual download configs updated.")

# -----------------------------
# Sample Data Inspection
# -----------------------------
def inspect_data(fp: Path):
    if fp.exists():
        print(f"\n--- Inspecting {fp.name} ---")
        df = pd.read_csv(fp, nrows=10)
        print(df.dtypes)
        print(df.head())
        print(df.describe())
    else:
        print(f"❌ Not found: {fp}")

inspect_data(raw_dir / "FI2010" / "FI2010_train.csv")
inspect_data(raw_dir / "CRYPTO_TICK" / "BTC_1min.csv")

# If HDF5 ITCH store exists, list its keys
if h5_files:
    with pd.HDFStore(h5_files[0], mode="r") as store:
        print("\nHDF5 ITCH store keys:")
        print(store.keys())

# -----------------------------
# Environment Setup Log
# -----------------------------
env_log = config_dir / "environment_setup.txt"
with open(env_log, "w") as f:
    f.write("# Library versions\n")
    for lib in ["numpy", "pandas", "scikit-learn", "torch", "tensorflow"]:
        try:
            v = __import__(lib).__version__
        except ImportError:
            v = "not installed"
        f.write(f"{lib}: {v}\n")
print("\n✅ Week 1 setup complete.")
