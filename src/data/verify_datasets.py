# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/verify_datasets.py -->
# File: code/src/data/verify_datasets.py
"""
Quick sanity check that all required raw files are present
and have the expected schema/keys, given this structure:

code/
 └ src/
    └ data/
       ├ raw/
       │  ├ FI2010/
       │  │   ├ FI2010_train.csv
       │  │   └ FI2010_test.csv
       │  ├ CRYPTO_TICK/
       │  │   ├ BTC_1min.csv
       │  │   ├ ETH_1min.csv
       │  │   └ ADA_1min.csv
       │  └ NASDAQ_ITCH/
       │      └ itch.h5
       └ processed/
"""
from pathlib import Path
import pandas as pd

# Base directories
script_dir = Path(__file__).resolve().parent            # .../code/src/data
raw_dir    = script_dir / "raw"
itch_dir   = raw_dir / "NASDAQ_ITCH"

# Expected FI-2010 files
fi_train = raw_dir / "FI2010" / "FI2010_train.csv"
fi_test  = raw_dir / "FI2010" / "FI2010_test.csv"

# Expected Crypto snapshots (1-min)
crypto_samples = [
    raw_dir / "CRYPTO_TICK" / "BTC_1min.csv",
    raw_dir / "CRYPTO_TICK" / "ETH_1min.csv",
    raw_dir / "CRYPTO_TICK" / "ADA_1min.csv",
]

print("🔍 Verifying FI-2010 files…")
for path in (fi_train, fi_test):
    if not path.exists():
        print(f"❌ MISSING: {path}")
    else:
        df = pd.read_csv(path, nrows=0)
        print(f"✔ {path.name}: {len(df.columns)} cols, sample cols {df.columns[:5].tolist()}")

print("\n🔍 Verifying Crypto LOB files (1-min)…")
for path in crypto_samples:
    if not path.exists():
        print(f"❌ MISSING: {path}")
    else:
        df = pd.read_csv(path, nrows=0)
        print(f"✔ {path.name}: {len(df.columns)} cols, sample cols {df.columns[:5].tolist()}")

print("\n🔍 Verifying NASDAQ ITCH HDF5 store…")
h5_files = list(itch_dir.glob("*.h5"))
if not h5_files:
    print(f"❌ No .h5 files found in {itch_dir}")
else:
    h5 = h5_files[0]
    print(f"✔ Found ITCH store: {h5.name}")
    with pd.HDFStore(h5, mode="r") as store:
        keys = store.keys()
        print(f"   • Groups: {keys[:5]}{(' …' if len(keys)>5 else '')}")
        # inspect one group, e.g. '/A'
        if "/A" in keys:
            dfA = store["/A"]
            print(f"   • '/A': {len(dfA)} rows × {len(dfA.columns)} columns")
print("\n✅ Dataset verification complete.")
