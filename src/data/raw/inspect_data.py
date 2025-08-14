# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/raw/inspect_data.py -->
import os
import pandas as pd
from dateutil.parser import parse

# This points INSIDE src/data/raw, so it'll inspect CRYPTO_TICK/, FI2010/, NASDAQ_ITCH/, etc.
DATA_DIR = os.path.dirname(__file__)  

def inspect_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(path, nrows=5)
        elif ext in ['.parquet', '.parq']:
            df = pd.read_parquet(path, nrows=5)
        elif ext in ['.json', '.ndjson']:
            df = pd.read_json(path, lines=True, nrows=5)
        else:
            # skip .py, .h5, etc.
            return
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    print(f"\n=== Inspecting: {os.path.basename(path)} ===")
    print("Columns and dtypes:")
    print(df.dtypes)
    print("\nSample rows:")
    print(df.head(3))

    # look for columns with date/time in the name
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            print(f"-> Potential timestamp column: '{col}'")
            samples = df[col].dropna().astype(str).unique()[:3]
            print("   Sample values:", samples)
            try:
                inferred = parse(samples[0])
                print("   Inferred datetime:", inferred.isoformat())
            except Exception:
                print("   Could not auto-parse format")

if __name__ == "__main__":
    # walk the raw/ folder
    for entry in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, entry)
        if os.path.isfile(full):
            inspect_file(full)
        elif os.path.isdir(full):
            # inspect all files in sub-folders
            for fname in os.listdir(full):
                inspect_file(os.path.join(full, fname))
