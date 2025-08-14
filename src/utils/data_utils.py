# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/utils/data_utils.py -->
# #!/usr/bin/env python3
# """
# Utility helpers for the HFT pipeline

# ▪ load_features_labels      – parquet → (X_df, y) keeping only numeric/bool
# ▪ scale_features            – z‑score with optional skip columns
# ▪ create_sequences          – sliding window constructor
# ▪ compute_class_weights
# """
# from __future__ import annotations
# from collections import Counter
# from typing import List, Tuple, Dict, Any
# import numpy as np
# import pandas as pd

# NUMERIC_ONLY = ["number", "bool"]  # keep numeric + flags

# def load_features_labels(
#     parquet_file: str,
#     label_col: str = "y",
#     drop_cols: List[str] | str | None = "timestamp",
# ) -> tuple[pd.DataFrame, np.ndarray]:
#     """
#     Load parquet → (X_df, y). X_df keeps only numeric/bool columns.
#     Sorts by the first available drop_col (usually 'timestamp') to keep order.
#     """
#     df = pd.read_parquet(parquet_file)

#     if drop_cols:
#         if isinstance(drop_cols, str):
#             drop_cols = [drop_cols]
#         for col in drop_cols:
#             if col in df.columns:
#                 df = df.sort_values(col)
#                 break

#     if label_col not in df.columns:
#         raise KeyError(f"Label column '{label_col}' not found in {parquet_file}")

#     y = df[label_col].to_numpy(copy=True)

#     cols_to_drop = [label_col] + ([*drop_cols] if drop_cols else [])
#     feat_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

#     # keep only numeric/bool features, in their original order
#     feat_df = feat_df.select_dtypes(include=NUMERIC_ONLY).copy()
#     return feat_df, y

# def scale_features(
#     X_train: np.ndarray,
#     X_val: np.ndarray | None = None,
#     X_test: np.ndarray | None = None,
#     *,
#     skip_indices: List[int] | None = None,
# ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
#     """Z‑score scaling (μ/σ) with optional per‑column skip."""
#     means = X_train.mean(0)
#     stds  = X_train.std(0)
#     stds[stds == 0] = 1e-8

#     if skip_indices:
#         for idx in skip_indices:
#             if idx < len(means):
#                 means[idx] = 0.0
#                 stds[idx]  = 1.0

#     def _apply(arr: np.ndarray | None):
#         if arr is None:
#             return None
#         return ((arr - means) / stds).astype(np.float32)

#     return (
#         _apply(X_train),
#         _apply(X_val),
#         _apply(X_test),
#         {"mean": means.tolist(), "std": stds.tolist()},
#     )

# def create_sequences(
#     X: np.ndarray,
#     y: np.ndarray,
#     seq_length: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Rolling window: X[i … i+L) → y[i+L‑1]."""
#     n = X.shape[0] - seq_length + 1
#     if n <= 0:
#         raise ValueError("Sequence length longer than data length.")
#     X_seq = np.stack([X[i:i+seq_length] for i in range(n)], 0)
#     y_seq = y[seq_length-1:]
#     return X_seq, y_seq

# def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
#     """Inverse‑frequency weights: N / (K * n_k)."""
#     counts = Counter(y.tolist())
#     N = float(len(y)); K = len(counts)
#     return {int(cls): (N / (K * cnt)) for cls, cnt in counts.items()}



# #!/usr/bin/env python3
# #!/usr/bin/env python3
# """
# Utility helpers for the HFT pipeline

# ▪ load_features_labels      – parquet → (X_df, y) keeping only numeric/bool
# ▪ scale_features            – z‑score with optional skip columns
# ▪ create_sequences          – sliding window constructor
# ▪ compute_class_weights
# """
# from __future__ import annotations
# from collections import Counter
# from typing import List, Tuple, Dict, Any
# import numpy as np
# import pandas as pd

# NUMERIC_ONLY = ["number", "bool"]  # keep numeric + flags

# def load_features_labels(
#     parquet_file: str,
#     label_col: str = "y",
#     drop_cols: List[str] | str | None = "timestamp",
# ) -> tuple[pd.DataFrame, np.ndarray]:
#     df = pd.read_parquet(parquet_file)

#     if drop_cols:
#         if isinstance(drop_cols, str):
#             drop_cols = [drop_cols]
#         for col in drop_cols:
#             if col in df.columns:
#                 df = df.sort_values(col)
#                 break

#     if label_col not in df.columns:
#         raise KeyError(f"Label column '{label_col}' not found in {parquet_file}")

#     y = df[label_col].to_numpy(copy=True)

#     cols_to_drop = [label_col] + ([*drop_cols] if drop_cols else [])
#     feat_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

#     feat_df = feat_df.select_dtypes(include=NUMERIC_ONLY).copy()
#     return feat_df, y

# def scale_features(
#     X_train: np.ndarray,
#     X_val: np.ndarray | None = None,
#     X_test: np.ndarray | None = None,
#     *,
#     skip_indices: List[int] | None = None,
# ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
#     means = X_train.mean(0)
#     stds  = X_train.std(0)
#     stds[stds == 0] = 1e-8

#     if skip_indices:
#         for idx in skip_indices:
#             if idx < len(means):
#                 means[idx] = 0.0
#                 stds[idx]  = 1.0

#     def _apply(arr: np.ndarray | None):
#         if arr is None:
#             return None
#         return ((arr - means) / stds).astype(np.float32)

#     return (
#         _apply(X_train),
#         _apply(X_val),
#         _apply(X_test),
#         {"mean": means.tolist(), "std": stds.tolist()},
#     )

# def create_sequences(
#     X: np.ndarray,
#     y: np.ndarray,
#     seq_length: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     n = X.shape[0] - seq_length + 1
#     if n <= 0:
#         raise ValueError("Sequence length longer than data length.")
#     X_seq = np.stack([X[i:i+seq_length] for i in range(n)], 0)
#     y_seq = y[seq_length-1:]
#     return X_seq, y_seq

# # def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
# #     counts = Counter(y.tolist())
# #     N = float(len(y)); K = max(1, len(counts))
# #     return {int(cls): float(N / (K * cnt)) for cls, cnt in counts.items()}

# def compute_class_weights(y: np.ndarray, max_weight: float = 5.0) -> Dict[int, float]:
#     """
#     Inverse‑frequency weights with an upper cap to avoid early collapse.
#     weight_k = min( max_weight,  N / (K * n_k) )
#     """
#     counts = Counter(y.tolist())
#     N = float(len(y)); K = max(1, len(counts))
#     weights = {int(cls): float(N / (K * cnt)) for cls, cnt in counts.items()}
#     # cap
#     for k in list(weights.keys()):
#         if np.isfinite(weights[k]):
#             weights[k] = float(min(max_weight, weights[k]))
#         else:
#             weights[k] = 1.0
#     return weights


#!/usr/bin/env python3
"""
Utility helpers for the HFT pipeline

▪ load_features_labels  – parquet → (X_df, y) keeping only numeric/bool
▪ scale_features        – z‑score with optional skip columns
▪ create_sequences      – sliding window constructor
▪ compute_class_weights – inverse-frequency with cap/floor/power/normalize
"""
from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

NUMERIC_ONLY = ["number", "bool"]  # keep numeric + flags

def load_features_labels(
    parquet_file: str,
    label_col: str = "y",
    drop_cols: List[str] | str | None = "timestamp",
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(parquet_file)

    # keep order by time if available
    if drop_cols:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        for col in drop_cols:
            if col in df.columns:
                df = df.sort_values(col)
                break

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {parquet_file}")

    y = df[label_col].to_numpy(copy=True)

    cols_to_drop = [label_col] + ([*drop_cols] if drop_cols else [])
    feat_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # numeric/bool features only, preserve original order
    feat_df = feat_df.select_dtypes(include=NUMERIC_ONLY).copy()
    return feat_df, y

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    *,
    skip_indices: List[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    means = X_train.mean(0)
    stds  = X_train.std(0)
    stds[stds == 0] = 1e-8

    if skip_indices:
        for idx in skip_indices:
            if idx < len(means):
                means[idx] = 0.0
                stds[idx]  = 1.0

    def _apply(arr: np.ndarray | None):
        if arr is None:
            return None
        return ((arr - means) / stds).astype(np.float32)

    return (
        _apply(X_train),
        _apply(X_val),
        _apply(X_test),
        {"mean": means.tolist(), "std": stds.tolist()},
    )

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling window: X[i … i+L) → y[i+L‑1]."""
    n = X.shape[0] - seq_length + 1
    if n <= 0:
        raise ValueError("Sequence length longer than data length.")
    X_seq = np.stack([X[i:i+seq_length] for i in range(n)], 0)
    y_seq = y[seq_length-1:]
    return X_seq, y_seq

def compute_class_weights(
    y: np.ndarray,
    *,
    cap: float = 5.0,
    floor: float = 1.0,
    power: float = 1.0,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Inverse-frequency weights with safety cap.
      base: w_k = (N / (K * n_k)) ** power
      then: clip to [floor, cap]
      normalize so average weight ≈ 1 (optional)
    """
    counts = Counter(y.tolist())
    K = max(1, len(counts))
    N = float(len(y))
    w = {int(k): float((N / (K * cnt)) ** power) for k, cnt in counts.items()}
    # clip
    for k in w:
        w[k] = float(np.clip(w[k], floor, cap))
    if normalize:
        mean_w = float(np.mean(list(w.values())))
        for k in w:
            w[k] = w[k] / (mean_w if mean_w > 0 else 1.0)
    return w