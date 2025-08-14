# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/loader.py -->
"""
Unified DataLoader for HFT datasets
----------------------------------
• YAML‑driven, leak‑free scaling, PyTorch/TensorFlow selectable.
"""

from __future__ import annotations
import json, yaml, os, joblib, math
import numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Union, Optional

# -----------------------------------------------------------------------------
# Optional backend detection
# -----------------------------------------------------------------------------
_BACKEND = None
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _BACKEND = "torch"
except ModuleNotFoundError:
    try:
        import tensorflow as tf
        _BACKEND = "tf"
    except ModuleNotFoundError:
        raise RuntimeError("Install torch or tensorflow before using DataLoader.")

# -----------------------------------------------------------------------------
# Scalers
# -----------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
def _build_scaler(kind: str):
    return {
        "standard": StandardScaler,
        "minmax":   MinMaxScaler,
        "robust":   RobustScaler,
        "none":     lambda: None,
    }[kind]()

# -----------------------------------------------------------------------------
# Sequence builder
# -----------------------------------------------------------------------------
def build_sequences(features: np.ndarray, labels: np.ndarray, seq: int):
    X, y = [], []
    for i in range(seq, len(features)):
        X.append(features[i-seq:i]); y.append(labels[i])
    return np.stack(X), np.stack(y)

# -----------------------------------------------------------------------------
# Backend‑specific dataset wrappers
# -----------------------------------------------------------------------------
class _TorchSeqDataset(_TorchDataset):
    def __init__(self,X,y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):  return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

def _tf_dataset(X,y,batch=32):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# -----------------------------------------------------------------------------
# Main factory
# -----------------------------------------------------------------------------
class DataLoaderFactory:
    def __init__(self, dataset_yaml, splits_json, split="train", backend=None):
        self.cfg     = yaml.safe_load(open(dataset_yaml))
        self.splits  = json.load(open(splits_json))
        self.split   = split.lower()
        self.backend = backend or _BACKEND
        if self.backend not in ("torch","tf"):
            raise ValueError("backend must be 'torch' or 'tf'.")

        self._load_raw(); self._apply_split(); self._scale_fit_or_load()
        X,y = build_sequences(self._X,self._y.values,self.cfg["sequence_length"])
        self.X,self.y = X,y

    # ------------------------------------------------------------------
    def make(self,batch_size=32):
        return (_TorchSeqDataset if self.backend=="torch" else _tf_dataset)(self.X,self.y) \
               if self.backend=="torch" else _tf_dataset(self.X,self.y,batch_size)

    # ------------------------------------------------------------------
    def _load_raw(self):
        root=Path(self.cfg["data_root"])
        df  = pd.read_parquet(root/self.cfg["features_file"])
        labels = np.load(root/self.cfg["labels_file"])
        self._df = df.sort_index(); self._y = pd.Series(labels,index=df.index)

    def _apply_split(self):
        start,end = self.splits[self.cfg["dataset_name"]][self.split]
        mask = (self._df.index>=pd.Timestamp(start))&(self._df.index<=pd.Timestamp(end))
        self._df,self._y = self._df.loc[mask],self._y.loc[mask]

    def _scale_fit_or_load(self):
        sc_cfg=self.cfg.get("scaler",{"type":"none"}); typ=sc_cfg["type"]
        if typ=="none": self._X=self._df.values.astype(np.float32); return
        path=Path(sc_cfg.get("save_path",""))
        if self.split=="train":
            scaler=_build_scaler(typ); scaler.fit(self._df); path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(scaler,path)
        else:
            scaler=joblib.load(path)
        self._X=scaler.transform(self._df).astype(np.float32)
