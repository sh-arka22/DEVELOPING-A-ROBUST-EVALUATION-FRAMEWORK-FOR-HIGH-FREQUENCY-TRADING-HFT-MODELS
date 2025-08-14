# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/strategies/ml_signal.py -->
"""
Backtrader strategy that loads a saved PyTorch model
---------------------------------------------------
• Expects a binary classification model (up/down).
• Thresholds can be changed via params.
"""
# src/strategies/ml_signal.py

import torch
from backtrader import Strategy, Indicator, Order
from pathlib import Path


class MLSignalStrategy(Strategy):
    params = dict(
        model_path="checkpoints/BTC_1sec/best.pt",
        seq_len=60,
        up_thresh=0.55,
        down_thresh=0.45,
        stake=1,
        cooldown=0,          # bars to wait after closing position
    )

    def __init__(self):
        self.model = torch.load(Path(self.p.model_path), map_location="cpu")
        self.model.eval()
        self.buffer = []

        self.last_action_bar = -9999

    def next(self):
        # build rolling window
        self.buffer.append([
            self.data.open[0], self.data.high[0],
            self.data.low[0],  self.data.close[0],
            self.data.volume[0],
        ])
        if len(self.buffer) < self.p.seq_len:
            return
        if len(self.buffer) > self.p.seq_len:
            self.buffer.pop(0)

        # Inference
        import numpy as np
        x = torch.tensor(np.array(self.buffer)[None, :, :]).float()
        with torch.no_grad():
            logits = self.model(x).squeeze().item()
            prob = torch.sigmoid(torch.tensor(logits)).item()

        if self.position:
            # already in trade – optional exit logic
            return

        if len(self) - self.last_action_bar < self.p.cooldown:
            return

        if prob >= self.p.up_thresh:
            self.buy(size=self.p.stake)
            self.last_action_bar = len(self)
        elif prob <= self.p.down_thresh:
            self.sell(size=self.p.stake)
            self.last_action_bar = len(self)
