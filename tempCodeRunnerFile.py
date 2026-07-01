"""
finguard_models.py
Shared model definitions and scoring logic.
Imported by both train.py and main.py (FastAPI).
"""

import numpy as np
import torch
import torch.nn as nn


class TransactionAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection on tabular transaction data.
    Trained only on normal transactions; fraud causes high reconstruction error.
    """
    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),          nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE. Higher = more anomalous."""
        with torch.no_grad():
            return ((x - self.forward(x)) ** 2).mean(dim=1)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
