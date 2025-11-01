# src/utils.py
from __future__ import annotations
from typing import Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -------------------- data -------------------- #

class VolDataset(Dataset):
    """
    Minimal dataset: expects columns ['maturity', 'strike', 'implied_vol'].
    Applies per-feature min-max normalization to inputs (T, K).
    """
    def __init__(self, df: pd.DataFrame):
        cols = ["maturity", "strike", "implied_vol"]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        df = df.dropna(subset=cols).copy()
        self.raw_T = torch.tensor(df["maturity"].values, dtype=torch.float32).unsqueeze(1)
        self.raw_K = torch.tensor(df["strike"].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df["implied_vol"].values, dtype=torch.float32).unsqueeze(1)

        # scale inputs for stable learning
        self.T_min, self.T_max = self.raw_T.min(), self.raw_T.max()
        self.K_min, self.K_max = self.raw_K.min(), self.raw_K.max()
        self.X = torch.cat([
            (self.raw_T - self.T_min) / (self.T_max - self.T_min + 1e-8),
            (self.raw_K - self.K_min) / (self.K_max - self.K_min + 1e-8),
        ], dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def make_loader(df: pd.DataFrame, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0) -> Tuple[VolDataset, DataLoader]:
    ds = VolDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return ds, dl


# -------------------- plotting -------------------- #

def plot_surface(points: np.ndarray, vols: np.ndarray, title: str = "Implied Vol Surface (scatter)") -> None:
    """
    3D scatter of surface points. 'points' shape [N,2] with columns (T_norm, K_norm); 'vols' shape [N,1].
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    T = points[:, 0]
    K = points[:, 1]
    V = vols.squeeze()
    ax.scatter(T, K, V, s=12)
    ax.set_xlabel("Maturity (norm)")
    ax.set_ylabel("Strike (norm)")
    ax.set_zlabel("Implied Vol")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# -------------------- arbitrage penalty (toy) -------------------- #

def butterfly_arbitrage_penalty(points: torch.Tensor, vols: torch.Tensor, k_bins: int = 8) -> torch.Tensor:
    """
    Encourage convexity along strike slices (butterfly arbitrage ↓).
    This is a lightweight proxy: sort by strike and penalize negative curvature.
    """
    if points.ndim != 2 or vols.ndim != 2:
        raise ValueError("Expected points [B,2], vols [B,1]")

    if points.size(0) < 5:
        return torch.tensor(0.0, device=points.device)

    # sort by K
    K = points[:, 1]
    idx = torch.argsort(K)
    v = vols[idx].squeeze(1)

    # finite-difference curvature on a sliding window
    curv = v[2:] - 2 * v[1:-1] + v[:-2]
    penalty = torch.relu(-curv).mean()
    return penalty


# -------------------- utils -------------------- #

def set_seed(seed: int = 1337) -> None:
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(epoch: int, G: torch.nn.Module, D: torch.nn.Module, out_dir: str = "results") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict()}, path)


def save_surface(points: np.ndarray, vols: np.ndarray, out_path: str = "results/surfaces/surface.npy") -> None:
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {"points": points, "vols": vols})
