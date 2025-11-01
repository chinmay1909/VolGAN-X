# src/generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VolGenConfig:
    """
    Configuration for VolGenerator.

    Args:
        input_dim: number of raw input features (default 2 for [T, K]; can be 3 if adding spot S)
        hidden_dim: width of hidden layers
        depth: number of hidden layers (not counting input/output)
        fourier_features: number of Fourier feature pairs per input dim (0 = off)
        fourier_sigma: stddev for random Fourier feature projection
        use_residual: use residual (pre-act) blocks
        dropout: dropout probability inside hidden layers
        activation: activation type: 'gelu' | 'relu' | 'silu' | 'leaky_relu'
        bounded_head: if True, enforce positive vols via softplus(+eps)
        vol_epsilon: small positive added by softplus head to keep vols > 0
    """
    input_dim: int = 2
    hidden_dim: int = 256
    depth: int = 5
    fourier_features: int = 16
    fourier_sigma: float = 10.0
    use_residual: bool = True
    dropout: float = 0.0
    activation: str = "silu"
    bounded_head: bool = True
    vol_epsilon: float = 1e-4


def _activation(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier features (RFF) encoder for continuous inputs.
    Maps x in R^d -> [sin(Bx), cos(Bx)] in R^{2*m}, where B ~ N(0, sigma^2).
    """
    def __init__(self, in_dim: int, m: int, sigma: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        self.register_buffer("B", torch.randn(in_dim, m) * sigma)

    @property
    def out_dim(self) -> int:
        return 2 * self.m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        proj = x @ self.B  # [N, m]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual MLP block: LN -> Act -> Linear -> Dropout -> LN -> Act -> Linear -> Skip
    """
    def __init__(self, dim: int, dropout: float, activation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = _activation(activation)
        self.lin1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = _activation(activation)
        self.lin2 = nn.Linear(dim, dim)

        # Kaiming init for stability
        nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.act1(h)
        h = self.lin1(h)
        h = self.drop(h)
        h = self.ln2(h)
        h = self.act2(h)
        h = self.lin2(h)
        return x + h


class MLPStack(nn.Module):
    """
    Plain MLP stack with optional residual blocks.
    """
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float, activation: str, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.act = _activation(activation)

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))

        if use_residual:
            self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout, activation) for _ in range(max(depth, 1))])
            self.proj = nn.Sequential(*layers)
        else:
            # plain MLP: [in -> hidden] + (depth-1)*[hidden->hidden]
            for _ in range(max(depth - 1, 0)):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
                layers.append(self.act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            h = self.proj(x)
            for blk in self.blocks:
                h = blk(h)
            return h
        else:
            return self.net(x)


class VolGenerator(nn.Module):
    """
    VolGAN-X Generator
    ------------------
    Maps continuous inputs (T, K[, S]) -> implied volatility.
    Designed for smooth 2D manifolds (surfaces) with optional Fourier features
    to capture high-frequency structure (e.g., skew/term ripples).

    Typical usage:
        cfg = VolGenConfig(input_dim=2, hidden_dim=256, depth=5, fourier_features=16)
        G = VolGenerator(cfg)
        vols = G(torch.tensor([[T, K], ...]))
    """
    def __init__(self, cfg: VolGenConfig = VolGenConfig()):
        super().__init__()
        self.cfg = cfg

        # Optional Fourier features front-end
        if cfg.fourier_features and cfg.fourier_features > 0:
            self.rff = RandomFourierFeatures(cfg.input_dim, cfg.fourier_features, cfg.fourier_sigma)
            feat_dim = self.rff.out_dim
        else:
            self.rff = None
            feat_dim = cfg.input_dim

        self.backbone = MLPStack(
            in_dim=feat_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_residual=cfg.use_residual
        )

        # Output head: map hidden -> 1 (implied vol)
        self.head = nn.Linear(cfg.hidden_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)

        self._bounded_head = cfg.bounded_head
        self._eps = cfg.vol_epsilon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess/encode inputs.
        Expects normalized inputs in [0,1] or similarly scaled (handled by dataset).
        """
        if self.rff is not None:
            return self.rff(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [B, input_dim] with columns (T, K[, S]).
               Inputs should already be numerically scaled/normalized by the data pipeline.

        Returns:
            vols: tensor of shape [B, 1], implied volatility (positive if bounded_head=True).
        """
        z = self.encode(x)
        h = self.backbone(z)
        out = self.head(h)

        if self._bounded_head:
            # Softplus ensures positivity (IV > 0); add tiny epsilon for numerical margin
            return F.softplus(out) + self._eps
        return out

    # ---------------- convenience helpers ---------------- #

    @torch.no_grad()
    def sample_grid(self, T: torch.Tensor, K: torch.Tensor, device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate surface on a T x K grid (assumes both are 1D tensors in [0,1] scaling).
        Returns:
            points: [N, 2] grid of (T, K)
            vols:   [N, 1] generated vols
        """
        device = device or next(self.parameters()).device
        TT, KK = torch.meshgrid(T.to(device), K.to(device), indexing="ij")
        pts = torch.stack([TT.reshape(-1), KK.reshape(-1)], dim=-1)
        vols = self.forward(pts)
        return pts.detach().cpu(), vols.detach().cpu()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
