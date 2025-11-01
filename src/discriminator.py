# src/discriminator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiscConfig:
    """
    Discriminator configuration.

    Args:
        input_dim: number of input features (default 3 for [T, K, vol])
        hidden_dim: width of hidden layers
        depth: number of hidden layers (not counting input/output)
        use_residual: enable pre-activation residual blocks
        dropout: dropout probability
        activation: 'relu' | 'silu' | 'gelu' | 'leaky_relu'
        spectral_norm: apply spectral norm for Lipschitz control (stability)
        use_minibatch_std: append minibatch stddev feature for mode-collapse resistance
    """
    input_dim: int = 3
    hidden_dim: int = 256
    depth: int = 5
    use_residual: bool = True
    dropout: float = 0.0
    activation: str = "leaky_relu"
    spectral_norm: bool = False
    use_minibatch_std: bool = True


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(layer) if use_sn else layer


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with LayerNorm.
    """
    def __init__(self, dim: int, dropout: float, activation: str, spectral_norm: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = _act(activation)
        self.fc1 = _maybe_sn(nn.Linear(dim, dim), spectral_norm)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = _act(activation)
        self.fc2 = _maybe_sn(nn.Linear(dim, dim), spectral_norm)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.act1(h)
        h = self.fc1(h)
        h = self.drop(h)
        h = self.ln2(h)
        h = self.act2(h)
        h = self.fc2(h)
        return x + h


class MLPBackbone(nn.Module):
    """
    Input projection + (residual blocks OR plain MLP).
    """
    def __init__(self, in_dim: int, hidden_dim: int, depth: int,
                 dropout: float, activation: str, use_residual: bool, spectral_norm: bool):
        super().__init__()
        self.use_residual = use_residual
        self.act = _act(activation)

        proj = _maybe_sn(nn.Linear(in_dim, hidden_dim), spectral_norm)
        nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
        self.proj = proj

        if use_residual:
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout, activation, spectral_norm)
                for _ in range(max(depth, 1))
            ])
        else:
            layers = []
            for _ in range(max(depth - 1, 0)):
                lin = _maybe_sn(nn.Linear(hidden_dim, hidden_dim), spectral_norm)
                nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                layers += [lin, self.act]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        if self.use_residual:
            for blk in self.blocks:
                h = blk(h)
        else:
            h = self.net(h) if len(self.net) > 0 else h
        return h


class MinibatchStdDev(nn.Module):
    """
    Appends one channel: per-feature scalar stddev across the minibatch.
    Helps the D detect lack of diversity in fakes (classic GAN trick).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        if x.size(0) < 2:
            # Not enough samples to compute a stable std; append zeros
            std_feat = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        else:
            std = torch.sqrt(torch.var(x, dim=0, unbiased=False) + self.eps)  # [F]
            mean_std = std.mean().unsqueeze(0).repeat(x.size(0), 1)           # [B,1]
            std_feat = mean_std
        return torch.cat([x, std_feat], dim=-1)


class VolDiscriminator(nn.Module):
    """
    VolGAN-X Discriminator
    ----------------------
    Input: concatenated [T, K, vol] (normalized).
    Output: probability in (0,1) via Sigmoid (compatible with BCE in your trainer).
    """
    def __init__(self, cfg: DiscConfig = DiscConfig()):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.input_dim
        self.use_mstd = cfg.use_minibatch_std

        # Backbone
        self.backbone = MLPBackbone(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_residual=cfg.use_residual,
            spectral_norm=cfg.spectral_norm
        )

        # Optional minibatch stddev feature (adds +1 to feature dim before head)
        head_in = cfg.hidden_dim + (1 if self.use_mstd else 0)
        head = nn.Linear(head_in, 1)
        if cfg.spectral_norm:
            head = nn.utils.spectral_norm(head)
        nn.init.xavier_uniform_(head.weight)
        self.head = head

        self.mstd = MinibatchStdDev() if self.use_mstd else nn.Identity()
        self.out_act = nn.Sigmoid()  # BCE-compatible

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3] -> (T, K, vol), normalized by the data pipeline/trainer.
        returns: [B, 1] in (0,1)
        """
        h = self.backbone(x)
        h = self.mstd(h)
        logits = self.head(h)
        return self.out_act(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
