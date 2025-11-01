# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 200
    lambda_arb: float = 1.0           # weight for arbitrage penalty
    log_every: int = 25               # steps
    ckpt_every: int = 100             # epochs
    device: Optional[str] = None
    mixed_precision: bool = False     # optional AMP


class GANTrainer:
    """
    BCE-GAN trainer with optional arbitrage penalty and an agentic hook.
    - D input: concat([T, K, vol])
    - G input: [T, K]  -> vol_hat
    """
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        arb_penalty_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        cfg: TrainerConfig = TrainerConfig(),
    ):
        self.G = generator
        self.D = discriminator
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)

        self.opt_g = Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_d = Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
        self.bce = nn.BCELoss()
        self.arb_penalty_fn = arb_penalty_fn

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

    def _step_discriminator(self, X: torch.Tensor, y_real: torch.Tensor) -> Dict[str, float]:
        B = X.size(0)
        real_in = torch.cat([X, y_real], dim=1)  # [B,3]
        real_lbl = torch.ones((B, 1), device=self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
            pred_real = self.D(real_in)
            loss_real = self.bce(pred_real, real_lbl)

            with torch.no_grad():
                y_fake = self.G(X)
            fake_in = torch.cat([X, y_fake], dim=1)
            fake_lbl = torch.zeros((B, 1), device=self.device)
            pred_fake = self.D(fake_in)
            loss_fake = self.bce(pred_fake, fake_lbl)

            loss_d = loss_real + loss_fake

        self.opt_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_d).backward()
        self.scaler.step(self.opt_d)
        return {"D": float(loss_d.detach().cpu())}

    def _step_generator(self, X: torch.Tensor) -> Dict[str, float]:
        B = X.size(0)
        real_lbl = torch.ones((B, 1), device=self.device)

        self.opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
            y_hat = self.G(X)
            fake_in = torch.cat([X, y_hat], dim=1)     # [B,3]
            pred_fake = self.D(fake_in)

            loss_g_adv = self.bce(pred_fake, real_lbl)

            loss_arb = torch.tensor(0.0, device=self.device)
            if self.arb_penalty_fn is not None:
                # encourage convexity / no-butterfly violations ↓
                loss_arb = self.arb_penalty_fn(X, y_hat)

            loss_g = loss_g_adv + self.cfg.lambda_arb * loss_arb

        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()

        return {
            "G": float(loss_g.detach().cpu()),
            "G_adv": float(loss_g_adv.detach().cpu()),
            "Arb": float(loss_arb.detach().cpu()),
        }

    def train(
        self,
        dataloader: DataLoader,
        agentic_hook: Optional[Callable[[Optional[torch.Tensor], Optional[torch.Tensor]], str]] = None,
        save_ckpt_fn: Optional[Callable[[int, nn.Module, nn.Module], None]] = None,
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        step = 0
        for epoch in range(1, self.cfg.epochs + 1):
            running = {"D": 0.0, "G": 0.0, "G_adv": 0.0, "Arb": 0.0}
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for X, y in pbar:
                X = X.to(self.device)  # [B,2]
                y = y.to(self.device)  # [B,1]

                # --- D update
                d_out = self._step_discriminator(X, y)
                # --- agentic move (optional) based on current batch
                if agentic_hook is not None:
                    try:
                        _regime = agentic_hook(X, y)
                    except TypeError:
                        _regime = agentic_hook()  # backwards-compatible signature

                # --- G update
                g_out = self._step_generator(X)

                # --- logging
                step += 1
                for k in running:
                    running[k] += (d_out.get(k, 0.0) + g_out.get(k, 0.0) if k == "D" else g_out.get(k, 0.0))

                if step % self.cfg.log_every == 0:
                    log_payload = {
                        "step": step,
                        "epoch": epoch,
                        "loss_D": d_out["D"],
                        "loss_G": g_out["G"],
                        "loss_G_adv": g_out["G_adv"],
                        "arb_penalty": g_out["Arb"],
                    }
                    pbar.set_postfix({k: f"{v:.3f}" for k, v in log_payload.items() if k.startswith("loss") or k == "arb_penalty"})
                    if log_fn:
                        log_fn(log_payload)

            # --- checkpoint
            if save_ckpt_fn and (epoch % self.cfg.ckpt_every == 0 or epoch == self.cfg.epochs):
                save_ckpt_fn(epoch, self.G, self.D)
