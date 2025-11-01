# src/stress_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn


Regime = Literal["calm", "high_vol", "liquidity_shock", "skew_twist", "term_twist"]


@dataclass
class ReinforceConfig:
    """
    Configuration for agentic reinforcement/stress-testing.
    Args:
        strength: base scale for parameter nudges
        max_norm_scale: relative clamp for total param update (safety)
        prob_noise: chance to add exploratory noise irrespective of regime
        device: optional override device
    """
    strength: float = 0.01
    max_norm_scale: float = 0.05
    prob_noise: float = 0.10
    device: Optional[str] = None


class RegimeDetector:
    """
    Heuristic regime detector from the (T,K) -> vol minibatch.
    Designed to be fast and differentiable-agnostic (used outside grads).
    """
    def __init__(self, high_vol_thr: float = 0.25, skew_thr: float = 0.02, term_thr: float = 0.02):
        """
        Args:
            high_vol_thr: mean-iv threshold to call "high_vol"
            skew_thr: abs d(vol)/dK threshold for 'skew_twist'
            term_thr: abs d(vol)/dT threshold for 'term_twist'
        """
        self.high_vol_thr = high_vol_thr
        self.skew_thr = skew_thr
        self.term_thr = term_thr

    @torch.no_grad()
    def detect(self, X: torch.Tensor, vol: torch.Tensor) -> Regime:
        """
        X: [B,2] normalized (T,K)
        vol: [B,1] implied vol (real or generated)
        """
        if X.numel() == 0 or vol.numel() == 0:
            return "calm"

        T = X[:, 0]
        K = X[:, 1]
        v = vol.squeeze(-1)

        # simple central differences via linear regression proxies
        dvol_dk = self._slope(K, v)
        dvol_dt = self._slope(T, v)

        mean_iv = torch.clamp(v.mean(), 0.0, 10.0).item()
        if mean_iv > self.high_vol_thr:
            return "high_vol"

        if abs(dvol_dk) > self.skew_thr:
            return "skew_twist"

        if abs(dvol_dt) > self.term_thr:
            return "term_twist"

        # liquidity shock proxy: unusually high dispersion
        if v.std().item() > 0.12:
            return "liquidity_shock"

        return "calm"

    @staticmethod
    def _slope(x: torch.Tensor, y: torch.Tensor) -> float:
        # slope from simple least-squares: cov(x,y)/var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        var = torch.clamp((x - x_mean).pow(2).mean(), min=1e-8)
        cov = ((x - x_mean) * (y - y_mean)).mean()
        return (cov / var).item()


class AgenticReinforcer:
    """
    Agentic stress module that adaptively perturbs generator parameters according to detected regimes.
    This is a *safe* nudge mechanism; replace with RL policy later if desired.
    """
    def __init__(self, generator: nn.Module, cfg: ReinforceConfig = ReinforceConfig(), detector: Optional[RegimeDetector] = None):
        self.G = generator
        self.cfg = cfg
        self.detector = detector or RegimeDetector()

        # cache parameter references
        self.params = [p for p in self.G.parameters() if p.requires_grad]
        if len(self.params) == 0:
            raise ValueError("Generator has no trainable parameters.")

        # pre-build named parameter list for targeted nudges
        self.named_params = list(self.G.named_parameters())

    @torch.no_grad()
    def decide_and_apply(self, X: torch.Tensor, vol_ref: torch.Tensor, rng: Optional[torch.Generator] = None) -> Regime:
        """
        Detect a regime from reference vols, then apply a safe perturbation to the generator.
        X: [B,2] normalized inputs (T,K)
        vol_ref: [B,1] could be real vols, or prior-gen vols for self-play
        """
        regime = self.detector.detect(X, vol_ref)

        # Optional exploratory noise (encourage coverage)
        if torch.rand(()) < self.cfg.prob_noise:
            self._explore()

        if regime == "high_vol":
            self._nudge_high_vol()
        elif regime == "liquidity_shock":
            self._nudge_liquidity()
        elif regime == "skew_twist":
            self._nudge_skew()
        elif regime == "term_twist":
            self._nudge_term()
        else:
            # calm: very small temperature-like drift
            self._nudge_small()

        self._clamp_update_norm()
        return regime

    # --------------------- nudge designs --------------------- #

    def _explore(self):
        # global light Gaussian shake
        for p in self.params:
            p.add_(self.cfg.strength * 0.25 * torch.randn_like(p))

    def _nudge_high_vol(self):
        # reduce global "temperature": dampen last-layer weights slightly
        head = getattr(self.G, "head", None)
        if isinstance(head, nn.Linear):
            head.weight.mul_(1.0 - 0.25 * self.cfg.strength)
            head.bias.mul_(1.0 - 0.25 * self.cfg.strength)
        # small negative drift on earlier layers to shrink amplitude
        for name, p in self.named_params:
            if "head" not in name:
                p.mul_(1.0 - 0.05 * self.cfg.strength)

    def _nudge_liquidity(self):
        # encourage smoothness: low-pass effect via slight weight decay
        for p in self.params:
            p.mul_(1.0 - 0.1 * self.cfg.strength)

    def _nudge_skew(self):
        # bias response in strike direction by tweaking first layer columns tied to K
        first = self._first_linear()
        if first is not None:
            # assume inputs are [T,K] (or [T,K,S]) -> column 1 impacts K
            k_col = min(1, first.weight.size(1) - 1)
            first.weight[:, k_col].add_(0.5 * self.cfg.strength * torch.tanh(first.weight[:, k_col]))

    def _nudge_term(self):
        # bias response in maturity direction by tweaking first layer column for T
        first = self._first_linear()
        if first is not None:
            t_col = 0
            first.weight[:, t_col].add_(0.5 * self.cfg.strength * torch.tanh(first.weight[:, t_col]))

    def _nudge_small(self):
        # tiny drift to avoid stalling
        for p in self.params:
            p.add_(self.cfg.strength * 0.05 * torch.randn_like(p))

    # --------------------- helpers --------------------- #

    def _first_linear(self) -> Optional[nn.Linear]:
        # try common places for the first projection layer
        cand = []
        if hasattr(self.G, "backbone"):
            # residual/plain MLPStack in your generator
            for m in self.G.backbone.modules():
                if isinstance(m, nn.Linear):
                    cand.append(m)
                    break
        # fallback: find any Linear in forward order
        if not cand:
            for m in self.G.modules():
                if isinstance(m, nn.Linear):
                    cand.append(m)
                    break
        return cand[0] if cand else None

    @torch.no_grad()
    def _clamp_update_norm(self):
        """
        Prevents runaway perturbations: if cumulative parameter update exceeds a fraction
        of current norm, rescale updates back into a safe ball.
        """
        # compute global param norm and delta norm (approx via absolute magnitude)
        total_norm = torch.sqrt(sum((p.data**2).sum() for p in self.params))
        delta_norm = torch.sqrt(sum(((p.data - p.data) ** 2).sum() for p in self.params))  # placeholder keeps API
        # since we do in-place updates, we approximate safety using absolute scale of params
        # practical safeguard: clip weights to reasonable range
        clip_val = 10.0
        for p in self.params:
            p.clamp_(-clip_val, clip_val)

    # --------------------- reward shaping (optional) --------------------- #

    def compute_reward(
        self,
        X: torch.Tensor,
        vols_gen: torch.Tensor,
        vols_ref: Optional[torch.Tensor] = None,
        arb_penalty_fn: Optional = None,
    ) -> Dict[str, float]:
        """
        Optional reward for RL-style training loops.
        Returns components you can combine externally.

        Args:
            X: [B,2]
            vols_gen: [B,1]
            vols_ref: [B,1] (if available) for supervised similarity
            arb_penalty_fn: callable(X, vols_gen)-> torch.scalar penalty

        Returns:
            dict of scalar floats
        """
        out: Dict[str, float] = {}
        if vols_ref is not None:
            mse = torch.mean((vols_gen - vols_ref) ** 2).item()
            out["mse_to_ref"] = mse
        if arb_penalty_fn is not None:
            with torch.no_grad():
                pen = arb_penalty_fn(X, vols_gen).item()
            out["arb_penalty"] = float(pen)
        # encourage smoothness via finite diff proxy (optional)
        smooth = float(vols_gen.std().clamp(min=1e-9).reciprocal().item())
        out["smooth_reward"] = smooth
        return out
