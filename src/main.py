# src/main.py
from __future__ import annotations
import argparse
import os
import torch
import numpy as np
import pandas as pd

# Robust imports whether called as "python src/main.py" or from root
try:
    from .generator import VolGenerator, VolGenConfig
    from .discriminator import VolDiscriminator, DiscConfig
    from .stress_module import AgenticReinforcer, ReinforceConfig, RegimeDetector
    from .trainer import GANTrainer, TrainerConfig
    from .utils import (
        load_csv, make_loader, save_checkpoint, set_seed,
        butterfly_arbitrage_penalty, save_surface, plot_surface
    )
except ImportError:
    from generator import VolGenerator, VolGenConfig
    from discriminator import VolDiscriminator, DiscConfig
    from stress_module import AgenticReinforcer, ReinforceConfig, RegimeDetector
    from trainer import GANTrainer, TrainerConfig
    from utils import (
        load_csv, make_loader, save_checkpoint, set_seed,
        butterfly_arbitrage_penalty, save_surface, plot_surface
    )


def parse_args():
    p = argparse.ArgumentParser(description="VolGAN-X training entrypoint")
    p.add_argument("--dataset", type=str, default="data/spx_options_sample.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--lambda_arb", type=float, default=1.0)
    p.add_argument("--fourier_features", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_grid", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---------- data ----------
    df = load_csv(args.dataset)
    ds, dl = make_loader(df, batch_size=args.batch_size, shuffle=True)

    # ---------- models ----------
    g_cfg = VolGenConfig(
        input_dim=2,
        hidden_dim=args.hidden,
        depth=args.depth,
        fourier_features=args.fourier_features,
        bounded_head=True
    )
    d_cfg = DiscConfig(input_dim=3, hidden_dim=args.hidden, depth=args.depth)

    G = VolGenerator(g_cfg)
    D = VolDiscriminator(d_cfg)

    # ---------- trainer ----------
    t_cfg = TrainerConfig(
        lr_g=args.lr_g, lr_d=args.lr_d,
        epochs=args.epochs,
        lambda_arb=args.lambda_arb,
        mixed_precision=args.mixed_precision,
    )
    trainer = GANTrainer(G, D, arb_penalty_fn=butterfly_arbitrage_penalty, cfg=t_cfg)

    # ---------- agentic module ----------
    reinforcer = AgenticReinforcer(G, ReinforceConfig(strength=0.01), RegimeDetector())

    def agentic_hook(X_batch=None, y_ref=None):
        # uses the current batch to detect regime & safely nudge generator
        if X_batch is not None and y_ref is not None:
            return reinforcer.decide_and_apply(X_batch, y_ref)
        return "calm"

    # ---------- callbacks ----------
    def ckpt_cb(epoch: int, Gm, Dm):
        save_checkpoint(epoch, Gm, Dm, out_dir="results")

    def log_cb(metrics: dict):
        # you can extend this to wandb/tensorboard
        print({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})

    # ---------- train ----------
    trainer.train(dl, agentic_hook=agentic_hook, save_ckpt_fn=ckpt_cb, log_fn=log_cb)

    # ---------- sample & visualize ----------
    with torch.no_grad():
        # create a 50x50 normalized grid over (T,K) in [0,1]
        T = torch.linspace(0.0, 1.0, 50)
        K = torch.linspace(0.0, 1.0, 50)
        pts, vols = G.sample_grid(T, K)

    os.makedirs("results/surfaces", exist_ok=True)
    save_surface(pts.numpy(), vols.numpy(), "results/surfaces/surface_final.npy")

    if args.plot:
        plot_surface(pts.numpy(), vols.numpy(), title="VolGAN-X Generated Surface")

    if args.save_grid:
        np.save("results/surfaces/grid_points.npy", {"T": T.numpy(), "K": K.numpy()})
        print("Saved grid to results/surfaces/grid_points.npy")

    print("✅ Training complete. Checkpoints in results/, surface in results/surfaces/")

if __name__ == "__main__":
    main()
