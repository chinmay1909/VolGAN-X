# VolGAN-X  
*Agentic AI-Enhanced Generative Framework for Arbitrage-Free Implied Volatility Surfaces*

---

## 📘 Overview

**VolGAN-X** integrates **Generative Adversarial Networks (GANs)** with an **agentic reinforcement module** to generate **arbitrage-free implied volatility surfaces**.  
The model learns nonlinear option-market dynamics from historical data and adaptively stress-tests its own surfaces under shifting market regimes such as volatility clustering, term-structure twists, and liquidity shocks.

This repository represents an open academic exploration of **AI-driven quantitative finance** — blending ideas from **stochastic modeling, deep learning, and agentic reinforcement intelligence**.

---

## 🧩 Key Features

- 🎯 **Arbitrage-Free Volatility Generation**  
  Enforces convexity and monotonicity constraints to ensure risk-neutral consistency.

- 🧠 **Agentic Reinforcement Layer**  
  Adaptive module monitors market regimes and retrains the GAN for regime stability.

- ⏱️ **High-Frequency Compatible**  
  Designed for nanosecond-precision replay engines and time-series backtests.

- 📈 **Dynamic Market Stress Testing**  
  Recreates volatility smiles and term structures under extreme macro or liquidity events.

- 🧮 **Quantitative Evaluation Suite**  
  Includes arbitrage-check functions, smoothness metrics, and visual diagnostics.

---

## ⚙️ Architecture

            ┌──────────────────────────────┐
            │        Market Data            │
            │ (Strikes, Maturities, IVs)    │
            └─────────────┬────────────────┘
                          │
                    Data Pipeline
                          │
              ┌───────────▼───────────┐
              │      Generator G      │
              │ (Vol Surface Synth.)  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Discriminator D    │
              │ (Arb. Violation Test) │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │ Agentic Reinforcer R  │
              │ (Market Regime AI)    │
              └───────────────────────┘

## Project Structure 
```bash
VolGAN-X/
│
├── README.md — project overview, installation \& usage guide
├── requirements.txt — Python dependencies
├── .gitignore — standard ignore file
│
├── data/
│ ├── spx_options_sample.csv — sample options dataset
│ └── README.md — explains expected data structure
│
├── src/
│ ├── init.py
│ ├── generator.py — GAN-based volatility surface generator
│ ├── discriminator.py — adversarial discriminator network
│ ├── stress_module.py — agentic AI reinforcement \& regime detector
│ ├── trainer.py — training loop, losses, and optimization logic
│ ├── utils.py — dataset loader, plotting, and arbitrage checks
│ └── main.py — CLI entry point for training and evaluation
│
├── notebooks/
│ ├── exploratory.ipynb — dataset EDA and visualization
│ └── stress_tests.ipynb — regime-based stress-testing demo
│
└── results/
├── surfaces/ — generated implied volatility surfaces .npy/.png
└── logs/ — training logs and metrics .csv
```
