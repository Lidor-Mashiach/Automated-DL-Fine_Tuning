# 🔮 Future Work

Planned extensions that are not part of the current version, listed in rough order of priority.

---

## 1. 🎮 Deep Reinforcement Learning (DRL)

The current framework is built around supervised learning (train/val loss, classification/regression). Adding DRL support would require:

- **Environment instead of "dataset"** — Gymnasium interface. Configure the env name in `main.py` (CartPole, FrozenLake, MountainCar, ...).
- **RL algorithms** — DQN, PPO, A2C, SAC. Each would get its own `configs/architectures/dqn.yaml`, `configs/architectures/ppo.yaml` with RL-specific parameters: `discount_factor` (γ), `gae_lambda`, `entropy_coefficient`, `replay_buffer_size`, `exploration_rate` (ε).
- **Different metrics** — instead of val_loss, track episode return, episode length, and exploration rate.
- **RL-aware diagnoses** — extend the Analyzer with failure patterns specific to RL: policy collapse, reward hacking, premature convergence, training instability.

The Orchestrator + Analyzer + Reporter architecture is still appropriate — only the Trainer and data_loaders need RL-specific replacements.

---

## 2. 🏛️ Additional Architectures

- **Autoencoder / VAE** — for representation learning.
- **Vision Transformer (ViT)** — a modern alternative to CNN.
- **GRU** — a lighter alternative to LSTM.
- **Graph Neural Networks** — would require PyG.

---

## 3. 🔎 Advanced Search Strategies

- **Hyperband / ASHA** — early stopping of weak trials (orders-of-magnitude speedup).
- **Population-Based Training (PBT)** — migrates hyperparameters between trials as they run.
- **BOHB** — combines Bayesian optimization with Hyperband.

---

## 4. 🎯 Multi-Objective Optimization

The system currently maximizes one scalar quality score. A future version could support Pareto optimization of multiple goals: accuracy + latency + model size + memory.

---

## 5. 🌐 Distributed Experiments

- Run on multiple GPUs concurrently with DistributedDataParallel.
- Queue-based scheduling of trials across multiple cluster nodes.

---

## 6. 💾 Checkpointing and Resume

Save the state of the best trial, and allow new trials to warm-start from it (fine-tuning a fine-tune). Also resume an interrupted run from where it stopped by serializing the ExperimentTree.

---

## 7. 🖥️ Live Dashboard

Real-time visualization (Streamlit / Gradio) with live curves, instead of just the final PNG.

---

## 🔗 Related Documents

- [`README.md`](README.md) — project overview
- [`SETUP_GUIDE.md`](SETUP_GUIDE.md) — current usage walkthrough
