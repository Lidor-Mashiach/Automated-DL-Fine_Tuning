# 🧠 AutoTune-NN

**Automated fine-tuning framework for neural networks.**

The framework runs a series of experiments, analyzes what happened in each one (failed to learn / learned too fast / peaked and dropped / converged), draws conclusions, and modifies parameters for the next experiment — all without user intervention.

---

## ✨ Key Features

- 🏗️ **5 architectures out of the box** — MLP, CNN, RNN, LSTM, Transformer.
- 🧪 **Smart Analyzer** — diagnoses training curves and proposes prioritized actions.
- 🌳 **FTTS (Fine-Tuning Tree Search)** — a tree-based search with backtracking and adaptive step sizes, fully explainable in the final report.
- 🔄 **Multiple search strategies** — FTTS (default), Bayesian (Optuna), Grid search.
- 📋 **Per-architecture and per-strategy configs** — YAML files with thematic sections, docstrings, and no magic.
- 📝 **Unified text report** — a single `report.txt` documenting every trial, its parent, its rationale, and its diagnosis.
- 🖥️ **Works on GPU, CPU, and SLURM** — automatic device fallback and non-blocking plots.

---

## 📂 Project Structure

```
autotune_nn/
├── main.py                 Entry point - top-level selections only.
├── requirements.txt        Python dependencies with inline notes.
├── README.md               This file.
├── FUTURE_WORK.md          Planned extensions (DRL, ViT, etc.).
│
├── configs/                📁 See configs/README.md
│   ├── architectures/      One YAML per architecture.
│   └── strategies/         One YAML per search strategy.
│
├── core/                   📁 See core/README.md
│   └── (Orchestrator, Analyzer, Trainer, quality scorer, ...)
│
├── models/                 📁 See models/README.md
├── data_loaders/           📁 See data_loaders/README.md
├── search_strategies/      📁 See search_strategies/README.md
├── reporting/              📁 See reporting/README.md
│
├── Data/                   📁 Your datasets go here (local mode).
├── experiments/            Auto-created per run.
└── slurm/                  sbatch template for HPC clusters.
```

Each subdirectory has its own README with deeper detail. This document stays high-level.

---

## 🚀 Getting Started

### 📦 First-Time Setup

> ⚠️ **Do this once**, before the very first run.

**1. Create a virtual environment** *(recommended)*

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> 💡 **GPU users:** If you want GPU acceleration, install the CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) **before** `pip install -r requirements.txt`.

---

### ⚙️ Configure Your Run

Open `main.py`. Only a few top-level selections live there:

| Setting | What it controls |
|---|---|
| `ARCHITECTURE` | `mlp`, `cnn`, `rnn`, `lstm`, or `transformer` |
| `SEARCH_STRATEGY` | `ftts` (default), `bayesian`, or `grid` |
| `TASK_TYPE` | `classification` or `regression` |
| `DATASET_MODE` | `local` (your file) or `imported` (MNIST, Iris, etc.) |
| `DEVICE` | `auto`, `gpu`, or `cpu` |

Every other setting (parameter ranges, stopping conditions, quality weights, etc.) lives in the YAML files under `configs/`.

---

### 📁 Prepare Your Dataset

#### Option A: `DATASET_MODE = "local"`

Place your file in the `Data/` folder:

- **Tabular / text:** one CSV at `Data/Data-Set.csv`.
- **Image:** an ImageFolder at `Data/Data-Set/` (one subfolder per class).

On the **first run**, the framework automatically creates:

- `Data/Train_set.csv` (`TRAIN_PCT` of the data — used to train the model)
- `Data/Val_set.csv` (`VAL_PCT` — used inside the tuning loop for early stopping and scoring)
- `Data/Test_set.csv` (`TEST_PCT` — held out, untouched during tuning)

Subsequent runs reuse these files for reproducibility. To re-split, delete them.

You control:
- `FEATURE_COLUMNS` — list of column names, or `None` for all except the label.
- `LABEL_COLUMN` — column name, or `None` for the last column.
- `TRAIN_PCT` / `VAL_PCT` / `TEST_PCT` — must sum to 1.0.

#### Option B: `DATASET_MODE = "imported"`

Pick a well-known dataset:

- **Image:** `mnist`, `fashion_mnist`, `cifar10`, `cifar100`
- **Tabular:** `iris`, `wine`, `breast_cancer`, `digits`

These are downloaded automatically to `~/.cache/autotune_nn/` — not into `Data/`.

---

### ▶️ Run It

```bash
python main.py
```

---

## 📤 Output

For each run, a new folder is created under `experiments/`:

```
experiments/20260421_143022_mlp_ftts/
├── report.txt              Unified, human-readable trial log (appended live).
└── best_trial.png          Training curves of the best trial (updates live).
```

### 📝 Inside `report.txt`

For every trial:

- 🔗 **Parent** — which trial this one descends from.
- 💡 **Rationale** — why these parameters were chosen.
- ⚙️ **Parameters** — grouped by section (architecture / methods / optimization / training).
- 📈 **Results** — raw best, smoothed best, training duration.
- 🎚️ **Quality** — total score + 4 components (best_metric / stability / convergence_speed / generalization_gap).
- 🔍 **Diagnosis** — the Analyzer's verdict and observations.
- 🎬 **Actions for next trial** — prioritized suggestions with reasons.
- 📝 **Conclusion** — one-line takeaway.

The final block is a run summary: total trials, best trial, stop reason, and best metric.

---

## 🔍 Search Strategies

The search strategy is **how** the system picks hyperparameters for each new trial.

### FTTS (default) 🌳

**Fine-Tuning Tree Search** — a tree of experiments where each child is created by applying an Analyzer-suggested Action to a parent's hyperparameters.

- Actions have priorities assigned by the Analyzer.
- Priority queue ranks pending children by `parent_quality × action_priority`.
- Adaptive step sizes: successful directions accelerate, failed directions slow down.
- **Fully explainable**: every parameter change in the report has a reason.

See `search_strategies/README.md` for the algorithm details.

### Bayesian (Optuna TPE) 📊

Uses Optuna's Tree-structured Parzen Estimator. Statistically effective but less interpretable than FTTS. The Analyzer still runs and its diagnosis appears in the report — Optuna alone decides the next parameters.

### Grid Search 🔲

Exhaustive. Best for small search spaces.

---

## 🖥️ RAM and Parallelism

The `max_parallel_experiments` setting (in `configs/strategies/*.yaml`) controls how many trials run concurrently.

### Rough RAM estimates

| Dataset | Architecture | Threads | RAM |
|---|---|---|---|
| CSV, 1K rows, 10 features | MLP | 1 | ~2 GB |
| CSV, 1K rows, 10 features | MLP | 4 | ~3 GB |
| MNIST (60K images) | CNN | 1 | ~4 GB |
| MNIST (60K images) | CNN | 4 | ~8 GB |
| CIFAR-10 | CNN | 4 | ~12 GB |
| NLP, 100K texts | Transformer | 2 | ~16 GB |
| NLP, 100K texts | Transformer | 4 | ~28 GB |

### Guidelines

- 🖥️ **GPU runs**: keep `max_parallel_experiments: 1`. Multiple trials on one GPU fight for memory and hurt throughput.
- 💻 **CPU with small dataset**: 2-4 threads often works well.
- 🏋️ **CPU with large dataset**: scale to available RAM.
- 🛡️ **Thread safety**: the experiment tree uses internal locks; no conflicts between threads.

---

## 🖥️ Running on SLURM

A template is provided at `slurm/run.sbatch`.

**1.** Edit the top of the file to match your cluster (partition, time, GPUs).

**2.** Submit:
```bash
sbatch slurm/run.sbatch
```

**3.** Monitor:
```bash
squeue -u $USER
```

Logs go to `slurm_logs/<jobid>_<jobname>.out`.

---

## 🔮 Future Work

See [`FUTURE_WORK.md`](FUTURE_WORK.md) for planned extensions, including:

- Deep Reinforcement Learning (DQN / PPO / SAC)
- Additional architectures (ViT, GRU, Autoencoders)
- Advanced search strategies (Hyperband, PBT)
- Multi-objective optimization

---

## ❓ Common Questions

**Q: What happens if I press Ctrl+C mid-run?**
The `report.txt` is written after every trial, so you keep everything done so far. The best trial's plot is also saved live.

**Q: Why is `STOP_TARGET_ACCURACY` in the strategy YAML, not in `main.py`?**
`main.py` holds high-level choices (which architecture, which strategy). Numeric knobs belong in the YAML config of the strategy you chose, so settings don't get tangled across strategies.

**Q: How do I force a specific learning rate?**
In `configs/architectures/<arch>.yaml`, find `learning_rate` and set `initial_value` to your preferred starting point. You can optionally narrow `range` too. The Analyzer will still explore from there.
