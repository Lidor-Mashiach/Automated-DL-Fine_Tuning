# 🧠 AutoTune-NN

**Automated fine-tuning framework for neural networks.**

The framework runs a series of experiments, analyzes what happened in each one (failed to learn / learned too fast / peaked and dropped / converged), draws conclusions, and modifies parameters for the next experiment — all without user intervention.

---

## ✨ Key Features

- 🏗️ **5 architectures out of the box** — MLP, CNN, RNN, LSTM, Transformer.
- 🎵 **Language modeling mode** — train an LSTM on lyrics (Word2Vec + MIDI features), generate text with 4 sampling strategies, and probe melody influence.
- 🧪 **Smart Analyzer** — diagnoses training curves and proposes prioritized actions.
- 🌳 **FTTS (Fine-Tuning Tree Search)** — tree-based search with backtracking, adaptive step sizes, and **DAG deduplication** (skips already-explored hyperparameter combinations).
- 🎯 **Context-aware layer shape selection (MLP)** — Analyzer picks the right pattern (uniform / funnel / pyramid / hourglass / bottleneck) based on the verdict; user doesn't have to choose manually.
- 🔄 **Multiple search strategies** — FTTS (default), Bayesian (Optuna), Grid search.
- 🚀 **Modern training optimizations** — Mixed Precision (fp16/AMP), Gradient Accumulation, Focal Loss with auto class-imbalance detection, Label Smoothing, **TensorBoard logging** (optional).
- 🎨 **Advanced augmentation** — Mixup + CutMix + CutOut for images; Token Dropout / Word Shuffle / N-gram Shuffle for text; Stochastic Depth for deep Transformers.
- 📋 **Per-architecture and per-strategy configs** — YAML files with thematic sections, docstrings, and no magic.
- 📝 **Unified text report** — a single `report.txt` documenting every trial, its parent, its rationale, and its diagnosis.
- 📦 **Standalone deliverable** — generates a self-contained `model.py` for the best trial.
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

> 📖 **For a step-by-step walkthrough, see [`SETUP_GUIDE.md`](SETUP_GUIDE.md).** It covers installation, data preparation, and configuration in order. The summary below is a condensed version for those already familiar with similar tools.

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
# Use the settings defined at the top of main.py:
python main.py

# Override any setting from the CLI:
python main.py --architecture lstm --run_name my_experiment

# Language modeling (lyrics generation, Assignment 3):
python main.py \
  --architecture lstm --task_type language_modeling --data_type lyrics \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --midi_dir ./midi_files --midi_variant simple \
  --word2vec_path ./GoogleNews-vectors-negative300.bin

# Generate-only (skip training, use existing checkpoint):
python main.py --mode generate \
  --checkpoint ./experiments/<run>/final/model_checkpoint.pt \
  --initial_words love the morning \
  --sampling_strategy nucleus --sampling_top_p 0.9
```

See `python main.py --help` for the full list of flags. Every constant at the top of `main.py` has a corresponding `--<lower_name>` argument.

---

## 📤 Output

Each run creates a folder named `<RUN_NAME>_<strategy>_<dataset>_<ranking>/` under `experiments/`:

```
experiments/experiment1_ftts_iris_qualityscore/
├── tuning/
│   ├── report.txt           Unified, human-readable trial log (appended live).
│   └── best_trial.png       Training curves of the best tuning trial.
└── final/
    ├── model.py             Standalone code of the chosen architecture.
    ├── final_training.png   Final refit training curves (Train+Val).
    └── test_evaluation.txt  Final evaluation results on Test_set.
```

> 📦 **`final/model.py` is the actual deliverable** — a standalone script that rebuilds the architecture, retrains it on Train+Val combined, and evaluates on Test_set. You can take this file alone to another project; it has no dependency on AutoTune-NN itself.

> 🔄 **The Final Test Evaluation phase** runs automatically after tuning. The best trial's hyperparameters are used to train a fresh model on Train+Val combined ("refit on full training set" — standard ML practice), and that model is evaluated on the held-out Test_set. This is the only time Test_set is touched.

### 📝 Inside `tuning/report.txt`

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

**Q: How do I get started?**
Read [`SETUP_GUIDE.md`](SETUP_GUIDE.md) — it walks you through installation, data preparation, and configuration in order.

**Q: What happens if I press Ctrl+C mid-run?**
The `report.txt` is written after every trial, so you keep everything done so far. The best trial's plot is also saved live.

**Q: How do I configure when the run stops?**
Stopping conditions live in `configs/strategies/<strategy>.yaml` under `stopping:`. There are four conditions; the run stops as soon as **any** one fires:

```yaml
stopping:
  max_trials: 50              # cap on number of trials
  time_limit_minutes: null    # wall-clock budget in minutes
  target_accuracy: null       # stop when smoothed val accuracy reaches this
  convergence_patience: 40    # stop after N trials with no improvement
```

To **disable** a condition, set it to `null`. **At least one must be active** (the orchestrator validates this at startup). See [`SETUP_GUIDE.md` §11](SETUP_GUIDE.md#11-strategy-configuration) for recommended combinations and detailed value ranges.

**Q: How do I force a specific learning rate?**
In the relevant `configs/architectures/<arch>.yaml`, set the `learning_rate` parameter's `range: [your_value, your_value]` (single value). The same approach works for any continuous parameter. For discrete parameters, set `choices: ["your_value"]`.

**Q: How does AutoTune-NN choose the MLP layer shape (uniform / funnel / bottleneck / etc.)?**
Automatically — based on the diagnosis. If the model overfits, the Analyzer suggests `bottleneck` or `funnel` (compress information). If it underfits, it suggests `hourglass` (richer middle). You don't pick manually. To restrict the search to one shape, set `choices: ["bottleneck"]` in `mlp.yaml`.

**Q: What's the `optimizer_name` parameter?**
Previously called `name` — renamed for clarity. It's the optimizer choice (`adam`, `adamw`, `sgd`, `rmsprop`). The Adam beta1/beta2 parameters are present but disabled by default (PyTorch's standard 0.9 / 0.999 are used).
In `configs/architectures/<arch>.yaml`, find `learning_rate` and set `initial_value` to your preferred starting point. You can optionally narrow `range` too. The Analyzer will still explore from there.

**Q: How does the Test_set get used?**
Only **once**, at the very end. After tuning is complete, the best hyperparameters are used to retrain a fresh model on Train+Val combined; that model is then evaluated on Test_set. This is standard practice in ML and avoids data leakage.

**Q: How does AutoTune-NN handle imbalanced classes?**
Automatically. When the data is loaded, the class distribution is checked. If `loss_function: "auto"` (the default), the system picks Focal Loss with a tuned `gamma` based on the imbalance ratio:
- ratio < 3:1 → CrossEntropy (balanced)
- 3:1–10:1 → Focal Loss with γ=1.5
- > 10:1 → Focal Loss with γ=2.5

You can override by setting `loss_function` to a specific value in the architecture YAML.

**Q: I want to run multiple datasets in parallel. Will they collide?**
No. Each local dataset gets its own split sub-folder (`Data/<filename>_split/`), so concurrent runs don't overwrite each other. Each run's output folder is also unique by `RUN_NAME`.

**Q: How big should my network be?**
See the [Sizing Your Network](SETUP_GUIDE.md#-sizing-your-network-how-deep-how-wide) section in `SETUP_GUIDE.md`. It has rule-of-thumb tables for each architecture and problem size.

**Q: How much RAM does AutoTune-NN use?**
Most cost is the model itself, not the framework. The FTTS tree only stores trial metadata (~2 KB per trial; an entire tree of 25,000 trials ≈ 50 MB). Typical model footprints:
- MLP: 50-200 MB
- CNN (medium): 1-2 GB
- Transformer (medium): 1-3 GB
- Transformer (12 layers, d_model=512): 8-16 GB

If you have 32+ GB of RAM, you can comfortably explore deep networks. With 128 GB, even very large transformers fit easily.

**Q: How do I make FTTS try deeper architectures?**
Edit the relevant `range` in `configs/architectures/<arch>.yaml`. For example, raise Transformer's `num_encoder_layers` `range: [2, 6]` to `[2, 24]`. FTTS won't waste trials going deep unnecessarily — it only suggests `add_depth` when the Analyzer detects it would help.
