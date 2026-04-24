"""
================================================================================
 AutoTune-NN  |  Automated Fine-Tuning Framework for Neural Networks
================================================================================

This file contains only the TOP-LEVEL SELECTIONS for a run:
  - Which architecture to use
  - Which search strategy to use
  - Where the dataset lives
  - Hardware settings

All other settings (parameter ranges, stopping conditions, quality weights,
parallelism, etc.) are in YAML files under configs/. This keeps the entry
point clean and focused on choices, not on numeric knobs.

================================================================================
 BEFORE FIRST RUN - do these once:
================================================================================

  Step 1 - Install dependencies:
      pip install -r requirements.txt

  Step 2 - Prepare your dataset:
      * For DATASET_MODE="local":
          Place your CSV at Data/Data-Set.csv (or edit LOCAL_DATASET_PATH).
          On first run, the system auto-splits into Train_set / Val_set / Test_set.
      * For DATASET_MODE="imported":
          Nothing to do - the dataset downloads automatically.

  Step 3 - Review the YAML configs:
      * configs/architectures/<ARCHITECTURE>.yaml -  parameter ranges & methods.
      * configs/strategies/<SEARCH_STRATEGY>.yaml - stopping, scoring, parallelism.

  Step 4 - Run:
      python main.py

  Results land in experiments/<timestamp>_<arch>_<strategy>/ :
      - report.txt       : full trial-by-trial log
      - best_trial.png   : plot of the best trial's learning curves
================================================================================
"""

# ==============================================================================
#                           TOP-LEVEL SELECTIONS
# ==============================================================================

# ARCHITECTURE
# ------------
# Which neural network type to tune.
#   "mlp"         : Multi-Layer Perceptron (for tabular data).
#   "cnn"         : Convolutional Neural Network (for images).
#   "rnn"         : Vanilla RNN (short sequences).
#   "lstm"        : Long Short-Term Memory (long sequences / basic NLP).
#   "transformer" : Transformer encoder (NLP and complex sequences).
ARCHITECTURE = "mlp"

# SEARCH_STRATEGY
# ---------------
# How hyperparameters are chosen across trials.
#   "ftts"     : Fine-Tuning Tree Search. Default - explainable and effective.
#   "bayesian" : Optuna TPE. Statistical approach. Requires optuna installed.
#   "grid"     : Exhaustive grid search. Best for small search spaces.
SEARCH_STRATEGY = "ftts"

# TASK_TYPE
# ---------
#   "classification" : predicts a class label (accuracy metric).
#   "regression"     : predicts a continuous value (RMSE metric).
TASK_TYPE = "classification"


# ==============================================================================
#                                DATASET
# ==============================================================================

# DATASET_MODE
# ------------
#   "local"    : read from a file/folder you provide.
#   "imported" : a well-known dataset from torchvision/sklearn.
DATASET_MODE = "local"

# --- Local mode settings ---

# Path to the dataset. For tabular/text: a CSV file. For images: an ImageFolder.
LOCAL_DATASET_PATH = "./Data/Data-Set.csv"

# Type of data (determines which data loader runs).
#   "tabular" | "image" | "text"
DATA_TYPE = "tabular"

# Feature columns (for tabular/text). None = all columns except the label.
FEATURE_COLUMNS = None

# Label column name. None = the last column.
LABEL_COLUMN = None

# Data split percentages. Must sum to 1.0.
# On the first run, the system creates:
#   Data/Train_set.csv (train_pct of the data)
#   Data/Val_set.csv   (val_pct of the data - used inside the tuning loop)
#   Data/Test_set.csv  (test_pct - held out, not used during tuning)
# Subsequent runs reuse these files for reproducibility. To re-split, delete them.
TRAIN_PCT = 0.6
VAL_PCT   = 0.2
TEST_PCT  = 0.2

# --- Imported mode settings ---

# Which well-known dataset to use. Supported:
#   Image: "mnist", "fashion_mnist", "cifar10", "cifar100"
#   Tabular: "iris", "wine", "breast_cancer", "digits"
IMPORTED_DATASET_NAME = "mnist"


# ==============================================================================
#                               HARDWARE
# ==============================================================================

# DEVICE
# ------
#   "auto" : pick GPU if available, else CPU.
#   "gpu"  : request GPU; fall back to CPU with a warning if unavailable.
#   "cpu"  : force CPU.
DEVICE = "auto"

# RANDOM_SEED
# -----------
# Seed for reproducibility. None = non-deterministic.
RANDOM_SEED = 42


# ==============================================================================
#                                OUTPUT
# ==============================================================================
EXPERIMENTS_ROOT = "./experiments"


# ==============================================================================
#                          END OF USER SETTINGS
# ==============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.device_utils import resolve_device
from core.orchestrator import Orchestrator
from core.run_config import RunConfig


def build_run_config() -> RunConfig:
    """Bundle the user's top-level choices into a RunConfig."""
    return RunConfig(
        architecture=ARCHITECTURE,
        search_strategy=SEARCH_STRATEGY,
        task_type=TASK_TYPE,

        dataset_mode=DATASET_MODE,
        local_dataset_path=LOCAL_DATASET_PATH,
        imported_dataset_name=IMPORTED_DATASET_NAME,
        data_type=DATA_TYPE,
        feature_columns=FEATURE_COLUMNS,
        label_column=LABEL_COLUMN,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        test_pct=TEST_PCT,

        device=resolve_device(DEVICE),
        random_seed=RANDOM_SEED,
        experiments_root=EXPERIMENTS_ROOT,
    )


def main():
    cfg = build_run_config()
    cfg.validate()

    orchestrator = Orchestrator(cfg)
    summary = orchestrator.run()

    print("\n" + "=" * 80)
    print(" AutoTune-NN finished ".center(80, "="))
    print("=" * 80)
    print(f"Total trials          : {summary['total_trials']}")
    print(f"Best trial id         : {summary['best_trial_id']}")
    print(f"Best quality score    : {summary['best_quality']:.4f}")
    print(f"Best metric (raw)     : {summary['best_metric_raw']:.4f}")
    print(f"Best metric (smoothed): {summary['best_metric_smoothed']:.4f}")
    print(f"Stop reason           : {summary['stop_reason']}")
    print(f"Results directory     : {summary['results_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
