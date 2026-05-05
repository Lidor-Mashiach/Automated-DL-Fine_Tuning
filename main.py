"""
================================================================================
 AutoTune-NN  |  Automated Fine-Tuning Framework for Neural Networks
================================================================================

This file contains only the TOP-LEVEL SELECTIONS for a run:
  - A label for this run (RUN_NAME)
  - Which architecture to use
  - Which search strategy to use
  - Where the dataset lives
  - Hardware settings

All other settings (parameter ranges, stopping conditions, quality weights,
parallelism, etc.) are in YAML files under configs/.

For a step-by-step setup walkthrough, see SETUP_GUIDE.md.
================================================================================
"""

# ==============================================================================
#                              RUN IDENTITY
# ==============================================================================

# RUN_NAME
# --------
# A short label for this run. Used as a prefix in the output folder name so
# multiple runs (different datasets / tasks) can coexist without overwriting.
# Examples: "experiment1", "iris_class", "customer_churn_v2"
RUN_NAME = "experiment1"


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

# Path to the dataset. Tabular: .csv / .npy / .parquet. Images: ImageFolder dir.
LOCAL_DATASET_PATH = "./Data/Data-Set.csv"

# Type of data (determines which data loader runs).
#   "tabular" | "image" | "text"
DATA_TYPE = "tabular"

# Feature columns (for tabular/text). None = all columns except the label.
FEATURE_COLUMNS = None

# Label column name. None = the last column.
LABEL_COLUMN = None

# Data split percentages. Must sum to 1.0.
# For LOCAL or imported datasets without a built-in split:
#   Train, Val, and Test sets are created from these percentages.
# For imported datasets WITH a built-in test set (MNIST, CIFAR, ...):
#   The built-in test set is preserved.
#   TRAIN_PCT:VAL_PCT becomes the ratio for splitting the original train set.
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
    return RunConfig(
        run_name=RUN_NAME,
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
