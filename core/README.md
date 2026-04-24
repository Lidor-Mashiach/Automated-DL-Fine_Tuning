# 🧠 Core

The internal engine of AutoTune-NN. You don't normally edit these files — they're documented here for developers and anyone curious about what's happening under the hood.

---

## 📂 Modules

| Module | Role |
|---|---|
| `run_config.py` | Dataclass bundling the user's `main.py` selections. Validates them. |
| `config_manager.py` | Loads and normalizes architecture YAMLs. Uniform `enabled`/`initial_value` parameter access. |
| `strategy_config.py` | Simple YAML loader for strategy configs. |
| `device_utils.py` | Resolves `DEVICE` (auto/gpu/cpu) to a real torch device, with clear messaging if GPU is unavailable. |
| `trainer.py` | Runs a single trial's training loop. Uses smoothed val_loss for early stopping. |
| `smoothing.py` | Moving-average and tail-average utilities for training curves. |
| `quality_scorer.py` | Computes the 4-component quality score (best_metric / stability / convergence_speed / generalization_gap). |
| `analyzer.py` | Diagnoses each trial's training behavior and emits prioritized Actions. |
| `orchestrator.py` | Top-level runner. Wires together the architecture, strategy, data loader, trainer, analyzer, and reporter. Manages the main loop and stopping conditions. |

---

## 🔄 How They Work Together

```
                      main.py
                        ↓
                   [RunConfig]
                        ↓
                  Orchestrator
                    ↙   ↓   ↘
          ConfigMgr  Strategy  DataLoader
                    ↙   ↓   ↘
              Trainer → Analyzer → QualityScorer
                           ↓
                       Reporter → report.txt
                           ↓
                       Plotter  → best_trial.png
```

---

## 📈 Smoothing Everywhere

All metric curves are **smoothed with a moving average** before being used for decisions. The window size comes from the strategy YAML's `scoring.smoothing_window`. This prevents single-epoch spikes from misleading any of:

- Early stopping (trial-level)
- Quality score computation
- Target accuracy comparisons
- Analyzer diagnosis

The report always shows **both** the raw best metric and the smoothed best, so you can see the difference.

---

## 🧪 The Analyzer

Given a completed trial's curves, the Analyzer produces:

1. **Verdict** — one of: `failed_to_learn`, `peaked_and_dropped`, `learning_too_fast`, `learning_too_slow`, `converged`, `healthy`, `diverged`, `failed`, `insufficient_data`.
2. **Observations** — human-readable facts about what happened.
3. **Actions** — concrete parameter changes to try next, each with a **priority** in `[0, 1]`.

Actions are sorted by priority before being returned. FTTS uses these priorities to rank tree branches.

### Example action priorities

| Verdict | Top-priority action |
|---|---|
| `diverged` | `decrease_lr` (0.95) |
| `learning_too_fast` | `decrease_lr` (0.90) |
| `peaked_and_dropped` | `increase_dropout` (0.85) |
| `failed_to_learn` | `increase_lr` (0.85) |
| `learning_too_slow` | `increase_lr` (0.80) |

---

## 🛡️ Thread Safety

When `max_parallel_experiments > 1`, multiple trials run concurrently.

- Each trial builds its own model — no shared mutable state.
- The `ExperimentTree` (in `search_strategies/experiment_tree.py`) uses an internal `threading.Lock` for all state mutations.
- The `Reporter` writes atomically after each trial.
- No race conditions by design.
