# 🔍 Strategy Configurations

One YAML file per search strategy. The strategy controls **how** hyperparameters are chosen across trials, and **when** the run stops.

---

## 📂 Available Strategies

### [`ftts.yaml`](ftts.yaml) — Fine-Tuning Tree Search (default) 🌳

Builds a tree where each node is a trial and each edge is an Analyzer-suggested action applied to the parent's hyperparameters. Fully explainable in the report.

### [`bayesian.yaml`](bayesian.yaml) — Optuna TPE 📊

Optuna's statistical optimizer. Effective but less interpretable. Requires `optuna` installed.

### [`grid.yaml`](grid.yaml) — Grid Search 🔲

Exhaustive combination search. Use only for small search spaces.

---

## 📋 Common Sections

All three strategies share these top-level sections.

### `scoring`

Controls how trials are scored into a single quality score in `[0, 1]`.

```yaml
scoring:
  smoothing_window: 5        # Moving average window for all curves
  profile: "balanced"        # performance / balanced / robust / custom
  custom_weights: null       # Required only if profile == "custom"
```

The quality score combines 4 components:

| Component | What it rewards |
|---|---|
| `best_metric` | Peak performance (smoothed) |
| `stability` | Low variance near the peak |
| `convergence_speed` | Healthy learning pace (not too fast, not too slow) |
| `generalization_gap` | Small train-val gap (good generalization) |

The three weight profiles:

| Profile | Best | Stability | Speed | Gap | When to use |
|---|---|---|---|---|---|
| `performance` | 0.70 | 0.10 | 0.10 | 0.10 | "I just want high accuracy" |
| `balanced` | 0.50 | 0.20 | 0.15 | 0.15 | Default — good tradeoff |
| `robust` | 0.35 | 0.30 | 0.15 | 0.20 | Production / research — need reliable models |
| `custom` | user-defined | | | | Full control |

### `stopping`

All four stopping conditions can be active simultaneously. The run stops on **whichever fires first**. At least one must be set (non-null), otherwise the system raises an error to prevent infinite runs.

```yaml
stopping:
  max_trials: 50                # null = no limit
  time_limit_minutes: null      # null = no limit
  target_accuracy: null         # smoothed metric target; null = no target
  convergence_patience: 40      # null = disabled
```

### `execution`

```yaml
execution:
  max_parallel_experiments: 1   # concurrent trials
```

---

## 🌳 FTTS-Specific Sections

### `step_control`

Adaptive step sizes for FTTS. When an action (e.g. "increase_lr") improves the model, the step grows for next time. When it fails, the step shrinks.

```yaml
step_control:
  successful_step_boost: 1.5    # grow step by this factor on success
  failed_step_shrink: 0.5       # shrink step by this factor on failure
  min_step_factor: 1.2          # never go below
  max_step_factor: 5.0          # never go above
```

---

## 📊 Bayesian-Specific Sections

### `n_startup_trials`

First N trials use random sampling before TPE kicks in. Optuna needs initial data.

```yaml
n_startup_trials: 10
```

---

## 🔲 Grid-Specific Sections

### `grid_points`

Number of points sampled from each continuous `range` parameter. `grid_points: 3` gives `[min, mid, max]`.

```yaml
grid_points: 3
```

⚠️ Higher values explode the combination count: `grid_points^num_params` trials.

---

## 🔗 Which Strategy Should I Use?

| Situation | Recommended |
|---|---|
| First run, don't know your search space | **FTTS** (default) |
| Want full explainability in the report | **FTTS** |
| Already know your hyperparameters roughly | **FTTS** with narrower `range` |
| Have Optuna installed, want proven statistical search | **Bayesian** |
| Few parameters (2-3), few choices each | **Grid** |
| Large search space, many choices | **FTTS** or **Bayesian** |

**FTTS is the default for a reason** — it runs out-of-the-box without extra dependencies, produces a fully-annotated report explaining every decision, and performs well across a wide range of tasks.
