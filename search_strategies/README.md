# 🔍 Search Strategies

How AutoTune-NN picks hyperparameters for each new trial. Three strategies are provided, each with its own algorithm.

---

## 📂 Files

| File | Role |
|---|---|
| `__init__.py` | `build_strategy()` factory |
| `ftts.py` | Fine-Tuning Tree Search (default) |
| `experiment_tree.py` | The thread-safe tree data structure used by FTTS |
| `bayesian.py` | Optuna TPE wrapper |
| `grid.py` | Exhaustive grid search |

---

## 🌳 FTTS (Fine-Tuning Tree Search)

The default strategy. Each trial is a node. Each edge represents an Analyzer-suggested Action applied to a parent's hyperparameters.

### The Priority Queue Formula

When a trial completes, it gets a `quality_score` in `[0, 1]` from the quality scorer. The Analyzer produces a list of Actions for it, each with a `priority` in `[0, 1]`.

For every Action, a queue entry is created with:

```
child_score = parent.quality_score × action.priority
```

The priority queue (a heap) keeps all pending entries sorted. The next trial is always derived from the entry with the highest `child_score`.

### Worked Example

Suppose the tree currently has two complete trials:

| Trial | Quality | Pending actions (priorities) |
|---|---|---|
| A (accuracy ≈ 0.92) | 0.85 | increase_dropout (0.85), increase_weight_decay (0.80), increase_augmentation (0.70) |
| B (accuracy ≈ 0.78) | 0.70 | decrease_lr (0.95), add_lr_scheduler (0.75) |

The priority queue has 5 entries:

| child_score | parent | action |
|---|---|---|
| 0.85 × 0.85 = **0.7225** | A | increase_dropout |
| 0.70 × 0.95 = **0.6650** | B | decrease_lr |
| 0.85 × 0.80 = **0.6800** | A | increase_weight_decay |
| 0.85 × 0.70 = **0.5950** | A | increase_augmentation |
| 0.70 × 0.75 = **0.5250** | B | add_lr_scheduler |

Sorted: the next trial will be **A → increase_dropout** (0.7225). Even though trial B has a high-priority action (decrease_lr at 0.95), trial A's strong `quality_score` makes its pending actions more promising overall.

### Why This Works

- **Good parents get priority** — children of the best trials are tried first.
- **Not just accuracy** — `quality_score` includes stability, convergence speed, and generalization gap. A spiky 0.95 accuracy loses to a stable 0.90.
- **Smart prioritization** — a great parent with a weak action can be beaten by a lesser parent with a confident action, so the search isn't blindly greedy.
- **Automatic backtracking** — when a branch stops improving, the queue naturally returns to other nodes with good pending actions.

### Adaptive Step Sizes

When an action like `increase_lr` succeeds, FTTS remembers and **grows the step** for next time:

- `successful_step_boost: 1.5` — each success multiplies the step factor by 1.5
- `failed_step_shrink: 0.5` — each failure multiplies by 0.5
- Clamped between `min_step_factor` (1.2) and `max_step_factor` (5.0)

This gives controlled acceleration in productive directions and automatic slowdown when an action stops helping.

### Tree State

Every node tracks:

- `trial_id`, `parent_id`, `hyperparameters`, `quality_score`, `verdict`
- `pending_actions` — Actions not yet spawned as children
- `consumed_actions` — Actions already turned into children
- `children_ids` — links in the tree
- `status` — `pending` / `running` / `done` / `failed` / `diverged`

---

## 📊 Bayesian (Optuna TPE)

A wrapper around Optuna's Tree-structured Parzen Estimator.

1. Optuna models `p(params | good trials)` and `p(params | bad trials)`.
2. For the next trial, it samples parameters that maximize `p_good / p_bad`.
3. After each trial, the quality score is fed back to update the model.

The Analyzer still runs and its diagnosis is included in the report — Optuna alone decides parameters, but explanations are still available for each trial.

The first `n_startup_trials` (default 10) use random sampling. Optuna needs some initial data before TPE is effective.

---

## 🔲 Grid Search

Exhaustively enumerates all combinations of parameter values.

- Discrete (`choices`) parameters: every choice.
- Continuous (`range`) parameters: `grid_points` evenly-spaced values (default 3 → `[min, mid, max]`).

**Combination count** = product of values per parameter. Explodes quickly: 5 parameters × 4 choices each = 1024 trials.

Best for: 2-3 parameters, few choices each.

---

## 🆚 Comparison

| Feature | FTTS 🌳 | Bayesian 📊 | Grid 🔲 |
|---|---|---|---|
| Explainability | ✅ Full | ⚠️ Partial | ✅ Full |
| External dependency | None | `optuna` | None |
| Adaptive | ✅ | ✅ | ❌ |
| Uses Analyzer diagnoses | ✅ to pick next | 📝 for report only | 📝 for report only |
| Tree structure | ✅ | ❌ | ❌ |
| Good for large search spaces | ✅ | ✅ | ❌ |
| Good for few parameters | ✅ | ⚠️ | ✅ |

---

## 🧵 Thread Safety

The `ExperimentTree` uses a `threading.Lock` around all state mutations. Multiple trials can run concurrently (controlled by `max_parallel_experiments` in the strategy YAML) without race conditions.

---

## ⚠️ Important: FTTS always uses quality_score internally

The `ranking_metric` parameter (in `configs/strategies/<strategy>.yaml`) controls **only** which trial is picked as the final "best" — the one written to `final/model.py`. It does **not** change FTTS's tree-search behavior.

Concretely:

- **Inside FTTS**: every node in the tree is ranked by `parent.quality_score × action.priority`. This is **always** the case, regardless of `ranking_metric`.
- **At the very end**: the system picks the "best trial" using the configured `ranking_metric`:
  - `quality_score` (default) — same as the internal ranking.
  - `smoothed_accuracy` — picks the highest smoothed val accuracy.
  - `raw_accuracy` — picks the highest single-epoch peak.

This separation is intentional: FTTS performs better when guided by the composite (stability matters during search), but for reporting you might want a different selection criterion.

---

## 🔗 Related Documents

- [`README.md`](../README.md) — project overview
- [`configs/strategies/README.md`](../configs/strategies/README.md) — strategy YAML format
- [`core/README.md`](../core/README.md) — Quality Score components

---

## 🧠 Context-Aware Layer Shape Suggestions

When the Analyzer wants to change `layer_shape` (MLP), it picks the specific pattern most likely to help, rather than cycling through all of them. Logic:

| Verdict | Suggested patterns (with priority) | Why |
|---|---|---|
| `overfit` | `bottleneck` (0.55), `funnel` (0.40) | Both compress information → implicit regularization |
| `failed_to_learn` | `hourglass` (0.55), `uniform` (0.40) | Hourglass widens in middle for more capacity |
| `slow` | `funnel` (0.45) | Fewer params in deep layers → faster training |
| `healthy` | `bottleneck` (0.30), `pyramid` (0.25) | Exploration only |

The current shape is skipped (would be a no-op). Implemented in `core/analyzer.py::_propose_layer_shapes`.

---

## 🔄 DAG Deduplication

FTTS tracks every hyperparameter configuration it has explored or queued, using a deterministic JSON signature. When a proposed action would produce a duplicate configuration (e.g., `try_layer_shape_uniform` from a node that already has `uniform` after some other path), FTTS silently skips it and moves to the next action.

This:
- Saves trial budget for genuinely new configurations.
- Prevents wasted compute on revisited nodes.
- Is logged when it happens: `[ftts] dedup: skipping action 'X' from TN - target HP already explored.`

Implemented in `search_strategies/ftts.py::_hp_signature` and used by `propose_next`.
