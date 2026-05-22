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

FTTS builds a **DAG** (directed acyclic graph) of trials, not a tree: the same hyperparameter configuration can be reached from multiple parents via different action sequences. To avoid wasting trial budget re-evaluating an identical configuration, FTTS deduplicates.

### How the signature works

`_hp_signature(hp)` produces a deterministic JSON string from the **entire** hyperparameter dict (all keys except `__`-prefixed metadata), with sorted keys. Two configurations have the same signature **if and only if every hyperparameter value matches**.

```
seq_len=256, lr=1e-3, hidden=256  ->  signature A
seq_len=512, lr=1e-3, hidden=256  ->  signature B   (different - seq differs)
seq_len=512, lr=3e-3, hidden=256  ->  signature C   (different - lr differs)
```

The signature is computed **after** `_apply_action` mutates `child_hp`, so it always reflects the post-action values. A child of `increase_sequence_length` from a `seq_len=256` parent gets `seq_len=512` in its dict before the signature is taken.

### When dedup fires (and why it's correct)

Dedup fires when two **different action paths converge to the identical full HP dict**. Example from a real run:

```
T0013: seq=512, lr=3.60e-3   (path: root -> increase_seq -> increase_lr -> increase_seq)
T0019: seq=512, lr=4.32e-3   (T0013 + increase_lr)
T0016: seq=256, lr=4.32e-3   (a different path)
T0024 attempt: T0016 + increase_sequence_length -> seq=512, lr=4.32e-3
               ^^^ identical to T0019 -> DEDUP FIRES, correctly.
```

Both paths legitimately arrived at `{seq=512, lr=4.32e-3, ...}`. Evaluating it twice would waste a trial. This is **correct DAG behavior, not a bug** — the action `increase_sequence_length` did its job (256→512), it just happened to land on an already-visited node.

### Why `increase_sequence_length` can *look* blocked

In a run where most trials get the `failed_to_learn` verdict, the analyzer keeps proposing both `increase_lr` (0.85) and `increase_sequence_length` (0.80). FTTS explores the seq_len axis fully: `128 → 256 → 512`. Once `seq_len=512` has been visited in combination with every reachable `lr` value, **any further `increase_sequence_length` necessarily lands on an already-seen node** and gets deduped. The log then shows repeated dedup messages for `increase_sequence_length`, which can look like it's "blocked" — but actually the entire reachable seq_len space was already covered. The remaining choices `{32, 64}` are *below* the initial value of 128 and are only reachable via `decrease_sequence_length`, which the `failed_to_learn` verdict doesn't emit (correctly — a model that won't learn shouldn't shrink its context).

### Diagnostic logging

Two log lines help diagnose dedup behavior:

```
[ftts] dedup: skipping 'increase_sequence_length' from T0016
       (target=sequence_length: 256 -> 512) - HP combination already
       explored via another path.
```
Shows the action, parent, and exactly which value the target param moved to. If you see `256 -> 512` repeatedly, FTTS *is* trying larger seq_len — the combination is just already covered.

```
[ftts] action 'increase_sequence_length' from T0008 produced no change
       (target may be at YAML boundary). Trying next.
```
Shows the action hit the end of the YAML `choices` list (e.g. `seq_len` already at its max of 512). `_adjust_choice_param` returns `None` in this case and FTTS moves on.

### Value-coverage tracking

`_value_coverage` is a `dict[param_name -> set of values]` that records which concrete values of each tunable have been visited anywhere in the DAG. It's a **diagnostic aid** (not used to block anything) — it lets you answer "did FTTS ever try `seq_len=512`?" by inspecting `strat._value_coverage['sequence_length']`.

### Implementation

- `_hp_signature(hp)` — full-dict JSON signature
- `_value_coverage_key(hp, target_param)` — `(param, value)` tuple for coverage tracking
- `_seen_signatures: set[str]` — every visited/queued signature
- `_value_coverage: dict[str, set]` — per-param visited values
- All wired in `propose_next`

The dedup is intentionally **full-HP-based**, not per-parameter. A per-parameter dedup ("never try seq_len=512 twice") would be wrong: `seq_len=512` paired with `lr=1e-3` and `seq_len=512` paired with `lr=5e-3` are genuinely different experiments worth running.

---

## 🎲 Exploration Diversity Boost (FTTS)

After empirical observation that FTTS would greedily pick the same single action (e.g. `increase_lr` with priority=0.80) trial after trial — starving lower-priority but still important LM-specific actions like `increase_sequence_length` (0.78) — the tree now applies a **diversity dampening factor** when scoring queue entries.

### The math
For each action queued in `register_completed`:
```
diversity_factor = 0.85 ^ n_already_consumed_of_this_type
effective_score = quality_score * action.priority * diversity_factor
```

After consuming the same action type N times across the tree:
- N=0: factor=1.00 (full priority)
- N=1: factor=0.85
- N=2: factor=0.72
- N=3: factor=0.61
- N=5: factor=0.44

This means a high-priority action stays competitive but eventually loses to alternatives that haven't been tried yet. Empirically, this balances exploit vs explore well.

### Why not just lower priorities in the analyzer?
The analyzer doesn't know how many trials of each action type have already been run. The tree does. Pushing this logic into the tree keeps the analyzer stateless (good) and gives the heap a global view of what's been tried.

### What's NOT changed
- The DAG dedup (skipping HP signatures already explored) is unchanged — it's correct and necessary.
- Per-parent action consumption tracking is unchanged — each parent emits each of its actions at most once.
- Action priorities in `analyzer.py` are unchanged.

The diversity boost only affects which action is picked next when multiple parents have similar candidates. It doesn't change WHAT'S available to pick.
