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

---

## 📐 Loss vs Accuracy vs Metric

These three terms are often confused, but they measure different things.

### Loss (technical / mathematical)

**Loss is NOT "the percentage of mistakes."** It is a mathematical function the optimizer tries to minimize. It captures how **confident** the model is in its (correct or wrong) answers.

**Cross-Entropy example** — all three predictions are CORRECT but loss differs:

| Model says...                       | Loss   |
|-------------------------------------|--------|
| 99% confident → answer A (correct)  | 0.01   |
| 60% confident → answer A (correct)  | 0.51   |
| 10% confident → answer A (correct)  | 2.30   |

The model that's MORE confident in correct answers gets a LOWER loss, even though all three are technically right. In production, a 99%-confident-and-right model is far more trustworthy than a 51%-confident-and-right one.

### Accuracy (human / interpretable)

`accuracy = correct_predictions / total_predictions`. A simple percentage in `[0, 1]`.

### Metric (the human-readable score)

A general term meaning "the number we report to humans," depending on the task:

- Classification → `accuracy`
- Regression → `-RMSE` (negative so "higher is better")

### Why we track both Loss and Metric

- **Loss** drives the optimizer (it is differentiable and well-behaved).
- **Loss on Val_set** detects overfitting: when train_loss keeps dropping but val_loss starts rising, the model is memorizing.
- **Metric (accuracy)** is what we report.

In our reports you'll see all four: `train_loss`, `val_loss`, `train_metric`, `val_metric`.

---

## 🚫 Why we don't put Softmax in the architectures

PyTorch's `nn.CrossEntropyLoss` already includes Softmax internally — it accepts raw logits and computes `Softmax + log + NLLLoss` in a single numerically-stable step.

**If you add `nn.Softmax` as the final layer, Softmax is applied twice**, which:
- Hurts gradient flow.
- Reduces numerical stability.
- Slows training.

Our architectures emit raw logits; the loss function handles the rest. This is the standard PyTorch pattern.

---

## 🎯 Loss function selection (auto)

When `loss_function: "auto"` (the default), the system picks based on `TASK_TYPE` and detected class imbalance:

| Task type | Imbalance ratio | Loss chosen |
|---|---|---|
| classification | < 3:1 | CrossEntropy |
| classification | 3:1 – 10:1 | Focal Loss (γ=1.5) |
| classification | > 10:1 | Focal Loss (γ=2.5) |
| regression | (always) | MSE |

The user can override by setting `loss_function` to a specific value in the architecture YAML (`cross_entropy`, `focal`, `mse`).

**Focal Loss** down-weights easy (high-confidence-correct) examples so the model focuses on hard ones — useful for imbalanced classes. The `focal_gamma` parameter controls how aggressive the focusing is.

---

## 🔚 The Final Refit & Test Evaluation Phase

After tuning finishes, AutoTune-NN runs one extra phase before exiting:

1. Pick the best trial (per the `ranking_metric` in the strategy YAML).
2. Build a fresh model with that trial's hyperparameters.
3. Train it on **Train + Val combined** ("refit on full training set" — standard ML practice).
4. Evaluate it on the held-out **Test_set** (the only time Test is touched).
5. Generate `final/model.py` (standalone, runnable) and `final/test_evaluation.txt`.

This phase is implemented in `core/final_trainer.py` and `core/code_generator.py`.


---

## 🔗 Related Documents

- [`README.md`](../README.md) — project overview
- [`SETUP_GUIDE.md`](../SETUP_GUIDE.md) — step-by-step usage
- [`search_strategies/README.md`](../search_strategies/README.md) — FTTS algorithm
- [`models/README.md`](../models/README.md) — architecture builders
- [`reporting/README.md`](../reporting/README.md) — output formats

---

## 🎵 Language Modeling Components

These modules are loaded only when `task_type="language_modeling"`.

### `sampling.py`
Pure functions for next-word selection during generation:
- **`sample_proportional(logits)`** - sample from softmax directly.
- **`sample_temperature(logits, T)`** - scale logits by 1/T before softmax.
- **`sample_top_k(logits, k)`** - keep top K logits, sample from those.
- **`sample_nucleus(logits, p)`** - smallest set whose cumulative prob > p.

Dispatcher: `sample(strategy, logits, **kwargs)`. Strategies are interchangeable at generation time without retraining.

### `generator.py`
End-to-end generation pipeline:
- **`generate_lyrics(model, vocab, initial_word, midi_features, ...)`** - autoregressive generation calling `model.step()` per token.
- **`format_lyrics(tokens, line_separator)`** - converts the line-separator token to newlines for human-readable output.
- **`melody_influence_probe(...)`** - generates twice (real MIDI vs shuffled MIDI) with the same RNG seed; reports Jaccard / sequence overlap / length diff.
- **`run_generation_for_test_set(...)`** - iterates over test songs x initial words, writes `generated_lyrics.txt` and (optionally) `melody_probe.json`.
- **`run_decoding_comparison(...)`** - generates the same prompt on a small subset of test songs (default: 2) with proportional / temperature / nucleus sampling side-by-side. Writes `decoding_comparison.txt`. Triggered by `--run_decoding_comparison true`.

### `model.py` for LM
Yes - LM runs produce a real, runnable `model.py` (just like other task types). Because LM has external file dependencies, the generator also creates:

- A **`data/`** subfolder next to `model.py` containing copies of the lyrics CSV and MIDI files (and Word2Vec if it's small enough to copy).
- **Global path constants** at the top of `model.py` (`LYRICS_CSV_PATH`, `MIDI_DIR`, `WORD2VEC_PATH`) - easy to edit.

The user can:
- Run `python model.py` as-is (it auto-resolves paths to `./data/...`).
- Swap files in `data/` to retrain on different inputs.
- Edit a path constant at the top to point elsewhere.

The script trains end-to-end OR loads `model_checkpoint.pt` if present (skipping training), then generates lyrics on the test split with the chosen sampling strategy.

---

## 📊 Perplexity as a reported metric (LM)

For `task_type="language_modeling"`, the trainer records `val_perplexity = exp(val_loss)` per epoch in `TrialResult.val_perplexity_curve`. Why both?

- **Loss** is the optimization signal — what backprop minimizes — and the `quality_score` is computed from it.
- **Perplexity** is the human-readable interpretation. PPL=20 means "the model is as confused as if it had to pick uniformly from 20 words at each step." Lower is better.

Where it appears:
- Per-epoch console log: `loss=2.41  ppl=11.13`
- `report.txt`: `val_perplexity range: 1234.56 -> 42.31 (min=38.20)`
- TensorBoard: `<trial_id>/val_perplexity` scalar curve
- `report.txt` final summary: best perplexity for the winning trial

Loss and perplexity are monotonically related (`ppl = e^loss`), so they always agree on ranking. The reason to report both is convention — the Assignment 3 expects perplexity numbers in the writeup, but the search is driven by loss.

## 🎼 LM-specific trainer additions

The trainer's `_run_epoch` accepts two extra parameters for language modeling:

- `tf_ratio` (default `1.0`) — teacher forcing ratio, applied as input-token dropout to `<unk>` during training. Pulled from `hp["teacher_forcing_ratio"]`.
- `unk_idx` (default `1`) — the `<unk>` token's vocab index, used when dropping inputs. Pulled from `data_info["vocab"].unk_idx`.

Both default to safe values (no perturbation) so non-LM tasks are unaffected.
