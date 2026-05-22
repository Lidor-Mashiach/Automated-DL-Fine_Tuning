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

---

## 🔥 Warm Start (continue tuning from a checkpoint)

When `--warm_start_checkpoint <path>` is passed in `tune` mode, the orchestrator loads the checkpoint's weights into the model **before T0001** instead of using random initialization.

### When to use
- Previous run produced a promising model but you have more compute budget
- You want to continue from a known-good baseline rather than restart
- Resuming after an interrupted run (use the run's `final/model_checkpoint.pt`)

### Behavior
- Applied **only at T0001**. T0002+ rebuild from FTTS-mutated hyperparameters and re-initialize naturally - if the FTTS changed `hidden_size` for T0002, the checkpoint weights wouldn't match anyway.
- Uses `strict=False`: parameters that don't match (different vocab size, different fusion layers, etc.) are reported but skipped.
- Failures are non-fatal: missing checkpoint or load error logs a warning and proceeds with random init.

### Example
```bash
# Continue tuning ex1 from its best trial
python main.py \
  --architecture lstm \
  --task_type language_modeling \
  --data_type lyrics \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --word2vec_path ./word2vec.bin \
  --run_name ex1_continued \
  --warm_start_checkpoint ./experiments/ex1_baseline_*/final/model_checkpoint.pt
```

### What it does NOT do
- Does not preserve optimizer state - the new optimizer is fresh
- Does not preserve the FTTS tree - exploration starts from T0001
- Does not skip training - T0001 still trains, just from better weights

---

## 🧭 Complete Analyzer Verdict → Actions Map

After the empirical-analysis pass, every ACTION_TYPE is emitted by at least one verdict. The full picture:

### `failed_to_learn` (loss not decreasing)
- `increase_lr` (0.85), `increase_sequence_length` (0.80, LM)
- `add_width` (0.70), `unfreeze_embeddings` (0.65, LM), `add_depth` (0.60)
- `enable_batch_norm` (0.55), `change_normalization` (0.50)
- `change_activation` (0.45), `change_optimizer` (0.35)
- + layer_shape suggestions

### `slow` (learning but slowly)
- `increase_lr` (0.80), `increase_sequence_length` (0.78, LM)
- `decrease_teacher_forcing` (0.70, LM), `unfreeze_embeddings` (0.65, LM)
- `reduce_batch_size` (0.60), `add_width` (0.55)
- `enable_mixed_precision` (0.50), `decrease_dropout` (0.45)
- `increase_batch_size` (0.40), `increase_grad_accumulation` (0.35)
- `change_fusion_method` (0.35, LM), `change_optimizer` (0.30)
- `adjust_attention_dropout` (0.30)
- `adjust_adam_beta1` (0.15), `adjust_adam_beta2` (0.15) [advanced, off-by-default]

### `healthy` (good training, can polish)
- `decrease_teacher_forcing` (0.70, LM)  ← top
- `decrease_lr` (0.60), `increase_lr` (0.50)
- `increase_dropout` (0.50), `increase_teacher_forcing` (0.50, LM)
- `add_width` (0.45)
- `increase_focal_gamma` (0.35), `decrease_focal_gamma` (0.30)
- `try_cross_entropy` (0.20)

### `overfit` (val diverging from train)
- `increase_dropout` (0.85), `increase_weight_decay` (0.80)
- `increase_augmentation` (0.70), `disable_bidirectional` (0.60, LM)
- `enable_mixup` (0.60), `enable_cutout` (0.55)
- `increase_stochastic_depth` (0.55), `enable_label_smoothing` (0.55)
- `enable_cutmix` (0.50), `increase_embedding_dropout` (0.50)
- `change_text_augmentation` (0.45), `reduce_width` (0.40)
- `increase_text_augmentation` (0.40), `reduce_depth` (0.35)
- `decrease_sequence_length` (0.35, LM)
- `toggle_bidirectional` (0.30), `try_focal_loss` (0.25)
- `increase_teacher_forcing` (0.25, LM)

### `fast` (converged too fast, may be unstable)
- `decrease_lr` (0.90), `add_lr_scheduler` (0.80)
- `increase_warmup` (0.70), `increase_dropout` (0.60)

### `converged` (plateaued, polish only)
- `add_width` (0.50), `decrease_weight_decay` (0.45)
- `change_activation` (0.30)

### `diverged` (loss became NaN/Inf)
- `decrease_lr` (0.95), `add_gradient_clipping` (0.85)
- (emitted in `analyze()` top-level, not in `_add_*`)

Each action is gated by `_is_tunable(cm, param)` — if the parameter is disabled in the architecture's YAML, the action is silently skipped.

---

## 🎵 Generator MIDI Safety Net

A bug was found where generation could crash with:
```
input.size(-1) must be equal to input_size. Expected 308, got 300
```

This happened when the LSTM was built with `midi_dim > 0` (e.g. 8 for the `simple` or `per_word` variants), but at generation time the song didn't have MIDI features attached (e.g. because the .mid file was missing, or `midi_dir` was None). The model expected a 308-dim LSTM input (300 word_emb + 8 MIDI), but only got 300.

### The fix
Two defensive layers:

1. **`generator.get_midi_for_step`** — when `midi_dim > 0` but `midi_features is None`, returns a zero tensor of the correct shape instead of `None`. This ensures the model always receives the input shape it was built for.

2. **`models/lstm.py::_LSTMLanguageModel.step`** — defensive check: if the model has `midi_dim > 0` but `midi_feat` arrives as `None`, fills in zeros automatically. This is a safety net for any other caller path.

### Behavior
- When the model was trained with MIDI conditioning but generation has no MIDI features available, the model behaves as if the melody were silent (all-zero features). It can still generate coherent text — just without melody influence.
- When MIDI features ARE available, behavior is unchanged.

The melody-influence probe still works correctly because it explicitly passes real MIDI vs. shuffled MIDI; both are non-None, so neither hits the zero-fill path.

---

## 🛡️ NaN/Divergence Defense (post-empirical fix)

After observing failed runs where a single diverged trial poisoned the entire final phase, the code now has three layers of defense against NaN values propagating through the system.

### Layer 1: Quality score zeroed for diverged trials
In `quality_scorer.compute_quality_score`, if the `val_metric` curve contains any `NaN` or `Inf`, the total quality is set to 0. Otherwise a trial that briefly looked good before exploding (e.g. captured loss 2.5 right before going NaN) could win the smoothed-best comparison and become the "best" trial despite producing unusable weights.

### Layer 2: Orchestrator filters by status + verdict
In `orchestrator._execute_and_record`, a trial is eligible to become `best` only if:
- `result.status in ("completed", "early_stopped")` — actually trained to completion
- `diagnosis.verdict not in ("diverged", "failed")` — didn't blow up

Even if Layer 1 somehow gave a non-zero score, this layer rejects diverged/failed trials at the orchestrator level.

### Layer 3: Final-phase abort on NaN
In `final_trainer._refit_and_eval`, if the training loss becomes NaN during refit, the function returns `(NaN, NaN)` immediately and prints a clear warning. The downstream code then:
- Writes `test_evaluation.txt` with NaN values (transparency: the user sees what happened)
- Skips lyrics generation entirely (`final_trainer.run_final_phase`), preventing the CUDA assertion crash that happens when sampling from a model with NaN logits.

This protects against the failure mode where the diverged trial passed Layers 1+2 (e.g. due to a corner case in the curve) but its hyperparameters still produce NaN on the larger combined Train+Val set.

### What the user sees on a divergent run
Instead of a CUDA assertion crash, the user gets:
```
[final] [WARN] refit diverged at epoch 20 (loss=nan). Aborting refit. ...
[final] [WARN] Skipping lyrics generation: refit diverged (test_loss=nan).
```
The model checkpoint is still saved (for debugging), but lyrics generation is skipped.

---

## 🛡️ Additional Defense Layers (post-empirical fix v2)

After another round of empirical analysis, two more bugs were found and fixed:

### Bug: `lr_warmup` was a dead hyperparameter
The `lr_warmup` parameter was defined in YAML (transformer.yaml), proposed by the analyzer (`increase_warmup` action), and adjusted by FTTS — but `_build_scheduler` never consumed it. Setting `lr_warmup=500` had zero effect on training.

**Fix**: `_build_scheduler` now uses `LambdaLR + SequentialLR` to chain a linear warmup phase (0→base_lr over N epochs) with the main scheduler (cosine/step/etc). `reduce_on_plateau` is incompatible with `SequentialLR` and skips warmup.

### Bug: `normalization` was a dead hyperparameter
The `normalization` parameter was tuned by FTTS, but the data loader only reads it from a `cfg._normalization_method` attribute that's never set. Result: changes to `hp["normalization"]` were silently ignored.

**Fix**: Marked `enabled: false` in `mlp.yaml`. The action `change_normalization` is still in ACTION_TYPES (because someone could enable the param manually after fixing the loader), but won't be emitted unless the YAML opts in.

### Bug: LSTM forward crashes on (B,T) input when midi_dim>0
Same root cause as the generator bug: if `midi_feats=None` is passed to `forward()` (e.g. dataset returns 2-tuple instead of 3-tuple), the LSTM gets a 300-dim input instead of the expected 308. Now defensively fills with zeros — consistent with the `step()` fix from the previous session.

### Bug: Generated standalone `model.py` had the same MIDI crash
The `lm_codegen.py` template inherited the original (buggy) `step()` and `midi_for_step()` patterns. Users running the deliverable `model.py` on their own machine would hit the same "Expected 308, got 300" crash. Fixed by mirroring the main-codebase defensive fills.
