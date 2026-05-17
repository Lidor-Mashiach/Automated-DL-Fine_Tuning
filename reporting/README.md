# 📝 Reporting

Two output artifacts are produced per run:

1. **`report.txt`** — a unified, human-readable log of every trial.
2. **`best_trial.png`** — a plot of the best trial's learning curves.

Both are updated **live** during the run. If the run is interrupted (Ctrl+C, crash, etc.), whatever was completed is already saved.

---

## 📂 Files

| File | Role |
|---|---|
| `reporter.py` | Writes `report.txt` (cumulative) |
| `plotter.py` | Writes `best_trial.png` (overwrites on improvement) |

---

## 📋 Report Structure

```
AutoTune-NN - Experiment Report
===============================
<Run configuration>

===============================
Trial-by-trial log
===============================

Trial #1  (trial_id: T0001)
    [Parent]       root (initial trial)
    [Rationale]    Root trial: starting from initial_value fields.
    [Architecture parameters]
        activation              : relu
        hidden_size             : 128
        num_hidden_layers       : 2
    [Structural variants]
        layer_shape             : uniform
    [Regularization / stability methods]
        batch_norm              : True
        dropout                 : 0.1
        ...
    [Optimization]
        learning_rate           : 1.000e-03
        name                    : adam
        ...
    [Training]
        batch_size              : 64
        ...
    [Results]
        status                  : completed
        epochs_completed        : 10
        raw best metric         : 0.843000 (epoch 7)
        smoothed best           : 0.831500 (epoch 6)
        duration (seconds)      : 12.5
    [Quality]
        total                   : 0.7823
        best_metric component   : 0.8315
        stability component     : 0.9210
        convergence_speed comp  : 1.0000
        generalization_gap comp : 0.5500
    [Diagnosis]
        verdict: healthy
        - Completed 10 epochs.
        - Smoothed val_loss: 1.0923 -> min 0.4412 at epoch 6 -> 0.4510 at end.
    [Actions suggested for next trial]
        1. priority=0.55 type=decrease_lr [learning_rate]
           reason: Fine-tune LR downward.
        2. priority=0.50 type=increase_dropout [dropout]
           reason: Stronger regularization may improve generalization.
        ...
    [Conclusion]  Healthy trial; explore minor tweaks.

Trial #2  (trial_id: T0002)
    [Parent]       T0001
    [Rationale]    Based on T0001 (verdict=healthy, quality=0.782). Applied
                   action 'decrease_lr' [priority=0.55]: Fine-tune LR
                   downward. -> learning_rate: 1.000e-03 -> 3.333e-04
    ...
```

### Key elements

- **`[Parent]` + `[Rationale]`** — makes it clear which previous trial influenced this one and why.
- **`[Quality]`** — the full breakdown so you can see whether the win was in accuracy, stability, etc.
- **`[Actions]`** — the Analyzer's prioritized suggestions. Each one says **why**.
- **Smoothed vs raw metrics** — both are shown for transparency.

---

## 📊 Plotting

`plot_best_trial()` writes a 2-panel figure:

- **Left:** train and val `loss` across epochs.
- **Right:** train and val `metric` (accuracy or -RMSE) across epochs.

### Non-blocking

Uses matplotlib's `Agg` backend — no windows are ever opened. This means:

- ✅ Works headless on SLURM clusters.
- ✅ Works in CI/Docker environments.
- ✅ Never blocks the tuning loop.

### Atomic writes

The plotter writes to `best_trial.tmp.png` first, then renames to `best_trial.png`. This prevents ever seeing a half-written file if the process is killed mid-save.

### Only the best

Only **one** plot is kept: the current best trial. When a new best emerges, the old plot is overwritten. The full history of every trial lives in `report.txt`, so no visual information is lost.

---

## 🔄 Recovering After a Crash

Because both artifacts are written live:

- ✅ `report.txt` contains everything up through the last completed trial.
- ✅ `best_trial.png` shows the best trial so far.
- ✅ You can inspect what happened, then either accept the partial result or re-run.

Future versions will support **full resume** (reconstructing the ExperimentTree from disk) — see `FUTURE_WORK.md`.

---

## 🔗 Related Documents

- [`README.md`](../README.md) — output folder structure
- [`core/README.md`](../core/README.md) — Final Refit & Test Evaluation phase

---

## 📊 TensorBoard Logger (`tensorboard_logger.py`)

Optional TensorBoard integration. Logs every trial's training/validation curves under `<run_dir>/tensorboard/`.

### Behavior
- Initializes via `torch.utils.tensorboard.SummaryWriter`.
- **No-ops gracefully** if `tensorboard` is not installed (no exception, just disabled).
- Logs per-trial:
  - `<trial_id>/train_loss`, `<trial_id>/val_loss`
  - `<trial_id>/train_metric`, `<trial_id>/val_metric`
- Logs per run:
  - `trials/quality_score` (per trial)
  - `trials/best_val_loss` (per trial)

### View
```bash
tensorboard --logdir experiments/<run_name>_*/tensorboard
```

### Used by
`core/orchestrator.py` instantiates one logger per run, logs after each trial, and closes it in `_finalize`. Particularly useful for language_modeling runs where the loss curve (not accuracy) is the key signal.

---

## 🎵 Language Modeling Outputs

For `task_type="language_modeling"`, the `final/` directory contains LM-specific deliverables:

- **`model_checkpoint.pt`** - PyTorch state_dict + vocab + hyperparameters + metadata (required for re-generation).
- **`generated_lyrics.txt`** - test-set generations: every test song x every initial word.
- **`melody_probe.json`** - (if `--melody_probe true`) per-song Jaccard, sequence overlap, length diff vs. corrupted MIDI.
- **`decoding_comparison.txt`** - (if `--run_decoding_comparison true`) side-by-side generation of proportional / temperature=0.7 / nucleus p=0.9 on 2 test songs. Used for diversity-vs-coherence analysis (Assignment 3 sec. 13).
- **`model.py`** - real, runnable script with the best hyperparameters inlined.
- **`data/`** - subfolder next to model.py containing the lyrics CSV, MIDI files (and Word2Vec if it fit). Edit/swap to retrain on new inputs.

The standard `report.txt` is still produced under `tuning/` exactly as for other task types.
