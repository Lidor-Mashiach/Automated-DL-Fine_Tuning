# ⚙️ Configuration

All tunable settings live in YAML files under two subfolders:

| Folder | What it controls |
|---|---|
| [`architectures/`](architectures/) | Per-architecture parameters (MLP, CNN, RNN, LSTM, Transformer) — depth, width, activations, regularization methods, optimizer, training settings. |
| [`strategies/`](strategies/) | Per-search-strategy settings (FTTS, Bayesian, Grid) — stopping conditions, quality score weights, step sizes, parallelism. |

---

## 📋 Parameter Configuration Format

All architecture YAML files use a uniform parameter format:

```yaml
learning_rate:
  enabled: true              # if false, this parameter is never used in any trial
  initial_value: 1.0e-3      # starting value for the root trial (null = system picks)
  range: [1.0e-5, 1.0e-1]    # hard boundary - Analyzer never proposes outside this
  log: true                  # log-scale sampling for parameters spanning orders of magnitude
```

### The four modes

| Configuration | Meaning |
|---|---|
| `enabled: true` + `initial_value: X` | Parameter active, root trial starts at X. Analyzer tunes from there. |
| `enabled: true` + `initial_value: null` | Parameter active, system picks the middle of `range`/`choices`. |
| `enabled: false` | Parameter never used, regardless of anything else. |

### `range` vs `choices`

- **`range: [min, max]`** — for continuous parameters (learning_rate, dropout, weight_decay). The Analyzer can move to any value in the range.
- **`choices: [a, b, c]`** — for discrete parameters (activation, optimizer name, layer_shape). The Analyzer can only pick from these.

### `log: true`

Use this for parameters that span **orders of magnitude**, like `learning_rate` (1e-5 to 1e-1 is 4 orders) or `weight_decay` (1e-6 to 1e-2).

> 💡 **Why?** Linear sampling of `[1e-5, 1e-1]` gives you mostly values near `1e-1`. Log-scale sampling gives equal weight to each order of magnitude (1e-5, 1e-4, 1e-3, ...) which is how these parameters actually behave.

---

## 📖 Full Parameter Reference

See individual files:

- 📁 [`architectures/README.md`](architectures/README.md) — sections in architecture YAMLs and per-architecture guidance.
- 📁 [`strategies/README.md`](strategies/README.md) — what each strategy config controls.

Each YAML file itself is heavily commented — every parameter has a docstring above it.
