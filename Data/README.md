# 📁 Data Folder

This folder holds user-provided datasets for **local mode**.

---

## 🗂️ Files

### Input file (you provide)

- `Data-Set.csv` — your tabular or text dataset.

Or, for image data, a folder `Data-Set/` in ImageFolder format:

```
Data-Set/
├── class_a/
│   ├── img01.jpg
│   └── img02.jpg
└── class_b/
    └── img03.jpg
```

### Auto-generated files (created on first run)

- `Train_set.csv` — `TRAIN_PCT` of the data. Used to train the model.
- `Val_set.csv` — `VAL_PCT` of the data. Used inside the tuning loop for scoring and early stopping.
- `Test_set.csv` — `TEST_PCT` of the data. **Held out** — not used during tuning.

The split is deterministic (uses `RANDOM_SEED` from `main.py`), so rerunning produces identical splits.

---

## 🔄 Re-splitting

To force a fresh split, delete `Train_set.csv`, `Val_set.csv`, and `Test_set.csv`. The next run will regenerate them from `Data-Set.csv`.

---

## 🖼️ Image Data

For image data, files are **not physically duplicated** into `Train_set/`, `Val_set/`, `Test_set/` folders (that would be expensive for large datasets). Instead, the framework deterministically partitions the indices of the `ImageFolder` on every run using `RANDOM_SEED`.

---

## 🌐 Imported Datasets

If `DATASET_MODE = "imported"` in `main.py`, this folder is **not used**. The dataset is downloaded to `~/.cache/autotune_nn/` to keep your project clean.

---

## 📄 Current Contents

`Data-Set.csv` is a synthetic XOR-like binary classification dataset (500 rows, 3 features) provided for quick smoke-testing. Replace it with your real data when ready.
