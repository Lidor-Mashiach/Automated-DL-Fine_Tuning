# 📁 Data Folder

This folder holds user-provided datasets for **local mode**.

---

## 🗂️ Supported Inputs

### Tabular (any of the three formats)

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | First row = headers. Last column = label by default. |
| NumPy | `.npy` | 2-D array. Last column = label. |
| Parquet | `.parquet` | Columnar; pandas-readable. |

### Image
A folder in ImageFolder format (one subfolder per class):
```
Data/Data-Set/
├── class_a/
│   ├── img01.jpg
│   └── img02.jpg
└── class_b/
    └── img03.jpg
```

### Text
CSV with `text` and `label` columns (customizable via `FEATURE_COLUMNS` / `LABEL_COLUMN` in `main.py`).

---

## 🔀 Auto-Generated Splits

On the first run, AutoTune-NN creates a sub-folder for each dataset to avoid collisions when running multiple datasets in parallel:

```
Data/
├── Data-Set.csv                  ← your file
├── Data-Set_split/               ← created automatically
│   ├── Train_set.csv             ← TRAIN_PCT of the data
│   ├── Val_set.csv               ← VAL_PCT of the data
│   └── Test_set.csv              ← TEST_PCT (held out, not touched in tuning)
│
├── Customers.csv                 ← another dataset, run in parallel
└── Customers_split/
    ├── Train_set.csv
    ├── Val_set.csv
    └── Test_set.csv
```

The split is deterministic via `RANDOM_SEED` from `main.py` — rerunning produces identical splits. To force a fresh split, delete the `<filename>_split/` folder.

---

## 🖼️ Image Data

For image data, files are NOT physically duplicated into split folders (would be expensive). Instead, the framework deterministically partitions the indices using `RANDOM_SEED` on every run.

---

## 🌐 Imported Datasets

When `DATASET_MODE = "imported"` (MNIST, Iris, etc.), this folder is **not used**. The dataset downloads to `~/.cache/autotune_nn/`.

Datasets with a built-in train/test split (MNIST, CIFAR) preserve the original test set; the original train set is split into Train+Val using `TRAIN_PCT:VAL_PCT` as a ratio. See [`data_loaders/README.md`](../data_loaders/README.md) for the full table.

---

## 📄 Sample File

`Data-Set.csv` is a synthetic XOR-like binary classification dataset (500 rows, 3 features) provided for quick smoke-testing. Replace with your real data when ready.

---

## 🎵 Lyrics Dataset (language modeling)

For `task_type="language_modeling"`, the CSV is expected in this format (no header required):

```csv
elton john,candle in the wind,goodbye norma jean & though i never knew you at all & you had the grace ...
gerry rafferty,baker street,winding your way down on baker street & lite in your head and dead on your feet & ...
```

| Column | Content |
|---|---|
| 1 | Artist name |
| 2 | Song name |
| 3 | Full lyrics (one cell). Lines are separated by `&` (configurable via `--line_separator_token`). |

### MIDI files

Place .mid files in a separate directory and pass via `--midi_dir`. Filenames must follow:

```
<artist_with_underscores>_-_<song_with_underscores>.mid
```

Example: `Elton_John_-_Candle_in_the_Wind.mid` matches `"elton john","candle in the wind"`.

Songs without a matching MIDI file get a zero feature vector (the model still trains as a baseline).

### Word2Vec embeddings

Pretrained Google News word2vec (300-dim) is recommended:
- Download `GoogleNews-vectors-negative300.bin.gz` (~1.5 GB)
- Pass with `--word2vec_path /path/to/GoogleNews-vectors-negative300.bin`

If the file is `.txt` format (GloVe-style), it's parsed directly without gensim.
If neither is available, embeddings are randomly initialized.
