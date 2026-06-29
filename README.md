# timeseries_with_chronos

Mortality prediction for NICU infants using heart rate (HR) and oxygen saturation (SpO₂) vital signs. Raw time series are windowed, embedded with [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2), and classified with a small neural network trained via stratified k-fold cross-validation.


## Overview

The pipeline predicts whether a NICU baby will die within **3, 5, or 7 days** of a given observation window. Non-NICU babies (survivors) serve as negative controls.

| Signal | Source | Role |
|--------|--------|------|
| HR, SpO₂ | `.mat` vital sign files | Input time series |
| Chronos-2 | Pretrained foundation model | Embedding extraction |
| MLP classifier | Trained on embeddings | Binary mortality prediction |

Each 80-hour context window is split into 5-minute bins (960 windows). HR and SpO₂ are averaged per bin, passed through Chronos-2, and the final token embeddings for both channels are concatenated into a single feature vector.

## Pipeline

```
Vital signs (.mat)
       │
       ▼
┌──────────────────────────┐
│ 1. Window metadata       │  Generate sliding 80h windows per baby
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 2. Sampling & labeling   │  Sample windows; assign labels per threshold
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 3. Chronos embeddings    │  Embed HR + SpO₂ via amazon/chronos-2
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 4. Classifier training   │  5-fold CV with focal loss (or BCE)
└──────────────────────────┘
```

### Labeling logic

- **Non-NICU babies** → label `0` (no imminent death)
- **NICU babies** → label `1` if the window ends within `threshold_days` of death, else `0`

Thresholds are evaluated independently: 3, 5, and 7 days.

## Directory layout

```
timeseries_with_chronos/
├── README.md
├── .gitignore
│
├── experiment_files/              # Main pipeline scripts
│   ├── main.py                    # Shared config, helpers, and pipeline entry point
│   ├── window_metadata.py         # Step 1 — build sliding-window metadata per baby
│   ├── sampling.py                # Step 2 — sample windows and assign labels
│   ├── training_data.py           # Step 3 — generate/load Chronos embeddings
│   └── train_classifier.py        # Step 4 — train classifier and save metrics/plots
│
├── EDA/
│   └── EDA.ipynb                  # Exploratory analysis of vital sign .mat files
│
├── old_files/                     # Earlier versions of the pipeline
│   ├── chronos.ipynb
│   ├── chronos_1.ipynb
│   └── chronos_1.py
│
└── data/                          # Data and generated artifacts (gitignored)
    ├── valid_summary_dic.pkl      # Baby ID summary; NICU vs non-NICU labels
    ├── final_df.pkl               # Processed dataframe from EDA
    ├── embeddings_global.pkl      # Cached Chronos embeddings for all windows
    ├── windows_to_embed.pkl       # Unique sampled windows to embed
    ├── sampled_windows_all_thresholds.pkl  # Sampled windows with labels (3/5/7 day)
    ├── windows_missing_embeddings.pkl      # Windows still needing embeddings
    └── download/
        ├── Vitals/                # Raw vital sign files (~293 .mat files)
        │   └── NICU_*_vitals.mat  # HR, SpO₂, timestamps, patient metadata
        └── Samples/               # (empty)
```

## Files

### `experiment_files/`

| File | Description |
|------|-------------|
| `main.py` | Constants, pickle helpers, Chronos embedding utilities, and `run_complete_pipeline()` |
| `window_metadata.py` | Creates threshold-independent 80-hour window metadata for each baby |
| `sampling.py` | Samples windows from metadata and labels them for 3-, 5-, and 7-day thresholds |
| `training_data.py` | Detects missing embeddings, runs Chronos-2, and prepares training sets |
| `train_classifier.py` | MLP classifier with k-fold CV; saves results and curve plots |

### `EDA/`

| File | Description |
|------|-------------|
| `EDA.ipynb` | Data quality checks — null rates, coverage, valid file filtering |

### `old_files/`

| File | Description |
|------|-------------|
| `chronos.ipynb` | Original notebook iteration of the pipeline |
| `chronos_1.ipynb` | Updated notebook version |
| `chronos_1.py` | Script export of the notebook pipeline |

### `data/`

| File / folder | Description |
|---------------|-------------|
| `download/Vitals/` | Raw `.mat` vital sign recordings (one file per baby) |
| `download/Samples/` | Placeholder for sample data |
| `valid_summary_dic.pkl` | Valid baby IDs and NICU indicator vector |
| `final_df.pkl` | Cleaned summary dataframe from EDA |
| `embeddings_global.pkl` | Global cache of Chronos embeddings keyed by window |
| `windows_to_embed.pkl` | Deduplicated list of windows selected for embedding |
| `sampled_windows_all_thresholds.pkl` | Labeled samples across all mortality thresholds |
| `windows_missing_embeddings.pkl` | Subset of windows not yet in the embedding cache |

Running the pipeline from `experiment_files/` also writes additional outputs to the working directory (e.g. `window_metadata.pkl`, `labels_*d.pkl`, `results_*d_focal.pkl`, `training_curves_*d_focal/`).

## Requirements

```bash
pip install torch torchvision chronos scikit-learn pandas numpy scipy matplotlib
```

You also need Hugging Face access to download `amazon/chronos-2` on first run.

## Data setup

Download the data from this google drive : 

[Link to data](https://drive.google.com/drive/folders/1MC7tVKk_1s_c2Tme9Wxl5g1wkHoC5G9v?usp=sharing)


Vital sign files live under `data/download/Vitals/`:

```
data/download/Vitals/*.mat
```

Each `.mat` file should contain:

- `vdata` — vital sign columns (HR at index 0, SpO₂ at index 2)
- `vt` — timestamps (seconds)
- `pdata` — patient metadata (birth time, age at death, etc.)

The pipeline also expects a precomputed summary file:

```
valid_summary_dic.pkl
```

This file maps baby IDs to NICU / non-NICU status via an `indicator_vector`. A copy is stored in `data/valid_summary_dic.pkl`.

## Usage

Run the full pipeline from `experiment_files/`:

```bash
cd experiment_files
python main.py
```

By default, `main.py` runs with **focal loss** (`γ=2.0`). To use cross-entropy instead, edit the call at the bottom of `main.py`:

```python
run_complete_pipeline(use_focal_loss=False, gamma=2.0)
```

Intermediate artifacts are cached so reruns skip completed steps:

| File | Description |
|------|-------------|
| `window_metadata.pkl` | All candidate windows per baby |
| `windows_to_embed.pkl` | Unique sampled windows |
| `sampled_windows_all_thresholds.pkl` | Sampled windows with labels per threshold |
| `embeddings_global.pkl` | Cached Chronos embeddings |
| `labels_{3,5,7}d.pkl` | Embedding keys and labels per threshold |
| `results_{3,5,7}d_focal.pkl` | Cross-validation metrics |
| `training_curves_{3,5,7}d_focal/` | Loss, PR, and ROC curves per fold |

Delete a cache file to regenerate that step.

## Key configuration

Defined in `experiment_files/main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONTEXT_SIZE_HOURS` | 80 | Length of each observation window |
| `WINDOW_SIZE_SECONDS` | 300 | Bin size for averaging vitals (5 min) |
| `THRESHOLDS_DAYS` | [3, 5, 7] | Mortality prediction horizons |
| `NUM_SAMPLE_WINDOWS_NICU` | 100 | Windows sampled per NICU baby |
| `NUM_SAMPLE_WINDOWS_NON_NICU` | 5 | Windows sampled per non-NICU baby |
| `NICU_LATE_WINDOW_BIAS` | 0.92 | Fraction of NICU samples drawn from later windows |

## Evaluation

The classifier is a 3-layer MLP trained with 5-fold stratified cross-validation. For each threshold, the pipeline reports:

- Accuracy, precision, recall, F1, ROC-AUC
- Per-fold and mean ± std metrics
- Training/validation loss curves, PR curves, and ROC curves

Focal loss uses an automatically computed `α` based on class imbalance.

## Notes

- The `data/` directory is listed in `.gitignore` and is not tracked by git.
- Pipeline scripts reference `./download/Vitals/*.mat`; symlink or copy `data/download/` into the working directory, or update the path in `main.py`, before running locally.
