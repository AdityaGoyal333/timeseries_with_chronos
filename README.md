# NICU Mortality Prediction with Chronos-2

Binary mortality forecasting for NICU infants using continuous **heart rate (HR)** and **oxygen saturation (SpO₂)** vital signs. Each 80-hour observation window is binned, embedded with the frozen foundation model [**Amazon Chronos-2**](https://huggingface.co/amazon/chronos-2), and classified with a small PyTorch MLP. The pipeline is **resumable** (cached intermediates), **threshold-independent** for embeddings (embed once, train three classifiers), and uses **baby-grouped cross-validation** to avoid validation leakage.

---

## Table of contents

1. [Research question](#research-question)
2. [End-to-end journey (data → model → outputs)](#end-to-end-journey-data--model--outputs)
3. [Cohort and labeling](#cohort-and-labeling)
4. [Pipeline steps (detailed)](#pipeline-steps-detailed)
5. [What the model actually sees](#what-the-model-actually-sees)
6. [Classifier, loss, and training](#classifier-loss-and-training)
7. [Evaluation and outputs](#evaluation-and-outputs)
8. [Cross-validation and leakage](#cross-validation-and-leakage)
9. [Repository layout](#repository-layout)
10. [Data setup](#data-setup)
11. [Installation](#installation)
12. [Running the pipeline](#running-the-pipeline)
13. [Configuration reference](#configuration-reference)
14. [Caching and regeneration](#caching-and-regeneration)
15. [Baselines and analysis notebooks](#baselines-and-analysis-notebooks)
16. [Known limitations and TODOs](#known-limitations-and-todos)

---

## Research question

> Given 80 hours of HR and SpO₂ vitals ending at time *t*, will this infant die within the next **3, 5, or 7 days**?

This is framed as **three separate binary classification tasks** (one classifier per horizon). The same physical windows are reused across thresholds; only the labels change.

| Cohort | Definition | Label |
|--------|------------|-------|
| **NICU (deceased)** | In-hospital death recorded in vitals metadata | `1` if window end falls inside the last *N* days before death; else `0` |
| **Non-NICU (survivors)** | No death timestamp | Always `0` (negative controls) |

---

## End-to-end journey (data → model → outputs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0 — Upstream (EDA, external)                                          │
│   Raw .mat vitals  →  quality filters  →  valid_summary_dic.pkl             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — Window enumeration (Step 1)                                       │
│   Each baby .mat  →  sliding 80h windows  →  window_metadata.pkl            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — Sampling + labeling (Step 2)                                      │
│   Downsample windows per baby  →  assign 3d/5d/7d labels                    │
│   →  windows_to_embed.pkl  +  sampled_windows_all_thresholds.pkl            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — Chronos embedding (Step 3)                                        │
│   Each unique window  →  HR/SpO₂ binned series  →  Chronos-2 embed          │
│   →  embeddings_global.pkl  (keyed by baby_id + time bounds)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4 — Training tensors (Step 4)                                         │
│   Join labels + cached embeddings per threshold  →  labels_{N}d.pkl          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5 — Classifier (Step 5)                                               │
│   StratifiedGroupKFold by baby_id  →  MLP + focal loss  →  metrics + curves  │
│   →  results_{N}d_focal.pkl  +  training_curves_{N}d_focal/                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**One training sample** = one tuple `(Chronos embedding vector, binary label, baby_id)` for a specific 80-hour window.

Typical scale (current run): **~19,015 windows**, **~4,244 unique babies**, **embedding dim 1,536**, positive rate ~3.7% (3d) / ~5.6% (5d) / ~7.2% (7d).

---

## Cohort and labeling

### Phase 0: Who enters the pipeline?

Before `main.py` runs, `EDA/EDA.ipynb` builds `data/valid_summary_dic.pkl`:

- Loads all `NICU_*_vitals.mat` files
- Validates timestamps vs birth time and age at death
- Computes **hit rate** (fraction of 5-minute bins with valid HR and SpO₂)
- Keeps babies with **≥ 70% coverage**
- Stores `baby_id`, coverage stats, and `indicator_vector` (`1` = NICU/deceased, `0` = survivor)

`main.py` then matches `.mat` filenames (full filename including `.mat`) against `valid_summary_dic['baby_id']` to split NICU vs non-NICU files.

### Label rule (Step 2)

For each threshold *N* ∈ {3, 5, 7} days:

```
absolute_death   = birth_time + death_time        # death_time in seconds since birth
threshold_start  = absolute_death - N * 86400

NICU:     label = 1  if  window_end > threshold_start
          label = 0  otherwise
Non-NICU: label = 0  always
```

**Timeline (7-day example):**

```
birth ──────── window_end? ──────── |← 7 days →| death
                                    threshold_start
```

A NICU baby can have **both** label-0 windows (early) and label-1 windows (late) in the same recording. Non-NICU babies contribute negatives only.

---

## Pipeline steps (detailed)

### Step 1 — Window metadata (`window_metadata.py`)

**Input:** list of NICU and non-NICU `.mat` paths  
**Output:** `data/window_metadata.pkl`

For each baby, slide an **80-hour** context window along the recording:

| Baby type | Overlap | Max candidate windows/baby |
|-----------|---------|----------------------------|
| Non-NICU | 0% | 15 |
| NICU | 90% | 200 |

If more windows fit than the cap, **evenly spaced indices** are kept (preserves early + late coverage).

Each row stores:

| Column | Meaning |
|--------|---------|
| `mat_file` | Path to `.mat` |
| `baby_id` | Filename without `.mat` |
| `window_start`, `window_end` | Absolute timestamps (seconds) |
| `baby_type` | `NICU` or `NON_NICU` |
| `birth_time` | Absolute timestamp (seconds) |
| `death_time` | Seconds since birth (NaN for survivors) |

**Threshold-independent** — labels are not assigned here.

---

### Step 2 — Sampling and labeling (`sampling.py`)

**Input:** `window_metadata.pkl`  
**Outputs:**
- `data/windows_to_embed.pkl` — unique windows (no labels; for Chronos)
- `data/sampled_windows_all_thresholds.pkl` — same windows × 3 thresholds with labels

**Downsampling** (class-balance aware):

| Baby type | Windows kept/baby | Strategy |
|-----------|-------------------|----------|
| Non-NICU | 5 | Evenly spaced |
| NICU | 100 | **92%** from late (pre-death) half of timeline, 8% from early half |

**Key design:** embed each window **once**; reuse the same embedding for 3d, 5d, and 7d classifiers with different labels.

---

### Step 3 — Chronos embeddings (`training_data.py`)

**Input:** `windows_to_embed.pkl`  
**Output:** `data/embeddings_global.pkl` (incremental dict cache)

For each window:

1. **Load vitals** from `.mat` in `[window_start, window_end)`
   - `vdata[:, 0]` → HR
   - `vdata[:, 2]` → SpO₂
   - Filter to non-NaN HR and SpO₂ pairs

2. **Bin into 960 × 5-minute sub-windows**
   - Mean HR and mean SpO₂ per bin
   - Empty bins → NaN (filled later)

3. **Prepare Chronos input** tensor shape `(2, 1, 960)`
   - Channel 0 = HR, channel 1 = SpO₂
   - NaN bins → channel mean (or 0 if all NaN)

4. **Run frozen Chronos-2** (`amazon/chronos-2`, CPU, `bfloat16`)
   - `pipeline.embed(chronos_input)`
   - Take **last-token** embedding from HR channel + last-token from SpO₂ channel
   - **Concatenate** → vector of dim **1536**

5. **Cache** under key:
   ```
   {baby_id}_{window_start}_{window_end}
   ```
   Example: `NICU_123_vitals_288000_576800`

Embedding is expensive; cache checkpoints every **500** new windows. Re-runs only embed missing keys.

---

### Step 4 — Prepare training data (`training_data.py`)

**Input:** `sampled_windows_all_thresholds.pkl` + `embeddings_global.pkl`  
**Output:** `labels_{N}d.pkl` (written to **current working directory**)

For threshold *N*:

```python
{
    'embedding_keys': [...],   # cache keys, parallel to samples
    'labels':         [...],   # 0/1 ints
    'baby_ids':       [...],   # one baby_id per sample (for grouped CV)
}
```

Windows missing from the embedding cache are **silently skipped**.

Returns in memory: `(embeddings, labels, baby_ids)` lists passed to training.

---

### Step 5 — Classifier training (`train_classifier.py`)

See [Classifier, loss, and training](#classifier-loss-and-training) and [Evaluation](#evaluation-and-outputs).

---

## What the model actually sees

The MLP **never** receives:

- Raw `.mat` data
- `baby_id` (used only for CV splitting)
- Patient metadata (gestational age, birth weight, etc.)
- Threshold identity (each horizon is a separate trained model)

The MLP **does** receive:

| Input | Shape | Source |
|-------|-------|--------|
| Chronos embedding | `(1536,)` | Frozen Chronos-2 last-token HR ∥ SpO₂ |
| Label (training only) | scalar 0/1 | Step 2 labeling rule |

**Forward pass:**

```
embedding (1536) → Linear(512) → ReLU
                 → Linear(512) → ReLU
                 → Linear(256) → ReLU
                 → Linear(1)   → logit
                 → sigmoid       → P(death within N days)
```

**Output layer bias** is initialized to `log(p / (1−p))` where `p` = positive rate in that fold's **training set** only.

---

## Classifier, loss, and training

### Architecture (`Classifier`)

4-layer MLP: `dim → 512 → 512 → 256 → 1`

### Loss

| Mode | Function | Default |
|------|----------|---------|
| Focal | `torchvision.ops.sigmoid_focal_loss` | `α=0.95`, `γ=2.0` |
| BCE | `binary_cross_entropy_with_logits` | optional via `use_focal_loss=False` |

Note: `_focal_alpha()` returns a **fixed** `0.95`, not a value computed from class prevalence (despite the docstring).

### Training loop (per fold)

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | `0.001` |
| Epochs | `250` |
| Batch size | `400` |
| Device | CPU (default) |
| Train batches | `UniformClassBatchSampler` — spreads positives evenly across batches |
| Val loss | Computed every **5** epochs |
| Model | **Fresh MLP per fold** (weights not shared across folds) |

### Imbalance handling

1. Reduced non-NICU sampling (5 vs 100 windows/baby)
2. NICU late-window bias (92%)
3. Focal loss
4. Prevalence-based output bias init
5. `UniformClassBatchSampler`

---

## Evaluation and outputs

### Metrics (per fold, then mean ± std)

| Metric | At threshold | Notes |
|--------|--------------|-------|
| Accuracy, precision, recall, F1 | 0.5 | From hard predictions |
| ROC-AUC | — | From predicted probabilities |
| Average precision (AP) | — | PR-AUC; compare to `prevalence` baseline |
| Prevalence | — | Positive rate in val fold |

### Files written per threshold

**In working directory** (where you run the pipeline):

| File | Contents |
|------|----------|
| `labels_{3,5,7}d.pkl` | `embedding_keys`, `labels`, `baby_ids` |
| `results_{3,5,7}d_focal.pkl` | Full experiment record (see below) |

**In `training_curves_{N}d_focal/`:**

| File | Contents |
|------|----------|
| `fold_{i}_loss.png` | Train loss (every epoch) + val loss (every 5 epochs) |
| `fold_{i}_interval_loss.png` | Train loss averaged every 5 batches |
| `fold_{i}_pr.png` | Per-fold precision–recall curve |
| `fold_{i}_roc.png` | Per-fold ROC curve |
| `PR curve.png` | **Cumulative** mean PR ± std across folds |
| `ROC curve.png` | **Cumulative** mean ROC ± std across folds |
| `fold_probabilities.pkl` | Per-fold val labels, probs, and precomputed curve arrays |

### `results_{N}d_focal.pkl` structure

```python
{
    'avg':                dict,   # mean metrics across folds
    'std':                dict,   # std across folds
    'fold_metrics':       list,   # scalar metrics per fold
    'fold_curve_data':    list,   # loss histories + pr/roc arrays per fold
    'cumulative_curves':  dict,   # mean PR/ROC grids + paths
    'fold_probabilities': dict,   # same as fold_probabilities.pkl
    'curve_output_dir':   str,
    'loss_type':          str,
    'gamma':              float,
}
```

### Cumulative PR/ROC

Built by **interpolating** each fold's PR (or ROC) curve onto a fixed grid (1000 points), then averaging pointwise with ±1 std shading. Legend AUC/AP = **mean of per-fold** AUC/AP values.

Per-fold curve arrays are computed **once** and reused (not recomputed from raw probs for the cumulative plot).

---

## Cross-validation and leakage

### Baby-grouped CV (current default)

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
split(..., groups=baby_ids)
```

| Property | Behavior |
|----------|----------|
| Split unit | **baby_id** — all windows from one infant stay in train **or** val |
| Class balance | Approximate across folds (stratified on window labels) |
| Leakage check | Runtime `assert` — no baby in both train and val |

**Why:** windows from the same baby share correlated physiology. Window-level CV lets the model memorize baby-specific patterns in training and evaluate on the same baby in validation — **optimistic metrics**.

### What is NOT leakage

| Concern | Status |
|---------|--------|
| Chronos pretrained on all windows before CV | OK — Chronos is **frozen** and does not see labels |
| Label uses future death time | OK — that's the **task definition** |
| Correlated windows within train | Expected — val still uses **unseen babies** |

### Validate the pipeline

```bash
cd experiment_files
python validate_pipeline.py
```

---

## Repository layout

```
timeseries_with_chronos/
├── README.md
├── requirements.txt              # Pinned dependencies (use venv/)
├── .gitignore
│
├── experiment_files/             # ★ Main pipeline (run from here or repo root)
│   ├── main.py                   # Orchestrator: run_complete_pipeline()
│   ├── constants.py              # Window geometry, thresholds, sampling caps
│   ├── window_metadata.py        # Step 1
│   ├── sampling.py               # Step 2
│   ├── training_data.py          # Steps 3–4
│   ├── train_classifier.py       # Step 5
│   └── validate_pipeline.py      # Alignment + leakage checks
│
├── data/                         # Gitignored — raw + cached artifacts
│   ├── valid_summary_dic.pkl     # Cohort filter (from EDA)
│   ├── window_metadata.pkl
│   ├── windows_to_embed.pkl
│   ├── sampled_windows_all_thresholds.pkl
│   ├── embeddings_global.pkl     # ~118 MB+; Chronos vectors
│   ├── windows_missing_embeddings.pkl
│   └── download/
│       └── Vitals/               # ~6,000 NICU_*_vitals.mat files
│
├── EDA/
│   ├── EDA.ipynb                 # Data QC → valid_summary_dic.pkl
│   └── fold_cumulated_curves.ipynb
│
├── lr_baseline/                  # Logistic regression on engineered features
│   ├── papermodeldata(in).csv
│   ├── test.ipynb
│   └── fold_cumulated_curves.ipynb
│
├── old_files/                    # Monolithic earlier pipeline versions
│   ├── chronos.ipynb
│   ├── chronos_1.ipynb
│   └── chronos_1.py
│
├── temporary/                    # Ad-hoc analysis scripts (gitignored)
├── docs/                         # Learning notes (e.g. baby-grouped CV)
│
├── labels_{3,5,7}d.pkl           # Written at training time (cwd-dependent)
├── results_{3,5,7}d_focal.pkl
└── training_curves_{3,5,7}d_focal/
```

**Note:** If you run from `experiment_files/`, `labels_*` and `results_*` land in `experiment_files/`. If you run from repo root (as in recent runs), they land in `timeseries_with_chronos/`.

---

## Data setup

### Download

[Google Drive data folder](https://drive.google.com/drive/folders/1xU-i8ymta5sn0TPjZKrNcSAj-yx7Sucf?usp=sharing)

### Place files

```
data/download/Vitals/NICU_*_vitals.mat
```

### `.mat` file structure

| Field | Description |
|-------|-------------|
| `vdata[:, 0]` | Heart rate |
| `vdata[:, 2]` | SpO₂ (column 1 unused) |
| `vt` | Sample timestamps (seconds) |
| `pdata[0][1]` | Birth time |
| `pdata[0][-1]` | Age at death in days (NaN if alive) |

### Precomputed cohort file

```
data/valid_summary_dic.pkl
```

Built by `EDA/EDA.ipynb`. Required before running `main.py`.

---

## Installation

```bash
cd timeseries_with_chronos
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `chronos-forecasting`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `transformers`, `huggingface_hub`.

**Chronos-2** downloads from Hugging Face on first embedding run (`amazon/chronos-2`). Ensure HF access / token if needed.

---

## Running the pipeline

```bash
cd experiment_files
python main.py
```

Default: **focal loss**, `γ=2.0`, baby-grouped 5-fold CV, all three thresholds sequentially.

### Switch to cross-entropy

Edit the bottom of `main.py`:

```python
run_complete_pipeline(use_focal_loss=False, gamma=2.0)
```

### Re-train one threshold from cache

```python
from training_data import load_training_data_for_threshold
from train_classifier import train_and_save_results

embeddings, labels, baby_ids = load_training_data_for_threshold(7)
train_and_save_results(7, embeddings, labels, baby_ids=baby_ids)
```

### Expected log lines

```
K-fold CV | Focal (alpha=0.950, gamma=2.0) | StratifiedGroupKFold (by baby_id)
Samples: 19015 | Pos rate: 0.072 (1362 positives)
Babies: 4244 | Folds: 5
```

---

## Configuration reference

All tunables live in `experiment_files/constants.py` unless noted.

### Window geometry

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CONTEXT_SIZE_HOURS` | 80 | Observation window length |
| `WINDOW_SIZE_SECONDS` | 300 | Bin size (5 min) |
| `NUM_WINDOWS` | 960 | Bins per window |

### Thresholds

| Parameter | Value |
|-----------|-------|
| `THRESHOLDS_DAYS` | `[3, 5, 7]` |

### Step 1 — metadata caps

| Parameter | Non-NICU | NICU |
|-----------|----------|------|
| `OVERLAP_PERCENTAGE_*` | 0% | 90% |
| `MAX_METADATA_WINDOWS_*` | 15 | 200 |

### Step 2 — sampling

| Parameter | Value |
|-----------|-------|
| `NUM_SAMPLE_WINDOWS_NON_NICU` | 5 |
| `NUM_SAMPLE_WINDOWS_NICU` | 100 |
| `NICU_LATE_WINDOW_BIAS` | 0.92 |

### Step 5 — training (`train_and_save_results` defaults)

| Parameter | Value |
|-----------|-------|
| `k` | 5 |
| `epochs` | 250 |
| `batch_size` | 400 |
| `lr` | 0.001 |
| `gamma` (focal) | 2.0 |
| `alpha` (focal) | 0.95 (fixed) |

### Paths (`main.py`)

| Variable | Location |
|----------|----------|
| `VALID_SUMMARY_PATH` | `data/valid_summary_dic.pkl` |
| `VITALS_DIR` | `data/download/Vitals/` |
| `WINDOW_METADATA_PATH` | `data/window_metadata.pkl` |
| `WINDOWS_TO_EMBED_PATH` | `data/windows_to_embed.pkl` |
| `SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH` | `data/sampled_windows_all_thresholds.pkl` |
| `EMBEDDINGS_CACHE_PATH` | `data/embeddings_global.pkl` |

---

## Caching and regeneration

The pipeline skips completed steps when cache files exist.

| Delete this file | To regenerate |
|------------------|---------------|
| `data/window_metadata.pkl` | Step 1 (window enumeration) |
| `data/windows_to_embed.pkl` **and** `data/sampled_windows_all_thresholds.pkl` | Step 2 (sampling + labels) |
| `data/embeddings_global.pkl` | Step 3 (all embeddings) |
| `labels_{N}d.pkl` | Step 4 for threshold *N* |
| `results_{N}d_focal.pkl` | Step 5 for threshold *N* |

**Important:** Step 2 skip only checks `windows_to_embed.pkl`. If you delete `sampled_windows_all_thresholds.pkl` but keep `windows_to_embed.pkl`, Step 2 will not re-run and labeling will break. Delete **both** to regenerate Step 2.

Step 3 is **incremental** — only missing keys are embedded.

---

## Baselines and analysis notebooks

| Location | Purpose |
|----------|---------|
| `lr_baseline/test.ipynb` | Logistic regression on hand-engineered HR/SpO₂ features (`papermodeldata(in).csv`) |
| `lr_baseline/fold_cumulated_curves.ipynb` | Aggregate PR/ROC for baseline |
| `EDA/EDA.ipynb` | Data quality, coverage filtering, cohort construction |
| `EDA/fold_cumulated_curves.ipynb` | Aggregate PR/ROC across Chronos pipeline folds |
| `old_files/` | Earlier all-in-one notebook/script versions (superseded by `experiment_files/`) |

---

## Known limitations and TODOs

- **NaN imputation:** empty 5-min bins filled with channel mean before Chronos (may bias embeddings)
- **Focal α:** hardcoded at 0.95; not computed from label prevalence
- **No patient metadata** in the classifier (gestational age, birth weight explored in EDA only)
- **Sampling bias:** positives are a subset of all possible windows (100 NICU windows/baby, late-biased)
- **CPU-only Chronos** by default — embedding is slow (~hours for full cache)
- **Metrics at 0.5 threshold** for F1/precision/recall may be suboptimal for rare positives
- **Old experiment results** (before baby-grouped CV) are not comparable to current runs

See inline TODOs in `main.py` for future work (W&B, masked tokens, metadata fusion, etc.).

---

## Quick reference: one sample's journey

```
NICU_456_vitals.mat
  └─ window [518400s, 806400s]           # 80 hours
       └─ 960 × (mean HR, mean SpO₂)      # 5-min bins
            └─ Chronos-2 embed             # frozen, no labels
                 └─ vector (1536,)         # HR_last_token ∥ SpO₂_last_token
                      └─ key: NICU_456_vitals_518400_806400
                           └─ label: 1     # if 7d threshold & near death
                                └─ baby_id: NICU_456_vitals
                                     └─ MLP(embedding) → P(death within 7d)
                                          └─ evaluated on val fold
                                               (all of NICU_456_vitals in train OR val, never both)
```

---

## Citation / model

- Chronos-2: [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon Science)
- Package: `chronos-forecasting` ([Chronos forecasting library](https://github.com/amazon-science/chronos-forecasting))
