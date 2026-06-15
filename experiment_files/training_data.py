"""Step 3–4: Generate missing Chronos embeddings and prepare training data.

Pipeline role
-------------
Step 3:  Run Amazon Chronos-2 on each unique sampled window → fixed-size
         embedding vector. Results are cached globally so we never recompute.

Step 4:  For each death-proximity threshold, pair cached embeddings with
         their threshold-specific labels → ready for classifier training.

Two-file design
---------------
embeddings_cache.pkl   —  large dict: window_key → embedding vector
labels_{N}d.pkl        —  small file: list of keys + labels for one threshold

Storing keys instead of duplicating embeddings saves disk space because
the same embedding is reused across all thresholds.
"""

import os
import pickle as pl
from pathlib import Path
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
import torch
import scipy.io

from constants import (
    CONTEXT_SIZE_SECONDS,
    WINDOW_SIZE_SECONDS
)



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()
MISSING_WINDOWS_PATH = (DATA_DIR / 'windows_missing_embeddings.pkl').resolve()
EMBEDDINGS_CACHE_PATH = (DATA_DIR / 'embeddings_global.pkl').resolve()



def prepare_chronos_input(hr_windows, spo2_windows):
    """Format HR + SpO2 sub-window arrays as Chronos-2 model input.

    - NaN bins are filled with the channel mean (or 0 if all NaN).
    - Returns a (2, 1, num_sub_windows) tensor: channel 0 = HR, channel 1 = SpO2.
    """
    hr_clean = np.where(
        np.isnan(hr_windows),
        np.nanmean(hr_windows) if not np.all(np.isnan(hr_windows)) else 0,
        hr_windows,
    ).astype(np.float32)

    spo2_clean = np.where(
        np.isnan(spo2_windows),
        np.nanmean(spo2_windows) if not np.all(np.isnan(spo2_windows)) else 0,
        spo2_windows,
    ).astype(np.float32)

    hr_tensor = torch.tensor(hr_clean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spo2_tensor = torch.tensor(spo2_clean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return torch.cat([hr_tensor, spo2_tensor], dim=0)


def get_chronos_embedding(chronos_input, pipeline):
    """Run Chronos-2 embed() and concatenate the last-token HR + SpO2 embeddings."""
    with torch.no_grad():
        embeddings, _ = pipeline.embed(chronos_input)

    # Take the final time-step embedding from each channel
    hr_emb = embeddings[0][0][-1].float()
    spo2_emb = embeddings[1][0][-1].float()
    return torch.cat([hr_emb, spo2_emb], dim=0).cpu().numpy()


def get_embedding(hr_windows, spo2_windows, pipeline):
    """End-to-end: sub-window arrays → Chronos input → concatenated embedding vector."""
    chronos_input = prepare_chronos_input(hr_windows, spo2_windows)
    return get_chronos_embedding(chronos_input, pipeline)

def extract_raw_timeseries(mat_file_path, start_time, context_size):
    """Load HR and SpO2 samples from a .mat file within [start, start + context).

    Args:
        mat_file_path: Path to the baby's vital-sign .mat file.
        start_time: Absolute window start (seconds, same units as vt).
        context_size: Window length in seconds (typically CONTEXT_SIZE_SECONDS).

    Returns:
        Dict with filtered timestamps, HR, SpO2 arrays — or None if no valid data.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return None


    # ((birth_time+age_at_death)*86400 - vt)/86400 < threshold_days

    birth_time = mat_data['pdata'][0][1]  # in days realtive to the global start time
    age_at_death = mat_data['pdata'][0][-1]  # the age in days at death
    death_ts = age_at_death * 86400 if not np.isnan(age_at_death) else None  # seconds since birth; None if alive  

    # vdata columns: 0 = heart rate, 2 = SpO2 (column 1 is unused here)
    hr_cols = mat_data['vdata'][:, 0]
    spo2_cols = mat_data['vdata'][:, 2]

    # vdata is seconds from birth time, not accounting the time of birth 
    timestamps = mat_data['vt'].flatten()

    end_time = start_time + context_size
    time_mask = (timestamps >= start_time) & (timestamps < end_time)
    valid_mask = ~np.isnan(hr_cols) & ~np.isnan(spo2_cols)
    final_mask = time_mask & valid_mask

    timestamps_valid = timestamps[final_mask]
    hr_valid = hr_cols[final_mask]
    spo2_valid = spo2_cols[final_mask]

    if len(timestamps_valid) == 0:
        return None

    return {
        'timestamps': timestamps_valid,
        'hr_raw': hr_valid,
        'spo2_raw': spo2_valid,
        'birth_time': birth_time,
        'death_ts': death_ts,
        'start_time': start_time,
        'context_size': context_size,
    }


def create_windowed_features_from_raw(raw_data, window_size):
    """Aggregate raw samples into fixed-length sub-window averages for Chronos.

    Divides the 80-hour context into NUM_WINDOWS bins (default 960 × 5 min).
    Each bin gets the mean HR and mean SpO2 of all samples falling in it.
    Bins with no samples remain NaN (filled later in prepare_chronos_input).
    """
    timestamps = raw_data['timestamps']
    hr_raw = raw_data['hr_raw']
    spo2_raw = raw_data['spo2_raw']
    start_time = raw_data['start_time']
    context_size = raw_data['context_size']

    # Assign each sample to a sub-window index (0, 1, 2, …)
    window_indices = np.floor((timestamps - start_time) / window_size).astype(int)
    num_windows = int(np.ceil(context_size / window_size))

    hr_windows = np.full(num_windows, np.nan)
    spo2_windows = np.full(num_windows, np.nan)

    for idx in range(num_windows):
        mask = window_indices == idx
        if np.any(mask):
            hr_windows[idx] = np.mean(hr_raw[mask])
            spo2_windows[idx] = np.mean(spo2_raw[mask])

    return hr_windows, spo2_windows


def create_windowed_features(mat_file_path, start_time, context_size, window_size):
    """Load .mat file and return (hr_windows, spo2_windows) sub-window arrays."""
    raw_data = extract_raw_timeseries(mat_file_path, start_time, context_size)
    if raw_data is None:
        return None
    return create_windowed_features_from_raw(raw_data, window_size)


def create_window_key(mat_file, window_start, window_end):
    """Unique cache key for one window: baby_id + start + end timestamps."""
    baby_id = mat_file.split('/')[-1].replace('.mat', '')
    return f"{baby_id}_{int(window_start)}_{int(window_end)}"


def ensure_numpy_pickle_compat():
    """Allow unpickling arrays saved with NumPy 2.x when running NumPy 1.x.

    NumPy 2.x renamed the internal module from numpy.core to numpy._core.
    Pickle files store the module path, so loading a 2.x pickle on 1.x fails
    unless we alias the old module names.
    """
    import sys

    try:
        import numpy._core  # noqa: F401  — already on NumPy 2.x, nothing to do
    except ModuleNotFoundError:
        import numpy.core as _core

        sys.modules.setdefault('numpy._core', _core)
        sys.modules.setdefault('numpy._core.multiarray', _core.multiarray)
        sys.modules.setdefault('numpy._core.umath', _core.umath)



def load_pickle(path):
    ensure_numpy_pickle_compat()
    with open(path, 'rb') as f:
        return pl.load(f)


def save_pickle(obj, path):
    """Atomic write: save to .tmp first, then rename (avoids corrupt partial files)."""
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        pl.dump(obj, f)
    os.replace(tmp_path, path)


def load_embeddings_cache():
    """Load the global embedding dict, or an empty dict if none exists yet."""
    if not os.path.exists(EMBEDDINGS_CACHE_PATH):
        return {}
    try:
        return load_pickle(EMBEDDINGS_CACHE_PATH)
    except Exception as e:
        # Common cause: cache pickled with NumPy 2.x, loaded with NumPy 1.x
        raise RuntimeError(
            f"Failed to load {EMBEDDINGS_CACHE_PATH} ({os.path.getsize(EMBEDDINGS_CACHE_PATH) / 1e6:.0f} MB). "
            "The cache was likely saved with a different NumPy version or is corrupted. "
            "Try upgrading NumPy (>=2.0 if the file uses numpy._core), or delete the file and regenerate embeddings. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def save_embeddings_cache(cache):
    save_pickle(cache, EMBEDDINGS_CACHE_PATH)


def detect_missing_windows(windows_df, cache):
    """Return rows whose (mat_file, start, end) key is not in the cache.

    Each window is uniquely identified by its source file + time bounds.
    This lets us skip windows that were embedded in a previous run.
    """
    missing_indices = []
    for idx, row in windows_df.iterrows():
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        if key not in cache:
            missing_indices.append(idx)
    if not missing_indices:
        return windows_df.iloc[0:0]  # empty DataFrame with same columns
    return windows_df.loc[missing_indices]


def generate_missing_embeddings(missing_df, cache, pipeline, save_every=500):
    """Run Chronos on each missing window; checkpoint cache periodically.

    For each window:
      1. create_windowed_features() slices the 80-hour vital-sign segment
         into smaller sub-windows for Chronos input.
      2. get_embedding() runs the Chronos-2 model and returns a vector.
      3. The vector is stored in cache under the window key.

    Failures (bad data, OOM, etc.) are counted but do not stop the loop.
    """
    total = len(missing_df)
    if total == 0:
        return cache

    print("\n" + "=" * 80)
    print(" " * 12 + "STEP 3b: GENERATING MISSING EMBEDDINGS")
    print("=" * 80 + "\n")
    print(f"Generating {total} missing embeddings (checkpoint every {save_every})...\n")

    initial_size = len(cache)
    generated = 0
    failed = 0
    new_since_save = 0

    for i, (_, row) in enumerate(missing_df.iterrows()):
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        try:
            # Extract vital-sign features for this 80-hour window
            result = create_windowed_features(
                row['mat_file'],
                row['window_start'],
                CONTEXT_SIZE_SECONDS,   # total context length (80 hours)
                WINDOW_SIZE_SECONDS,    # sub-window size fed to Chronos
            )
            if result is not None:
                # result = (feature_array, mask) → pass through Chronos-2
                cache[key] = get_embedding(result[0], result[1], pipeline)
                generated += 1
                new_since_save += 1
            else:
                failed += 1  # e.g. insufficient data in this time range
        except Exception:
            failed += 1

        # Checkpoint every save_every new embeddings (embedding is expensive)
        if new_since_save >= save_every:
            save_embeddings_cache(cache)
            print(f"  💾 Checkpoint at {i + 1}/{total} (cache: {len(cache)}, +{new_since_save} new)")
            new_since_save = 0

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total} | Generated: {generated}, Failed: {failed}")

    if new_since_save > 0:
        save_embeddings_cache(cache)
        print(f"\n  💾 Final checkpoint ({new_since_save} new embeddings)")

    print(f"\n{'=' * 80}")
    print("✓ MISSING EMBEDDING GENERATION COMPLETE")
    print(f"  Attempted: {total}")
    print(f"  Generated: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Global cache: {initial_size} → {len(cache)} (+{len(cache) - initial_size})")
    print(f"{'=' * 80}\n")

    return cache


def ensure_embeddings_for_windows(windows_df, save_every=500):
    """Detect missing embeddings, generate them, and return the updated cache.

    This is the main Step 3 entry point. It is incremental: re-running after
    a crash will only embed windows not already in the cache.
    """
    print("\n" + "=" * 80)
    print(" " * 10 + "STEP 3: EMBEDDINGS (DETECT MISSING → GENERATE)")
    print("=" * 80 + "\n")

    cache = load_embeddings_cache()

    # --- Step 3a: figure out what's already done ---
    print("STEP 3a: Detecting missing embeddings...")
    missing_df = detect_missing_windows(windows_df, cache)
    n_total = len(windows_df)
    n_missing = len(missing_df)

    print(f"  ✓ Global cache: {len(cache)} embeddings")
    print(f"  ✓ Windows required: {n_total}")
    print(f"  ✓ Already in cache: {n_total - n_missing}")
    print(f"  ✓ Missing: {n_missing}")

    missing_df.to_pickle(MISSING_WINDOWS_PATH)
    print(f"  ✓ Saved: {MISSING_WINDOWS_PATH}\n")

    if n_missing == 0:
        print("✓ Nothing missing — skipping generation\n")
        return cache

    # Load Chronos-2 once and reuse for all missing windows
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    cache = generate_missing_embeddings(missing_df, cache, pipeline, save_every=save_every)

    # Sanity check: warn if any windows still couldn't be embedded
    still_missing = len(detect_missing_windows(windows_df, cache))
    if still_missing:
        print(f"⚠ Warning: {still_missing} windows still missing after generation\n")

    return cache


def prepare_training_data_for_threshold(threshold_days, samples_windows_path, embeddings_cache=None):
    """Join labeled windows with cached embeddings for one threshold (Step 4).

    Reads the labeled data from Step 2, filters to one threshold, and
    looks up each window's embedding in the global cache.

    Windows whose embedding failed in Step 3 are silently skipped here,
    so the final sample count may be slightly less than the labeled count.
    """
    print(f"\nPreparing {threshold_days}-day threshold...")

    if embeddings_cache is None:
        embeddings_cache = load_embeddings_cache()

    sampled_all = pd.read_pickle(samples_windows_path)
    threshold_data = sampled_all[sampled_all['threshold_days'] == threshold_days]

    embedding_keys = []
    labels = []

    for _, row in threshold_data.iterrows():
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        if key in embeddings_cache:
            embedding_keys.append(key)
            labels.append(int(row['label']))

    print(f"  ✓ {len(embedding_keys)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")

    # Save keys + labels only; embeddings stay in the global cache
    with open(f'labels_{threshold_days}d.pkl', 'wb') as f:
        pl.dump({'embedding_keys': embedding_keys, 'labels': labels}, f)

    print(f"  ✓ Saved: labels_{threshold_days}d.pkl (~{(len(embedding_keys) * 100) / 1024:.2f} KB)")

    # Resolve keys → embedding vectors for immediate use in training
    embeddings = [embeddings_cache[key] for key in embedding_keys]
    return embeddings, labels


def load_training_data_for_threshold(threshold_days):
    """Reload a previously saved threshold's keys + labels and fetch embeddings.

    Use this to re-train without re-running Steps 1–4, as long as the
    labels file and embeddings cache still exist.
    """
    print(f"\nLoading {threshold_days}-day threshold from saved files...")

    with open(f'labels_{threshold_days}d.pkl', 'rb') as f:
        data = pl.load(f)

    embeddings_cache = load_embeddings_cache()
    embeddings = [embeddings_cache[key] for key in data['embedding_keys']]
    labels = data['labels']

    print(f"  ✓ Loaded {len(embeddings)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    return embeddings, labels
