"""Step 3–4: Generate missing Chronos embeddings and prepare training data.

Pipeline role
-------------
Step 3:  Run Amazon Chronos-2 on each unique sampled window → fixed-size
         embedding vector. Results are cached globally so we never recompute.

Step 4:  For each death-proximity threshold, pair cached embeddings with
         their threshold-specific labels → ready for classifier training.

Binning modes (see constants.BINNING_*):
  seasonal      — 960 × 300s equal bins
  fine_grained  — 960 geometric bins (15 min → 1 min), separate cache file
"""

import os
import pickle as pl
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
from chronos import Chronos2Pipeline

from constants import (
    BINNING_FINE_GRAINED,
    BINNING_SEASONAL,
    CONTEXT_SIZE_SECONDS,
    EMBEDDINGS_CACHE_PATHS,
    FINE_GRAINED_MAX_BIN_SECONDS,
    FINE_GRAINED_MIN_BIN_SECONDS,
    FINE_GRAINED_OUTPUT_SUFFIX,
    MISSING_WINDOWS_PATHS,
    NUM_WINDOWS,
    WINDOW_SIZE_SECONDS,
)
from log_utils import tprint

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()

# Computed once; 960 integer widths summing to CONTEXT_SIZE_SECONDS
_fine_grained_bin_widths = None


def output_suffix(binning_mode):
    """Filename suffix for labels/results when using fine-grained binning."""
    if binning_mode == BINNING_FINE_GRAINED:
        return FINE_GRAINED_OUTPUT_SUFFIX
    return ''


def cache_paths(binning_mode):
    """Return (embeddings_cache_path, missing_windows_path) for this binning mode."""
    return (
        (DATA_DIR / EMBEDDINGS_CACHE_PATHS[binning_mode]).resolve(),
        (DATA_DIR / MISSING_WINDOWS_PATHS[binning_mode]).resolve(),
    )


def geometric_bin_widths(context_length, n_values, min_width, max_width):
    """Build n_values bin widths (seconds) that sum exactly to context_length.

    Widths follow a geometric progression from max_width (index 0, early in
    the window) down to min_width (index n-1, late in the window).
    """
    ratio = (min_width / max_width) ** (1.0 / (n_values - 1))
    weights = max_width * (ratio ** np.arange(n_values, dtype=float))
    widths = weights / weights.sum() * context_length

    floored = np.floor(widths).astype(int)
    remainder = int(context_length - floored.sum())
    # Distribute leftover seconds to bins with largest fractional parts
    order = np.argsort(widths - floored)[::-1]
    floored[order[:remainder]] += 1
    return floored


def get_fine_grained_bin_widths():
    """Lazy-load the 960-bin geometric schedule (shared across all windows)."""
    global _fine_grained_bin_widths
    if _fine_grained_bin_widths is None:
        _fine_grained_bin_widths = geometric_bin_widths(
            CONTEXT_SIZE_SECONDS,
            NUM_WINDOWS,
            FINE_GRAINED_MIN_BIN_SECONDS,
            FINE_GRAINED_MAX_BIN_SECONDS,
        )
    return _fine_grained_bin_widths


def prepare_chronos_input(hr_windows, spo2_windows):
    """Format HR + SpO2 sub-window arrays as Chronos-2 model input."""
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

    hr_emb = embeddings[0][0][-1].float()
    spo2_emb = embeddings[1][0][-1].float()
    return torch.cat([hr_emb, spo2_emb], dim=0).cpu().numpy()


def get_embedding(hr_windows, spo2_windows, pipeline):
    chronos_input = prepare_chronos_input(hr_windows, spo2_windows)
    return get_chronos_embedding(chronos_input, pipeline)


def extract_raw_timeseries(mat_file_path, start_time, context_size):
    """Load HR and SpO2 samples from a .mat file within [start, start + context)."""
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        tprint(f"Error loading {mat_file_path}: {e}")
        return None

    birth_time = mat_data['pdata'][0][1]
    age_at_death = mat_data['pdata'][0][-1]
    death_ts = age_at_death * 86400 if not np.isnan(age_at_death) else None

    hr_cols = mat_data['vdata'][:, 0]
    spo2_cols = mat_data['vdata'][:, 2]
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


def _aggregate_into_bins(timestamps, hr_raw, spo2_raw, start_time, num_windows, bin_idx):
    """Mean HR/SpO2 per bin index; empty bins stay NaN."""
    hr_windows = np.full(num_windows, np.nan)
    spo2_windows = np.full(num_windows, np.nan)

    for idx in range(num_windows):
        mask = bin_idx == idx
        if np.any(mask):
            hr_windows[idx] = np.mean(hr_raw[mask])
            spo2_windows[idx] = np.mean(spo2_raw[mask])

    return hr_windows, spo2_windows


def create_windowed_features_from_raw(raw_data, binning_mode):
    """Aggregate raw samples into 960 Chronos bins (seasonal or fine-grained)."""
    timestamps = raw_data['timestamps']
    hr_raw = raw_data['hr_raw']
    spo2_raw = raw_data['spo2_raw']
    start_time = raw_data['start_time']

    if binning_mode == BINNING_SEASONAL:
        window_indices = np.floor(
            (timestamps - start_time) / WINDOW_SIZE_SECONDS
        ).astype(int)
        num_windows = int(np.ceil(raw_data['context_size'] / WINDOW_SIZE_SECONDS))
        return _aggregate_into_bins(
            timestamps, hr_raw, spo2_raw, start_time, num_windows, window_indices
        )

    # fine_grained: variable-width bins via cumulative edges
    bin_widths = get_fine_grained_bin_widths()
    num_windows = len(bin_widths)
    edges = np.concatenate([[0], np.cumsum(bin_widths)])
    rel = timestamps - start_time
    bin_idx = np.searchsorted(edges, rel, side='right') - 1
    in_range = (bin_idx >= 0) & (bin_idx < num_windows)
    return _aggregate_into_bins(
        timestamps[in_range],
        hr_raw[in_range],
        spo2_raw[in_range],
        start_time,
        num_windows,
        bin_idx[in_range],
    )


def create_windowed_features(mat_file_path, start_time, context_size, binning_mode):
    """Load .mat file and return (hr_windows, spo2_windows) for Chronos."""
    raw_data = extract_raw_timeseries(mat_file_path, start_time, context_size)
    if raw_data is None:
        return None
    return create_windowed_features_from_raw(raw_data, binning_mode)


def baby_id_from_window_key(key):
    return key.rsplit('_', 2)[0]


def create_window_key(mat_file, window_start, window_end):
    baby_id = mat_file.split('/')[-1].replace('.mat', '')
    return f"{baby_id}_{int(window_start)}_{int(window_end)}"


def ensure_numpy_pickle_compat():
    import sys

    try:
        import numpy._core  # noqa: F401
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
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        pl.dump(obj, f)
    os.replace(tmp_path, path)


def load_embeddings_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        return load_pickle(cache_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {cache_path} ({os.path.getsize(cache_path) / 1e6:.0f} MB). "
            "Try upgrading NumPy (>=2.0 if the file uses numpy._core), or delete and regenerate. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def save_embeddings_cache(cache, cache_path):
    save_pickle(cache, cache_path)


def detect_missing_windows(windows_df, cache):
    missing_indices = []
    for idx, row in windows_df.iterrows():
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        if key not in cache:
            missing_indices.append(idx)
    if not missing_indices:
        return windows_df.iloc[0:0]
    return windows_df.loc[missing_indices]


def generate_missing_embeddings(missing_df, cache, pipeline, cache_path, binning_mode, save_every=500):
    total = len(missing_df)
    if total == 0:
        return cache

    tprint("\n" + "=" * 80)
    tprint(" " * 12 + "STEP 3b: GENERATING MISSING EMBEDDINGS")
    tprint(f"  Binning: {binning_mode}")
    tprint("=" * 80 + "\n")
    tprint(f"Generating {total} missing embeddings (checkpoint every {save_every})...\n")

    initial_size = len(cache)
    generated = 0
    failed = 0
    new_since_save = 0

    for i, (_, row) in enumerate(missing_df.iterrows()):
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        try:
            result = create_windowed_features(
                row['mat_file'],
                row['window_start'],
                CONTEXT_SIZE_SECONDS,
                binning_mode,
            )
            if result is not None:
                cache[key] = get_embedding(result[0], result[1], pipeline)
                generated += 1
                new_since_save += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if new_since_save >= save_every:
            save_embeddings_cache(cache, cache_path)
            tprint(f"  💾 Checkpoint at {i + 1}/{total} (cache: {len(cache)}, +{new_since_save} new)")
            new_since_save = 0

        if (i + 1) % 100 == 0 or (i + 1) == total:
            tprint(f"  Progress: {i + 1}/{total} | Generated: {generated}, Failed: {failed}")

    if new_since_save > 0:
        save_embeddings_cache(cache, cache_path)
        tprint(f"\n  💾 Final checkpoint ({new_since_save} new embeddings)")

    tprint(f"\n{'=' * 80}")
    tprint("✓ MISSING EMBEDDING GENERATION COMPLETE")
    tprint(f"  Attempted: {total}")
    tprint(f"  Generated: {generated}")
    tprint(f"  Failed: {failed}")
    tprint(f"  Global cache: {initial_size} → {len(cache)} (+{len(cache) - initial_size})")
    tprint(f"{'=' * 80}\n")

    return cache


def ensure_embeddings_for_windows(windows_df, binning_mode, save_every=500):
    """Step 3 entry point: detect missing keys, embed, checkpoint incrementally."""
    embeddings_path, missing_path = cache_paths(binning_mode)

    tprint("\n" + "=" * 80)
    tprint(" " * 10 + "STEP 3: EMBEDDINGS (DETECT MISSING → GENERATE)")
    tprint(f"  Binning: {binning_mode}")
    tprint(f"  Cache: {embeddings_path}")
    tprint("=" * 80 + "\n")

    cache = load_embeddings_cache(embeddings_path)

    tprint("STEP 3a: Detecting missing embeddings...")
    missing_df = detect_missing_windows(windows_df, cache)
    n_total = len(windows_df)
    n_missing = len(missing_df)

    tprint(f"  ✓ Global cache: {len(cache)} embeddings")
    tprint(f"  ✓ Windows required: {n_total}")
    tprint(f"  ✓ Already in cache: {n_total - n_missing}")
    tprint(f"  ✓ Missing: {n_missing}")

    missing_df.to_pickle(missing_path)
    tprint(f"  ✓ Saved: {missing_path}\n")

    if n_missing == 0:
        tprint("✓ Nothing missing — skipping generation\n")
        return cache

    chronos_device = "cuda" if torch.cuda.is_available() else "cpu"
    tprint(f"Chronos-2 device: {chronos_device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=chronos_device,
        dtype=torch.bfloat16,
    )
    cache = generate_missing_embeddings(
        missing_df, cache, pipeline, embeddings_path, binning_mode, save_every=save_every
    )

    still_missing = len(detect_missing_windows(windows_df, cache))
    if still_missing:
        tprint(f"⚠ Warning: {still_missing} windows still missing after generation\n")

    return cache


def prepare_training_data_for_threshold(
    threshold_days, samples_windows_path, binning_mode, embeddings_cache=None
):
    """Join labeled windows with cached embeddings for one threshold (Step 4)."""
    suffix = output_suffix(binning_mode)
    labels_path = f'labels_{threshold_days}d{suffix}.pkl'

    tprint(f"\nPreparing {threshold_days}-day threshold (binning={binning_mode})...")

    if embeddings_cache is None:
        embeddings_path, _ = cache_paths(binning_mode)
        embeddings_cache = load_embeddings_cache(embeddings_path)

    sampled_all = pd.read_pickle(samples_windows_path)
    threshold_data = sampled_all[sampled_all['threshold_days'] == threshold_days]

    embedding_keys = []
    labels = []
    baby_ids = []

    for _, row in threshold_data.iterrows():
        key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
        if key in embeddings_cache:
            embedding_keys.append(key)
            labels.append(int(row['label']))
            baby_ids.append(baby_id_from_window_key(key))

    tprint(f"  ✓ {len(embedding_keys)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    tprint(f"  ✓ {len(set(baby_ids))} unique babies")

    with open(labels_path, 'wb') as f:
        pl.dump({'embedding_keys': embedding_keys, 'labels': labels, 'baby_ids': baby_ids}, f)

    tprint(f"  ✓ Saved: {labels_path} (~{(len(embedding_keys) * 100) / 1024:.2f} KB)")

    embeddings = [embeddings_cache[key] for key in embedding_keys]
    return embeddings, labels, baby_ids


def load_training_data_for_threshold(threshold_days, binning_mode):
    """Reload keys + labels and fetch embeddings from the cache for this binning mode."""
    suffix = output_suffix(binning_mode)
    labels_path = f'labels_{threshold_days}d{suffix}.pkl'

    tprint(f"\nLoading {threshold_days}-day threshold from {labels_path}...")

    with open(labels_path, 'rb') as f:
        data = pl.load(f)

    embeddings_path, _ = cache_paths(binning_mode)
    embeddings_cache = load_embeddings_cache(embeddings_path)
    embeddings = [embeddings_cache[key] for key in data['embedding_keys']]
    labels = data['labels']
    baby_ids = data.get('baby_ids')
    if baby_ids is None:
        baby_ids = [baby_id_from_window_key(key) for key in data['embedding_keys']]

    tprint(f"  ✓ Loaded {len(embeddings)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    tprint(f"  ✓ {len(set(baby_ids))} unique babies")
    return embeddings, labels, baby_ids
