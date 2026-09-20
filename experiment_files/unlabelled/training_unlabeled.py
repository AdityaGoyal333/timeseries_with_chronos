"""Step 3 (unlabeled): Chronos embeddings from parquet vitals.

Pipeline role
-------------
Same seasonal / FG cache contract as training_data, in microseconds, reading
age_us / HR / SPO2. Does not join labels or train the MLP.

FG keys are 4-part (80h bounds + slice), matching labeled FG — not
{id}_{start}_{end} only.

Outputs
-------
  data/unlabeled/embeddings_global.pkl
  data/unlabeled/embeddings_global_fine_grained.pkl
  data/unlabeled/windows_missing_embeddings[_fine_grained].pkl
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from constants import (
    BINNING_FINE_GRAINED,
    CONTEXT_SIZE_US,
    EMBEDDINGS_CACHE_PATHS,
    FG_CONTEXT_SIZE_US,
    FG_LOOKBACK_US,
    FG_WINDOW_SIZE_US,
    MISSING_WINDOWS_PATHS,
    NUM_FG_EMBEDDINGS,
    NUM_WINDOWS_FG,
    UNLABELED_DIR,
    UNLABELED_VITALS_DIR,
    WINDOW_SIZE_US,
)
from log_utils import tprint
from training_data import (
    create_fg_key,
    create_window_key,
    create_windowed_features_from_raw,
    get_embedding,
    load_embeddings_cache,
    save_embeddings_cache,
)


def cache_paths(binning_mode):
    """Return (embeddings_cache_path, missing_windows_path) under data/unlabeled/."""
    return (
        (UNLABELED_DIR / EMBEDDINGS_CACHE_PATHS[binning_mode]).resolve(),
        (UNLABELED_DIR / MISSING_WINDOWS_PATHS[binning_mode]).resolve(),
    )


def resolve_parquet_path(mat_ref):
    """Absolute parquet path under data/download/vitals from baby id."""
    baby_id = Path(str(mat_ref)).name.replace('.parquet', '')
    return UNLABELED_VITALS_DIR / f"{baby_id}.parquet"


def embed_slice(window_start):
    """Seasonal load: full 80h in microseconds."""
    return window_start, CONTEXT_SIZE_US, WINDOW_SIZE_US


def fg_slice_bounds(mat_file, window_start, window_end):
    """K uniform starts on [we-lookback, we-C] in µs; rng seeded by 80h key."""
    end = int(window_end)
    last_start = end - FG_CONTEXT_SIZE_US
    first_start = end - FG_LOOKBACK_US
    seed_key = create_window_key(mat_file, window_start, window_end)
    seed = int.from_bytes(hashlib.md5(seed_key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    starts = rng.integers(first_start, last_start + 1, size=NUM_FG_EMBEDDINGS)
    starts = sorted(set(int(s) for s in starts))
    return [(s, s + FG_CONTEXT_SIZE_US) for s in starts]


def extract_raw_timeseries(parquet_path, start_time, context_size):
    """Load HR and SpO2 from parquet in [start, start + context) on age_us."""
    try:
        df = pd.read_parquet(parquet_path, columns=['age_us', 'HR', 'SPO2'])
    except Exception as e:
        tprint(f"Error loading {parquet_path}: {e}")
        return None

    timestamps = df['age_us'].to_numpy()
    hr_cols = df['HR'].to_numpy()
    spo2_cols = df['SPO2'].to_numpy()

    end_time = start_time + context_size
    time_mask = (timestamps >= start_time) & (timestamps < end_time)
    valid_mask = ~np.isnan(hr_cols) & ~np.isnan(spo2_cols)
    final_mask = time_mask & valid_mask

    timestamps_valid = timestamps[final_mask]
    if len(timestamps_valid) == 0:
        return None

    return {
        'timestamps': timestamps_valid,
        'hr_raw': hr_cols[final_mask],
        'spo2_raw': spo2_cols[final_mask],
        'start_time': start_time,
        'context_size': context_size,
    }


def create_windowed_features(parquet_path, start_time, context_size, window_size):
    """Load parquet and return (hr_windows, spo2_windows) for Chronos."""
    raw_data = extract_raw_timeseries(parquet_path, start_time, context_size)
    if raw_data is None:
        return None
    return create_windowed_features_from_raw(raw_data, window_size)


def detect_missing_windows(windows_df, cache, binning_mode):
    """Rows whose required unlabeled cache key(s) are absent (empty DF if none)."""
    if binning_mode != BINNING_FINE_GRAINED:
        missing_indices = []
        for idx, row in windows_df.iterrows():
            key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
            if key not in cache:
                missing_indices.append(idx)
        if not missing_indices:
            return windows_df.iloc[0:0]
        return windows_df.loc[missing_indices]

    records = []
    for _, row in windows_df.iterrows():
        for fg_start, fg_end in fg_slice_bounds(
            row['mat_file'], row['window_start'], row['window_end']
        ):
            key = create_fg_key(
                row['mat_file'], row['window_start'], row['window_end'], fg_start, fg_end
            )
            if key not in cache:
                rec = row.to_dict()
                rec['fg_start'] = fg_start
                rec['fg_end'] = fg_end
                records.append(rec)
    if not records:
        return pd.DataFrame(columns=list(windows_df.columns) + ['fg_start', 'fg_end']).drop_duplicates()
    return pd.DataFrame(records).drop_duplicates()


def generate_missing_embeddings(missing_df, cache, pipeline, cache_path, binning_mode, save_every=500):
    """Embed missing unlabeled keys; checkpoint every save_every successes."""
    total = len(missing_df)
    if total == 0:
        return cache

    tprint("\n" + "=" * 80)
    tprint(" " * 8 + "STEP 3b (UNLABELED): GENERATING MISSING EMBEDDINGS")
    tprint(f"  Binning: {binning_mode}")
    if binning_mode == BINNING_FINE_GRAINED:
        tprint(
            f"  FG: last {FG_LOOKBACK_US}µs, {NUM_FG_EMBEDDINGS} slices of "
            f"{FG_CONTEXT_SIZE_US}µs, bin {FG_WINDOW_SIZE_US}µs → {NUM_WINDOWS_FG} steps"
        )
    tprint("=" * 80 + "\n")
    tprint(f"Generating {total} missing embeddings (checkpoint every {save_every})...\n")

    initial_size = len(cache)
    generated = 0
    failed = 0
    new_since_save = 0

    for i, (_, row) in enumerate(missing_df.iterrows()):
        if binning_mode == BINNING_FINE_GRAINED:
            key = create_fg_key(
                row['mat_file'],
                row['window_start'],
                row['window_end'],
                row['fg_start'],
                row['fg_end'],
            )
            load_start = int(row['fg_start'])
            context_size = FG_CONTEXT_SIZE_US
            window_size = FG_WINDOW_SIZE_US
        else:
            key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
            load_start, context_size, window_size = embed_slice(row['window_start'])
        try:
            result = create_windowed_features(
                str(resolve_parquet_path(row['mat_file'])),
                load_start,
                context_size,
                window_size,
            )
            if result is not None:
                cache[key] = get_embedding(result[0], result[1], pipeline)
                generated += 1
                new_since_save += 1
            else:
                failed += 1
        except Exception as e:
            print(e)
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
    """Unlabeled Step 3: detect missing keys, embed from parquet, checkpoint."""
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path, missing_path = cache_paths(binning_mode)

    tprint("\n" + "=" * 80)
    tprint(" " * 6 + "STEP 3 (UNLABELED): EMBEDDINGS (DETECT MISSING → GENERATE)")
    tprint(f"  Binning: {binning_mode}")
    tprint(f"  Cache: {embeddings_path}")
    tprint("=" * 80 + "\n")

    cache = load_embeddings_cache(embeddings_path)

    tprint("STEP 3a: Detecting missing embeddings...")
    missing_df = detect_missing_windows(windows_df, cache, binning_mode)
    if binning_mode == BINNING_FINE_GRAINED:
        n_required = len(windows_df) * NUM_FG_EMBEDDINGS
        unit = 'slices'
    else:
        n_required = len(windows_df)
        unit = 'windows'
    n_missing = len(missing_df)

    tprint(f"  ✓ Global cache: {len(cache)} embeddings")
    tprint(f"  ✓ {unit.capitalize()} required: {n_required}")
    tprint(f"  ✓ Already in cache: {n_required - n_missing}")
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
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    cache = generate_missing_embeddings(
        missing_df, cache, pipeline, embeddings_path, binning_mode, save_every=save_every
    )

    still_missing = len(detect_missing_windows(windows_df, cache, binning_mode))
    if still_missing:
        tprint(f"⚠ Warning: {still_missing} {unit} still missing after generation\n")

    return cache
