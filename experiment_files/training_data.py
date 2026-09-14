"""Steps 3–4: Chronos embeddings and join of Step 2 labels to cache(s).

Pipeline role
-------------
Step 3:  Seasonal: one vector per 80h key. FG: one vector per overlapping
         slice key (80h bounds + fg_start/fg_end) inside the lookback.
Step 4:  Labels table is the roster. Cache lookup only — no slice geometry.
         Singular seasonal: one row per 80h key. Singular FG / concat: one
         row per FG cache key that belongs to that 80h window.

Called from main.train (Step 3) and main.classify (Step 4). Does not train
the MLP.

Binning
-------
  seasonal      — [window_start, +80h), WINDOW_SIZE_SECONDS bins
  fine_grained  — K random FG_CONTEXT slices in the last FG_LOOKBACK seconds;
                  each slice ends at or before window_end

Outputs
-------
  data/embeddings_global.pkl
  data/embeddings_global_fine_grained.pkl
  data/windows_missing_embeddings[_fine_grained].pkl
  results/labels_{N}d[_fine_grained|_concat].pkl
"""

import hashlib
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
    CONCAT_OUTPUT_SUFFIX,
    CONTEXT_SIZE_SECONDS,
    EMBEDDINGS_CACHE_PATHS,
    FG_CONTEXT_SIZE_SECONDS,
    FG_LOOKBACK_SECONDS,
    FG_WINDOW_SIZE_SECONDS,
    FINE_GRAINED_OUTPUT_SUFFIX,
    HEAD_CONCAT,
    MISSING_WINDOWS_PATHS,
    NUM_FG_EMBEDDINGS,
    NUM_WINDOWS_FG,
    RESULTS_DIR,
    WINDOW_SIZE_SECONDS,
)
from log_utils import tprint

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()
VITALS_DIR = (DATA_DIR / "download" / "Vitals").resolve()


def baby_id_from_mat_ref(mat_ref):
    """Accept a baby id or a leftover path; return the id (filename without .mat)."""
    return Path(str(mat_ref)).name.replace('.mat', '')


def resolve_mat_path(mat_ref):
    """Absolute .mat path under data/download/Vitals from baby id (or old path)."""
    return VITALS_DIR / f"{baby_id_from_mat_ref(mat_ref)}.mat"


def labels_output_path(threshold_days, suffix):
    """Absolute path under repo results/ for a labels pickle; create the dir."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f'labels_{threshold_days}d{suffix}.pkl'


def output_suffix(binning_mode=None, head=None):
    """Return artifact suffix for labels / results / curve dirs.

    Concat wins over mode so a leftover --mode cannot write _fine_grained
    files when the head is concat.
    """
    if head == HEAD_CONCAT:
        return CONCAT_OUTPUT_SUFFIX
    if binning_mode == BINNING_FINE_GRAINED:
        return FINE_GRAINED_OUTPUT_SUFFIX
    return ''


def cache_paths(binning_mode):
    """Return (embeddings_cache_path, missing_windows_path) for this binning mode."""
    return (
        (DATA_DIR / EMBEDDINGS_CACHE_PATHS[binning_mode]).resolve(),
        (DATA_DIR / MISSING_WINDOWS_PATHS[binning_mode]).resolve(),
    )


def embed_slice(binning_mode, window_start, window_end):
    """Return (load_start, context_seconds, bin_width) for seasonal (full 80h)."""
    return window_start, CONTEXT_SIZE_SECONDS, WINDOW_SIZE_SECONDS


def fg_slice_bounds(mat_file, window_start, window_end):
    """K uniform starts on [we-lookback, we-C]; duplicates dropped; rng seeded by 80h key."""
    end = int(window_end)
    last_start = end - FG_CONTEXT_SIZE_SECONDS
    first_start = end - FG_LOOKBACK_SECONDS
    seed_key = create_window_key(mat_file, window_start, window_end)
    seed = int.from_bytes(hashlib.md5(seed_key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    starts = rng.integers(first_start, last_start + 1, size=NUM_FG_EMBEDDINGS)
    starts = sorted(set(int(s) for s in starts))
    return [(s, s + FG_CONTEXT_SIZE_SECONDS) for s in starts]


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
    """NaN-fill HR/SpO2 bins, embed with Chronos-2, return one concatenated vector."""
    chronos_input = prepare_chronos_input(hr_windows, spo2_windows)
    return get_chronos_embedding(chronos_input, pipeline)


def extract_raw_timeseries(mat_file_path, start_time, context_size):
    """Load HR and SpO2 samples from a .mat file within [start, start + context).

    Returns None if the file cannot be read or the half-open interval has no
    paired (HR, SpO2) samples. Empty FG tails therefore skip embedding.
    """
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

    # Half-open interval so adjacent bins do not double-count a timestamp.
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


def create_windowed_features_from_raw(raw_data, window_size):
    """Equal-width mean HR/SpO2 bins over [start, start + context)."""
    timestamps = raw_data['timestamps']
    hr_raw = raw_data['hr_raw']
    spo2_raw = raw_data['spo2_raw']
    start_time = raw_data['start_time']
    bin_idx = np.floor((timestamps - start_time) / window_size).astype(int)
    num_windows = int(np.ceil(raw_data['context_size'] / window_size))
    return _aggregate_into_bins(
        timestamps, hr_raw, spo2_raw, start_time, num_windows, bin_idx
    )


def create_windowed_features(mat_file_path, start_time, context_size, window_size):
    """Load .mat file and return (hr_windows, spo2_windows) for Chronos."""
    raw_data = extract_raw_timeseries(mat_file_path, start_time, context_size)
    if raw_data is None:
        return None
    return create_windowed_features_from_raw(raw_data, window_size)


def baby_id_from_window_key(key, binning_mode=BINNING_SEASONAL):
    """Strip trailing timestamps: 4 for fine-grained slices, 2 for seasonal."""
    drop = 4 if binning_mode == BINNING_FINE_GRAINED else 2
    return key.rsplit('_', drop)[0]


def create_window_key(mat_file, window_start, window_end):
    """Seasonal cache key: baby + 80h window bounds."""
    baby_id = baby_id_from_mat_ref(mat_file)
    return f"{baby_id}_{int(window_start)}_{int(window_end)}"


def create_fg_key(mat_file, window_start, window_end, fg_start, fg_end):
    """FG cache key: 80h bounds plus this slice's [fg_start, fg_end)."""
    return (
        f"{create_window_key(mat_file, window_start, window_end)}"
        f"_{int(fg_start)}_{int(fg_end)}"
    )


def index_fg_keys_by_80h(fg_cache):
    """Map seasonal key baby_ws_we → FG slice keys stored for that window."""
    index = {}
    for key in fg_cache:
        prefix = key.rsplit('_', 2)[0]
        index.setdefault(prefix, []).append(key)
    return index


def ensure_numpy_pickle_compat():
    """Alias numpy._core so pickles written on NumPy 2 load on NumPy 1."""
    import sys

    try:
        import numpy._core  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core as _core

        sys.modules.setdefault('numpy._core', _core)
        sys.modules.setdefault('numpy._core.multiarray', _core.multiarray)
        sys.modules.setdefault('numpy._core.umath', _core.umath)


def load_pickle(path):
    """Unpickle after applying NumPy 1/2 module aliases."""
    ensure_numpy_pickle_compat()
    with open(path, 'rb') as f:
        return pl.load(f)


def save_pickle(obj, path):
    """Atomic write so a crash mid-dump does not truncate the cache."""
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        pl.dump(obj, f)
    os.replace(tmp_path, path)


def load_embeddings_cache(cache_path):
    """Load {window_key: vector} or {} if the cache file is absent."""
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
    """Persist the full embeddings dict via atomic pickle."""
    save_pickle(cache, cache_path)


def detect_missing_windows(windows_df, cache, binning_mode):
    """Rows whose required cache key(s) are absent (empty DF if none).

    Seasonal: one 80h key per row. FG: one row per missing slice, with
    fg_start/fg_end so generate does not re-embed slices already in cache.
    """
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
    """Embed missing keys; checkpoint every save_every successes (FG: per slice)."""
    total = len(missing_df)
    if total == 0:
        return cache

    tprint("\n" + "=" * 80)
    tprint(" " * 12 + "STEP 3b: GENERATING MISSING EMBEDDINGS")
    tprint(f"  Binning: {binning_mode}")
    if binning_mode == BINNING_FINE_GRAINED:
        tprint(
            f"  FG: last {FG_LOOKBACK_SECONDS}s, {NUM_FG_EMBEDDINGS} slices of "
            f"{FG_CONTEXT_SIZE_SECONDS}s, bin {FG_WINDOW_SIZE_SECONDS}s → {NUM_WINDOWS_FG} steps"
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
            context_size = FG_CONTEXT_SIZE_SECONDS
            window_size = FG_WINDOW_SIZE_SECONDS
        else:
            key = create_window_key(row['mat_file'], row['window_start'], row['window_end'])
            load_start, context_size, window_size = embed_slice(
                binning_mode, row['window_start'], row['window_end']
            )
        try:
            result = create_windowed_features(
                str(resolve_mat_path(row['mat_file'])),
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
    """Step 3 entry point: detect missing keys, embed, checkpoint incrementally."""
    embeddings_path, missing_path = cache_paths(binning_mode)

    tprint("\n" + "=" * 80)
    tprint(" " * 10 + "STEP 3: EMBEDDINGS (DETECT MISSING → GENERATE)")
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

    # Load Chronos only when there is work; classify never reaches this path.
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


def prepare_training_data_for_threshold(
    threshold_days, samples_windows_path, binning_mode, embeddings_cache=None
):
    """Step 4 singular: join Step 2 labels to whatever keys exist in this cache.

    Seasonal: one row per 80h key. FG: one row per stored slice for that 80h window.
    """
    suffix = output_suffix(binning_mode)
    labels_path = labels_output_path(threshold_days, suffix)

    tprint(f"\nPreparing {threshold_days}-day threshold (binning={binning_mode})...")

    if embeddings_cache is None:
        embeddings_path, _ = cache_paths(binning_mode)
        embeddings_cache = load_embeddings_cache(embeddings_path)

    sampled_all = pd.read_pickle(samples_windows_path)
    threshold_data = sampled_all[sampled_all['threshold_days'] == threshold_days]

    embedding_keys = []
    labels = []
    baby_ids = []
    fg_index = (
        index_fg_keys_by_80h(embeddings_cache)
        if binning_mode == BINNING_FINE_GRAINED
        else None
    )

    for _, row in threshold_data.iterrows():
        baby_id = baby_id_from_mat_ref(row['mat_file'])
        window_key = create_window_key(
            row['mat_file'], row['window_start'], row['window_end']
        )
        if binning_mode == BINNING_FINE_GRAINED:
            slice_keys = fg_index.get(window_key, [])
        else:
            slice_keys = [window_key] if window_key in embeddings_cache else []
        for key in slice_keys:
            embedding_keys.append(key)
            labels.append(int(row['label']))
            baby_ids.append(baby_id)

    tprint(f"  ✓ {len(embedding_keys)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    tprint(f"  ✓ {len(set(baby_ids))} unique babies")

    with open(labels_path, 'wb') as f:
        pl.dump({'embedding_keys': embedding_keys, 'labels': labels, 'baby_ids': baby_ids}, f)

    tprint(f"  ✓ Saved: {labels_path} (~{(len(embedding_keys) * 100) / 1024:.2f} KB)")

    embeddings = [embeddings_cache[key] for key in embedding_keys]
    return embeddings, labels, baby_ids


def prepare_concat_training_data_for_threshold(threshold_days, samples_windows_path):
    """Step 4 concat: labels table + cache lookup. No slice geometry here.

    One row per FG cache key for that 80h window, if the seasonal 80h vector exists.
    """
    suffix = output_suffix(head=HEAD_CONCAT)
    labels_path = labels_output_path(threshold_days, suffix)

    seasonal_cache = load_embeddings_cache(cache_paths(BINNING_SEASONAL)[0])
    fg_cache = load_embeddings_cache(cache_paths(BINNING_FINE_GRAINED)[0])
    fg_index = index_fg_keys_by_80h(fg_cache)

    tprint(f"\nPreparing {threshold_days}-day threshold (head=concat)...")
    tprint(f"  Seasonal cache: {len(seasonal_cache)}")
    tprint(f"  Fine-grained cache: {len(fg_cache)}")

    sampled_all = pd.read_pickle(samples_windows_path)
    threshold_data = sampled_all[sampled_all['threshold_days'] == threshold_days]

    embedding_keys = []
    labels = []
    baby_ids = []
    embeddings = []
    n_no_seasonal = 0

    for _, row in threshold_data.iterrows():
        seasonal_key = create_window_key(
            row['mat_file'], row['window_start'], row['window_end']
        )
        if seasonal_key not in seasonal_cache:
            n_no_seasonal += 1
            continue
        baby_id = baby_id_from_mat_ref(row['mat_file'])
        for fg_key in fg_index.get(seasonal_key, []):
            embedding_keys.append(fg_key)
            labels.append(int(row['label']))
            baby_ids.append(baby_id)
            embeddings.append(
                np.concatenate([seasonal_cache[seasonal_key], fg_cache[fg_key]])
            )

    tprint(f"  ✓ 80h rows with no seasonal vector: {n_no_seasonal}")
    tprint(f"  ✓ {len(embedding_keys)} concat samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    tprint(f"  ✓ {len(set(baby_ids))} unique babies")

    with open(labels_path, 'wb') as f:
        pl.dump({'embedding_keys': embedding_keys, 'labels': labels, 'baby_ids': baby_ids}, f)

    tprint(f"  ✓ Saved: {labels_path} (~{(len(embedding_keys) * 100) / 1024:.2f} KB)")
    return embeddings, labels, baby_ids


def load_training_data_for_threshold(threshold_days, binning_mode):
    """Reload a singular labels pickle and look up vectors in that mode's cache.

    Does not handle concat (those vectors are not stored in one cache).
    """
    suffix = output_suffix(binning_mode)
    labels_path = labels_output_path(threshold_days, suffix)

    tprint(f"\nLoading {threshold_days}-day threshold from {labels_path}...")

    with open(labels_path, 'rb') as f:
        data = pl.load(f)

    embeddings_path, _ = cache_paths(binning_mode)
    embeddings_cache = load_embeddings_cache(embeddings_path)
    embeddings = [embeddings_cache[key] for key in data['embedding_keys']]
    labels = data['labels']
    baby_ids = data.get('baby_ids')
    # Older pickles stored keys only; derive baby_id the same way as write path.
    if baby_ids is None:
        baby_ids = [baby_id_from_window_key(key, binning_mode) for key in data['embedding_keys']]

    tprint(f"  ✓ Loaded {len(embeddings)} samples (L0:{labels.count(0)}, L1:{labels.count(1)})")
    tprint(f"  ✓ {len(set(baby_ids))} unique babies")
    return embeddings, labels, baby_ids
