"""
Mortality prediction pipeline — shared config, helpers, and orchestrator.
"""

import glob
import os
import pickle as pl

import numpy as np
import pandas as pd
import scipy.io
import torch

# =============================================================================
# Constants and paths
# =============================================================================

CONTEXT_SIZE_HOURS = 80
CONTEXT_SIZE_SECONDS = CONTEXT_SIZE_HOURS * 3600
WINDOW_SIZE_SECONDS = 300
NUM_WINDOWS = CONTEXT_SIZE_SECONDS // WINDOW_SIZE_SECONDS

THRESHOLDS_DAYS = [3, 5, 7]
THRESHOLDS_SECONDS = [d * 86400 for d in THRESHOLDS_DAYS]

OVERLAP_PERCENTAGE_NON_NICU = 0.0
OVERLAP_PERCENTAGE_NICU = 0.90

MAX_METADATA_WINDOWS_NON_NICU = 15
MAX_METADATA_WINDOWS_NICU = 200

NUM_SAMPLE_WINDOWS_NON_NICU = 5
NUM_SAMPLE_WINDOWS_NICU = 100
NICU_LATE_WINDOW_BIAS = 0.92

WINDOW_METADATA_PATH = 'window_metadata.pkl'
WINDOWS_TO_EMBED_PATH = 'windows_to_embed.pkl'
SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH = 'sampled_windows_all_thresholds.pkl'
MISSING_WINDOWS_PATH = 'windows_missing_embeddings.pkl'
EMBEDDINGS_CACHE_PATH = 'embeddings_global.pkl'
VALID_SUMMARY_PATH = 'valid_summary_dic.pkl'
VITALS_GLOB = './download/Vitals/*.mat'


# =============================================================================
# Pickle helpers (NumPy 1.x / 2.x compatibility)
# =============================================================================

def ensure_numpy_pickle_compat():
    """Allow unpickling arrays saved with NumPy 2.x when running NumPy 1.x."""
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


# =============================================================================
# Vitals and Chronos helpers
# =============================================================================

def create_window_key(mat_file, window_start, window_end):
    baby_id = mat_file.split('/')[-1].replace('.mat', '')
    return f"{baby_id}_{int(window_start)}_{int(window_end)}"


def extract_raw_timeseries(mat_file_path, start_time, context_size):
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
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


def create_windowed_features_from_raw(raw_data, window_size):
    timestamps = raw_data['timestamps']
    hr_raw = raw_data['hr_raw']
    spo2_raw = raw_data['spo2_raw']
    start_time = raw_data['start_time']
    context_size = raw_data['context_size']

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
    raw_data = extract_raw_timeseries(mat_file_path, start_time, context_size)
    if raw_data is None:
        return None
    return create_windowed_features_from_raw(raw_data, window_size)


def prepare_chronos_input(hr_windows, spo2_windows):
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
    with torch.no_grad():
        embeddings, _ = pipeline.embed(chronos_input)

    hr_emb = embeddings[0][0][-1].float()
    spo2_emb = embeddings[1][0][-1].float()
    return torch.cat([hr_emb, spo2_emb], dim=0).cpu().numpy()


def get_embedding(hr_windows, spo2_windows, pipeline):
    chronos_input = prepare_chronos_input(hr_windows, spo2_windows)
    return get_chronos_embedding(chronos_input, pipeline)


# =============================================================================
# Pipeline orchestrator
# =============================================================================

def run_complete_pipeline(use_focal_loss, gamma):
    from sampling import sample_and_label_all_thresholds
    from train_classifier import train_and_save_results
    from training_data import ensure_embeddings_for_windows, prepare_training_data_for_threshold
    from window_metadata import generate_all_window_metadata

    loss_name = f"Focal Loss (γ={gamma})" if use_focal_loss else "Cross-Entropy"
    print("\n" + "=" * 100)
    print(" " * 15 + f"MORTALITY PREDICTION PIPELINE WITH {loss_name.upper()}")
    print("=" * 100 + "\n")

    with open(VALID_SUMMARY_PATH, 'rb') as f:
        valid_summary_dic = pl.load(f)

    nicu_mask = valid_summary_dic['indicator_vector'] == 1
    nicu_baby_ids = set(valid_summary_dic['baby_id'][nicu_mask])
    non_nicu_baby_ids = set(valid_summary_dic['baby_id'][~nicu_mask])

    all_mat_files = sorted(glob.glob(VITALS_GLOB))
    nicu_files = [f for f in all_mat_files if f.split('/')[-1] in nicu_baby_ids]
    non_nicu_files = [f for f in all_mat_files if f.split('/')[-1] in non_nicu_baby_ids]

    print(f"✓ {len(nicu_files)} NICU files")
    print(f"✓ {len(non_nicu_files)} Non-NICU files\n")

    if os.path.exists(WINDOW_METADATA_PATH):
        metadata_df = pd.read_pickle(WINDOW_METADATA_PATH)
        print(f"✓ Loaded metadata: {len(metadata_df)} windows\n")
    else:
        metadata_df = generate_all_window_metadata(nicu_files, non_nicu_files)

    if os.path.exists(WINDOWS_TO_EMBED_PATH):
        unique_windows_df = pd.read_pickle(WINDOWS_TO_EMBED_PATH)
        print(f"✓ Loaded sampled windows: {len(unique_windows_df)}\n")
    else:
        _, unique_windows_df = sample_and_label_all_thresholds(metadata_df)

    embeddings_cache = ensure_embeddings_for_windows(unique_windows_df)

    for threshold_days in THRESHOLDS_DAYS:

        embeddings, labels = prepare_training_data_for_threshold(threshold_days, embeddings_cache)
        train_and_save_results(
            threshold_days,
            embeddings,
            labels,
            use_focal_loss=use_focal_loss,
            gamma=gamma,
            loss_name=loss_name,
        )

    print("\n" + "=" * 100)
    print(" " * 30 + "✓ PIPELINE COMPLETE")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_complete_pipeline(use_focal_loss=True, gamma=2.0)
