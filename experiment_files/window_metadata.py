"""Step 1: Generate window metadata for all babies."""

import numpy as np
import pandas as pd
import scipy.io

from main import (
    CONTEXT_SIZE_SECONDS,
    MAX_METADATA_WINDOWS_NICU,
    MAX_METADATA_WINDOWS_NON_NICU,
    OVERLAP_PERCENTAGE_NICU,
    OVERLAP_PERCENTAGE_NON_NICU,
    WINDOW_METADATA_PATH,
)


def generate_window_metadata_for_baby(mat_file_path, overlap_percentage, max_windows):
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except Exception:
        return pd.DataFrame()

    birth_time = mat_data['pdata'][0][1]
    age_at_death = mat_data['pdata'][0][-1]
    death_ts = age_at_death * 86400 if not np.isnan(age_at_death) else np.nan

    if np.isnan(birth_time):
        return pd.DataFrame()

    timestamps = mat_data['vt'].flatten()
    last_ts = timestamps[-1].item()

    baby_type = 'NICU' if not np.isnan(death_ts) else 'NON_NICU'
    baby_id = mat_file_path.split('/')[-1].replace('.mat', '')

    available_time = last_ts - birth_time
    if available_time < CONTEXT_SIZE_SECONDS:
        return pd.DataFrame()

    step_size = CONTEXT_SIZE_SECONDS * (1 - overlap_percentage)
    num_possible_windows = int((available_time - CONTEXT_SIZE_SECONDS) / step_size) + 1

    windows = []
    if num_possible_windows <= max_windows:
        indices = range(num_possible_windows)
    else:
        indices = np.linspace(0, num_possible_windows - 1, max_windows, dtype=int)

    for idx in indices:
        window_start = birth_time + idx * step_size
        window_end = window_start + CONTEXT_SIZE_SECONDS
        if window_end <= last_ts:
            windows.append({
                'mat_file': mat_file_path,
                'baby_id': baby_id,
                'window_start': window_start,
                'window_end': window_end,
                'baby_type': baby_type,
                'death_time': death_ts,
                'birth_time': birth_time,
            })

    return pd.DataFrame(windows)


def generate_all_window_metadata(nicu_files, non_nicu_files):
    print("\n" + "=" * 80)
    print(" " * 20 + "STEP 1: GENERATING WINDOW METADATA")
    print(" " * 15 + "(Uniformly Spaced, Threshold-Independent)")
    print("=" * 80 + "\n")

    all_metadata = []

    print(f"Processing {len(non_nicu_files)} Non-NICU babies...")
    print(f"  Overlap: {OVERLAP_PERCENTAGE_NON_NICU * 100:.0f}%")
    print(f"  Max metadata windows: {MAX_METADATA_WINDOWS_NON_NICU}\n")

    for i, file_path in enumerate(non_nicu_files):
        baby_id = file_path.split('/')[-1].replace('.mat', '')
        if (i + 1) % 20 == 0 or i + 1 == len(non_nicu_files):
            print(f"  Progress: {i + 1}/{len(non_nicu_files)} babies")
        try:
            df = generate_window_metadata_for_baby(
                file_path,
                overlap_percentage=OVERLAP_PERCENTAGE_NON_NICU,
                max_windows=MAX_METADATA_WINDOWS_NON_NICU,
            )
            all_metadata.append(df)
        except Exception as e:
            print(f"  ✗ Error with baby {baby_id}: {e}")

    print(f"\nProcessing {len(nicu_files)} NICU babies...")
    print(f"  Overlap: {OVERLAP_PERCENTAGE_NICU * 100:.0f}%")
    print(f"  Max metadata windows: {MAX_METADATA_WINDOWS_NICU}\n")

    for i, file_path in enumerate(nicu_files):
        baby_id = file_path.split('/')[-1].replace('.mat', '')
        if (i + 1) % 10 == 0 or i + 1 == len(nicu_files):
            print(f"  Progress: {i + 1}/{len(nicu_files)} babies")
        try:
            df = generate_window_metadata_for_baby(
                file_path,
                overlap_percentage=OVERLAP_PERCENTAGE_NICU,
                max_windows=MAX_METADATA_WINDOWS_NICU,
            )
            all_metadata.append(df)
        except Exception as e:
            print(f"  ✗ Error with baby {baby_id}: {e}")

    metadata_df = pd.concat(all_metadata, ignore_index=True)

    print(f"\n{'=' * 80}")
    print(f"✓ Total metadata windows: {len(metadata_df)}")
    print(f"  Non-NICU: {len(metadata_df[metadata_df['baby_type'] == 'NON_NICU'])}")
    print(f"  NICU: {len(metadata_df[metadata_df['baby_type'] == 'NICU'])}")
    print(f"{'=' * 80}\n")

    metadata_df.to_pickle(WINDOW_METADATA_PATH)
    print(f"✓ Saved: {WINDOW_METADATA_PATH}\n")

    return metadata_df
