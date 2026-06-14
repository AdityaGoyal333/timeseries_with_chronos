"""Step 1: Generate window metadata for all babies.

Pipeline role
-------------
This is the first step. For every baby .mat file we enumerate all valid
80-hour sliding windows and store their (start, end) timestamps.

Why threshold-independent?
--------------------------
Death-proximity thresholds (3-day, 7-day, etc.) are applied later in Step 2.
Generating windows once here avoids re-scanning .mat files for each threshold.

Outputs
-------
window_metadata.pkl  —  one row per candidate window, all babies combined
"""

import numpy as np
import pandas as pd
import scipy.io

from constants import (
    CONTEXT_SIZE_SECONDS,
    MAX_METADATA_WINDOWS_NICU,
    MAX_METADATA_WINDOWS_NON_NICU,
    OVERLAP_PERCENTAGE_NICU,
    OVERLAP_PERCENTAGE_NON_NICU
)


def generate_window_metadata_for_baby(mat_file_path, overlap_percentage, max_windows):
    """Return candidate windows for one baby, uniformly spaced when capped.

    Args:
        mat_file_path: Path to the baby's .mat recording.
        overlap_percentage: Fraction of overlap between consecutive windows
            (e.g. 0.5 → each window starts halfway through the previous one).
        max_windows: Cap on windows per baby; if exceeded, uniformly subsample.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except Exception:
        # Corrupt or unreadable file — skip this baby silently
        return pd.DataFrame()

    # --- Extract timing info from the .mat file ---
    # pdata fields are stored in days; vt timestamps are in seconds
    birth_time = mat_data['pdata'][0][1]       # absolute recording start
    age_at_death = mat_data['pdata'][0][-1]    # days since birth at death (NaN if alive)
    death_ts = age_at_death * 86400 if not np.isnan(age_at_death) else np.nan

    if np.isnan(birth_time):
        return pd.DataFrame()

    timestamps = mat_data['vt'].flatten()      # vital-sign sample timestamps
    last_ts = timestamps[-1].item()            # last available data point

    # NICU = recorded death; NON_NICU = survived / no death timestamp
    baby_type = 'NICU' if not np.isnan(death_ts) else 'NON_NICU'
    baby_id = mat_file_path.split('/')[-1].replace('.mat', '')

    # Total usable recording length for this baby
    available_time = last_ts - birth_time
    if available_time < CONTEXT_SIZE_SECONDS:
        # Recording too short to fit even one 80-hour window
        return pd.DataFrame()

    # --- Sliding window geometry ---
    # step_size: how far we advance between consecutive window starts
    # 50% overlap → step = 50% of window length → 2× more windows
    step_size = CONTEXT_SIZE_SECONDS * (1 - overlap_percentage)

    # Number of windows that fit: first window starts at birth_time,
    # last valid start is when window_end would still be ≤ last_ts
    num_possible_windows = int((available_time - CONTEXT_SIZE_SECONDS) / step_size) + 1

    windows = []
    if num_possible_windows <= max_windows:
        # Case A: manageable count — keep every window
        indices = range(num_possible_windows)
    else:
        # Case B: too many windows — pick max_windows evenly spaced indices
        # (preserves coverage of early AND late parts of the timeline)
        indices = np.linspace(0, num_possible_windows - 1, max_windows, dtype=int)

    for idx in indices:
        window_start = birth_time + idx * step_size
        window_end = window_start + CONTEXT_SIZE_SECONDS

        # Safety check: int/float rounding can push window_end past last_ts
        if window_end <= last_ts:
            windows.append({
                'mat_file': mat_file_path,
                'baby_id': baby_id,
                'window_start': window_start,   # absolute timestamp (seconds)
                'window_end': window_end,       # absolute timestamp (seconds)
                'baby_type': baby_type,
                'death_time': death_ts,         # seconds since birth (NOT absolute)
                'birth_time': birth_time,       # absolute timestamp (seconds)
            })

    return pd.DataFrame(windows)


def generate_all_window_metadata(nicu_files, non_nicu_files):
    """Generate and persist metadata for all babies, with separate settings per type.

    NICU and Non-NICU babies can use different overlap % and window caps
    because their recording lengths and class sizes differ.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "STEP 1: GENERATING WINDOW METADATA")
    print(" " * 15 + "(Uniformly Spaced, Threshold-Independent)")
    print("=" * 80 + "\n")

    all_metadata = []

    # --- Non-NICU babies (survivors) ---
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
            all_metadata.append(df)  # empty DataFrame is fine — concat handles it
        except Exception as e:
            print(f"  ✗ Error with baby {baby_id}: {e}")

    # --- NICU babies (deceased) ---
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

    return metadata_df
