"""Step 1 (unlabeled): candidate 80h windows from parquet vitals.

Pipeline role
-------------
Same overlap/caps as labeled window_metadata, but t0 is min(age_us) rather
than birth, and clocks stay in microseconds.

birth_time / death_time are first and last age_us (absolute), not calendar
birth or in-hospital death. NICU vs NON_NICU still comes from died_yn.

Outputs
-------
data/unlabeled/window_metadata.pkl  —  written by main
"""

from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    CONTEXT_SIZE_US,
    MAX_METADATA_WINDOWS_NICU,
    MAX_METADATA_WINDOWS_NON_NICU,
    OVERLAP_PERCENTAGE_NICU,
    OVERLAP_PERCENTAGE_NON_NICU,
)
from log_utils import tprint


def generate_window_metadata_for_baby(parquet_path, overlap_percentage, max_windows, baby_type):
    """Return candidate 80h windows for one parquet, uniformly spaced when capped."""
    try:
        df = pd.read_parquet(parquet_path, columns=['age_us'])
    except Exception:
        return pd.DataFrame()

    if df.empty or df['age_us'].isna().all():
        return pd.DataFrame()

    t0 = int(df['age_us'].min())
    last_ts = int(df['age_us'].max())
    available_time = last_ts - t0
    if available_time < CONTEXT_SIZE_US:
        return pd.DataFrame()

    baby_id = Path(parquet_path).stem
    step_size = CONTEXT_SIZE_US * (1 - overlap_percentage)
    num_possible_windows = int((available_time - CONTEXT_SIZE_US) / step_size) + 1

    if num_possible_windows <= max_windows:
        indices = range(num_possible_windows)
    else:
        indices = np.linspace(0, num_possible_windows - 1, max_windows, dtype=int)

    windows = []
    for idx in indices:
        window_start = t0 + idx * step_size
        window_end = window_start + CONTEXT_SIZE_US
        if window_end <= last_ts:
            windows.append({
                # Identity only; training_unlabeled resolves the parquet from this id.
                'mat_file': baby_id,
                'baby_id': baby_id,
                'window_start': window_start,
                'window_end': window_end,
                'baby_type': baby_type,
                # Absolute last sample (µs), not labeled "seconds since birth".
                'death_time': last_ts,
                'birth_time': t0,
            })

    return pd.DataFrame(windows)


def generate_all_window_metadata(nicu_files, non_nicu_files):
    """Generate unlabeled metadata for all parquet babies, NICU vs NON_NICU settings."""
    tprint("\n" + "=" * 80)
    tprint(" " * 16 + "STEP 1 (UNLABELED): WINDOW METADATA")
    tprint(" " * 10 + "(First age_us, microseconds, threshold-independent)")
    tprint("=" * 80 + "\n")

    all_metadata = []

    tprint(f"Processing {len(non_nicu_files)} Non-NICU babies...")
    tprint(f"  Overlap: {OVERLAP_PERCENTAGE_NON_NICU * 100:.0f}%")
    tprint(f"  Max metadata windows: {MAX_METADATA_WINDOWS_NON_NICU}\n")

    for i, file_path in enumerate(non_nicu_files):
        baby_id = Path(file_path).stem
        if (i + 1) % 20 == 0 or i + 1 == len(non_nicu_files):
            tprint(f"  Progress: {i + 1}/{len(non_nicu_files)} babies")
        try:
            df = generate_window_metadata_for_baby(
                file_path,
                overlap_percentage=OVERLAP_PERCENTAGE_NON_NICU,
                max_windows=MAX_METADATA_WINDOWS_NON_NICU,
                baby_type='NON_NICU',
            )
            all_metadata.append(df)
        except Exception as e:
            tprint(f"  ✗ Error with baby {baby_id}: {e}")

    tprint(f"\nProcessing {len(nicu_files)} NICU babies...")
    tprint(f"  Overlap: {OVERLAP_PERCENTAGE_NICU * 100:.0f}%")
    tprint(f"  Max metadata windows: {MAX_METADATA_WINDOWS_NICU}\n")

    for i, file_path in enumerate(nicu_files):
        baby_id = Path(file_path).stem
        if (i + 1) % 10 == 0 or i + 1 == len(nicu_files):
            tprint(f"  Progress: {i + 1}/{len(nicu_files)} babies")
        try:
            df = generate_window_metadata_for_baby(
                file_path,
                overlap_percentage=OVERLAP_PERCENTAGE_NICU,
                max_windows=MAX_METADATA_WINDOWS_NICU,
                baby_type='NICU',
            )
            all_metadata.append(df)
        except Exception as e:
            tprint(f"  ✗ Error with baby {baby_id}: {e}")

    if not all_metadata:
        tprint("✗ No unlabeled metadata windows generated\n")
        return pd.DataFrame()

    metadata_df = pd.concat(all_metadata, ignore_index=True)

    tprint(f"\n{'=' * 80}")
    tprint(f"✓ Total metadata windows: {len(metadata_df)}")
    if len(metadata_df) > 0:
        tprint(f"  Non-NICU: {len(metadata_df[metadata_df['baby_type'] == 'NON_NICU'])}")
        tprint(f"  NICU: {len(metadata_df[metadata_df['baby_type'] == 'NICU'])}")
    tprint(f"{'=' * 80}\n")

    return metadata_df
