"""Step 2 (unlabeled): downsample candidate windows. No death-proximity labels.

Pipeline role
-------------
Same per-baby counts and NICU late bias as labeled sampling. Unlabeled
parquet has no death timestamp, so this step does not write
sampled_windows_all_thresholds.pkl.

Outputs
-------
data/unlabeled/windows_to_embed.pkl  —  written by main
"""

import numpy as np
import pandas as pd

from constants import (
    NICU_LATE_WINDOW_BIAS,
    NUM_SAMPLE_WINDOWS_NICU,
    NUM_SAMPLE_WINDOWS_NON_NICU,
)
from log_utils import tprint


def sample_windows_once(metadata_df):
    """Downsample unlabeled metadata windows (class-balance aware, no labels)."""
    tprint("\n" + "=" * 80)
    tprint(" " * 12 + "STEP 2 (UNLABELED): SAMPLING WINDOWS")
    tprint("=" * 80 + "\n")

    all_sampled_windows = []

    tprint("Sampling Non-NICU babies (REDUCED for balance)...")
    non_nicu_babies = metadata_df[metadata_df['baby_type'] == 'NON_NICU']['baby_id'].unique()

    non_nicu_count = 0
    for baby_id in non_nicu_babies:
        baby_windows = metadata_df[metadata_df['baby_id'] == baby_id]
        n_available = len(baby_windows)
        n_needed = NUM_SAMPLE_WINDOWS_NON_NICU

        if n_available <= n_needed:
            selected = baby_windows.copy()
        else:
            indices = np.linspace(0, n_available - 1, n_needed, dtype=int)
            selected = baby_windows.iloc[indices].copy()

        all_sampled_windows.append(selected)
        non_nicu_count += len(selected)

    n_non = len(non_nicu_babies)
    tprint(f"  ✓ Sampled {non_nicu_count} windows from {n_non} Non-NICU babies")
    if n_non:
        tprint(f"    Average: {non_nicu_count / n_non:.1f} windows/baby\n")
    else:
        tprint("")

    tprint("Sampling NICU babies (BIASED toward later windows)...")
    nicu_babies = metadata_df[metadata_df['baby_type'] == 'NICU']['baby_id'].unique()

    nicu_count = 0
    early_count = 0
    late_count = 0

    for baby_id in nicu_babies:
        baby_windows = metadata_df[metadata_df['baby_id'] == baby_id].sort_values('window_start')
        n_available = len(baby_windows)
        n_needed = NUM_SAMPLE_WINDOWS_NICU

        if n_available <= n_needed:
            selected = baby_windows.copy()
            all_sampled_windows.append(selected)
            nicu_count += len(selected)
        else:
            midpoint = len(baby_windows) // 2
            early_windows = baby_windows.iloc[:midpoint]
            late_windows = baby_windows.iloc[midpoint:]

            n_late = int(n_needed * NICU_LATE_WINDOW_BIAS)
            n_early = n_needed - n_late
            selected_windows = []

            if len(early_windows) > 0 and n_early > 0:
                n_early_actual = min(n_early, len(early_windows))
                indices = np.linspace(0, len(early_windows) - 1, n_early_actual, dtype=int)
                selected_windows.append(early_windows.iloc[indices])
                early_count += n_early_actual

            if len(late_windows) > 0 and n_late > 0:
                n_late_actual = min(n_late, len(late_windows))
                indices = np.linspace(0, len(late_windows) - 1, n_late_actual, dtype=int)
                selected_windows.append(late_windows.iloc[indices])
                late_count += n_late_actual

            if selected_windows:
                selected = pd.concat(selected_windows)
                all_sampled_windows.append(selected)
                nicu_count += len(selected)

    n_nicu = len(nicu_babies)
    tprint(f"  ✓ Sampled {nicu_count} windows from {n_nicu} NICU babies")
    if n_nicu:
        tprint(f"    Average: {nicu_count / n_nicu:.1f} windows/baby")
    if nicu_count:
        tprint(f"    Early windows: {early_count} ({early_count / nicu_count * 100:.1f}%)")
        tprint(f"    Late windows: {late_count} ({late_count / nicu_count * 100:.1f}%)\n")
    else:
        tprint("")

    if not all_sampled_windows:
        raise ValueError("No unlabeled windows to sample. Check parquet + outcomes join.")

    sampled_windows_df = pd.concat(all_sampled_windows, ignore_index=True)

    tprint(f"{'=' * 80}")
    tprint(f"✓ Total sampled windows: {len(sampled_windows_df)}")
    tprint(f"  Non-NICU: {len(sampled_windows_df[sampled_windows_df['baby_type'] == 'NON_NICU'])}")
    tprint(f"  NICU: {len(sampled_windows_df[sampled_windows_df['baby_type'] == 'NICU'])}")
    tprint(f"  Bias applied: {NICU_LATE_WINDOW_BIAS * 100:.0f}% late, {(1 - NICU_LATE_WINDOW_BIAS) * 100:.0f}% early")
    tprint(f"{'=' * 80}\n")

    return sampled_windows_df
