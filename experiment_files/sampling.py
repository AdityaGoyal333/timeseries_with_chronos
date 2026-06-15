"""Step 2: Sample windows and assign labels for all thresholds.

Pipeline role
-------------
Step 1 produces thousands of candidate windows per baby. This step:
  1. Downsamples to a tractable set for Chronos embedding (Step 3).
  2. Assigns binary labels under each death-proximity threshold.

Key design choice
-----------------
The SAME sampled windows are reused for every threshold — only the labels
change. That way we embed each window once but train classifiers for
3-day, 7-day, 14-day, etc. without re-running Chronos.

Outputs
-------
sampled_windows_all_thresholds.pkl  —  labeled rows (windows × thresholds)
windows_to_embed.pkl                —  unique windows only (for Step 3)
"""

import numpy as np
import pandas as pd

from constants import (
    NICU_LATE_WINDOW_BIAS,
    NUM_SAMPLE_WINDOWS_NICU,
    NUM_SAMPLE_WINDOWS_NON_NICU,
    THRESHOLDS_DAYS,
    THRESHOLDS_SECONDS
)


def sample_windows_once(metadata_df):
    """Downsample metadata windows before embedding (class-balance aware).

    Problem: Non-NICU babies far outnumber NICU babies, and each Non-NICU
    baby has a long recording → many windows. Without downsampling the
    classifier would see mostly label-0 (non-death) examples.

    Strategy:
      - Non-NICU: take NUM_SAMPLE_WINDOWS_NON_NICU per baby, evenly spaced.
      - NICU: take NUM_SAMPLE_WINDOWS_NICU per baby, biased toward the
        late (pre-death) half of the timeline where label-1 windows live.
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "STEP 2: SAMPLING WINDOWS (BIASED FOR BALANCE)")
    print("=" * 80 + "\n")

    all_sampled_windows = []

    # ------------------------------------------------------------------
    # Non-NICU babies — uniform sampling, fewer windows per baby
    # ------------------------------------------------------------------
    print("Sampling Non-NICU babies (REDUCED for balance)...")
    non_nicu_babies = metadata_df[metadata_df['baby_type'] == 'NON_NICU']['baby_id'].unique()

    non_nicu_count = 0
    for baby_id in non_nicu_babies:
        baby_windows = metadata_df[metadata_df['baby_id'] == baby_id]
        n_available = len(baby_windows)
        n_needed = NUM_SAMPLE_WINDOWS_NON_NICU

        if n_available <= n_needed:
            # Baby has fewer windows than we want — take them all
            selected = baby_windows.copy()
        else:
            # Pick n_needed evenly spaced rows (by index position, not time)
            indices = np.linspace(0, n_available - 1, n_needed, dtype=int)
            selected = baby_windows.iloc[indices].copy()

        all_sampled_windows.append(selected)
        non_nicu_count += len(selected)

    print(f"  ✓ Sampled {non_nicu_count} windows from {len(non_nicu_babies)} Non-NICU babies")
    print(f"    Average: {non_nicu_count / len(non_nicu_babies):.1f} windows/baby\n")

    # ------------------------------------------------------------------
    # NICU babies — biased sampling toward late (pre-death) windows
    # ------------------------------------------------------------------
    print("Sampling NICU babies (BIASED toward later windows)...")
    nicu_babies = metadata_df[metadata_df['baby_type'] == 'NICU']['baby_id'].unique()

    nicu_count = 0
    early_count = 0   # for logging: how many windows came from first half
    late_count = 0    # for logging: how many came from second half

    for baby_id in nicu_babies:
        # Must sort so "early" = first half of timeline, "late" = second half
        baby_windows = metadata_df[metadata_df['baby_id'] == baby_id].sort_values('window_start')
        n_available = len(baby_windows)
        n_needed = NUM_SAMPLE_WINDOWS_NICU

        if n_available <= n_needed:
            # Not enough windows to subsample — keep all
            selected = baby_windows.copy()
            all_sampled_windows.append(selected)
            nicu_count += len(selected)
            # early_count / late_count stay 0 (stats only apply when subsampling)
        else:
            # Split timeline in half by window count (not by time duration)
            midpoint = len(baby_windows) // 2
            early_windows = baby_windows.iloc[:midpoint]
            late_windows = baby_windows.iloc[midpoint:]

            # e.g. NICU_LATE_WINDOW_BIAS=0.7 → 70% of slots go to late half
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

    print(f"  ✓ Sampled {nicu_count} windows from {len(nicu_babies)} NICU babies")
    print(f"    Average: {nicu_count / len(nicu_babies):.1f} windows/baby")
    print(f"    Early windows: {early_count} ({early_count / nicu_count * 100:.1f}%)")
    print(f"    Late windows: {late_count} ({late_count / nicu_count * 100:.1f}%)\n")

    sampled_windows_df = pd.concat(all_sampled_windows, ignore_index=True)

    print(f"{'=' * 80}")
    print(f"✓ Total sampled windows: {len(sampled_windows_df)}")
    print(f"  Non-NICU: {len(sampled_windows_df[sampled_windows_df['baby_type'] == 'NON_NICU'])}")
    print(f"  NICU: {len(sampled_windows_df[sampled_windows_df['baby_type'] == 'NICU'])}")
    print(f"  Bias applied: {NICU_LATE_WINDOW_BIAS * 100:.0f}% late, {(1 - NICU_LATE_WINDOW_BIAS) * 100:.0f}% early")
    print(f"{'=' * 80}\n")

    return sampled_windows_df


def assign_labels_for_all_thresholds(sampled_windows_df):
    """Label the same windows under each death-proximity threshold.

    Label definition (NICU babies only):
      Label 1 (positive): window_end falls inside the last `threshold` seconds
                          before death  →  "close to death"
      Label 0 (negative): window_end is earlier  →  "far from death"

    Non-NICU babies always get label 0 (they did not die in-hospital).

    Timeline example (7-day threshold):
      birth -------- window_end? -------- |← 7 days →| death
                                          threshold_start
      If window_end > threshold_start  →  label 1
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "ASSIGNING LABELS FOR ALL THRESHOLDS")
    print(" " * 20 + "(Same windows, different labels)")
    print("=" * 80 + "\n")

    all_labeled_data = []

    for threshold_days, threshold_seconds in zip(THRESHOLDS_DAYS, THRESHOLDS_SECONDS):
        print(f"Assigning labels for {threshold_days}-day threshold...")

        df = sampled_windows_df.copy()

        # All Non-NICU → label 0 (negative class / no imminent death)
        df.loc[df['baby_type'] == 'NON_NICU', 'label'] = 0

        nicu_mask = df['baby_type'] == 'NICU'

        # Convert relative death_time to absolute timestamp for comparison
        # death_time = seconds since birth; window_end = absolute seconds
        absolute_death = df.loc[nicu_mask, 'birth_time'] + df.loc[nicu_mask, 'death_time']
        threshold_start = absolute_death - threshold_seconds

        # Window is "positive" if it ends anywhere inside the threshold window
        df.loc[nicu_mask, 'label'] = (df.loc[nicu_mask, 'window_end'] > threshold_start).astype(int)
        df['threshold_days'] = threshold_days

        all_labeled_data.append(df)

        n_total = len(df)
        n_label_0 = (df['label'] == 0).sum()
        n_label_1 = (df['label'] == 1).sum()
        print(f"  ✓ Total: {n_total}, Label 0: {n_label_0}, Label 1: {n_label_1}")
        print(f"    Class balance: {n_label_0 / n_total * 100:.1f}% / {n_label_1 / n_total * 100:.1f}%\n")

    # Stack all thresholds — same windows repeated with different labels
    all_labeled_df = pd.concat(all_labeled_data, ignore_index=False)

    # Step 3 only needs unique windows (labels don't affect the embedding)
    unique_windows_df = sampled_windows_df.copy()

    print(f"\n{'=' * 80}")
    print(f"✓ Total labeled entries: {len(all_labeled_df)}")
    print(f"✓ Unique windows: {len(unique_windows_df)}")
    print(f"  (Each window appears {len(THRESHOLDS_DAYS)} times with different labels)")
    print(f"{'=' * 80}\n")

    return all_labeled_df, unique_windows_df


def sample_and_label_all_thresholds(metadata_df):
    """Convenience wrapper: run Step 2 sampling then labeling in one call."""
    sampled_windows_df = sample_windows_once(metadata_df)
    return assign_labels_for_all_thresholds(sampled_windows_df)



# window_start + window_duration + threshold > absolute_death_time