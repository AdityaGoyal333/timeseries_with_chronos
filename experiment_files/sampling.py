"""Step 2: Sample windows and assign labels for all thresholds."""

import numpy as np
import pandas as pd

from main import (
    NICU_LATE_WINDOW_BIAS,
    NUM_SAMPLE_WINDOWS_NICU,
    NUM_SAMPLE_WINDOWS_NON_NICU,
    SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH,
    THRESHOLDS_DAYS,
    THRESHOLDS_SECONDS,
    WINDOWS_TO_EMBED_PATH,
)


def sample_windows_once(metadata_df):
    print("\n" + "=" * 80)
    print(" " * 15 + "STEP 2: SAMPLING WINDOWS (BIASED FOR BALANCE)")
    print("=" * 80 + "\n")

    all_sampled_windows = []

    print("Sampling Non-NICU babies (REDUCED for balance)...")
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

    print(f"  ✓ Sampled {non_nicu_count} windows from {len(non_nicu_babies)} Non-NICU babies")
    print(f"    Average: {non_nicu_count / len(non_nicu_babies):.1f} windows/baby\n")

    print("Sampling NICU babies (BIASED toward later windows)...")
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
    print("\n" + "=" * 80)
    print(" " * 15 + "ASSIGNING LABELS FOR ALL THRESHOLDS")
    print(" " * 20 + "(Same windows, different labels)")
    print("=" * 80 + "\n")

    all_labeled_data = []

    for threshold_days, threshold_seconds in zip(THRESHOLDS_DAYS, THRESHOLDS_SECONDS):
        print(f"Assigning labels for {threshold_days}-day threshold...")

        df = sampled_windows_df.copy()
        df.loc[df['baby_type'] == 'NON_NICU', 'label'] = 0

        nicu_mask = df['baby_type'] == 'NICU'
        threshold_start = df.loc[nicu_mask, 'death_time'] - threshold_seconds
        df.loc[nicu_mask, 'label'] = (df.loc[nicu_mask, 'window_end'] > threshold_start).astype(int)
        df['threshold_days'] = threshold_days

        all_labeled_data.append(df)

        n_total = len(df)
        n_label_0 = (df['label'] == 0).sum()
        n_label_1 = (df['label'] == 1).sum()
        print(f"  ✓ Total: {n_total}, Label 0: {n_label_0}, Label 1: {n_label_1}")
        print(f"    Class balance: {n_label_0 / n_total * 100:.1f}% / {n_label_1 / n_total * 100:.1f}%\n")

    all_labeled_df = pd.concat(all_labeled_data, ignore_index=False)
    all_labeled_df.to_pickle(SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH)
    print(f"✓ Saved: {SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH}")

    unique_windows_df = sampled_windows_df.copy()
    unique_windows_df.to_pickle(WINDOWS_TO_EMBED_PATH)
    print(f"✓ Saved: {WINDOWS_TO_EMBED_PATH}")

    print(f"\n{'=' * 80}")
    print(f"✓ Total labeled entries: {len(all_labeled_df)}")
    print(f"✓ Unique windows: {len(unique_windows_df)}")
    print(f"  (Each window appears {len(THRESHOLDS_DAYS)} times with different labels)")
    print(f"{'=' * 80}\n")

    return all_labeled_df, unique_windows_df


def sample_and_label_all_thresholds(metadata_df):
    sampled_windows_df = sample_windows_once(metadata_df)
    return assign_labels_for_all_thresholds(sampled_windows_df)
