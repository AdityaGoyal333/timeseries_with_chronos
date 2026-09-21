"""Per-epoch / per-batch gap labels for unlabeled died-file windows.

Classify only. Survivors stay 0; labeled 0/1 are never gapped.
Unknown rows (label NaN): death = death_time + gap, then 0/1 vs threshold.

sample_gap is a placeholder (gap >= 0, same units as age_us).
"""

import numpy as np


def sample_gap(rng, size=None):
    """Draw a non-negative gap. Replace this distribution later."""
    if size is None:
        return float(rng.exponential(1.0))
    return rng.exponential(1.0, size=size)


def label_from_gap(window_end, death_time, gap, threshold_us):
    """1 iff the window ends inside the threshold before (death_time + gap)."""
    return int(window_end > (death_time + gap) - threshold_us)


def apply_gap_labels(realized_labels, unknown_mask, window_end, death_time, threshold_us, rng):
    """Copy realized 0/1; overwrite unknown rows with a fresh gap draw."""
    labels = np.array(realized_labels, dtype=np.float64, copy=True)
    idx = np.flatnonzero(unknown_mask)
    if idx.size == 0:
        return labels
    gap = sample_gap(rng, size=idx.size)
    labels[idx] = (window_end[idx] > death_time[idx] + gap - threshold_us).astype(np.float64)
    return labels
