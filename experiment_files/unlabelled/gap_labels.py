"""Gap labels for windows whose death time is unknown.

Known 0/1 labels are left alone. Unknown rows get a sampled gap, then 0/1
from whether the window ends inside the threshold of (death_time + gap).
"""

import numpy as np
import pandas as pd

from constants import DEATH_GAP_PATH, US_PER_SECOND

# Fitted once per process (seconds). None until get_gap_sampler() runs.
_SAMPLER = None


class GammaKDE:
    """Non-negative gap distribution fit once from observed gaps (seconds)."""

    def __init__(self, bandwidth=180.0, max_gap=7 * 24 * 60 * 60):
        self.bandwidth = bandwidth
        self.max_gap = max_gap

    def fit(self, gaps, grid_size=30000):
        """Build a discrete density on [0, max_gap] from observed gaps."""
        from statsmodels.nonparametric.kernels_asymmetric import pdf_kernel_asym
        self.x = np.linspace(0, self.max_gap, grid_size)
        self.density = pdf_kernel_asym(
            x=self.x,
            sample=np.asarray(gaps, dtype=np.float64),
            bw=self.bandwidth,
            kernel_type="gamma2",
        )
        mass = np.nan_to_num(self.density, nan=0.0, posinf=0.0, neginf=0.0)
        mass = np.clip(mass, 0, None)
        total = float(mass.sum())
        if total <= 0:
            raise ValueError("Gap KDE density is empty; check death_gap_seconds.")
        self.probabilities = mass / total
        return self

    def sample(self, n=1, random_state=None):
        """Draw n gaps (seconds) from the fitted density."""
        rng = np.random.default_rng(random_state)
        return rng.choice(self.x, size=n, p=self.probabilities)


def _fit_gap_sampler(path):
    """Load death_gap_seconds, drop NaN/negatives, fit GammaKDE in seconds."""
    gaps = pd.read_parquet(path, columns=["death_gap_seconds"])["death_gap_seconds"].to_numpy()
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.size == 0:
        raise ValueError(f"No finite non-negative death_gap_seconds in {path}")
    return GammaKDE().fit(gaps)


def get_gap_sampler(path=None):
    """Return the process-wide fitted sampler; load+fit on the first call."""
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = _fit_gap_sampler(DEATH_GAP_PATH if path is None else path)
    return _SAMPLER


def sample_gap(rng, size=None):
    """Draw one or more non-negative gaps (same time units as the window times).

    Args:
        rng: sampler used to draw the gap (any object/callable this function
            is written to call).
        size: None for a single gap; otherwise how many independent gaps.

    Returns:
        One gap, or a sequence of gaps of length size. Each gap is >= 0.
    """
    n = 1 if size is None else int(size)
    # Sampler is fit in seconds; unlabeled window times are microseconds.
    gaps = np.asarray(rng.sample(n=n), dtype=np.float64).reshape(-1) * US_PER_SECOND
    if size is None:
        return float(gaps[0])
    return gaps


def label_from_gap(window_end, death_time, gap, threshold_us):
    """0/1 for one window given a gap: 1 if the window is within the horizon of death.

    death = death_time + gap.
    Label 1 iff window_end > death - threshold.

    Args:
        window_end: exclusive end of this window.
        death_time: recorded last time for this file (before adding gap).
        gap: offset sampled for this window (same units as the times).
        threshold_us: mortality horizon (same units as the times).

    Returns:
        0 or 1.
    """
    return int(window_end > (death_time + gap) - threshold_us)


def apply_gap_labels(realized_labels, unknown_mask, window_end, death_time, threshold_us, rng):
    """Fill unknown rows with a new gap label; copy known 0/1 through.

    Row i of every array is the same window.

    Args:
        realized_labels: per-window label if already known; unset if unknown.
        unknown_mask: True where a gap should be sampled and the label computed.
        window_end: per-window exclusive end time.
        death_time: per-window recorded last time (before gap).
        threshold_us: mortality horizon, same for every window.
        rng: sampler passed into sample_gap.

    Returns:
        Per-window 0/1 labels. Unknown rows replaced; known rows unchanged.
    """
    labels = np.array(realized_labels, dtype=np.float64, copy=True)
    idx = np.flatnonzero(unknown_mask)
    # Nothing unknown: return known labels as-is.
    if idx.size == 0:
        return labels
    gap = sample_gap(rng, size=idx.size)
    # One gap per unknown window; same rule as label_from_gap.
    labels[idx] = (window_end[idx] > death_time[idx] + gap - threshold_us).astype(np.float64)
    return labels
