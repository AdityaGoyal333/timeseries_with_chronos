"""Shared knobs for the NICU mortality pipeline (Steps 1–5).

Pipeline role
-------------
Imported by window_metadata, sampling, training_data, train_classifier, and
main. Change geometry, thresholds, cache filenames, and CLI tokens here so
those modules stay in sync.

Not a runnable step. Values below are the contract for pickle names and
window keys; changing them invalidates existing caches unless you regenerate.
"""

from pathlib import Path

# Repo root from this file (not cwd), same pattern as DATA_DIR in the step modules.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = (REPO_ROOT / "results").resolve()

# =============================================================================
# Constants and paths
# =============================================================================

# --- Window geometry ---
# Each model input is an 80-hour context window sliced into 5-minute sub-windows
CONTEXT_SIZE_HOURS = 80
CONTEXT_SIZE_SECONDS = CONTEXT_SIZE_HOURS * 3600   # 288,000 s
WINDOW_SIZE_SECONDS = 300                             # 5 minutes per sub-window
NUM_WINDOWS = CONTEXT_SIZE_SECONDS // WINDOW_SIZE_SECONDS  # 960 sub-windows

# --- Death-proximity thresholds ---
# A window gets label 1 if it ends within this many days before death
THRESHOLDS_DAYS = [3, 5, 7]
THRESHOLDS_SECONDS = [d * 86400 for d in THRESHOLDS_DAYS]

# --- Step 1: metadata generation (window_metadata.py) ---
# Overlap between consecutive 80-hour windows; NICU uses high overlap to
# capture fine-grained pre-death dynamics
OVERLAP_PERCENTAGE_NON_NICU = 0.0
OVERLAP_PERCENTAGE_NICU = 0.90

# Max candidate windows stored per baby in metadata (uniformly subsampled if exceeded)
MAX_METADATA_WINDOWS_NON_NICU = 15
MAX_METADATA_WINDOWS_NICU = 200

# --- Step 2: sampling (sampling.py) ---
# How many windows to keep per baby after downsampling
NUM_SAMPLE_WINDOWS_NON_NICU = 5
NUM_SAMPLE_WINDOWS_NICU = 100
# Fraction of NICU sample slots allocated to the late (pre-death) half of timeline
NICU_LATE_WINDOW_BIAS = 0.92

# --- Step 3: binning modes (how one 80h sample is sliced before Chronos) ---
BINNING_SEASONAL = 'seasonal'
BINNING_FINE_GRAINED = 'fine_grained'
BINNING_CHOICES = (BINNING_SEASONAL, BINNING_FINE_GRAINED)

# Fine-grained: last C seconds of the same 80h sample, equal bins of W seconds.
# Chronos length = FG_CONTEXT_SIZE_SECONDS // FG_WINDOW_SIZE_SECONDS (tune this pair).
FG_CONTEXT_SIZE_SECONDS = 11520
FG_WINDOW_SIZE_SECONDS = 12
NUM_WINDOWS_FG = FG_CONTEXT_SIZE_SECONDS // FG_WINDOW_SIZE_SECONDS

# FG region = last FG_LOOKBACK_SECONDS before window_end. Each Chronos call
# still sees only C seconds; NUM_FG_EMBEDDINGS starts are drawn uniformly
# in [window_end - lookback, window_end - C] (seeded per 80h window).
FG_LOOKBACK_SECONDS = 10 * 3600
NUM_FG_EMBEDDINGS = 4

# Separate embedding caches per binning mode (window pickles are shared)
EMBEDDINGS_CACHE_PATHS = {
    BINNING_SEASONAL: 'embeddings_global.pkl',
    BINNING_FINE_GRAINED: 'embeddings_global_fine_grained.pkl',
}
MISSING_WINDOWS_PATHS = {
    BINNING_SEASONAL: 'windows_missing_embeddings.pkl',
    BINNING_FINE_GRAINED: 'windows_missing_embeddings_fine_grained.pkl',
}

# Filename suffixes so seasonal / FG / concat artifacts do not overwrite each other.
FINE_GRAINED_OUTPUT_SUFFIX = '_fine_grained'
CONCAT_OUTPUT_SUFFIX = '_concat'

# classify --head: one cache (singular) vs seasonal∥FG (concat). Concat ignores --mode.
HEAD_SINGULAR = 'singular'
HEAD_CONCAT = 'concat'
HEAD_CHOICES = (HEAD_SINGULAR, HEAD_CONCAT)