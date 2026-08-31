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

# --- Step 3: binning modes (how one 80h window is sliced before Chronos) ---
BINNING_SEASONAL = 'seasonal'           # 960 equal 300s bins (current default workflow)
BINNING_FINE_GRAINED = 'fine_grained'   # 960 geometric bins, coarse early → fine late
BINNING_CHOICES = (BINNING_SEASONAL, BINNING_FINE_GRAINED)

# Fine-grained: first bin (start of window) is widest, last bin (near window_end) is narrowest
FINE_GRAINED_MIN_BIN_SECONDS = 60    # 1 minute
FINE_GRAINED_MAX_BIN_SECONDS = 900   # 15 minutes

# Separate embedding caches per binning mode (window pickles are shared)
EMBEDDINGS_CACHE_PATHS = {
    BINNING_SEASONAL: 'embeddings_global.pkl',
    BINNING_FINE_GRAINED: 'embeddings_global_fine_grained.pkl',
}
MISSING_WINDOWS_PATHS = {
    BINNING_SEASONAL: 'windows_missing_embeddings.pkl',
    BINNING_FINE_GRAINED: 'windows_missing_embeddings_fine_grained.pkl',
}

# Suffix for labels / results / curve dirs when binning != seasonal
FINE_GRAINED_OUTPUT_SUFFIX = '_fine_grained'