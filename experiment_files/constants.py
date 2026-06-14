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