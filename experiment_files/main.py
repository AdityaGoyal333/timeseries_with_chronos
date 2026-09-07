"""
Mortality prediction pipeline — shared config, helpers, and orchestrator.

Central hub for the experiment. All step modules (window_metadata, sampling,
training_data, train_classifier) import constants and helpers from here.

Pipeline steps (run in order by run_complete_pipeline):
  1. window_metadata  — enumerate 80-hour sliding windows per baby
  2. sampling         — downsample windows + assign death-proximity labels
  3. training_data    — embed windows with Chronos-2, cache vectors
  4. training_data    — pair embeddings with threshold-specific labels
  5. train_classifier — k-fold CV on a small MLP head

Run directly:
  python main.py --binning seasonal
  python main.py --binning fine_grained
"""

import argparse
import os
import pickle as pl
import pandas as pd
from pathlib import Path


# Lazy imports so constants/helpers are available without circular deps
from sampling import sample_and_label_all_thresholds
from train_classifier import train_and_save_results
from training_data import ensure_embeddings_for_windows, prepare_training_data_for_threshold
from window_metadata import generate_all_window_metadata
from constants import (
    BINNING_CHOICES,
    THRESHOLDS_DAYS,
)
from log_utils import tprint


# Pre-built list of valid babies (NICU vs Non-NICU classification)
# Absolute path to the directory containing this Python file

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()

# Absolute paths
VALID_SUMMARY_PATH = (DATA_DIR / "valid_summary_dic.pkl").resolve()
VITALS_DIR = (DATA_DIR / "download" / "Vitals").resolve()

# Glob pattern to find all .mat files in the vitals directory
VITALS_FILES = list(VITALS_DIR.glob("*.mat"))
VITALS_FILES = [str(f) for f in VITALS_FILES]  # Convert Path objects to strings


# --- Intermediate / output file paths (written to cwd) ---
WINDOW_METADATA_PATH = (DATA_DIR / 'window_metadata.pkl').resolve()
WINDOWS_TO_EMBED_PATH = (DATA_DIR / 'windows_to_embed.pkl').resolve()
SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH = (DATA_DIR / 'sampled_windows_all_thresholds.pkl').resolve()

# =============================================================================
# Pipeline orchestrator
# =============================================================================

def run_complete_pipeline(use_focal_loss, gamma, binning_mode):
    """Run all 5 pipeline steps end-to-end for every death-proximity threshold.

    Skips steps whose output files already exist (incremental / resumable):
      - window_metadata.pkl  → skip Step 1
      - windows_to_embed.pkl → skip Step 2

    Args:
        use_focal_loss: If True, train with focal loss; else binary cross-entropy.
        gamma: Focal-loss focusing parameter (only used when use_focal_loss=True).
        binning_mode: 'seasonal' (full 80h, 300s bins) or 'fine_grained'
            (last FG_CONTEXT hours of the same 80h sample, FG_WINDOW bins).
    """

    loss_name = f"Focal Loss (γ={gamma})" if use_focal_loss else "Cross-Entropy"
    tprint("\n" + "=" * 100)
    tprint(" " * 15 + f"MORTALITY PREDICTION PIPELINE WITH {loss_name.upper()}")
    tprint(f"  Binning: {binning_mode}")
    tprint("=" * 100 + "\n")






    # --- Discover which .mat files belong to NICU vs Non-NICU babies ---
    with open(VALID_SUMMARY_PATH, 'rb') as f:
        valid_summary_dic = pl.load(f)

    # valid_summary_dic only contains files with atleast 70 percent hit rate in both spo2 and hr
    nicu_mask = valid_summary_dic['indicator_vector'] == 1
    nicu_baby_ids = set(valid_summary_dic['baby_id'][nicu_mask])
    non_nicu_baby_ids = set(valid_summary_dic['baby_id'][~nicu_mask])

    tprint(f"✓ {len(nicu_baby_ids)} NICU babies in valid summary dic")
    tprint(f"✓ {len(non_nicu_baby_ids)} Non-NICU babies in valid summary dic\n")    
    
    all_mat_files = sorted(VITALS_FILES)
    mat_file_as_baby_ids = set(f.split('/')[-1] for f in all_mat_files)
    
    nicu_files = [f for f in all_mat_files if f.split('/')[-1] in nicu_baby_ids]
    non_nicu_files = [f for f in all_mat_files if f.split('/')[-1] in non_nicu_baby_ids]

    tprint(f"⚠ {len(all_mat_files) - (len(nicu_files) + len(non_nicu_files))} .mat files are not in the valid summary dic but are in vitals directory")
    tprint(f"✓ {len(all_mat_files)} total .mat files found in vitals directory")
    tprint(f"✓ {len(nicu_files)} NICU files")
    tprint(f"✓ {len(non_nicu_files)} Non-NICU files\n")


    # In valid summary but NOT in vitals dir  ← this is what you asked for
    all_summary_ids = nicu_baby_ids | non_nicu_baby_ids
    tprint(len(all_summary_ids))
    tprint(len(mat_file_as_baby_ids))
    in_summary_not_local = all_summary_ids - mat_file_as_baby_ids
    tprint(f"⚠ {len(in_summary_not_local)} babies are in valid summary dic but have no local .mat file")





    # --- Step 1: window metadata ---
    if os.path.exists(WINDOW_METADATA_PATH):
        metadata_df = pd.read_pickle(WINDOW_METADATA_PATH)
        tprint(f"✓ Loaded metadata: {len(metadata_df)} windows\n")
    else:
        metadata_df = generate_all_window_metadata(nicu_files, non_nicu_files)
        metadata_df.to_pickle(WINDOW_METADATA_PATH)
        tprint(f"✓ Saved: {WINDOW_METADATA_PATH}\n")




    # --- Step 2: sample windows + assign labels ---
    if os.path.exists(WINDOWS_TO_EMBED_PATH):
        unique_windows_df = pd.read_pickle(WINDOWS_TO_EMBED_PATH)
        tprint(f"✓ Loaded sampled windows: {len(unique_windows_df)}\n")
    else:

        # all_labelled_df contains the sampled window metadata with labels for all thresholds
        # unique_windows_df contains the sampled window metadata with no labels (for embedding)
        all_labeled_df, unique_windows_df = sample_and_label_all_thresholds(metadata_df)
        unique_windows_df.to_pickle(WINDOWS_TO_EMBED_PATH)
        tprint(f"✓ Saved: {WINDOWS_TO_EMBED_PATH}")

        all_labeled_df.to_pickle(SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH)
        tprint(f"✓ Saved: {SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH}")



    # --- Step 3: embed unique windows (incremental cache) ---
    embeddings_cache = ensure_embeddings_for_windows(unique_windows_df, binning_mode)


    # --- Steps 4 + 5: for each threshold, prepare data and train classifier ---
    for threshold_days in THRESHOLDS_DAYS:

        embeddings, labels, baby_ids = prepare_training_data_for_threshold(
            threshold_days,
            SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH,
            binning_mode,
            embeddings_cache,
        )
        train_and_save_results(
            threshold_days,
            embeddings,
            labels,
            baby_ids=baby_ids,
            use_focal_loss=use_focal_loss,
            gamma=gamma,
            loss_name=loss_name,
            binning_mode=binning_mode,
        )

    tprint("\n" + "=" * 100)
    tprint(" " * 30 + "✓ PIPELINE COMPLETE")
    tprint("=" * 100 + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NICU mortality prediction pipeline")
    parser.add_argument(
        "--binning",
        required=True,
        choices=BINNING_CHOICES,
        help="seasonal: full 80h, 300s bins; fine_grained: last 2h of the same sample, 2s bins",
    )
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument(
        "--ce",
        action="store_true",
        help="Use cross-entropy instead of focal loss",
    )
    args = parser.parse_args()

    run_complete_pipeline(
        use_focal_loss=not args.ce,
        gamma=args.gamma,
        binning_mode=args.binning,
    )
