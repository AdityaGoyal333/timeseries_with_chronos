"""CLI orchestrator: `train` (Steps 1–3) or `classify` (Steps 4–5).

Pipeline role
-------------
Wires window_metadata → sampling → Chronos caches → label join → MLP.
Does not implement those steps; it chooses which ones to run.

Commands
--------
  train --mode seasonal|fine_grained
      Metadata + sampling if missing, then embeddings for that mode only.
  classify --head singular --mode …
      Join Step 2 labels to one cache, train MLP, write labels/results/curves.
  classify --head concat
      Inner-join both caches (seasonal ∥ FG). --mode is unused.

Outputs (via callees; cwd-relative except data/ pickles)
--------------------------------------------------------
  data/window_metadata.pkl, windows_to_embed.pkl, sampled_windows_all_thresholds.pkl
  data/embeddings_global[_fine_grained].pkl
  labels_{N}d[_fine_grained|_concat].pkl
  results_{N}d_{focal|ce}[_fine_grained|_concat].pkl
  training_curves_{N}d_{focal|ce}[…]
"""

import argparse
import os
import pickle as pl
from pathlib import Path

import pandas as pd

from sampling import sample_and_label_all_thresholds
from train_classifier import train_and_save_results
from training_data import (
    ensure_embeddings_for_windows,
    output_suffix,
    prepare_concat_training_data_for_threshold,
    prepare_training_data_for_threshold,
)
from window_metadata import generate_all_window_metadata
from constants import (
    BINNING_CHOICES,
    HEAD_CHOICES,
    HEAD_CONCAT,
    HEAD_SINGULAR,
    THRESHOLDS_DAYS,
)
from log_utils import tprint


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()

VALID_SUMMARY_PATH = (DATA_DIR / "valid_summary_dic.pkl").resolve()
VITALS_DIR = (DATA_DIR / "download" / "Vitals").resolve()

VITALS_FILES = list(VITALS_DIR.glob("*.mat"))
VITALS_FILES = [str(f) for f in VITALS_FILES]

WINDOW_METADATA_PATH = (DATA_DIR / 'window_metadata.pkl').resolve()
WINDOWS_TO_EMBED_PATH = (DATA_DIR / 'windows_to_embed.pkl').resolve()
SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH = (DATA_DIR / 'sampled_windows_all_thresholds.pkl').resolve()


def _discover_mat_files():
    """Partition local .mat paths into NICU / non-NICU using valid_summary_dic.

    Used only when Step 1 metadata is missing. Filenames must match baby_id
    keys in the summary; files in only one source are logged, not invented.
    """
    with open(VALID_SUMMARY_PATH, 'rb') as f:
        valid_summary_dic = pl.load(f)

    # indicator_vector == 1 is the NICU cohort used for overlap/sampling rates.
    nicu_mask = valid_summary_dic['indicator_vector'] == 1
    nicu_baby_ids = set(valid_summary_dic['baby_id'][nicu_mask])
    non_nicu_baby_ids = set(valid_summary_dic['baby_id'][~nicu_mask])

    tprint(f"✓ {len(nicu_baby_ids)} NICU babies in valid summary dic")
    tprint(f"✓ {len(non_nicu_baby_ids)} Non-NICU babies in valid summary dic\n")

    all_mat_files = sorted(VITALS_FILES)
    mat_file_as_baby_ids = set(f.split('/')[-1] for f in all_mat_files)

    # Match on basename so DATA_DIR layout does not affect cohort assignment.
    nicu_files = [f for f in all_mat_files if f.split('/')[-1] in nicu_baby_ids]
    non_nicu_files = [f for f in all_mat_files if f.split('/')[-1] in non_nicu_baby_ids]

    tprint(f"⚠ {len(all_mat_files) - (len(nicu_files) + len(non_nicu_files))} .mat files are not in the valid summary dic but are in vitals directory")
    tprint(f"✓ {len(all_mat_files)} total .mat files found in vitals directory")
    tprint(f"✓ {len(nicu_files)} NICU files")
    tprint(f"✓ {len(non_nicu_files)} Non-NICU files\n")

    all_summary_ids = nicu_baby_ids | non_nicu_baby_ids
    in_summary_not_local = all_summary_ids - mat_file_as_baby_ids
    tprint(f"⚠ {len(in_summary_not_local)} babies are in valid summary dic but have no local .mat file")
    return nicu_files, non_nicu_files


def _ensure_steps_1_and_2():
    """Load or create shared 80h metadata and labeled samples.

    Mode-independent: seasonal and fine-grained reuse the same windows and
    labels. Skip is keyed on windows_to_embed.pkl (existing contract), not
    on sampled_windows_all_thresholds.pkl.
    """
    # Step 1: candidate 80h windows per baby.
    if os.path.exists(WINDOW_METADATA_PATH):
        metadata_df = pd.read_pickle(WINDOW_METADATA_PATH)
        tprint(f"✓ Loaded metadata: {len(metadata_df)} windows\n")
    else:
        nicu_files, non_nicu_files = _discover_mat_files()
        metadata_df = generate_all_window_metadata(nicu_files, non_nicu_files)
        metadata_df.to_pickle(WINDOW_METADATA_PATH)
        tprint(f"✓ Saved: {WINDOW_METADATA_PATH}\n")

    # Step 2: downsample + write unique windows and the full labeled table.
    if os.path.exists(WINDOWS_TO_EMBED_PATH):
        unique_windows_df = pd.read_pickle(WINDOWS_TO_EMBED_PATH)
        tprint(f"✓ Loaded sampled windows: {len(unique_windows_df)}\n")
    else:
        all_labeled_df, unique_windows_df = sample_and_label_all_thresholds(metadata_df)
        unique_windows_df.to_pickle(WINDOWS_TO_EMBED_PATH)
        tprint(f"✓ Saved: {WINDOWS_TO_EMBED_PATH}")
        all_labeled_df.to_pickle(SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH)
        tprint(f"✓ Saved: {SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH}")

    return unique_windows_df


def run_train(binning_mode):
    """Steps 1–3 only: shared windows/labels, then embeddings for one mode.

    Does not join labels or train the MLP. Writes that mode's cache and
    missing-windows pickle.
    """
    tprint("\n" + "=" * 100)
    tprint(f"  TRAIN  mode={binning_mode}")
    tprint("=" * 100 + "\n")

    unique_windows_df = _ensure_steps_1_and_2()
    # Step 3 fills only keys absent from this mode's cache.
    ensure_embeddings_for_windows(unique_windows_df, binning_mode)

    tprint("\n" + "=" * 100)
    tprint(" " * 30 + "✓ TRAIN COMPLETE")
    tprint("=" * 100 + "\n")


def run_classify(head, binning_mode, use_focal_loss, gamma):
    """Steps 4–5: join Step 2 labels to cache(s), then k-fold MLP.

    Does not call Chronos. Labels always come from sampled_windows_all_thresholds.
    Concat ignores binning_mode; singular requires it (enforced in __main__).
    """
    if not os.path.exists(SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH):
        raise FileNotFoundError(
            f"Missing {SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH}. Run train first."
        )

    loss_name = f"Focal Loss (γ={gamma})" if use_focal_loss else "Cross-Entropy"
    suffix = output_suffix(binning_mode, head)

    tprint("\n" + "=" * 100)
    tprint(f"  CLASSIFY  head={head}  mode={binning_mode or 'n/a'}  {loss_name}")
    tprint("=" * 100 + "\n")

    for threshold_days in THRESHOLDS_DAYS:
        # Step 4: same labeled rows; feature vector is one cache or seasonal∥FG.
        if head == HEAD_CONCAT:
            embeddings, labels, baby_ids = prepare_concat_training_data_for_threshold(
                threshold_days,
                SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH,
            )
        else:
            embeddings, labels, baby_ids = prepare_training_data_for_threshold(
                threshold_days,
                SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH,
                binning_mode,
            )
        # Step 5: write labels_* already done in prepare_*; this trains and dumps results.
        train_and_save_results(
            threshold_days,
            embeddings,
            labels,
            baby_ids=baby_ids,
            use_focal_loss=use_focal_loss,
            gamma=gamma,
            loss_name=loss_name,
            binning_mode=binning_mode or 'seasonal',
            output_suffix=suffix,
        )

    tprint("\n" + "=" * 100)
    tprint(" " * 30 + "✓ CLASSIFY COMPLETE")
    tprint("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NICU mortality prediction pipeline")
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Steps 1–3: metadata, labels, embeddings')
    p_train.add_argument('--mode', required=True, choices=BINNING_CHOICES)

    p_cls = sub.add_parser('classify', help='Steps 4–5: join labels to cache(s), MLP')
    p_cls.add_argument('--head', required=True, choices=HEAD_CHOICES)
    p_cls.add_argument('--mode', choices=BINNING_CHOICES, default=None)
    p_cls.add_argument('--gamma', type=float, default=2.0)
    p_cls.add_argument('--ce', action='store_true', help='Use cross-entropy instead of focal loss')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.mode)
    else:
        # Concat does not use --mode; singular must pick which cache to join.
        if args.head == HEAD_SINGULAR and args.mode is None:
            parser.error('classify --head singular requires --mode')
        run_classify(
            head=args.head,
            binning_mode=args.mode,
            use_focal_loss=not args.ce,
            gamma=args.gamma,
        )
