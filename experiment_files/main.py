"""CLI orchestrator: `train` (Steps 1–3) or `classify` (Steps 4–5).

Pipeline role
-------------
Wires window_metadata → sampling → Chronos caches → label join → MLP.
Does not implement those steps; it chooses which ones to run.

Commands
--------
  train --source labeled|site_1|site_2|site_3|site_4 --mode seasonal|fine_grained
      Metadata + sampling if missing, then embeddings for that mode only.
      site_* is unlabeled (one dataset per site). Death-proximity labels are
      0 (survivors) or NaN (died); classify fills NaN later.
  classify --head singular --mode …
      Join Step 2 labels to cache(s), train MLP. Default train+eval = labeled.
      --sites site_i … adds unlabeled to train. --no-labeled requires --sites
      (eval stays labeled). Concat ignores --mode.

Outputs (via callees; paths from repo root via __file__, not cwd)
--------------------------------------------------------
  data/labeled/window_metadata.pkl, windows_to_embed.pkl, sampled_windows_all_thresholds.pkl
  data/labeled/embeddings_global[_fine_grained].pkl
  data/unlabeled/site_i/window_metadata.pkl, windows_to_embed.pkl, sampled_windows_all_thresholds.pkl
  data/unlabeled/site_i/embeddings_global[_fine_grained].pkl
  results/labels_{N}d[_fine_grained|_concat].pkl
  results/results_{N}d_{focal|ce}[_fine_grained|_concat][_lab|_unlab_site_i].pkl
  results/training_curves_{N}d_{focal|ce}[_fine_grained|_concat][_lab|_unlab_site_i]
"""

import argparse
import os
import pickle as pl
from pathlib import Path

import pandas as pd

from labelled.sampling import sample_and_label_all_thresholds
from labelled.training_data import (
    ensure_embeddings_for_windows,
    output_suffix,
    prepare_concat_training_data_for_threshold,
    prepare_training_data_for_threshold,
)
from labelled.window_metadata import generate_all_window_metadata
from unlabelled.sampling_unlabeled import sample_and_label_all_thresholds as sample_and_label_unlabeled
from unlabelled.training_unlabeled import ensure_embeddings_for_windows as ensure_embeddings_unlabeled
from unlabelled.window_metadata_unlabeled import generate_all_window_metadata as generate_all_window_metadata_unlabeled
from train_classifier import train_and_save_results
from constants import (
    BINNING_CHOICES,
    HEAD_CHOICES,
    HEAD_CONCAT,
    HEAD_SINGULAR,
    LABELED_DIR,
    OUTCOMES_PATH,
    SITE_CHOICES,
    SOURCE_CHOICES,
    THRESHOLDS_DAYS,
    classify_mix_suffix,
    unlabeled_pickle_dir,
    unlabeled_vitals_dir,
)
from log_utils import tprint


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / "data").resolve()

VALID_SUMMARY_PATH = (DATA_DIR / "valid_summary_dic.pkl").resolve()
VITALS_DIR = (DATA_DIR / "download" / "Vitals").resolve()

VITALS_FILES = list(VITALS_DIR.glob("*.mat"))
VITALS_FILES = [str(f) for f in VITALS_FILES]

WINDOW_METADATA_PATH = (LABELED_DIR / 'window_metadata.pkl').resolve()
WINDOWS_TO_EMBED_PATH = (LABELED_DIR / 'windows_to_embed.pkl').resolve()
SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH = (LABELED_DIR / 'sampled_windows_all_thresholds.pkl').resolve()


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


def _discover_parquet_files(site):
    """Partition one site's parquet paths using outcomes.parquet died_yn.

    Join is int(stem) == subject_number_deid (EDA). NICU = died_yn == 'Yes'.
    """
    if not OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing {OUTCOMES_PATH}")

    outcomes = pd.read_parquet(OUTCOMES_PATH, columns=['subject_number_deid', 'died_yn'])
    died_map = dict(
        zip(outcomes['subject_number_deid'].astype(int), outcomes['died_yn'])
    )

    vitals_dir = unlabeled_vitals_dir(site)
    all_parquet = sorted(vitals_dir.glob('*.parquet'))
    nicu_files = []
    non_nicu_files = []
    unmatched = 0
    bad_stem = 0

    for path in all_parquet:
        try:
            sid = int(path.stem)
        except ValueError:
            bad_stem += 1
            continue
        if sid not in died_map:
            unmatched += 1
            continue
        if died_map[sid] == 'Yes':
            nicu_files.append(str(path))
        else:
            non_nicu_files.append(str(path))

    tprint(f"✓ {len(died_map)} rows in outcomes.parquet")
    tprint(f"✓ {len(all_parquet)} parquet files in {vitals_dir}")
    tprint(f"✓ {len(nicu_files)} NICU (died_yn == Yes)")
    tprint(f"✓ {len(non_nicu_files)} Non-NICU (died_yn != Yes)")
    tprint(f"⚠ {unmatched} parquet stems not in outcomes")
    tprint(f"⚠ {bad_stem} parquet files with non-integer stems\n")
    return nicu_files, non_nicu_files


def _ensure_steps_1_and_2():
    """Load or create shared 80h metadata and labeled samples.

    Mode-independent: seasonal and fine-grained reuse the same windows and
    labels. Skip is keyed on windows_to_embed.pkl (existing contract), not
    on sampled_windows_all_thresholds.pkl.
    """
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

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


def _ensure_unlabeled_steps_1_and_2(site):
    """Load or create unlabeled 80h metadata, unique windows, and the 0/NaN roster.

    Skip is keyed on data/unlabeled/site_i/windows_to_embed.pkl.
    """
    pickle_dir = unlabeled_pickle_dir(site)
    pickle_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = pickle_dir / 'window_metadata.pkl'
    windows_path = pickle_dir / 'windows_to_embed.pkl'
    thresholds_path = pickle_dir / 'sampled_windows_all_thresholds.pkl'

    if os.path.exists(metadata_path):
        metadata_df = pd.read_pickle(metadata_path)
        tprint(f"✓ Loaded unlabeled metadata ({site}): {len(metadata_df)} windows\n")
    else:
        nicu_files, non_nicu_files = _discover_parquet_files(site)
        metadata_df = generate_all_window_metadata_unlabeled(nicu_files, non_nicu_files)
        metadata_df.to_pickle(metadata_path)
        tprint(f"✓ Saved: {metadata_path}\n")

    if os.path.exists(windows_path):
        unique_windows_df = pd.read_pickle(windows_path)
        tprint(f"✓ Loaded unlabeled sampled windows ({site}): {len(unique_windows_df)}\n")
    else:
        all_labeled_df, unique_windows_df = sample_and_label_unlabeled(metadata_df)
        unique_windows_df.to_pickle(windows_path)
        tprint(f"✓ Saved: {windows_path}")
        all_labeled_df.to_pickle(thresholds_path)
        tprint(f"✓ Saved: {thresholds_path}")

    return unique_windows_df


def run_train(source, binning_mode):
    """Steps 1–3: shared windows, then embeddings for one mode.

    Does not join labels or train the MLP. Writes that mode's cache and
    missing-windows pickle under data/labeled or data/unlabeled/site_i.
    """
    tprint("\n" + "=" * 100)
    tprint(f"  TRAIN  source={source}  mode={binning_mode}")
    tprint("=" * 100 + "\n")

    if source in SITE_CHOICES:
        unique_windows_df = _ensure_unlabeled_steps_1_and_2(source)
        ensure_embeddings_unlabeled(unique_windows_df, binning_mode, source)
    else:
        unique_windows_df = _ensure_steps_1_and_2()
        ensure_embeddings_for_windows(unique_windows_df, binning_mode)

    tprint("\n" + "=" * 100)
    tprint(" " * 30 + "✓ TRAIN COMPLETE")
    tprint("=" * 100 + "\n")


def _prepare_threshold(head, binning_mode, threshold_days, roster_path, pickle_dir=None, write_labels=True):
    """Join one source's roster to that source's cache(s)."""
    if head == HEAD_CONCAT:
        return prepare_concat_training_data_for_threshold(
            threshold_days, roster_path, pickle_dir=pickle_dir, write_labels=write_labels
        )
    return prepare_training_data_for_threshold(
        threshold_days,
        roster_path,
        binning_mode,
        pickle_dir=pickle_dir,
        write_labels=write_labels,
    )


def run_classify(head, binning_mode, use_focal_loss, gamma, include_labeled=True, sites=None):
    """Steps 4–5: join Step 2 labels to cache(s), then MLP.

    Eval is always labeled. Train is labeled and/or unlabeled sites.
    Does not call Chronos. Concat ignores binning_mode; singular requires it.
    """
    sites = list(sites or [])
    if not os.path.exists(SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH):
        raise FileNotFoundError(
            f"Missing {SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH}. Run train --source labeled first."
        )
    for site in sites:
        roster = unlabeled_pickle_dir(site) / 'sampled_windows_all_thresholds.pkl'
        if not roster.exists():
            raise FileNotFoundError(f"Missing {roster}. Run train --source {site} first.")

    loss_name = f"Focal Loss (γ={gamma})" if use_focal_loss else "Cross-Entropy"
    suffix = output_suffix(binning_mode, head) + classify_mix_suffix(include_labeled, sites)

    tprint("\n" + "=" * 100)
    tprint(
        f"  CLASSIFY  head={head}  mode={binning_mode or 'n/a'}  {loss_name}  "
        f"labeled_train={include_labeled}  sites={sites or 'none'}"
    )
    tprint("=" * 100 + "\n")

    for threshold_days in THRESHOLDS_DAYS:
        lab_emb, lab_y, lab_baby, _, _ = _prepare_threshold(
            head, binning_mode, threshold_days, SAMPLED_WINDOWS_ALL_THRESHOLDS_PATH,
            pickle_dir=None, write_labels=True,
        )

        un_emb, un_y, un_we, un_dt = [], [], [], []
        for site in sites:
            roster = unlabeled_pickle_dir(site) / 'sampled_windows_all_thresholds.pkl'
            e, y, _b, we, dt = _prepare_threshold(
                head, binning_mode, threshold_days, roster,
                pickle_dir=unlabeled_pickle_dir(site), write_labels=False,
            )
            un_emb.extend(e)
            un_y.extend(y)
            un_we.extend(we)
            un_dt.extend(dt)

        train_and_save_results(
            threshold_days,
            lab_emb,
            lab_y,
            baby_ids=lab_baby,
            use_focal_loss=use_focal_loss,
            gamma=gamma,
            loss_name=loss_name,
            binning_mode=binning_mode or 'seasonal',
            output_suffix=suffix,
            unlabeled={
                'embeddings': un_emb,
                'labels': un_y,
                'window_end': un_we,
                'death_time': un_dt,
            } if un_emb else None,
            include_labeled=include_labeled,
        )

    tprint("\n" + "=" * 100)
    tprint(" " * 30 + "✓ CLASSIFY COMPLETE")
    tprint("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NICU mortality prediction pipeline")
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Steps 1–3: metadata, labels, embeddings')
    p_train.add_argument('--source', required=True, choices=SOURCE_CHOICES)
    p_train.add_argument('--mode', required=True, choices=BINNING_CHOICES)

    p_cls = sub.add_parser('classify', help='Steps 4–5: join labels to cache(s), MLP')
    p_cls.add_argument('--head', required=True, choices=HEAD_CHOICES)
    p_cls.add_argument('--mode', choices=BINNING_CHOICES, default=None)
    p_cls.add_argument('--gamma', type=float, default=2.0)
    p_cls.add_argument('--ce', action='store_true', help='Use cross-entropy instead of focal loss')
    p_cls.add_argument(
        '--sites', nargs='*', default=[], choices=SITE_CHOICES,
        help='Unlabeled sites to add to training (eval stays labeled)',
    )
    p_cls.add_argument(
        '--no-labeled', action='store_true',
        help='Omit labeled data from training (requires --sites; eval stays labeled)',
    )

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args.source, args.mode)
    else:
        # Concat does not use --mode; singular must pick which cache to join.
        if args.head == HEAD_SINGULAR and args.mode is None:
            parser.error('classify --head singular requires --mode')
        include_labeled = not args.no_labeled
        sites = args.sites or []
        if not include_labeled and not sites:
            parser.error('classify --no-labeled requires --sites')
        run_classify(
            head=args.head,
            binning_mode=args.mode,
            use_focal_loss=not args.ce,
            gamma=args.gamma,
            include_labeled=include_labeled,
            sites=sites,
        )
