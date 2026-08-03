"""Step 5: Train binary classifier on Chronos embeddings with k-fold CV."""

import os
import pickle as pl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss

import math
from torch.utils.data import Sampler


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_matrix, labels):
        self.embeddings = torch.FloatTensor(embedding_matrix)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def _output_bias_from_prevalence(positive_ratio, eps=1e-6):
    """Initialize output bias as log(p / (1 - p)), p = positives / total."""
    p = float(np.clip(positive_ratio, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


class Classifier(nn.Module):
    """MLP head: embedding dim → 512 → 256 → 1 logit."""

    def __init__(self, dim, hidden=512, positive_ratio=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        if positive_ratio is not None:
            with torch.no_grad():
                self.net[-1].bias.fill_(_output_bias_from_prevalence(positive_ratio))

    def forward(self, x):
        # squeeze(-1) keeps batch dim when batch_size == 1
        return self.net(x.view(x.size(0), -1)).squeeze(-1)



# we are randomly shuffling the positive indices and the negative indices 
# we are then use np.array_split for as equal splitting as possible 
# then we add the required negatives and shuffle within the batch as well 
class UniformClassBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size

        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]

        self.num_samples = len(self.labels)
        self.num_batches = math.ceil(
            self.num_samples / self.batch_size
        )

    def __iter__(self):
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()

        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        # Divide positives as uniformly as possible among batches.
        positive_batches = list(
            np.array_split(positive_indices, self.num_batches)
        )

        np.random.shuffle(positive_batches)

        negative_pointer = 0

        for positive_batch in positive_batches:
            batch = positive_batch.tolist()

            negatives_needed = self.batch_size - len(batch)

            batch.extend(
                negative_indices[
                    negative_pointer:
                    negative_pointer + negatives_needed
                ].tolist()
            )

            negative_pointer += negatives_needed

            np.random.shuffle(batch)

            if batch:
                yield batch

    def __len__(self):
        return self.num_batches




def _focal_alpha(labels):
    """torchvision focal loss: positives get alpha, negatives get (1 - alpha).

    Set alpha = fraction of negatives so the rare positive class is up-weighted.
    """
    return 0.95


def _batch_loss(logits, labels, use_focal_loss, alpha, gamma):
    if use_focal_loss:
        return sigmoid_focal_loss(
            logits, labels, alpha=alpha, gamma=gamma, reduction='mean'
        )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    use_focal_loss=False,
    alpha=None,
    gamma=None,
    interval_losses=None,
):
    """Train if optimizer is given, else evaluate loss only."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0

    interval_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch_idx, (embs, labels) in enumerate(loader, start=1):
            embs, labels = embs.to(device), labels.to(device)
            batch_size = labels.size(0)
            logits = model(embs)
            loss = _batch_loss(logits, labels, use_focal_loss, alpha, gamma)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # batch loss is mean over samples; multiply back to sum, then divide by total samples
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if training and interval_losses is not None:
                interval_loss += loss.item()

                if batch_idx % 5 == 0:
                    interval_losses.append(interval_loss / 5)
                    interval_loss = 0.0

    remaining_batches = len(loader) % 5

    if training and interval_losses is not None and remaining_batches > 0:
        interval_losses.append(interval_loss / remaining_batches)

    return total_loss / max(total_samples, 1)


def _evaluate(model, loader, device):
    """Return scalar metrics plus raw labels/probs for curve plotting."""
    model.eval()
    labels_all, probs_all = [], []

    with torch.no_grad():
        for embs, labels in loader:
            embs = embs.to(device)
            logits = model(embs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            labels_all.extend(labels.numpy().ravel())
            probs_all.extend(probs)

    labels_all = np.asarray(labels_all, dtype=int)
    probs_all = np.asarray(probs_all, dtype=float)
    preds = (probs_all >= 0.5).astype(int)

    has_both_classes = len(np.unique(labels_all)) > 1
    return {
        'accuracy': accuracy_score(labels_all, preds),
        'precision': precision_score(labels_all, preds, zero_division=0),
        'recall': recall_score(labels_all, preds, zero_division=0),
        'f1': f1_score(labels_all, preds, zero_division=0),
        'auc': roc_auc_score(labels_all, probs_all) if has_both_classes else float('nan'),
        'average_precision': average_precision_score(labels_all, probs_all) if has_both_classes else float('nan'),
        'prevalence': labels_all.mean(),
        'labels': labels_all,
        'probs': probs_all,
    }


def _save_loss_curve(train_losses, val_losses, val_epochs, path):
    train_epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_losses, label='Train (every epoch)')
    plt.plot(val_epochs, val_losses, label='Val (every 5 epochs)', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_interval_loss_curve(interval_losses, path):
    intervals = np.arange(1, len(interval_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(intervals, interval_losses)
    plt.xlabel('5-batch interval')
    plt.ylabel('Average loss')
    plt.title('Training loss averaged every 5 batches')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_cumulative_loss_curve(fold_curve_data, output_dir, k):
    """Per-fold curves (faint) + mean ± std (bold), on separate train and val plots."""
    train_by_fold = np.array([fd['train_losses'] for fd in fold_curve_data], dtype=float)
    val_by_fold = np.array([fd['val_losses'] for fd in fold_curve_data], dtype=float)
    val_epochs = np.asarray(fold_curve_data[0]['val_epochs'], dtype=int)

    n_epochs = train_by_fold.shape[1]
    for fd in fold_curve_data:
        if len(fd['train_losses']) != n_epochs:
            raise ValueError('All folds must have the same number of train loss points.')
        if list(fd['val_epochs']) != list(val_epochs):
            raise ValueError('All folds must use the same val epoch schedule.')

    train_epochs = np.arange(1, n_epochs + 1)
    mean_train = train_by_fold.mean(axis=0)
    std_train = train_by_fold.std(axis=0)
    mean_val = val_by_fold.mean(axis=0)
    std_val = val_by_fold.std(axis=0)

    n_folds = train_by_fold.shape[0]
    cmap = plt.cm.tab10 if n_folds <= 10 else plt.cm.tab20
    fold_colors = [cmap(i % cmap.N) for i in range(n_folds)]
    fold_alpha = 0.65
    mean_color = 'black'

    def _legend_kwargs():
        return {'fontsize': 8, 'ncol': 2, 'loc': 'upper right'}

    # --- Train ---
    plt.figure(figsize=(9, 5))
    for i in range(n_folds):
        plt.plot(
            train_epochs,
            train_by_fold[i],
            color=fold_colors[i],
            alpha=fold_alpha,
            linewidth=1.2,
            label=f'Fold {i + 1}',
        )
    plt.fill_between(
        train_epochs,
        mean_train - std_train,
        mean_train + std_train,
        color=mean_color,
        alpha=0.15,
        zorder=5,
    )
    plt.plot(
        train_epochs,
        mean_train,
        color=mean_color,
        linewidth=2.5,
        zorder=6,
        label='Mean ± std',
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Cumulative train loss ({k}-fold CV)')
    plt.legend(**_legend_kwargs())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    train_path = os.path.join(output_dir, 'Loss curve train.png')
    plt.savefig(train_path, dpi=150)
    plt.close()

    # --- Val (every 5 epochs) ---
    plt.figure(figsize=(9, 5))
    for i in range(n_folds):
        plt.plot(
            val_epochs,
            val_by_fold[i],
            color=fold_colors[i],
            alpha=fold_alpha,
            linewidth=1.2,
            marker='o',
            markersize=3,
            label=f'Fold {i + 1}',
        )
    plt.fill_between(
        val_epochs,
        mean_val - std_val,
        mean_val + std_val,
        color=mean_color,
        alpha=0.15,
        zorder=5,
    )
    plt.plot(
        val_epochs,
        mean_val,
        color=mean_color,
        linewidth=2.5,
        marker='o',
        markersize=5,
        zorder=6,
        label='Mean ± std',
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Cumulative val loss ({k}-fold CV, every 5 epochs)')
    plt.legend(**_legend_kwargs())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    val_path = os.path.join(output_dir, 'Loss curve val.png')
    plt.savefig(val_path, dpi=150)
    plt.close()

    return {'train_path': train_path, 'val_path': val_path}


def _compute_pr_curve(labels, probs):
    """Return precision/recall arrays and AP for one validation fold."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    prevalence = float(labels.mean())

    if len(np.unique(labels)) < 2:
        return {
            'precision': np.array([prevalence, prevalence]),
            'recall': np.array([0.0, 1.0]),
            'average_precision': prevalence,
            'prevalence': prevalence,
        }

    precision, recall, _ = precision_recall_curve(labels, probs)
    return {
        'precision': precision,
        'recall': recall,
        'average_precision': float(average_precision_score(labels, probs)),
        'prevalence': prevalence,
        'length': len(labels)
    }


def _plot_pr_curve(pr_data, path):
    """Save a per-fold PR curve PNG from precomputed curve data."""
    precision, recall = pr_data['precision'], pr_data['recall']
    ap, prevalence = pr_data['average_precision'], pr_data['prevalence']

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'Model (AP = {ap:.3f})')
    plt.axhline(prevalence, color='gray', linestyle='--', label=f'Baseline(prevalence) ({prevalence:.3f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall curve')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _compute_roc_curve(labels, probs):
    """Return FPR/TPR arrays and AUC for one validation fold."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if len(np.unique(labels)) < 2:
        return {'fpr': np.array([]), 'tpr': np.array([]), 'auc': float('nan')}

    fpr, tpr, _ = roc_curve(labels, probs)
    return {
        'fpr': fpr, 
        'tpr': tpr, 
        'auc': float(auc(fpr, tpr))
    }


def _plot_roc_curve(roc_data, path):
    """Save a per-fold ROC curve PNG from precomputed curve data."""
    fpr, tpr = roc_data['fpr'], roc_data['tpr']
    roc_auc = roc_data['auc']

    plt.figure(figsize=(6, 5))
    if len(fpr) == 0:
        plt.text(0.5, 0.5, 'ROC undefined\n(only one class in val fold)', ha='center', va='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(path, dpi=150)
        plt.close()
        return

    plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline (0.5)')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# Same threshold as _evaluate() uses for accuracy, precision, recall, and F1.
DECISION_THRESHOLD = 0.5


def _predictions_from_probs(probs, threshold=DECISION_THRESHOLD):
    """Turn model probabilities into hard 0/1 predictions."""
    probs = np.asarray(probs, dtype=float)
    return (probs >= threshold).astype(int)


def _confusion_matrix_counts(y_true, y_pred):
    """2x2 counts: rows = true class, columns = predicted class (labels 0 then 1).

    Layout:
        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def _save_confusion_matrix_plot(counts, path, title):
    """Save a heatmap PNG for one confusion matrix (integer counts in each cell)."""
    counts = np.asarray(counts, dtype=int)
    class_names = ['0 (negative)', '1 (positive)']

    plt.figure(figsize=(6, 5))
    plt.imshow(counts, cmap='Blues')
    plt.colorbar(label='Count')

    # Write the count inside each cell.
    for row in range(2):
        for col in range(2):
            plt.text(
                col,
                row,
                str(counts[row, col]),
                ha='center',
                va='center',
                color='black' if counts[row, col] < counts.max() / 2 else 'white',
                fontsize=14,
            )

    plt.xticks([0, 1], class_names, rotation=15, ha='right')
    plt.yticks([0, 1], class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_cumulative_confusion_matrix(fold_confusion_matrices, output_dir, k):
    """Sum per-fold confusion matrices and save one pooled (out-of-fold) plot."""
    # Each fold's val set is disjoint, so summing counts = one CM over all data.
    cumulative = np.sum(np.asarray(fold_confusion_matrices, dtype=int), axis=0)
    path = os.path.join(output_dir, 'Confusion matrix.png')
    _save_confusion_matrix_plot(
        cumulative,
        path,
        title=f'Cumulative confusion matrix ({k}-fold CV, threshold={DECISION_THRESHOLD})',
    )
    return {'counts': cumulative, 'path': path}


"""
this is with interpolation
"""
# def _save_cumulative_roc(fold_roc_curves, output_dir, k):
#     """Mean ROC across folds with ±1 std band (reuses per-fold FPR/TPR arrays)."""
#     fpr_grid = np.linspace(0, 1, 1000)
#     tpr_interp, aucs = [], []

#     for roc in fold_roc_curves:
#         fpr, tpr = roc['fpr'], roc['tpr']
#         if len(fpr) == 0:
#             continue
#         aucs.append(roc['auc'])
#         tpr_interp.append(np.interp(fpr_grid, fpr, tpr))

#     if not tpr_interp:
#         return None

#     tpr_interp = np.asarray(tpr_interp)
#     mean_tpr = tpr_interp.mean(axis=0)
#     std_tpr = tpr_interp.std(axis=0)
#     mean_auc = float(np.mean(aucs))

#     plt.figure(figsize=(6, 6))
#     plt.plot(fpr_grid, mean_tpr, label=f'ROC Curve (AUC = {mean_auc:.3f})')
#     plt.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
#     plt.plot([0, 1], [0, 1], linestyle='--', color='k', alpha=0.5, linewidth=1)
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title(f'ROC Curve ({k}-fold CV)', fontsize=13)
#     plt.legend()
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.tick_params(axis='both', which='major', labelsize=12)
#     plt.tight_layout()
#     path = os.path.join(output_dir, 'ROC curve interpolation.png')
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()

#     return {
#         'fpr_grid': fpr_grid,
#         'mean_tpr': mean_tpr,
#         'std_tpr': std_tpr,
#         'mean_auc': mean_auc,
#         'fold_aucs': aucs,
#         'path': path,
#     }



"""
this is with mapping and creating bins
"""
def _save_cumulative_roc(fold_roc_curves, output_dir, k):
    """Mean ROC across folds using FPR bins (no interpolation)."""

    n_bins = 60
    fpr_grid = np.linspace(0, 1, n_bins)
    aucs = []

    fold_binned_tprs = []

    # we map the fpr to the bins and then we average all the tpr values for a particular bin for a fold 
    # then we do the second averaging across folds for a particular bin

    for roc in fold_roc_curves:
        fpr = roc["fpr"]
        tpr = roc["tpr"]

        if len(fpr) == 0:
            continue

        aucs.append(roc["auc"])

        # Store TPR values belonging to each FPR bin
        bins = [[] for _ in range(n_bins)]

        # One pass over the ROC points
        for f, t in zip(fpr, tpr):
            idx = min(int(f * (n_bins - 1)), n_bins - 1)
            bins[idx].append(t)

        # Average TPR inside each bin
        binned_tpr = np.full(n_bins, np.nan)

        for i, values in enumerate(bins):
            if values:
                binned_tpr[i] = np.mean(values)

        fold_binned_tprs.append(binned_tpr)

    if not fold_binned_tprs:
        return None

    fold_binned_tprs = np.asarray(fold_binned_tprs)

    # Number of folds contributing to each FPR bin
    counts = np.sum(~np.isnan(fold_binned_tprs), axis=0)

    mean_tpr = np.nanmean(fold_binned_tprs, axis=0)
    std_tpr = np.nanstd(fold_binned_tprs, axis=0)

    # Ignore bins containing values from only one fold
    mean_tpr[counts <= 1] = np.nan
    std_tpr[counts <= 1] = np.nan

    mean_auc = float(np.mean(aucs))

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr_grid,
        mean_tpr,
        label=f"ROC Curve (AUC = {mean_auc:.3f})",
    )

    plt.fill_between(
        fpr_grid,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        alpha=0.2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="k",
        alpha=0.5,
        linewidth=1,
    )

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve ({k}-fold CV)", fontsize=13)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "ROC curve binning.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    # plt.show()

    return {
        "fpr_grid": fpr_grid,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": mean_auc,
        "fold_aucs": aucs,
        "path": path,
    }
    
    
    
"""
this is with everything added 
"""
# def _save_cumulative_roc(fold_roc_curves, output_dir, k):
#     """Mean ROC across folds using FPR bins (no interpolation)."""

#     aucs = np.array([])
#     fold_binned_tprs = np.array([])
#     fpr_grid = np.array([])

#     for roc in fold_roc_curves:
#         fpr = roc["fpr"]
#         tpr = roc["tpr"]

#         if len(fpr) == 0:
#             continue

#         aucs = np.append(aucs,roc["auc"])
#         fold_binned_tprs = np.append(fold_binned_tprs,tpr)
#         fpr_grid = np.append(fpr_grid,fpr)

#     if not fold_binned_tprs.any():
#         return None

#     mean_auc = float(np.mean(aucs))

#     plt.figure(figsize=(6, 6))

#     plt.plot(
#         fpr_grid,
#         fold_binned_tprs,
#         label=f"ROC Curve (AUC = {mean_auc:.3f})",
#     )

#     plt.plot(
#         [0, 1],
#         [0, 1],
#         linestyle="--",
#         color="k",
#         alpha=0.5,
#         linewidth=1,
#     )

#     plt.xlabel("False Positive Rate", fontsize=12)
#     plt.ylabel("True Positive Rate", fontsize=12)
#     plt.title(f"ROC Curve ({k}-fold CV)", fontsize=13)
#     plt.legend(fontsize=11)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.tick_params(axis="both", which="major", labelsize=12)
#     plt.tight_layout()

#     path = os.path.join(output_dir, "ROC curve cumulative probability.png")

#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     # plt.show()

#     return {
#         "fpr_grid": fpr_grid,
#         "mean_auc": mean_auc,
#         "fold_aucs": aucs,
#         "path": path,
#     }
    
    
    
    
    
    

"""
this is with interpolation
"""
# def _save_cumulative_pr(fold_pr_curves, output_dir, k):
#     """Mean PR across folds with ±1 std band (reuses per-fold precision/recall arrays)."""
#     recall_grid = np.linspace(0, 1, 100)
#     precision_interp, average_precisions = [], []

#     sum_length = sum( [fd['length'] for fd in fold_pr_curves])
#     weighted_prevalence = sum([  (fd['prevalence']*fd['length'])/sum_length for fd in fold_pr_curves])

#     for pr in fold_pr_curves:
#         precision, recall = pr['precision'], pr['recall']
#         average_precisions.append(pr['average_precision'])
#         precision_interp.append(np.interp(recall_grid, recall[::-1], precision[::-1]))

#     if not precision_interp:
#         return None

#     precision_interp = np.asarray(precision_interp)
#     mean_precision = precision_interp.mean(axis=0)
#     std_precision = precision_interp.std(axis=0)
#     mean_ap = float(np.mean(average_precisions))

#     plt.figure(figsize=(6, 6))
#     plt.plot(recall_grid, mean_precision, label=f'PR Curve (AUPRC = {mean_ap:.3f})')
#     plt.fill_between(
#         recall_grid,
#         mean_precision - std_precision,
#         mean_precision + std_precision,
#         alpha=0.2,
#     )
#     plt.xlabel('Recall', fontsize=12)
#     plt.ylabel('Precision', fontsize=12)
#     plt.title(f'Precision/Recall Curve ({k}-fold CV)', fontsize=13)
#     plt.axhline(weighted_prevalence, color='gray', linestyle='--', label=f'Baseline(prevalence) ({weighted_prevalence:.3f})')
#     plt.legend(fontsize=11)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.tick_params(axis='both', which='major', labelsize=12)
#     plt.tight_layout()
#     path = os.path.join(output_dir, 'PR curve interpolation.png')
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()

#     return {
#         'recall_grid': recall_grid,
#         'mean_precision': mean_precision,
#         'std_precision': std_precision,
#         'mean_ap': mean_ap,
#         'fold_aps': average_precisions,
#         'path': path,
#     }



"""
this is with mapping and creating bins
"""
def _save_cumulative_pr(fold_pr_curves, output_dir,k):
    """Mean PR across folds using recall bins (no interpolation)."""

    n_bins = 150
    recall_grid = np.linspace(0, 1, n_bins)
    average_precisions = []


    # we map the fpr to the bins and then we average all the tpr values for a particular bin for a fold 
    # then we do the second averaging across folds for a particular bin
    
    

    sum_length = sum( [fd['length'] for fd in fold_pr_curves])
    weighted_prevalence = sum([  (fd['prevalence']*fd['length'])/sum_length for fd in fold_pr_curves])
    
    fold_binned_precisions = []

    for pr in fold_pr_curves:
        precision = pr["precision"]
        recall = pr["recall"]

        average_precisions.append(pr["average_precision"])

        # Store precision values belonging to each recall bin
        bins = [[] for _ in range(n_bins)]

        # One pass over the PR points
        for r, p in zip(recall, precision):
            idx = min(int(r * (n_bins - 1)), n_bins - 1)
            bins[idx].append(p)

        # Average precision inside each bin
        binned_precision = np.full(n_bins, np.nan)
        for i, values in enumerate(bins):
            if values:
                binned_precision[i] = np.mean(values)

        fold_binned_precisions.append(binned_precision)

    if not fold_binned_precisions:
        return None

    fold_binned_precisions = np.asarray(fold_binned_precisions)

    counts = np.sum(~np.isnan(fold_binned_precisions), axis=0)

    mean_precision = np.nanmean(fold_binned_precisions, axis=0)
    std_precision = np.nanstd(fold_binned_precisions, axis=0)

    # Ignore bins with values from only one fold
    mean_precision[counts <= 1] = np.nan
    std_precision[counts <= 1] = np.nan

    mean_ap = float(np.mean(average_precisions))

    plt.figure(figsize=(6, 6))
    plt.plot(recall_grid, mean_precision,
             label=f"PR Curve (AUPRC = {mean_ap:.3f})")
    plt.fill_between(
        recall_grid,
        mean_precision - std_precision,
        mean_precision + std_precision,
        alpha=0.2,
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision/Recall Curve ({k}-fold CV)", fontsize=13)
    plt.axhline(weighted_prevalence, color='gray', linestyle='--', label=f'Baseline(prevalence) ({weighted_prevalence:.3f})')
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "PR curve binning.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    # plt.show()

    return {
        "recall_grid": recall_grid,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_ap": mean_ap,
        "fold_aps": average_precisions,
        "path": path,
    }
    
    
    
    


"""
this is with everything added
"""
# def _save_cumulative_pr(fold_pr_curves, output_dir,k):
#     """Mean PR across folds using recall bins (no interpolation)."""

#     average_precisions =  np.array([])
#     recall_grid = np.array([])
#     fold_binned_precisions =  np.array([])

#     # we map the fpr to the bins and then we average all the tpr values for a particular bin for a fold 
#     # then we do the second averaging across folds for a particular bin

#     sum_length = sum( [fd['length'] for fd in fold_pr_curves])
#     weighted_prevalence = sum([  (fd['prevalence']*fd['length'])/sum_length for fd in fold_pr_curves])
        
#     for pr in fold_pr_curves:
#         precision = pr["precision"]
#         recall = pr["recall"]
        
#         average_precisions = np.append(average_precisions,pr["average_precision"])
#         recall_grid = np.append(recall_grid,recall)
#         fold_binned_precisions = np.append(fold_binned_precisions,precision)

#     if not fold_binned_precisions.any():
#         return None

#     mean_ap = float(np.mean(average_precisions))

#     idx = np.argsort(recall_grid)
#     recall_grid = recall_grid[idx]
#     fold_binned_precisions = fold_binned_precisions[idx]
    
#     plt.figure(figsize=(6, 6))
#     plt.plot(recall_grid, fold_binned_precisions, alpha=0.4,
#              label=f"PR Curve (AUPRC = {mean_ap:.3f})")

#     plt.xlabel("Recall", fontsize=12)
#     plt.ylabel("Precision", fontsize=12)
#     plt.title(f"Precision/Recall Curve ({k}-fold CV)", fontsize=13)
#     plt.axhline(weighted_prevalence, color='gray', linestyle='--', label=f'Baseline(prevalence) ({weighted_prevalence:.3f})')
#     plt.legend(fontsize=11)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.tick_params(axis="both", which="major", labelsize=12)
#     plt.tight_layout()


#     path = os.path.join(output_dir, "PR curve cumulative probability.png")
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     # plt.show()

#     return {
#         "recall_grid": recall_grid,
#         "mean_ap": mean_ap,
#         "fold_aps": average_precisions,
#         "path": path,
#     }
    
    
def _save_fold_probabilities(y_true_folds, y_score_folds, fold_pr_curves, fold_roc_curves, output_dir, k):
    """Persist per-fold validation outputs and precomputed curve arrays."""
    fold_probabilities = {
        'y_true_folds': [np.asarray(y, dtype=int) for y in y_true_folds],
        'y_score_folds': [np.asarray(s, dtype=float) for s in y_score_folds],
        'pr_curves': fold_pr_curves,
        'roc_curves': fold_roc_curves,
        'k': k,
    }
    path = os.path.join(output_dir, 'fold_probabilities.pkl')
    with open(path, 'wb') as f:
        pl.dump(fold_probabilities, f)
    return fold_probabilities, path


def k_fold_cv(
    embeddings,
    labels,
    baby_ids,
    k=5,
    epochs=50,
    batch_size=32,
    lr=0.001,
    device='cpu',
    use_focal_loss=True,
    alpha=None,
    gamma=2.0,
    output_dir=None,
):
    if len(embeddings) == 0:
        print('No embeddings to train on!')
        return None, None, None, None, None, None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    labels = np.asarray(labels, dtype=int)
    embedding_matrix = np.stack([np.asarray(e, dtype=np.float32).flatten() for e in embeddings])
    dim = embedding_matrix.shape[1]

    if use_focal_loss and alpha is None:
        alpha = _focal_alpha(labels)

    loss_name = f'Focal (alpha={alpha:.3f}, gamma={gamma})' if use_focal_loss else 'BCE'
    pos_rate = labels.mean()
    n_babies = len(set(baby_ids)) if baby_ids is not None else None
    split_name = 'StratifiedGroupKFold (by baby_id)' if baby_ids is not None else 'StratifiedKFold (by window)'
    print(f"\n{'=' * 80}")
    print(f'K-fold CV | {loss_name} | {split_name}')
    print(f'Samples: {len(labels)} | Pos rate: {pos_rate:.3f} ({labels.sum()} positives)')
    if n_babies is not None:
        print(f'Babies: {n_babies} | Folds: {k}')
    else:
        print(f'Folds: {k}')
    print(f'Embedding dim: {dim}')
    if output_dir:
        print(f'Curves: {output_dir}')
    print('=' * 80)


    baby_ids = np.asarray(baby_ids)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    split_iter = splitter.split(embedding_matrix, labels, groups=baby_ids)

    fold_metrics, fold_curve_data = [], []
    fold_y_true, fold_y_probs = [], []
    fold_confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if baby_ids is not None:
            train_babies = set(baby_ids[train_idx])
            val_babies = set(baby_ids[val_idx])
            overlap = train_babies & val_babies
            assert not overlap, f"Fold {fold + 1}: {len(overlap)} babies in both train and val"

        print(f'Fold {fold + 1}/{k}...', end=' ')

        train_dataset = EmbeddingDataset(
            embedding_matrix[train_idx],
            labels[train_idx],
        )

        batch_sampler = UniformClassBatchSampler(
            labels=labels[train_idx],
            batch_size=batch_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
        )

        # train_loader = DataLoader(
        #     EmbeddingDataset(embedding_matrix[train_idx], labels[train_idx]),
        #     batch_size=batch_size,
        #     shuffle=True,
        # )

        val_loader = DataLoader(
            EmbeddingDataset(embedding_matrix[val_idx], labels[val_idx]),
            batch_size=batch_size,
        )

        train_pos_rate = labels[train_idx].mean()
        model = Classifier(dim, positive_ratio=train_pos_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses, val_epochs = [], [], []
        train_interval_losses = []

        for epoch in range(1, epochs + 1):
            train_losses.append(
                _run_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer,
                    use_focal_loss,
                    alpha,
                    gamma,
                    train_interval_losses,
                )
            )
            if epoch % 5 == 0:
                val_losses.append(
                    _run_epoch(model, val_loader, device, None, use_focal_loss, alpha, gamma)
                )
                val_epochs.append(epoch)

        # One full val pass after training (final epoch weights). Same labels/probs as metrics.
        val_result = _evaluate(model, val_loader, device)
        fold_y_true.append(val_result['labels'])
        fold_y_probs.append(val_result['probs'])
        metrics = {name: val_result[name] for name in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision', 'prevalence')}
        fold_metrics.append(metrics)

        val_preds = _predictions_from_probs(val_result['probs'])
        fold_cm = _confusion_matrix_counts(val_result['labels'], val_preds)
        fold_confusion_matrices.append(fold_cm)

        pr_curve = _compute_pr_curve(val_result['labels'], val_result['probs'])
        roc_curve_data = _compute_roc_curve(val_result['labels'], val_result['probs'])

        curve_data = {
            'train_losses': train_losses,
            'train_interval_losses': train_interval_losses,
            'val_losses': val_losses,
            'val_epochs': val_epochs,
            'pr_curve': pr_curve,
            'roc_curve': roc_curve_data,
        }

        if output_dir:
            prefix = os.path.join(output_dir, f'fold_{fold + 1}')
            _save_loss_curve(train_losses, val_losses, val_epochs, f'{prefix}_loss.png')
            _save_interval_loss_curve(train_interval_losses, f'{prefix}_interval_loss.png')
            _plot_pr_curve(pr_curve, f'{prefix}_pr.png')
            _plot_roc_curve(roc_curve_data, f'{prefix}_roc.png')
            _save_confusion_matrix_plot(
                fold_cm,
                f'{prefix}_confusion_matrix.png',
                title=f'Fold {fold + 1} validation confusion matrix (threshold={DECISION_THRESHOLD})',
            )

        fold_curve_data.append(curve_data)
        print(
            f"F1={metrics['f1']:.3f} | Recall={metrics['recall']:.3f} | "
            f"AP={metrics['average_precision']:.3f} | AUC={metrics['auc']:.3f} "
            f"(baseline AP={metrics['prevalence']:.3f})"
        )

    avg = {key: float(np.nanmean([f[key] for f in fold_metrics])) for key in fold_metrics[0]}
    std = {key: float(np.nanstd([f[key] for f in fold_metrics])) for key in fold_metrics[0]}

    cumulative_curves = {}
    fold_probabilities = None
    fold_probabilities_path = None

    if output_dir and fold_curve_data:
        fold_pr_curves = [fd['pr_curve'] for fd in fold_curve_data]
        fold_roc_curves = [fd['roc_curve'] for fd in fold_curve_data]

        cumulative_curves['roc'] = _save_cumulative_roc(fold_roc_curves, output_dir, k)
        cumulative_curves['pr'] = _save_cumulative_pr(fold_pr_curves, output_dir, k)
        cumulative_curves['confusion_matrix'] = _save_cumulative_confusion_matrix(
            fold_confusion_matrices, output_dir, k
        )
        cumulative_curves['loss'] = _save_cumulative_loss_curve(fold_curve_data, output_dir, k)
        fold_probabilities, fold_probabilities_path = _save_fold_probabilities(
            fold_y_true, fold_y_probs, fold_pr_curves, fold_roc_curves, output_dir, k
        )
        print(f'\nSaved cumulative ROC: {cumulative_curves["roc"]["path"]}')
        print(f'Saved cumulative PR:  {cumulative_curves["pr"]["path"]}')
        print(f'Saved cumulative CM:  {cumulative_curves["confusion_matrix"]["path"]}')
        print(f'Saved cumulative loss (train): {cumulative_curves["loss"]["train_path"]}')
        print(f'Saved cumulative loss (val):   {cumulative_curves["loss"]["val_path"]}')
        print(f'Saved fold probs:     {fold_probabilities_path}')

    print(f"\n{'=' * 80}")
    print('Average metrics')
    for key in avg:
        print(f'  {key:18s}: {avg[key]:.4f} ± {std[key]:.4f}')
    print('=' * 80)

    return avg, std, fold_metrics, fold_curve_data, cumulative_curves, fold_probabilities


def train_and_save_results(
    threshold_days,
    embeddings,
    labels,
    baby_ids=None,
    use_focal_loss=True,
    gamma=2.0,
    loss_name='Focal Loss',
    k=5,
    epochs=250,
    batch_size=400,
    lr=0.001,
    device='cpu',
):
    print(f"\n{'=' * 80}\nThreshold: {threshold_days} days ({loss_name})\n{'=' * 80}")

    suffix = 'focal' if use_focal_loss else 'ce'
    output_dir = f'training_curves_{threshold_days}d_{suffix}'

    avg, std, fold_metrics, fold_curve_data, cumulative_curves, fold_probabilities = k_fold_cv(
        embeddings,
        labels,
        baby_ids,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        use_focal_loss=use_focal_loss,
        alpha=None,
        gamma=gamma,
        output_dir=output_dir,
    )

    filename = f'results_{threshold_days}d_{suffix}.pkl'
    with open(filename, 'wb') as f:
        pl.dump(
            {
                'avg': avg,
                'std': std,
                'fold_metrics': fold_metrics,
                'fold_curve_data': fold_curve_data,
                'cumulative_curves': cumulative_curves,
                'fold_probabilities': fold_probabilities,
                'curve_output_dir': output_dir,
                'loss_type': loss_name,
                'gamma': gamma if use_focal_loss else None,
            },
            f,
        )
    print(f'\nSaved: {filename}')
    print(f'Saved curves: {output_dir}/')

    return avg, std, fold_metrics


# * alpha controls class importance.
# * gamma controls easy versus difficult example importance.


# Precision: Of everything the model predicted as positive, what fraction was actually positive?
# Recall: Of all the samples that were actually positive, what fraction did the model correctly identify?
# recall decreases or stays the same 
# precision can increase or decrease


# An ROC curve shows how a model's performance changes across classification thresholds.
#  The ROC curve plots TPR against FPR at different thresholds. 
# * True Positive Rate (Recall): Of all actual positives, how many did the model detect?
# \text{TPR}=\frac{TP}{TP+FN}
# * False Positive Rate: Of all actual negatives, how many were incorrectly predicted as positive?
# \text{FPR}=\frac{FP}{FP+TN}
# basically it comaprares how the model says the no of positives across the two classes. this helps is learn if the 
# model is learning soemthign abotu the class specifically or not, whether it is able to distinguish or not


# explore class balance among k folds...there may be too less label 1 examples in one of the folds, because 
# of which everything is being predicted as label 0.
# see if there is some seed , whcih ensires that experiemnts are repeorducible, since we are getting same results every time
# we store the validation values after every 5 epochs sicne we need to see that the model generalizes...it may be susscpetibrl, to the trainign data, so we do it over a longer no of epochs
# we store the loss and the val loss curve, since we need to see how the model is learning and not just overfitting.
# Precision–Recall is centered mainly on the positive class
# ROC looks at performance across both sides