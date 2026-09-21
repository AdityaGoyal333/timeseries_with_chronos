"""Step 5: k-fold MLP on Chronos embeddings (no Chronos in this module).

Pipeline role
-------------
Takes labeled (and optional unlabeled) embeddings from Step 4. Eval is
always labeled. Unlabeled died-file NaNs are gapped per epoch / batch fill.

Writes per-fold and pooled curves plus
results_{N}d_{focal|ce}[suffix].pkl.

Called from main.classify via train_and_save_results. Artifact suffix
comes from classify (seasonal='', FG=_fine_grained, concat=_concat, mix).

Outputs
-------
  results/training_curves_{N}d_{focal|ce}[suffix]/
  results/results_{N}d_{focal|ce}[suffix].pkl
"""

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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss

import math
from torch.utils.data import Sampler

from log_utils import tprint

from constants import (
    BINNING_FINE_GRAINED,
    FINE_GRAINED_OUTPUT_SUFFIX,
    RESULTS_DIR,
    UNLABELED_BATCH_POSITIVE_RATIO,
    US_PER_SECOND,
)
from unlabelled.gap_labels import apply_gap_labels


class EmbeddingDataset(Dataset):
    """Indexable (embedding, label) pairs for DataLoader."""

    def __init__(self, embedding_matrix, labels):
        """Store N×D embeddings and N labels as float tensors."""
        self.embeddings = torch.FloatTensor(embedding_matrix)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        """Number of windows in this split."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return (embedding, label) for one window."""
        return self.embeddings[idx], self.labels[idx]


def _output_bias_from_prevalence(positive_ratio, eps=1e-6):
    """Initialize output bias as log(p / (1 - p)), p = positives / total."""
    p = float(np.clip(positive_ratio, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


class Classifier(nn.Module):
    """MLP head: embedding dim → 512 → 256 → 1 logit."""

    def __init__(self, dim, hidden=512, positive_ratio=None):
        """Build the MLP; optional last-layer bias from train-fold prevalence."""
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
        # Bias init matches train-fold prevalence so the first logit is not ~0.5.
        if positive_ratio is not None:
            with torch.no_grad():
                self.net[-1].bias.fill_(_output_bias_from_prevalence(positive_ratio))

    def forward(self, x):
        """Logits of shape (batch,). squeeze(-1) keeps dim when batch_size == 1."""
        return self.net(x.view(x.size(0), -1)).squeeze(-1)



class UniformClassBatchSampler(Sampler):
    """Spread positives across batches, then fill each batch with negatives.

    Default (positive_ratio=None): each batch gets an equal share of all
    positives (~n_pos / num_batches), remainder filled with negatives.

    With positive_ratio set (e.g. 0.1): each batch starts from that equal
    share, then tops up to positive_ratio * batch_size by sampling extra
    positives from indices not already in the batch (reuse across batches OK).
    """

    def __init__(self, labels, batch_size, positive_ratio=None):
        """Index positives/negatives once; batch count is ceil(N / batch_size)."""
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio

        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]

        self.num_samples = len(self.labels)
        self.num_batches = math.ceil(self.num_samples / self.batch_size)

    def _target_positives_per_batch(self):
        """Hard cap of positives per batch when positive_ratio is set."""
        if self.positive_ratio is None:
            return None
        return min(int(round(self.positive_ratio * self.batch_size)), self.batch_size)

    def _positives_for_batch(self, base_chunk, all_positives):
        """Uniform chunk plus optional top-up / trim to hit positive_ratio."""
        batch_pos = list(base_chunk)
        target = self._target_positives_per_batch()
        if target is None or len(all_positives) == 0:
            return batch_pos

        if len(batch_pos) > target:
            np.random.shuffle(batch_pos)
            return batch_pos[:target]

        need = target - len(batch_pos)
        if need == 0:
            return batch_pos

        in_batch = set(batch_pos)
        pool = [idx for idx in all_positives if idx not in in_batch]
        if not pool:
            return batch_pos

        extra = np.random.choice(pool, size=need, replace=need > len(pool))
        batch_pos.extend(extra.tolist())
        return batch_pos

    def __iter__(self):
        """Yield one shuffled batch of indices per epoch pass."""
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()

        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        positive_batches = list(np.array_split(positive_indices, self.num_batches))
        np.random.shuffle(positive_batches)

        negative_pointer = 0

        for base_chunk in positive_batches:
            batch = self._positives_for_batch(base_chunk, positive_indices)

            negatives_needed = self.batch_size - len(batch)
            batch.extend(
                negative_indices[negative_pointer: negative_pointer + negatives_needed].tolist()
            )
            negative_pointer += negatives_needed

            np.random.shuffle(batch)
            if batch:
                yield batch

    def __len__(self):
        """Number of batches DataLoader should draw this epoch."""
        return self.num_batches


# Cap re-gaps on a short fill so a missing class cannot hang an epoch.
_MAX_FILL_REGAPS = 32


class GapEpochLoader:
    """Train loader: first-pass gap for P-chunks, then a new gap before each fill."""

    def __init__(
        self,
        embedding_matrix,
        realized_labels,
        unknown_mask,
        window_end,
        death_time,
        threshold_us,
        batch_size,
        positive_ratio,
    ):
        self.emb = np.asarray(embedding_matrix, dtype=np.float32)
        self.realized = np.asarray(realized_labels, dtype=np.float64)
        self.unknown_mask = np.asarray(unknown_mask, dtype=bool)
        self.window_end = np.asarray(window_end, dtype=np.float64)
        self.death_time = np.asarray(death_time, dtype=np.float64)
        self.threshold_us = threshold_us
        self.batch_size = batch_size
        self.n = len(self.realized)
        self.num_batches = math.ceil(self.n / batch_size) if self.n else 0
        self.T = min(int(round(positive_ratio * batch_size)), batch_size)
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.num_batches

    def _impute(self):
        return apply_gap_labels(
            self.realized, self.unknown_mask, self.window_end,
            self.death_time, self.threshold_us, self.rng,
        )

    def first_pass_positive_rate(self):
        """MLP bias: this train set after one gap pass (unknown rows included)."""
        y = self._impute()
        return float(y.mean()) if len(y) else 0.0

    def _extra_ones(self, need, in_batch, fill_y, realized_ones):
        if need <= 0:
            return []
        imputed_ones = np.where(self.unknown_mask & (fill_y == 1))[0]
        pool = [int(i) for i in np.concatenate([realized_ones, imputed_ones]) if int(i) not in in_batch]
        if not pool:
            return []
        extra = np.random.choice(pool, size=need, replace=need > len(pool))
        return [int(i) for i in extra]

    def _fill_zeros(self, batch, fill_y):
        in_batch = {i for i, _ in batch}
        zeros = np.where(fill_y == 0)[0]
        np.random.shuffle(zeros)
        for i in zeros:
            if len(batch) >= self.batch_size:
                break
            i = int(i)
            if i not in in_batch:
                batch.append((i, 0))
                in_batch.add(i)

    def __iter__(self):
        if self.n == 0 or self.num_batches == 0:
            return
        y0 = self._impute()
        positives = np.where(y0 == 1)[0]
        np.random.shuffle(positives)
        chunks = list(np.array_split(positives, self.num_batches))
        np.random.shuffle(chunks)
        realized_ones = np.where((~self.unknown_mask) & (self.realized == 1))[0]

        for chunk in chunks:
            batch = [(int(i), 1) for i in np.asarray(chunk)]
            if len(batch) > self.T:
                np.random.shuffle(batch)
                batch = batch[:self.T]

            for attempt in range(_MAX_FILL_REGAPS + 1):
                fill_y = self._impute()
                in_batch = {i for i, _ in batch}
                n_pos = sum(y for _, y in batch)
                if n_pos < self.T:
                    extras = self._extra_ones(self.T - n_pos, in_batch, fill_y, realized_ones)
                    batch.extend((i, 1) for i in extras)
                if len(batch) < self.batch_size:
                    self._fill_zeros(batch, fill_y)
                n_pos = sum(y for _, y in batch)
                if n_pos >= self.T and len(batch) >= self.batch_size:
                    break
                if attempt == _MAX_FILL_REGAPS:
                    break

            if not batch:
                continue
            np.random.shuffle(batch)
            idx, y = zip(*batch)
            yield torch.FloatTensor(self.emb[list(idx)]), torch.FloatTensor(y)




def _focal_alpha(labels):
    """torchvision focal loss: positives get alpha, negatives get (1 - alpha).

    Set alpha = fraction of negatives so the rare positive class is up-weighted.
    """
    return 0.95


def _batch_loss(logits, labels, use_focal_loss, alpha, gamma):
    """Mean focal or BCE-with-logits over the batch (shared train/eval path)."""
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
    """Per-fold train (every epoch) and val (every 5 epochs) loss PNG."""
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
    """Per-fold mean loss over successive 5-batch windows."""
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
        plt.plot(train_epochs, train_by_fold[i], color=fold_colors[i], alpha=fold_alpha, linewidth=1.2, label=f'Fold {i + 1}')
    plt.fill_between(train_epochs, mean_train - std_train, mean_train + std_train, color=mean_color, alpha=0.15, zorder=5)
    plt.plot(train_epochs, mean_train, color=mean_color, linewidth=2.5, zorder=6, label='Mean ± std')
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
        plt.plot(val_epochs, val_by_fold[i], color=fold_colors[i], alpha=fold_alpha, linewidth=1.2, marker='o', markersize=3, label=f'Fold {i + 1}')
    plt.fill_between(val_epochs, mean_val - std_val, mean_val + std_val, color=mean_color, alpha=0.15, zorder=5)
    plt.plot(val_epochs, mean_val, color=mean_color, linewidth=2.5, marker='o', markersize=5, zorder=6, label='Mean ± std')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Cumulative val loss ({k}-fold CV, every 5 epochs)')
    plt.legend(**_legend_kwargs())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    val_path = os.path.join(output_dir, 'Loss curve val.png')
    plt.savefig(val_path, dpi=150)
    plt.close()

    return {
        'train_path': train_path, 
        'val_path': val_path
    }


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

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    return {
        'precision': precision[:-1],
        'recall': recall[:-1],
        'average_precision': float(average_precision_score(labels, probs)),
        'prevalence': prevalence,
        'length': len(labels),
        'thresholds': thresholds
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
            plt.text(col, row, str(counts[row, col]), ha='center', va='center', color='black' if counts[row, col] < counts.max() / 2 else 'white', fontsize=14)

    plt.xticks([0, 1], class_names, rotation=15, ha='right')
    plt.yticks([0, 1], class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_cumulative_confusion_matrix(labels, probs, output_dir, threshold):
    """Pool all-fold labels/probs, then one CM at the given probability threshold."""
    labels_cumulative = np.array([])
    probs_cumulative = np.array([])
    
    for fold_label in labels:
        labels_cumulative = np.append(labels_cumulative,fold_label)
        
    for prob in probs:    
        probs_cumulative = np.append(probs_cumulative,prob)
        
    predicted_labels = (probs_cumulative >=threshold).astype(int)
    
    fold_cm = _confusion_matrix_counts(labels_cumulative, predicted_labels)
    path = os.path.join(output_dir, "confusion_matrix.png")
    _save_confusion_matrix_plot(fold_cm, path, title=f'Cumulative confusion matrix (threshold={threshold})')
    
    return {
        "path": path,
        "counts": fold_cm
    }
    
    
def _save_cumulative_roc_added(labels, probs, output_dir, k):
    """ROC from all fold (fpr, tpr) points concatenated and sorted by FPR."""

    labels_cumulative = np.array([])
    probs_cumulative = np.array([])
    
    for fold_label in labels:
        labels_cumulative = np.append(labels_cumulative,fold_label)
        
    for prob in probs:    
        probs_cumulative = np.append(probs_cumulative,prob)
    
    result = _compute_roc_curve(labels_cumulative, probs_cumulative)
    fpr = result['fpr']
    tpr = result['tpr']
    auc = result['auc']

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="k", alpha=0.5, linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve ({k}-fold CV)", fontsize=13)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "ROC curve cumulative probability.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    
    return {
        "result": result,
        "path": path
    }
    
    

def _save_cumulative_pr_added(labels, probs, output_dir, k):
    """
    PR from all fold (recall, precision) points concatenated and sorted by recall.
    """
  
    labels_cumulative = np.array([])
    probs_cumulative = np.array([])
    
    for fold_label in labels:
        labels_cumulative = np.append(labels_cumulative,fold_label)
        
    for prob in probs:    
        probs_cumulative = np.append(probs_cumulative,prob)
    
    result = _compute_pr_curve(labels_cumulative, probs_cumulative)
    precision = result['precision']
    recall = result['recall']
    prevalence = result['prevalence']
    average_precision = result['average_precision']
    thresholds = result['thresholds']
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, alpha=0.4, label=f"PR Curve (AUPRC = {average_precision :.3f})")
    
    i_80 = np.argmin(np.abs(recall - 0.8))
    precision_at_80_recall = float(precision[i_80])
    prob_threshold_80 = thresholds[i_80]
    
    plt.axvline(0.8, color='C3', linestyle=':', linewidth=1.5, alpha=0.85)
    plt.plot( recall[i_80], precision_at_80_recall, 'o', color='C3', markersize=8, label=f'Precision @ ~80% recall = {precision_at_80_recall:.3f}')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision/Recall Curve ({k}-fold CV)", fontsize=13)
    plt.axhline(prevalence, color='gray', linestyle='--', label=f'Baseline(prevalence) ({prevalence:.3f})')
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "PR curve cumulative probability.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    
    return {
        "result": result,
        "threshold": prob_threshold_80,
        "path" : path
    }
    
    

    
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


def _stack_emb(embeddings):
    """N×D float32 matrix; concat heads are already flattened vectors."""
    return np.stack([np.asarray(e, dtype=np.float32).flatten() for e in embeddings])


def _pack_unlabeled(unlabeled, dim):
    """Stack unlabeled train arrays, or None if that pool is empty."""
    if not unlabeled or not unlabeled.get('embeddings'):
        return None
    mat = _stack_emb(unlabeled['embeddings'])
    if mat.shape[1] != dim:
        raise ValueError(f"Labeled embedding dim {dim} != unlabeled dim {mat.shape[1]}")
    y = np.asarray(unlabeled['labels'], dtype=float)
    return {
        'mat': mat,
        'y': y,
        'we': np.asarray(unlabeled['window_end'], dtype=float),
        'dt': np.asarray(unlabeled['death_time'], dtype=float),
        'unknown': np.isnan(y),
    }


def _append_unlabeled(lab_mat_train, lab_y_train, un):
    """Train fold = labeled train rows + all unlabeled (gap arrays aligned)."""
    n_lab = len(lab_y_train)
    train_mat = np.concatenate([lab_mat_train, un['mat']], axis=0)
    gap_spec = (
        np.concatenate([lab_y_train.astype(float), un['y']]),
        np.concatenate([np.zeros(n_lab, dtype=bool), un['unknown']]),
        np.concatenate([np.full(n_lab, np.nan), un['we']]),
        np.concatenate([np.full(n_lab, np.nan), un['dt']]),
    )
    return train_mat, gap_spec


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
    batch_positive_ratio=None,
    unlabeled=None,
    include_labeled=True,
    threshold_us=None,
):
    """Grouped k-fold MLP. Split is labeled babies only; unlabeled never in val."""
    has_unlab = unlabeled is not None and len(unlabeled.get('embeddings') or []) > 0
    if include_labeled and len(embeddings) == 0:
        tprint('No embeddings to train on!')
        return None, None, None, None, None, None
    if not include_labeled and not has_unlab:
        tprint('No embeddings to train on!')
        return None, None, None, None, None, None
    if len(embeddings) == 0:
        tprint('No labeled embeddings to evaluate on!')
        return None, None, None, None, None, None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lab_y = np.asarray(labels, dtype=int)
    lab_mat = _stack_emb(embeddings)
    lab_baby = np.asarray(baby_ids)
    dim = lab_mat.shape[1]
    un = _pack_unlabeled(unlabeled, dim) if has_unlab else None
    has_unlab = un is not None

    if use_focal_loss and alpha is None:
        alpha = _focal_alpha(lab_y)

    loss_name = f'Focal (alpha={alpha:.3f}, gamma={gamma})' if use_focal_loss else 'BCE'
    pos_rate = lab_y.mean()
    n_babies = len(set(lab_baby.tolist()))
    if include_labeled:
        split_name = (
            'StratifiedGroupKFold (labeled baby_id; unlabeled all-in-train)'
            if has_unlab else 'StratifiedGroupKFold (by baby_id)'
        )
    else:
        split_name = 'Unlabeled train / full labeled eval'
        k = 1
    tprint(f"\n{'=' * 80}")
    tprint(f'K-fold CV | {loss_name} | {split_name}')
    tprint(f'Labeled samples: {len(lab_y)} | Pos rate: {pos_rate:.3f} ({lab_y.sum()} positives)')
    if has_unlab:
        tprint(f'Unlabeled train rows: {len(un["y"])} | unknown (NaN): {int(un["unknown"].sum())}')
    tprint(f'Babies: {n_babies} | Folds: {k}')
    tprint(f'Embedding dim: {dim}')
    if output_dir:
        tprint(f'Curves: {output_dir}')
    tprint('=' * 80)

    if include_labeled:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        split_iter = list(splitter.split(lab_mat, lab_y, groups=lab_baby))
    else:
        split_iter = [(None, None)]

    fold_metrics, fold_curve_data = [], []
    fold_y_true, fold_y_probs = [], []

    # One model per fold; val metrics use the last epoch's weights.
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if include_labeled:
            train_babies = set(lab_baby[train_idx])
            val_babies = set(lab_baby[val_idx])
            overlap = train_babies & val_babies
            assert not overlap, f"Fold {fold + 1}: {len(overlap)} babies in both train and val"
            val_mat, val_y = lab_mat[val_idx], lab_y[val_idx]
            if has_unlab:
                train_mat, gap_spec = _append_unlabeled(lab_mat[train_idx], lab_y[train_idx], un)
                train_y = None
            else:
                train_mat = lab_mat[train_idx]
                train_y = lab_y[train_idx]
                gap_spec = None
        else:
            train_mat = un['mat']
            train_y = None
            gap_spec = (un['y'], un['unknown'], un['we'], un['dt'])
            val_mat, val_y = lab_mat, lab_y

        tprint(f'Fold {fold + 1}/{k}...', end=' ')

        if gap_spec is None:
            train_loader = DataLoader(
                EmbeddingDataset(train_mat, train_y),
                batch_sampler=UniformClassBatchSampler(
                    labels=train_y,
                    batch_size=batch_size,
                    positive_ratio=batch_positive_ratio,
                ),
            )
            train_pos_rate = train_y.mean()
        else:
            realized, unknown, we, dt = gap_spec
            train_loader = GapEpochLoader(
                train_mat, realized, unknown, we, dt,
                threshold_us, batch_size, batch_positive_ratio,
            )
            train_pos_rate = train_loader.first_pass_positive_rate()

        val_loader = DataLoader(
            EmbeddingDataset(val_mat, val_y), batch_size=batch_size
        )
        model = Classifier(dim, positive_ratio=train_pos_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses, val_epochs = [], [], []
        train_interval_losses = []

        # Val loss is sampled every 5 epochs to keep the inner loop cheap.
        for epoch in range(1, epochs + 1):
            train_losses.append(_run_epoch( model, train_loader, device, optimizer, use_focal_loss, alpha, gamma, train_interval_losses))
            
            if epoch % 5 == 0:
                val_losses.append(_run_epoch(model, val_loader, device, None, use_focal_loss, alpha, gamma))
                val_epochs.append(epoch)

        # One full val pass after training (final epoch weights). Same labels/probs as metrics.
        val_result = _evaluate(model, val_loader, device)
        fold_y_true.append(val_result['labels'])
        fold_y_probs.append(val_result['probs'])
        metrics = {name: val_result[name] for name in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision', 'prevalence')}
        fold_metrics.append(metrics)

        val_preds = _predictions_from_probs(val_result['probs'])
        fold_cm = _confusion_matrix_counts(val_result['labels'], val_preds)

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
            _save_confusion_matrix_plot(fold_cm, f'{prefix}_confusion_matrix.png', title=f'Fold {fold + 1} validation confusion matrix (threshold={DECISION_THRESHOLD})')

        fold_curve_data.append(curve_data)
        
        tprint(
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
        # Pooled-probability plots; CM uses the PR-derived ~80% recall threshold.
        fold_pr_curves = [fd['pr_curve'] for fd in fold_curve_data]
        fold_roc_curves = [fd['roc_curve'] for fd in fold_curve_data]
        
        fold_probabilities, fold_probabilities_path = _save_fold_probabilities(
            fold_y_true, fold_y_probs, fold_pr_curves, fold_roc_curves, output_dir, k
        )

        cumulative_curves['roc'] = _save_cumulative_roc_added(fold_probabilities['y_true_folds'],fold_probabilities['y_score_folds'], output_dir, k)
        cumulative_curves['pr'] = _save_cumulative_pr_added(fold_probabilities['y_true_folds'],fold_probabilities['y_score_folds'], output_dir, k)
        cumulative_curves['confusion_matrix'] = _save_cumulative_confusion_matrix(fold_probabilities['y_true_folds'], fold_probabilities['y_score_folds'], output_dir, cumulative_curves['pr']['threshold'])
        cumulative_curves['loss'] = _save_cumulative_loss_curve(fold_curve_data, output_dir, k)

        tprint(f'\nSaved cumulative ROC: {cumulative_curves["roc"]["path"]}')
        tprint(f'Saved cumulative PR:  {cumulative_curves["pr"]["path"]}')
        tprint(f'Saved cumulative CM:  {cumulative_curves["confusion_matrix"]["path"]}')
        tprint(f'Saved cumulative loss (train): {cumulative_curves["loss"]["train_path"]}')
        tprint(f'Saved cumulative loss (val):   {cumulative_curves["loss"]["val_path"]}')
        tprint(f'Saved fold probs:     {fold_probabilities_path}')

    tprint(f"\n{'=' * 80}")
    tprint('Average metrics')
    for key in avg:
        tprint(f'  {key:18s}: {avg[key]:.4f} ± {std[key]:.4f}')
    tprint('=' * 80)

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
    epochs=30,
    batch_size=400,
    lr=0.0006,
    batch_positive_ratio=None,
    binning_mode='seasonal',
    output_suffix='',
    unlabeled=None,
    include_labeled=True,
):
    """Train one threshold and write results/curves using the classify suffix.

    If output_suffix is set (preferred), it names artifacts. Else fall back
    to fine-grained-from-binning_mode so older call sites stay valid.
    unlabeled is {embeddings, labels, window_end, death_time} or None.
    """
    tprint(f"\n{'=' * 80}\nThreshold: {threshold_days} days ({loss_name})\n{'=' * 80}")

    labels_arr = np.asarray(labels, dtype=int)
    dataset_pos_rate = float(labels_arr.mean()) if len(labels_arr) else float('nan')
    # Labeled in this run: 2× full labeled table (same R every fold). Unlabeled-only: config.
    if batch_positive_ratio is None:
        batch_positive_ratio = (
            min(2.0 * dataset_pos_rate, 1.0) if include_labeled else UNLABELED_BATCH_POSITIVE_RATIO
        )
    r_note = (
        f"2 × labeled dataset rate {dataset_pos_rate:.4f}"
        if include_labeled else "UNLABELED_BATCH_POSITIVE_RATIO"
    )
    tprint(f"Train batch positive ratio: {batch_positive_ratio:.4f} ({r_note})")

    loss_suffix = 'focal' if use_focal_loss else 'ce'
    # Explicit suffix from main (incl. _concat / mix) overrides mode-only naming.
    if output_suffix:
        bin_suffix = output_suffix
    else:
        bin_suffix = FINE_GRAINED_OUTPUT_SUFFIX if binning_mode == BINNING_FINE_GRAINED else ''
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(RESULTS_DIR / f'training_curves_{threshold_days}d_{loss_suffix}{bin_suffix}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kfold device: {device}")

    avg, std, fold_metrics, fold_curve_data, cumulative_curves, fold_probabilities = k_fold_cv(
        embeddings,
        labels,
        baby_ids,
        k=k if include_labeled else 1,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        use_focal_loss=use_focal_loss,
        alpha=None,
        gamma=gamma,
        output_dir=output_dir,
        batch_positive_ratio=batch_positive_ratio,
        unlabeled=unlabeled,
        include_labeled=include_labeled,
        threshold_us=threshold_days * 86400 * US_PER_SECOND,
    )

    filename = str(RESULTS_DIR / f'results_{threshold_days}d_{loss_suffix}{bin_suffix}.pkl')
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
    tprint(f'\nSaved: {filename}')
    tprint(f'Saved curves: {output_dir}/')

    return avg, std, fold_metrics