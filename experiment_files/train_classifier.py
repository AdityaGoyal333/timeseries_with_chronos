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


def k_fold_cv(embeddings, labels, baby_ids, k=5, epochs=50, batch_size=32, lr=0.001, device='cpu', use_focal_loss=True, alpha=None, gamma=2.0, output_dir=None,):
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

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if baby_ids is not None:
            train_babies = set(baby_ids[train_idx])
            val_babies = set(baby_ids[val_idx])
            overlap = train_babies & val_babies
            assert not overlap, f"Fold {fold + 1}: {len(overlap)} babies in both train and val"

        print(f'Fold {fold + 1}/{k}...', end=' ')

        train_dataset = EmbeddingDataset( embedding_matrix[train_idx], labels[train_idx])
        batch_sampler = UniformClassBatchSampler(labels=labels[train_idx], batch_size=batch_size)

        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        val_loader = DataLoader( EmbeddingDataset(embedding_matrix[val_idx], labels[val_idx]), batch_size=batch_size)

        train_pos_rate = labels[train_idx].mean()
        model = Classifier(dim, positive_ratio=train_pos_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses, val_epochs = [], [], []
        train_interval_losses = []

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
        
        fold_probabilities, fold_probabilities_path = _save_fold_probabilities(
            fold_y_true, fold_y_probs, fold_pr_curves, fold_roc_curves, output_dir, k
        )

        cumulative_curves['roc'] = _save_cumulative_roc_added(fold_probabilities['y_true_folds'],fold_probabilities['y_score_folds'], output_dir, k)
        cumulative_curves['pr'] = _save_cumulative_pr_added(fold_probabilities['y_true_folds'],fold_probabilities['y_score_folds'], output_dir, k)
        cumulative_curves['confusion_matrix'] = _save_cumulative_confusion_matrix(fold_probabilities['y_true_folds'], fold_probabilities['y_score_folds'], output_dir, cumulative_curves['pr']['threshold'])
        cumulative_curves['loss'] = _save_cumulative_loss_curve(fold_curve_data, output_dir, k)

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