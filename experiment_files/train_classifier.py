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
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_matrix, labels):
        self.embeddings = torch.FloatTensor(embedding_matrix)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class Classifier(nn.Module):
    """MLP head: embedding dim → 512 → 256 → 1 logit."""

    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # squeeze(-1) keeps batch dim when batch_size == 1
        return self.net(x.view(x.size(0), -1)).squeeze(-1)


def _focal_alpha(labels):
    """torchvision focal loss: positives get alpha, negatives get (1 - alpha).

    Set alpha = fraction of negatives so the rare positive class is up-weighted.
    """
    labels = np.asarray(labels)
    n_neg = (labels == 0).sum()
    return n_neg / len(labels)


def _batch_loss(logits, labels, use_focal_loss, alpha, gamma):
    if use_focal_loss:
        return sigmoid_focal_loss(
            logits, labels, alpha=alpha, gamma=gamma, reduction='mean'
        )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _run_epoch(model, loader, device, optimizer=None, use_focal_loss=False, alpha=None, gamma=None):
    """Train if optimizer is given, else evaluate loss only."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for embs, labels in loader:
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


def _save_pr_curve(labels, probs, path):
    """PR curve with random-guess baseline (= positive class prevalence)."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    prevalence = labels.mean()

    if len(np.unique(labels)) < 2:
        ap = prevalence
        precision = np.array([prevalence, prevalence])
        recall = np.array([0.0, 1.0])
    else:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        # Drop sklearn sentinel point (recall=0, precision=1) — it creates a fake vertical cliff
        # precision, recall = precision[1:], recall[1:]

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'Model (AP = {ap:.3f})')
    plt.axhline(prevalence, color='gray', linestyle='--', label=f'Random guess ({prevalence:.3f})')
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

    return {'precision': precision, 'recall': recall, 'average_precision': ap, 'prevalence': prevalence}


def _save_roc_curve(labels, probs, path):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    plt.figure(figsize=(6, 5))
    if len(np.unique(labels)) < 2:
        plt.text(0.5, 0.5, 'ROC undefined\n(only one class in val fold)', ha='center', va='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(path, dpi=150)
        plt.close()
        return {'fpr': np.array([]), 'tpr': np.array([]), 'auc': float('nan')}

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (0.5)')
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

    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}


def k_fold_cv(
    embeddings,
    labels,
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
        return None, None, None, None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    labels = np.asarray(labels, dtype=int)
    embedding_matrix = np.stack([np.asarray(e, dtype=np.float32).flatten() for e in embeddings])
    dim = embedding_matrix.shape[1]

    if use_focal_loss and alpha is None:
        alpha = _focal_alpha(labels)

    loss_name = f'Focal (alpha={alpha:.3f}, gamma={gamma})' if use_focal_loss else 'BCE'
    pos_rate = labels.mean()
    print(f"\n{'=' * 80}")
    print(f'K-fold CV | {loss_name}')
    print(f'Samples: {len(labels)} | Pos rate: {pos_rate:.3f} ({labels.sum()} positives)')
    print(f'Folds: {k} | Embedding dim: {dim}')
    if output_dir:
        print(f'Curves: {output_dir}')
    print('=' * 80)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics, fold_curve_data = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embedding_matrix, labels)):
        print(f'Fold {fold + 1}/{k}...', end=' ')

        train_loader = DataLoader(
            EmbeddingDataset(embedding_matrix[train_idx], labels[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            EmbeddingDataset(embedding_matrix[val_idx], labels[val_idx]),
            batch_size=batch_size,
        )

        model = Classifier(dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses, val_epochs = [], [], []
        for epoch in range(1, epochs + 1):
            train_losses.append(
                _run_epoch(model, train_loader, device, optimizer, use_focal_loss, alpha, gamma)
            )
            if epoch % 5 == 0:
                val_losses.append(
                    _run_epoch(model, val_loader, device, None, use_focal_loss, alpha, gamma)
                )
                val_epochs.append(epoch)

        val_result = _evaluate(model, val_loader, device)
        metrics = {k: val_result[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision', 'prevalence')}
        fold_metrics.append(metrics)

        curve_data = {'train_losses': train_losses, 'val_losses': val_losses, 'val_epochs': val_epochs}
        if output_dir:
            prefix = os.path.join(output_dir, f'fold_{fold + 1}')
            _save_loss_curve(train_losses, val_losses, val_epochs, f'{prefix}_loss.png')
            curve_data['pr_curve'] = _save_pr_curve(val_result['labels'], val_result['probs'], f'{prefix}_pr.png')
            curve_data['roc_curve'] = _save_roc_curve(val_result['labels'], val_result['probs'], f'{prefix}_roc.png')

        fold_curve_data.append(curve_data)
        print(
            f"F1={metrics['f1']:.3f} | Recall={metrics['recall']:.3f} | "
            f"AP={metrics['average_precision']:.3f} | AUC={metrics['auc']:.3f} "
            f"(baseline AP={metrics['prevalence']:.3f})"
        )

    avg = {key: float(np.nanmean([f[key] for f in fold_metrics])) for key in fold_metrics[0]}
    std = {key: float(np.nanstd([f[key] for f in fold_metrics])) for key in fold_metrics[0]}

    print(f"\n{'=' * 80}")
    print('Average metrics')
    for key in avg:
        print(f'  {key:18s}: {avg[key]:.4f} ± {std[key]:.4f}')
    print('=' * 80)

    return avg, std, fold_metrics, fold_curve_data


def train_and_save_results(
    threshold_days,
    embeddings,
    labels,
    use_focal_loss=True,
    gamma=2.0,
    loss_name='Focal Loss',
    k=5,
    epochs=50,
    batch_size=32,
    lr=0.001,
    device='cpu',
):
    print(f"\n{'=' * 80}\nThreshold: {threshold_days} days ({loss_name})\n{'=' * 80}")

    suffix = 'focal' if use_focal_loss else 'ce'
    output_dir = f'training_curves_{threshold_days}d_{suffix}'

    avg, std, fold_metrics, fold_curve_data = k_fold_cv(
        embeddings,
        labels,
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
                'curve_output_dir': output_dir,
                'loss_type': loss_name,
                'gamma': gamma if use_focal_loss else None,
            },
            f,
        )
    print(f'\nSaved: {filename}')
    print(f'Saved curves: {output_dir}/')

    return avg, std, fold_metrics
