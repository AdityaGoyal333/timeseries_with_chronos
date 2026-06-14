"""Step 5: Train binary classifier on Chronos embeddings with k-fold CV.

Pipeline role
-------------
Chronos embeddings are frozen (not fine-tuned). We train a small MLP head
on top to predict death-proximity (label 1) vs not (label 0).

Evaluation
----------
Stratified k-fold cross-validation preserves class balance in each fold.
Metrics: accuracy, precision, recall, F1, ROC-AUC.
Loss curves, PR curves, and ROC curves are saved per fold.

Outputs
-------
results_{N}d_focal.pkl          —  avg/std metrics + per-fold details
training_curves_{N}d_focal/     —  PNG plots for each fold
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
    """PyTorch Dataset wrapping precomputed Chronos embedding vectors."""

    def __init__(self, embeddings, labels):
        # Pre-convert to tensors once (avoids repeated conversion in __getitem__)
        self.embeddings = [torch.FloatTensor(e) for e in embeddings]
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class Classifier(nn.Module):
    """Simple 3-layer MLP head on top of frozen Chronos embeddings.

    Architecture: dim → 512 → 256 → 1 (single logit for binary classification)
    Output is a raw logit; apply sigmoid for probability.
    """

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
        # Flatten in case embedding has extra dimensions
        return self.net(x.view(x.size(0), -1)).squeeze()


def _batch_loss(logits, labels, use_focal_loss, alpha, gamma):
    """Compute loss for one batch — focal loss or standard BCE."""
    if use_focal_loss:
        # Focal loss down-weights easy examples, focuses on hard positives
        return sigmoid_focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction='mean')
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one stable op
    return nn.BCEWithLogitsLoss()(logits, labels)


def compute_loader_loss(model, loader, device, use_focal_loss, alpha, gamma):
    """Average loss over all batches in a DataLoader (used for val loss tracking)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for embs, labels in loader:
            embs, labels = embs.to(device), labels.to(device)
            logits = model(embs)
            total_loss += _batch_loss(logits, labels, use_focal_loss, alpha, gamma).item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer, device, use_focal_loss, alpha, gamma, verbose=False):
    """Run one training epoch; return mean loss over all batches."""
    model.train()
    total_loss = 0.0
    label_0_losses = []
    label_1_losses = []

    for embs, labels in loader:
        embs, labels = embs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(embs)

        if use_focal_loss:
            # Per-sample loss (reduction='none') for optional class-wise logging
            loss_per_sample = sigmoid_focal_loss(
                logits, labels, alpha=alpha, gamma=gamma, reduction='none'
            )
            label_0_mask = labels == 0
            label_1_mask = labels == 1
            if label_0_mask.any():
                label_0_losses.append(loss_per_sample[label_0_mask].mean().item())
            if label_1_mask.any():
                label_1_losses.append(loss_per_sample[label_1_mask].mean().item())
            loss = loss_per_sample.mean()
        else:
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if verbose:
        if label_0_losses:
            print(f"    Label 0 avg loss: {np.mean(label_0_losses):.4f}")
        if label_1_losses:
            print(f"    Label 1 avg loss: {np.mean(label_1_losses):.4f}")

    return total_loss / len(loader)


def validate(model, loader, device):
    """Evaluate model on a held-out set; return metrics dict."""
    model.eval()
    preds, lbls, probs = [], [], []

    with torch.no_grad():
        for embs, labels in loader:
            embs, labels = embs.to(device), labels.to(device)
            logits = model(embs)
            probs_batch = torch.sigmoid(logits).cpu().numpy()
            preds_batch = (probs_batch > 0.5).astype(int)  # default 0.5 threshold
            probs.extend(probs_batch)
            preds.extend(preds_batch)
            lbls.extend(labels.cpu().numpy())

    lbls = np.array(lbls)
    probs = np.array(probs)
    preds = np.array(preds)

    return {
        'accuracy': accuracy_score(lbls, preds),
        'precision': precision_score(lbls, preds, zero_division=0),
        'recall': recall_score(lbls, preds, zero_division=0),
        'f1': f1_score(lbls, preds, zero_division=0),
        # AUC undefined if val fold has only one class (can happen with small k)
        'auc': roc_auc_score(lbls, probs) if len(set(lbls)) > 1 else 0.0,
        'labels': lbls,
        'probs': probs,
    }


def _save_loss_curve(train_losses, val_losses, path):
    """Plot train vs val loss over epochs for one fold."""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Val loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_pr_curve(labels, probs, path):
    """Plot precision-recall curve; return curve data for pickle storage."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs) if len(set(labels)) > 1 else 0.0

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'average_precision': ap,
    }


def _save_roc_curve(labels, probs, path):
    """Plot ROC curve; handle single-class folds gracefully."""
    if len(set(labels)) < 2:
        plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, 'ROC undefined\n(single class in val fold)', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return {'fpr': np.array([]), 'tpr': np.array([]), 'thresholds': np.array([]), 'auc': 0.0}

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}


def k_fold_cv(
    embeddings,
    labels,
    k,
    epochs,
    batch_size,
    lr,
    device,
    use_focal_loss,
    alpha,
    gamma,
    output_dir=None,
):
    """Run stratified k-fold CV; return average metrics and per-fold details.

    A fresh model is trained from scratch in each fold (no weight sharing).
    StratifiedKFold ensures each fold has roughly the same class ratio.
    """
    if len(embeddings) == 0:
        print("No embeddings to train on!")
        return None, None, None, None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    dim = embeddings[0].flatten().shape[0]

    # Auto-set focal-loss alpha from class frequency if not provided
    # alpha = fraction of negatives → balances positive/negative contribution
    if use_focal_loss and alpha is None:
        n_neg = (labels == 0).sum()
        alpha = n_neg / len(labels)

    loss_type = f"Focal Loss (α={alpha:.3f}, γ={gamma})" if use_focal_loss else "Binary CE"
    print(f"\n{'=' * 80}")
    print(f"K-Fold CV with {loss_type}")
    print(f"Samples: {len(labels)}, Folds: {k}, Embedding dim: {dim}")
    print(f"Class distribution: Label 0: {(labels == 0).sum()}, Label 1: {(labels == 1).sum()}")
    if output_dir:
        print(f"Saving curves to: {output_dir}")
    print(f"{'=' * 80}\n")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    fold_curve_data = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        print(f"Fold {fold + 1}/{k}...", end=" ")

        train_loader = DataLoader(
            EmbeddingDataset(embeddings[train_idx], labels[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            EmbeddingDataset(embeddings[val_idx], labels[val_idx]),
            batch_size=batch_size,
        )

        # New model + optimizer for each fold (standard CV practice)
        model = Classifier(dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, device,
                use_focal_loss=use_focal_loss, alpha=alpha, gamma=gamma, verbose=False,
            )
            val_loss = compute_loader_loss(
                model, val_loader, device,
                use_focal_loss=use_focal_loss, alpha=alpha, gamma=gamma,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        val_result = validate(model, val_loader, device)
        metrics = {key: val_result[key] for key in ('accuracy', 'precision', 'recall', 'f1', 'auc')}
        fold_metrics.append(metrics)

        curve_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

        if output_dir:
            fold_prefix = os.path.join(output_dir, f'fold_{fold + 1}')
            _save_loss_curve(train_losses, val_losses, f'{fold_prefix}_loss.png')
            curve_data['pr_curve'] = _save_pr_curve(
                val_result['labels'], val_result['probs'], f'{fold_prefix}_pr.png'
            )
            curve_data['roc_curve'] = _save_roc_curve(
                val_result['labels'], val_result['probs'], f'{fold_prefix}_roc.png'
            )

        fold_curve_data.append(curve_data)
        print(
            f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, AUC: {metrics['auc']:.4f}"
        )

    avg = {key: np.mean([f[key] for f in fold_metrics]) for key in fold_metrics[0]}
    std = {key: np.std([f[key] for f in fold_metrics]) for key in fold_metrics[0]}

    print(f"\n{'=' * 80}")
    print(f"AVERAGE METRICS ({loss_type})")
    print('─' * 80)
    for key in avg:
        print(f"{key.capitalize():12s}: {avg[key]:.4f} ± {std[key]:.4f}")
    print('=' * 80)

    return avg, std, fold_metrics, fold_curve_data


def train_and_save_results(
    threshold_days,
    embeddings,
    labels,
    use_focal_loss,
    gamma,
    loss_name,
    k=5,
    epochs=50,
    batch_size=32,
    lr=0.001,
    device='cpu',
):
    """Train classifier for one threshold and persist results + curve plots."""
    print("\n" + "=" * 100)
    print(f"{' ' * 30}THRESHOLD: {threshold_days} DAYS ({loss_name})")
    print("=" * 100)

    suffix = 'focal' if use_focal_loss else 'ce'
    output_dir = f'training_curves_{threshold_days}d_{suffix}'

    avg, std, fold_metrics, fold_curve_data = k_fold_cv(
        embeddings, labels,
        k=k, epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, use_focal_loss=use_focal_loss, alpha=None, gamma=gamma,
        output_dir=output_dir,
    )

    # Save everything needed to reproduce / compare results later
    filename = f'results_{threshold_days}d_{suffix}.pkl'
    with open(filename, 'wb') as f:
        pl.dump({
            'avg': avg,
            'std': std,
            'fold_metrics': fold_metrics,
            'fold_curve_data': fold_curve_data,
            'curve_output_dir': output_dir,
            'loss_type': loss_name,
            'gamma': gamma if use_focal_loss else None,
        }, f)
    print(f"\n✓ Saved: {filename}")
    print(f"✓ Saved curves: {output_dir}/")

    return avg, std, fold_metrics
