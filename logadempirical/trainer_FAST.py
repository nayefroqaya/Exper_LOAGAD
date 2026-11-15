"""Fast, exact-equivalent validation top-k selection for the existing DeepLog Trainer.

Place beside the DeepLog runner. Original logadempirical/trainer.py stays unchanged.
Only _valid_epoch is overridden; original training, optimizer, and test prediction
methods are inherited. Scikit-learn tie convention: higher class index wins ties.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from logadempirical.trainer import Trainer as OriginalTrainer


def fast_topk_accuracy(y_true, probabilities, target_accuracy=0.997, chunk_size=512):
    """Return (smallest k with accuracy >= target, accuracy at that k).

    Exactly matches sklearn.metrics.top_k_accuracy_score(y_true, scores,
    k=k, labels=np.arange(C)) for C>2 and 1<=k<=C, including score ties.
    Only true-class rank is needed, avoiding C repeated sorts of all samples.
    """
    scores = np.asarray(probabilities)
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != truth.size:
        raise ValueError('Expected scores shape [N,C] and labels shape [N].')
    n, classes = scores.shape
    if n == 0 or classes < 3:
        raise ValueError(f'Need nonempty multiclass next-event predictions (N={n}, C={classes}).')
    if not np.isfinite(scores).all() or (truth < 0).any() or (truth >= classes).any():
        raise ValueError('Invalid target index or nonfinite probability scores.')
    ranks = np.empty(n, dtype=np.int32)
    class_indices = np.arange(classes, dtype=np.int64)[None, :]
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        s = scores[start:stop]
        t = truth[start:stop]
        target_scores = s[np.arange(stop-start), t][:, None]
        # sklearn tie rule: larger label ID ranks earlier when scores are equal.
        ahead = (s > target_scores) | ((s == target_scores) & (class_indices > t[:, None]))
        ranks[start:stop] = 1 + ahead.sum(axis=1)
    count_by_rank = np.bincount(ranks, minlength=classes+1)
    cumulative = np.cumsum(count_by_rank, dtype=np.int64)
    desired = target_accuracy * n
    passed = np.flatnonzero(cumulative[1:] >= desired)
    if passed.size == 0:
        raise RuntimeError('No valid top-k found, check probabilities and labels.')
    k = int(passed[0] + 1)
    acc = float(cumulative[k] / n)
    return k, acc


class Trainer(OriginalTrainer):
    """Drop-in original Trainer with one-pass validation top-k calculation."""

    def _valid_epoch(self, val_loader, device, topk=1):
        self.model.eval()
        y_pred, y_true, losses = [], [], []
        for batch in val_loader:
            del batch['idx']
            with torch.no_grad():
                outputs = self.model(batch, device=device)
            probabilities = self.accelerator.gather(outputs.probabilities)
            y_pred.append(probabilities.detach().clone().cpu().numpy())
            losses.append(outputs.loss.item())
            label = self.accelerator.gather(batch['label'])
            y_true.append(label.detach().clone().cpu().numpy())
        if not y_pred:
            raise ValueError('Empty validation loader')
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = float(np.mean(losses))
        if topk > 1:
            if y_pred.shape[1] != self.num_classes:
                raise ValueError(f'num_classes mismatch: {self.num_classes} vs {y_pred.shape[1]} outputs')
            k, acc = fast_topk_accuracy(y_true, y_pred)
            self.logger.info('Top-%d accuracy: %.12f (fast equivalent)', k, acc)
            return loss, acc, k
        acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        return loss, acc, 1
