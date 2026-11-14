import os
import os
import time   # ✅ ADD THIS
import pickle
from collections import Counter
import pickle
from collections import Counter
import colorama
import shutil

#<<<<<<< HEAD
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
#=======
import pandas as pd
#>>>>>>> 6198c6c (update dataset portion)
from random import seed

import torch
import yaml
from sklearn.utils import shuffle
#from logadempirical.data.data_loader import process_dataset_from_df  #process_dataset

#from logadempirical.data import process_dataset
from logadempirical.data.vocab import Vocab
#from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.data_loader_LABELFIX import process_dataset_from_df
from logadempirical.data.feature_extraction_LABELFIX import load_features, sliding_window

from logadempirical.data.dataset import LogDataset, MaskedDataset
from logadempirical.helpers import arg_parser, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer
from accelerate import Accelerator
import logging
from logging import getLogger, Logger
import argparse
from typing import List, Tuple, Optional
import numpy as np
from logadempirical.models.LogBert.predict_log import predict
import pdb
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


def build_vocab(vocab_path: str,
                data_dir: str,
                train_path: str,
                embeddings: str,
                embedding_dim: int = 300,
                is_unsupervised: bool = False,
                logger: Logger = getLogger("__name__")) -> Vocab:
    """
    Build vocab from training data
    Parameters
    ----------
    vocab_path: str: Path to save vocab
    data_dir: str: Path to data directory
    train_path: str: Path to training data
    embeddings: str: Path to pretrained embeddings
    embedding_dim: int: Dimension of embeddings
    is_unsupervised: bool: Whether the model is unsupervised or not
    logger: Logger: Logger

    Returns
    -------
    vocab: Vocab: Vocabulary
    """
    if not os.path.exists(vocab_path):
        with open(train_path, 'rb') as f:
            data = pickle.load(f)

        print("DEBUG sample item:", data[0])
        print("AVAILABLE KEYS:", data[0].keys())
#        exit()
 
        if is_unsupervised:
           logs = [
               x['EventTemplate']
               for x in data
               if np.max(x['Label']) == 0
           ]

        else:
                logs = [x['EventTemplate'] for x in data]

#            logs = [x['EventTemplate'] for x in data if np.max(x['Label']) == 0]

             # Access nested 'sequential' dictionary for EventTemplate and Label
##             logs = [
 ##                x['sequential']['EventTemplate']
  ##               for x in data
   ##              if np.max(x['sequential']['Label']) == 0
     ##        ]


       ## else:
        ##    logs = [x['sequential']['EventTemplate'] for x in data]
        vocab = Vocab(logs, os.path.join(data_dir, embeddings), embedding_dim=embedding_dim)
        logger.info(f"Vocab size: {len(vocab)}")
        logger.info(f"Save vocab in {vocab_path}")
        vocab.save_vocab(vocab_path)
    else:
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Vocab size: {len(vocab)}")
    return vocab


def build_model(args, vocab_size):
    """
    Build model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    vocab_size: int: Size of vocabulary

    Returns
    -------

    """
    if args.model_name == "DeepLog":
        model_config = ModelConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0)
        )
    elif args.model_name == "LogAnomaly":
        model_config = ModelConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
            use_semantic=args.semantic
        )
    elif args.model_name == "LogBERT":
        model_config = ModelConfig(
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
            vocab_size=vocab_size,
            use_semantic=args.semantic,
            num_heads=args.num_heads,
        )
    elif args.model_name == "LogRobust":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            is_bilstm=True,
            n_class=args.n_class,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss()
        )
    elif args.model_name == "CNN":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            max_seq_len=args.history_size,
            n_class=args.n_class,
            dropout=args.dropout,
            out_channels=args.hidden_size,
            criterion=torch.nn.CrossEntropyLoss()
        )
    elif args.model_name == "PLELog":
        raise NotImplementedError
    elif args.model_name == "NeuralLog":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            n_class=args.n_class,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            num_heads=args.num_heads,
            criterion=torch.nn.CrossEntropyLoss()
        )
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented")
    model = get_model(args.model_name, model_config)
    return model



# ============================================================
# RL-compatible FINAL TEST REPORT
# Minimal addition: original training/validation/test prediction unchanged.
# ============================================================

def _rl_as_fraction(x):
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def _rl_binary_label(label):
    if isinstance(label, (list, tuple, np.ndarray)):
        label = np.max(label)
    if isinstance(label, (bool, np.bool_)):
        return 1 if bool(label) else 0
    if isinstance(label, (int, float, np.integer, np.floating)):
        return 0 if float(label) == 0.0 else 1
    s = str(label).strip().lower()
    return 0 if s in {"0", "0.0", "-", "normal", "n", "false", "benign"} else 1


def _rl_safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def _rl_f1(p, r):
    return (2.0 * p * r / (p + r)) if (p + r) else 0.0


def _rl_weighted_supports(session_labels, num_sessions):
    """Restore real test supports after the original sequence deduplication."""
    normal = 0
    anomaly = 0
    for idx, label in session_labels.items():
        w = int(num_sessions[idx]) if idx < len(num_sessions) else 1
        w = max(1, w)
        if _rl_binary_label(label) == 1:
            anomaly += w
        else:
            normal += w
    return normal, anomaly


# The obsolete _rl_predict_one_deeplog_prefix and legacy metric wrappers
# were removed. The auditable evaluation below is the ONLY active test path.

def _dl_f2(p, r):
    return (5.0 * p * r / (4.0 * p + r)) if (4.0 * p + r) else 0.0


def _dl_mcc(tp, tn, fp, fn):
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return ((tp * tn - fp * fn) / den) if den else 0.0


def _dl_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    precision = _rl_safe_div(tp, tp + fp)
    recall = _rl_safe_div(tp, tp + fn)
    specificity = _rl_safe_div(tn, tn + fp)
    f1 = _rl_f1(precision, recall)
    f2 = _dl_f2(precision, recall)
    accuracy = _rl_safe_div(tp + tn, total)
    fpr = _rl_safe_div(fp, fp + tn)
    fnr = _rl_safe_div(fn, fn + tp)
    mcc = _dl_mcc(tp, tn, fp, fn)
    alert_rate = _rl_safe_div(tp + fp, total)

    no_p = _rl_safe_div(tn, tn + fn)
    no_r = specificity
    no_f1 = _rl_f1(no_p, no_r)

    total_normals = tn + fp
    total_anomalies = tp + fn

    macro_p = (no_p + precision) / 2.0
    macro_r = (no_r + recall) / 2.0
    macro_f1 = (no_f1 + f1) / 2.0

    weighted_p = _rl_safe_div(
        no_p * total_normals + precision * total_anomalies, total
    )
    weighted_r = _rl_safe_div(
        no_r * total_normals + recall * total_anomalies, total
    )
    weighted_f1 = _rl_safe_div(
        no_f1 * total_normals + f1 * total_anomalies, total
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "alert_rate": alert_rate,
        "normal_precision": no_p,
        "normal_recall": no_r,
        "normal_f1": no_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "total": total,
        "total_normals": total_normals,
        "total_anomalies": total_anomalies,
    }




# ============================================================
# Auditable output for journal evaluation (shared by both variants)
# ============================================================
def _dl_save_audit_artifacts(args, variant_name, tp, tn, fp, fn,
                             early_metrics, rows, diagnostics=None):
    """Persist every evaluated input sequence and one consistent metric set.

    No test labels are passed to the predictor; they are used here only to
    score its output. Preserve repeated sequences as separate observations.
    """
    import json
    import csv
    from pathlib import Path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{variant_name}_{getattr(args, 'dataset_name', 'dataset')}"

    n = tp + tn + fp + fn
    p = _rl_safe_div(tp, tp + fp)
    r = _rl_safe_div(tp, tp + fn)
    fpr = _rl_safe_div(fp, fp + tn)
    f1 = _rl_f1(p, r)
    fp_unit = float(getattr(args, 'fp_unit_cost', 10.0))
    fn_unit = float(getattr(args, 'fn_unit_cost', 20.0))
    delay_unit = float(getattr(args, 'delay_unit_cost', 5.0))
    cfp = float(tp * 0 + fp * fp_unit)
    cfn = float(fn * fn_unit)
    cd = float(sum(float(x) for x in early_metrics['detection_ratios']) * delay_unit)

    # Cross-check that metrics are backed by exactly the saved records.
    assert n == len(rows), ("test sequence count", n, len(rows))
    check_tp = sum(x['true_label'] == 1 and x['predicted_label'] == 1 for x in rows)
    check_tn = sum(x['true_label'] == 0 and x['predicted_label'] == 0 for x in rows)
    check_fp = sum(x['true_label'] == 0 and x['predicted_label'] == 1 for x in rows)
    check_fn = sum(x['true_label'] == 1 and x['predicted_label'] == 0 for x in rows)
    assert (tp, tn, fp, fn) == (check_tp, check_tn, check_fp, check_fn), 'Metric/CSV disagreement'

    metrics = {
        'method': variant_name, 'dataset': getattr(args, 'dataset_name', 'dataset'),
        'num_test_sequences': n, 'precision': p, 'recall': r, 'f1': f1,
        'fpr': fpr, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'detection_coverage': r,
        'avg_ratio': (float(early_metrics['avg_ratio']) if tp else None),
        'edr_denominator': 'ALL true anomalous test sequences',
        'edr25': float(early_metrics['edr25']),
        'edr50': float(early_metrics['edr50']),
        'edr75': float(early_metrics['edr75']),
        'cost_weights': {'fp': fp_unit, 'fn': fn_unit, 'delay': delay_unit},
        'C_FP': cfp, 'C_FN': cfn, 'C_D': cd,
        'C_total': cfp + cfn + cd,
        'avg_cost_per_sequence': _rl_safe_div(cfp + cfn + cd, n),
        'evaluation_mode': ('full-sequence decision' if 'Standard' in variant_name
                            else 'causal first-alert sequential inference'),
        'diagnostics': diagnostics or {},
    }
    assert all(0 <= metrics[f'edr{x}'] <= r + 1e-9 for x in ('25', '50', '75'))
    assert metrics['edr25'] <= metrics['edr50'] + 1e-9 <= metrics['edr75'] + 1e-9

    pred_path = output_dir / f'{prefix}_predictions.csv'
    fields = ['sequence_id', 'true_label', 'predicted_label', 'sequence_length',
              'first_alert_step', 'detection_ratio', 'prefixes_evaluated']
    with pred_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_dir / f'{prefix}_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    config_path = output_dir / f'{prefix}_run_config.json'
    config_path.write_text(json.dumps({k: v for k, v in vars(args).items()
                                       if not k.startswith('_dl_audit_')},
                                      default=str, indent=2), encoding='utf-8')

    print('\n' + '=' * 72)
    print(f'{variant_name}: VERIFIED PAPER METRICS')
    print('=' * 72)
    print(f"Precision={p:.6f}  Recall={r:.6f}  F1={f1:.6f}  FPR={fpr:.6f}")
    if tp:
        print(f"AvgRatio={metrics['avg_ratio']:.6f}")
    else:
        print('AvgRatio=N/A (no true-positive detections)')
    print(f"EDR@25={metrics['edr25']:.6f}  EDR@50={metrics['edr50']:.6f}  EDR@75={metrics['edr75']:.6f}")
    print(f"C_FP={cfp:.2f}  C_FN={cfn:.2f}  C_D={cd:.2f}  C_total={metrics['C_total']:.2f}")
    print(f'TP={tp} TN={tn} FP={fp} FN={fn}')
    print('Saved:', pred_path.resolve())
    print('Saved:', metrics_path.resolve())
    print('Saved:', config_path.resolve())
    print('=' * 72)
    return metrics


def _dl_neutral_label_prediction(trainer, candidate_sequence, vocab, args, device,
                                  *, require_one_window=False):
    """Infer a DeepLog top-k alert with a synthetic NORMAL placeholder label.

    The true test label is deliberately NEVER passed to sliding_window or
    trainer.predict_unsupervised. For a normal-labelled single session, the
    returned session accuracy is 1 if the detector says normal, 0 if it says
    anomaly. This interface assumption MUST be tested on the exact local
    logadempirical fork before paper results are finalized.
    """
    # We reuse the original next-event prediction logic and preprocessing.
    import contextlib
    import io
    candidate = [(list(candidate_sequence), 0)]
    with contextlib.redirect_stderr(io.StringIO()):
        seqs, quants, sems, labels, idxs, session_labels = sliding_window(
            candidate, vocab=vocab, window_size=int(args.history_size),
            is_train=False, semantic=args.semantic,
            quantitative=args.quantitative, sequential=args.sequential,
            is_unsupervised=True, logger=logging.getLogger('DEEPLOG_AUDIT'))

    if seqs is None or len(seqs) == 0:
        raise RuntimeError('No next-event examples generated; inspect history_size and sliding_window.')
    dataset = LogDataset(sequentials=seqs, quantitatives=quants,
                         semantics=sems, labels=labels, idxs=idxs,
                         is_unsupervised=True, remove_duplicates=False)
    if not len(dataset):
        raise RuntimeError('Empty prediction dataset after LogDataset construction.')
    if require_one_window and len(dataset) != 1:
        raise RuntimeError(f'Expected one next-event example, got {len(dataset)}. '
                           'Verify the local logadempirical sliding_window implementation.')
    with torch.no_grad():
        accuracy, _, _, _ = trainer.predict_unsupervised(
            dataset, session_labels, topk=args.topk,
            device=device, is_valid=False, num_sessions=[1])
    session_accuracy = _rl_as_fraction(accuracy)
    if not (0 <= session_accuracy <= 1):
        raise RuntimeError(f'Unexpected single-session accuracy: {accuracy!r}')
    # Only a full-session binary accuracy (not per-window accuracy) can be
    # inverted safely to recover a normal-vs-anomaly decision.
    if min(abs(session_accuracy), abs(session_accuracy - 1.0)) > 1e-5:
        raise RuntimeError(f'Expected binary per-session accuracy, got {session_accuracy}. '
                           'Inspect Trainer.predict_unsupervised before continuing.')
    return 0 if session_accuracy > 0.5 else 1


def _dl_print_report(
    args,
    variant_name,
    tp,
    tn,
    fp,
    fn,
    early_metrics,
    evaluation_time_min,
    training_time_min,
    protocol_note,
):
    m = _dl_metrics_from_counts(tp, tn, fp, fn)

    fp_unit = float(getattr(args, "fp_unit_cost", 10.0))
    fn_unit = float(getattr(args, "fn_unit_cost", 20.0))
    delay_unit = float(getattr(args, "delay_unit_cost", 5.0))

    fp_cost = fp * fp_unit
    fn_cost = fn * fn_unit
    delay_cost = float(
        sum(delay_unit * float(r) for r in early_metrics["detection_ratios"])
    )
    total_cost = fp_cost + fn_cost + delay_cost
    avg_cost = _rl_safe_div(total_cost, m["total"])

    dataset_name = getattr(args, "dataset_name", "dataset")
    cm = f"[[{tn:6d} {fp:6d}]\\n [{fn:6d} {tp:6d}]]"

    lines = [
        "",
        "=" * 78,
        f"{variant_name} - FINAL TEST METRICS  # {dataset_name}",
        "=" * 78,
        "",
        "[Classification Metrics - Anomaly Class]",
        "Positive class        : 1 = anomaly",
        f"Accuracy              : {m['accuracy']:.4f}",
        f"Precision             : {m['precision']:.4f}",
        f"Recall / TPR          : {m['recall']:.4f}",
        f"Specificity / TNR     : {m['specificity']:.4f}",
        f"F1-score              : {m['f1']:.4f}",
        f"F2-score              : {m['f2']:.4f}",
        f"FPR                   : {m['fpr']:.4f}",
        f"FNR                   : {m['fnr']:.4f}",
        f"MCC                   : {m['mcc']:.4f}",
        "AUROC                 : N/A",
        "AUPRC                 : N/A",
        f"Alert rate            : {m['alert_rate']:.4f}",
        "",
        "[Classification Report - Class 0 and Class 1]",
        "Class 0               : Normal",
        "Class 1               : Anomaly",
        "",
        "              precision    recall  f1-score   support",
        "",
        (
            f"  Normal (0)     {m['normal_precision']:.4f}    "
            f"{m['normal_recall']:.4f}    {m['normal_f1']:.4f}  "
            f"{m['total_normals']:8d}"
        ),
        (
            f" Anomaly (1)     {m['precision']:.4f}    "
            f"{m['recall']:.4f}    {m['f1']:.4f}  "
            f"{m['total_anomalies']:8d}"
        ),
        "",
        f"    accuracy                         {m['accuracy']:.4f}  {m['total']:8d}",
        (
            f"   macro avg     {m['macro_precision']:.4f}    "
            f"{m['macro_recall']:.4f}    {m['macro_f1']:.4f}  {m['total']:8d}"
        ),
        (
            f"weighted avg     {m['weighted_precision']:.4f}    "
            f"{m['weighted_recall']:.4f}    {m['weighted_f1']:.4f}  {m['total']:8d}"
        ),
        "",
        "[Confusion Matrix]",
        "Labels: 0=normal, 1=anomaly",
        cm,
        f"TP={tp} TN={tn} FP={fp} FN={fn}",
        "",
        "[Early Detection Metrics]",
        f"Total anomalies       : {early_metrics['total_anomalies']}",
        f"Detected anomalies    : {early_metrics['detected_anomalies']}",
        f"Detection coverage    : {early_metrics['coverage']:.4f}",
        f"Avg detection step    : {early_metrics['avg_step']:.4f}",
        f"Avg detection ratio   : {early_metrics['avg_ratio']:.4f}",
        f"Median detect. ratio  : {early_metrics['median_ratio']:.4f}",
        f"EDR@25                : {early_metrics['edr25']:.4f}",
        f"EDR@50                : {early_metrics['edr50']:.4f}",
        f"EDR@75                : {early_metrics['edr75']:.4f}",
        "",
        "[Cost-Sensitive Metrics]",
        f"FP unit cost          : {fp_unit:.4f}",
        f"FN unit cost          : {fn_unit:.4f}",
        f"Delay unit cost       : {delay_unit:.4f}",
        f"False-positive cost   : {fp_cost:.4f}",
        f"False-negative cost   : {fn_cost:.4f}",
        f"Delay cost            : {delay_cost:.4f}",
        f"Total cost            : {total_cost:.4f}",
        f"Avg cost per sequence : {avg_cost:.4f}",
        "",
        "[Protocol]",
        protocol_note,
        "",
        "[Score-based metrics note]",
        (
            "The current predict_unsupervised() interface does not expose a "
            "continuous per-sequence anomaly score. AUROC/AUPRC are therefore "
            "reported as N/A rather than fabricated from hard decisions."
        ),
    ]

    full_report = "\n".join(lines)
    print(full_report)

    safe_variant = variant_name.replace(" ", "_").replace("/", "_")

    try:
        full_txt = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_all_metrics.txt",
        )
        with open(full_txt, "w", encoding="utf-8") as f:
            f.write(full_report + "\n")
        print("Saved all metrics:", os.path.abspath(full_txt))
    except Exception as exc:
        print("[WARNING] Could not save all-metrics TXT:", exc)

    compact_lines = [
        f"{variant_name} - COMPACT PAPER RESULTS  # {dataset_name}",
        "=" * 80,
        f"Precision           : {m['precision']:.4f}",
        f"Recall              : {m['recall']:.4f}",
        f"FPR                 : {m['fpr']:.4f}",
        f"F1                  : {m['f1']:.4f}",
        f"Detection Coverage  : {early_metrics['coverage']:.4f}",
        f"Avg Det. Ratio      : {early_metrics['avg_ratio']:.4f}",
        f"EDR@25              : {early_metrics['edr25']:.4f}",
        f"EDR@50              : {early_metrics['edr50']:.4f}",
        f"EDR@75              : {early_metrics['edr75']:.4f}",
        f"False-positive Cost : {fp_cost:.4f}",
        f"False-negative Cost : {fn_cost:.4f}",
        f"Delay Cost          : {delay_cost:.4f}",
        f"Total Cost          : {total_cost:.4f}",
        f"Evaluation Time     : {evaluation_time_min:.4f} min",
        f"Training Time       : {training_time_min:.4f} min",
        "=" * 80,
    ]
    compact_report = "\n".join(compact_lines)
    print(compact_report)

    try:
        compact_txt = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_compact_results.txt",
        )
        with open(compact_txt, "w", encoding="utf-8") as f:
            f.write(compact_report + "\\n")

        compact_csv = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_compact_results.csv",
        )
        pd.DataFrame([{
            "Method": variant_name,
            "Dataset": dataset_name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "FPR": m["fpr"],
            "F1": m["f1"],
            "Detection Coverage": early_metrics["coverage"],
            "Avg Det. Ratio": early_metrics["avg_ratio"],
            "EDR@25": early_metrics["edr25"],
            "EDR@50": early_metrics["edr50"],
            "EDR@75": early_metrics["edr75"],
            "False-positive Cost": fp_cost,
            "False-negative Cost": fn_cost,
            "Delay Cost": delay_cost,
            "Total Cost": total_cost,
            "Evaluation Time (min)": evaluation_time_min,
            "Training Time (min)": training_time_min,
        }]).to_csv(compact_csv, index=False)

        print("Saved compact TXT:", os.path.abspath(compact_txt))
        print("Saved compact CSV:", os.path.abspath(compact_csv))
    except Exception as exc:
        print("[WARNING] Could not save compact results:", exc)

    return m



def _dl_make_fixed_next_event_sequence(sequence, end_pos, vocab, history_size):
    """
    Create one native DeepLog next-event example for a prefix endpoint.

    end_pos is 1-based and identifies the target event.
    Context is truncated to the most recent history_size events and
    LEFT-padded when shorter.
    """
    seq = list(sequence)
    if end_pos < 2 or end_pos > len(seq):
        return None

    target_event = seq[end_pos - 1]
    observed = seq[: end_pos - 1]

    if len(observed) >= history_size:
        context = observed[-history_size:]
    else:
        context = [vocab.pad_token] * (history_size - len(observed)) + observed

    return context + [target_event]


def _dl_predict_one_prefix(trainer, sequence, end_pos, vocab, args, device):
    """Causal DeepLog next-event decision at an observed prefix endpoint.

    Unlike the previous implementation, this function never receives or uses
    the true sequence label. The event at end_pos is known when it is scored.
    """
    fixed_seq = _dl_make_fixed_next_event_sequence(
        sequence, end_pos, vocab, int(args.history_size))
    if fixed_seq is None:
        raise RuntimeError(f'Invalid next-event endpoint {end_pos}.')
    return _dl_neutral_label_prediction(
        trainer, fixed_seq, vocab, args, device, require_one_window=True)


def _dl_evaluate_sequential_early(
    trainer,
    test_data,
    num_sessions,
    vocab,
    args,
    device,
):
    """
    Final Test evaluation for DeepLog-Early.

    The Test set is used only here, after training and DEV validation.
    A sequence is anomalous if any tested prefix triggers DeepLog's
    existing top-k anomaly rule. First alert gives the detection time.
    """
    stride = max(1, int(getattr(args, "early_eval_stride", 1)))
    min_events = max(2, int(getattr(args, "early_prefix_min_events", 2)))

    y_true = []
    y_pred = []
    tp_steps = []
    tp_ratios = []
    rows = []
    evaluated_prefixes = 0
    unique_processed = 0

    print("\n" + "=" * 72)
    print("DEEPLOG-EARLY CAUSAL SEQUENTIAL TEST INFERENCE")
    print("=" * 72)
    print("history_size           :", int(args.history_size))
    print("prefix stride          :", stride)
    print("first evaluated target :", min_events)
    print("prediction rule        : native DeepLog top-k next-event rule")
    print("test labels used in prediction: NEVER")

    # Preserve every test record: deduplication can collapse contradictory
    # labels and make per-sequence error counts untraceable.
    for i, (sequence, true_label) in enumerate(test_data):
        seq = list(sequence)
        T = len(seq)
        true_binary = _rl_binary_label(true_label)
        unique_processed += 1
        first_alert = None
        prefix_count = 0
        if T >= min_events:
            endpoints = list(range(min_events, T + 1, stride))
            if endpoints and endpoints[-1] != T:
                endpoints.append(T)
            for end_pos in endpoints:
                try:
                    pred = _dl_predict_one_prefix(
                        trainer=trainer, sequence=seq, end_pos=end_pos,
                        vocab=vocab, args=args, device=device)
                except Exception as exc:
                    raise RuntimeError(
                        f'DeepLog-Early inference FAILED for sequence {i} '
                        f'(length={T}), prefix endpoint={end_pos}. '
                        'Do not treat failed predictions as normal.') from exc
                prefix_count += 1
                evaluated_prefixes += 1
                if pred == 1:
                    first_alert = int(end_pos)
                    break
        seq_pred = int(first_alert is not None)
        y_true.append(true_binary)
        y_pred.append(seq_pred)
        ratio = (float(first_alert)/T) if first_alert is not None and T else None
        rows.append({
            'sequence_id': i, 'true_label': true_binary,
            'predicted_label': seq_pred, 'sequence_length': T,
            'first_alert_step': first_alert if first_alert is not None else '',
            'detection_ratio': ratio if ratio is not None else '',
            'prefixes_evaluated': prefix_count,
        })
        if true_binary == 1 and seq_pred == 1:
            tp_steps.append(float(first_alert))
            tp_ratios.append(float(ratio))
        if unique_processed % 100 == 0:
            print(f'Evaluated {unique_processed} test sequences; '
                  f'{evaluated_prefixes} next-event predictions', flush=True)

    args._dl_audit_rows = rows
    args._dl_audit_diagnostics = {'evaluated_prefixes': evaluated_prefixes,
                                  'first_eligible_prefix': min_events,
                                  'prefix_stride': stride,
                                  'test_deduplicated': False}

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total_anomalies = tp + fn

    if tp_steps:
        ratios = np.asarray(tp_ratios, dtype=float)
        avg_step = float(np.mean(tp_steps))
        avg_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))

        # Same strict definition used for the paper:
        # EDR denominator = ALL anomalous test sequences.
        edr25 = float(np.sum(ratios <= 0.25) / max(1, total_anomalies))
        edr50 = float(np.sum(ratios <= 0.50) / max(1, total_anomalies))
        edr75 = float(np.sum(ratios <= 0.75) / max(1, total_anomalies))
    else:
        avg_step = 0.0
        avg_ratio = 0.0
        median_ratio = 0.0
        edr25 = edr50 = edr75 = 0.0

    print("\\n[DeepLog-Early diagnostics]")
    print("Evaluated prefixes       :", evaluated_prefixes)
    print("Failed prefix calls      :", 0)
    print("Supervised prefix labels : False")
    print("Test used before final   : False")

    early_metrics = {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(tp),
        "coverage": _rl_safe_div(tp, total_anomalies),
        "avg_step": avg_step,
        "avg_ratio": avg_ratio,
        "median_ratio": median_ratio,
        "edr25": edr25,
        "edr50": edr50,
        "edr75": edr75,
        "detection_ratios": tp_ratios,
    }

    return tp, tn, fp, fn, early_metrics


def _dl_resolve_splits(data_root, dataset):
    from pathlib import Path
    base = Path(data_root).expanduser().resolve()
    candidates = [base / dataset / f'1_{dataset}_Splitted_Datasets',
                  base / dataset, base / f'1_{dataset}_Splitted_Datasets', base]
    for directory in candidates:
        if all((directory / x).is_file() for x in ('train_df.pkl', 'val_df.pkl', 'test_df.pkl')):
            print('Using PKL folder:', directory)
            return str(directory)
    raise FileNotFoundError(f'{dataset}: PKL splits not found under {base}; checked {candidates}')


def _dl_verify_splits(train, val, test, output_dir):
    from pathlib import Path
    import json
    import pandas as pd
    sets = {}
    for name, frame in (('train', train), ('val', val), ('test', test)):
        if 'Node_block_id' not in frame.columns:
            raise ValueError(f'{name}: missing Node_block_id; refusing to train')
        ids = frame['Node_block_id'].astype('string').str.strip()
        if ids.isna().any() or ids.eq('').any():
            raise ValueError(f'{name}: null/blank Node_block_id; refusing to train')
        sets[name] = set(ids.tolist())
    overlaps = {f'{a}_{b}': sorted(sets[a] & sets[b])
                for a,b in (('train','val'),('train','test'),('val','test'))}
    audit = {'overlap_counts': {k:len(v) for k,v in overlaps.items()},
             'examples': {k:v[:10] for k,v in overlaps.items()},
             'status': 'FAIL' if any(overlaps.values()) else 'PASS'}
    path = Path(output_dir) / 'node_block_id_audit.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if audit['status']=='FAIL':
        raise ValueError(f'CROSS-SPLIT Node_block_id OVERLAP: {audit["overlap_counts"]}. Stop. Audit: {path}')
    print('PASS: Node_block_id disjoint across train/val/test')


def _dl_evaluate_original_standard(trainer, test_data, vocab, args, device):
    """Native next-event DeepLog full-session aggregation, without early stopping.
    Uses original sliding_window and original Trainer.predict_unsupervised.
    """
    rows=[]; ratios=[]; steps=[]
    y_true=[]; y_pred=[]
    for i,(sequence,label) in enumerate(test_data):
        events=list(sequence)
        if len(events)<2:
            raise ValueError(f'Sequence {i} has fewer than 2 events; cannot predict next event')
        pred=_dl_neutral_label_prediction(trainer,events,vocab,args,device,
                                           require_one_window=False)
        truth=int(_rl_binary_label(label))
        T=len(events)
        rows.append({'sequence_id':i,'true_label':truth,'predicted_label':pred,
                     'sequence_length':T,'first_alert_step':T if pred else '',
                     'detection_ratio':1.0 if pred else '',
                     'prefixes_evaluated':1})
        y_true.append(truth);y_pred.append(pred)
        if truth==1 and pred==1:
            ratios.append(1.0);steps.append(float(T))
        if (i+1)%100==0:print('Standard test sequences:',i+1,flush=True)
    yt=np.array(y_true,dtype=int);yp=np.array(y_pred,dtype=int)
    tp=int(((yt==1)&(yp==1)).sum());tn=int(((yt==0)&(yp==0)).sum())
    fp=int(((yt==0)&(yp==1)).sum());fn=int(((yt==1)&(yp==0)).sum())
    metrics={'total_anomalies':tp+fn,'detected_anomalies':tp,
             'coverage':_rl_safe_div(tp,tp+fn),
             'avg_step':float(np.mean(steps)) if steps else 0.,
             'avg_ratio':1.0 if tp else 0.,
             'median_ratio':1.0 if tp else 0.,
             'edr25':0.,'edr50':0.,'edr75':0.,'detection_ratios':ratios}
    return (tp,tn,fp,fn),metrics,rows


def _dl_final_two_column_report(args,std,std_metrics,on,on_metrics):
    def metrics(counts, extra):
        tp,tn,fp,fn=counts
        out=_dl_metrics_from_counts(tp,tn,fp,fn)
        fp_cost=fp*float(getattr(args,'fp_unit_cost',10))
        fn_cost=fn*float(getattr(args,'fn_unit_cost',20))
        delay_cost=sum(extra['detection_ratios'])*float(getattr(args,'delay_unit_cost',5))
        return {'Precision':out['precision'],'Recall':out['recall'],
                'F1':out['f1'],'FPR':out['fpr'],
                'AvgRatio':extra['avg_ratio'] if tp else float('nan'),
                'EDR@25':extra['edr25'],'EDR@50':extra['edr50'],
                'EDR@75':extra['edr75'],'FP Cost':fp_cost,'FN Cost':fn_cost,
                'Delay Cost':delay_cost,'Total Cost':fp_cost+fn_cost+delay_cost}
    a=metrics(std,std_metrics);b=metrics(on,on_metrics)
    lines=['='*72,f'DeepLog {args.dataset_name} - ORIGINAL TRAINING, STANDARD vs ONLINE','='*72,
           f'{"Metric":<25} {"Standard":>18} {"Online":>18}','-'*72]
    for k in a:
        lines.append(f'{k:<25} {a[k]:>18.6f} {b[k]:>18.6f}')
    lines.append('='*72)
    report='\n'.join(lines)
    print(report)
    outfile=os.path.join(args.output_dir,f'{args.dataset_name}_DeepLog_Standard_vs_Online.txt')
    with open(outfile,'w',encoding='utf-8') as fh: fh.write(report+'\n')
    print('Saved:',outfile)


def train_and_eval(args: argparse.Namespace,
                   train_path: str,
                   test_path: str,
                   valid_path: str,
                   vocab: Vocab,
                   model: torch.nn.Module,
                   is_unsupervised=False,
                   logger: Logger = getLogger("__name__")) -> Tuple[float, float, float, float]:
    """
    Run model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    train_path: str: Path to training data
    test_path: str: Path to test data
    vocab: Vocab: Vocabulary
    model: torch.nn.Module: Model
    is_unsupervised: bool: Whether the model is unsupervised or not
    logger: Logger: Logger

    Returns
    -------
    Accuracy metrics
    """
    print("Loading train dataset\n")
    #data, stat = load_features(train_path,
    #                           is_unsupervised=is_unsupervised,
    #                           is_train=True)
    #logger.info(f"Train data statistics: {stat}")
    #data = shuffle(data)
    #n_valid = int(len(data) * args.valid_ratio)
    #train_data, valid_data = data[:-n_valid], data[-n_valid:]
    import pickle
    import numpy as np

    with open(train_path, "rb") as f:
        sessions = pickle.load(f)

    normal = 0
    abnormal = 0
    unexpected = 0

    for session in sessions:
        labels = np.asarray(session["Label"]).ravel()

        if labels.size == 0:
            unexpected += 1
        elif np.all(labels == 0):
            normal += 1
        elif np.any(labels == 1):
            abnormal += 1
        else:
            unexpected += 1

    print("Total generated training sessions:", len(sessions))
    print("Fully normal sessions:", normal)
    print("Sessions containing anomalies:", abnormal)
    print("Unexpected or empty labels:", unexpected)
    print("First session labels:", sessions[0]["Label"] if sessions else "EMPTY")

    train_data, train_stat = load_features(train_path, is_unsupervised=is_unsupervised, is_train=True)
    valid_data, valid_stat = load_features(valid_path, is_unsupervised=is_unsupervised, is_train=False)
    logger.info(f"Train data statistics: {train_stat}")
    logger.info(f"Valid data statistics: {valid_stat}")

    # ORIGINAL DeepLog training only. No prefix augmentation, no extra labels.

    sequentials, quantitatives, semantics, labels, idxs, _ = sliding_window(
        train_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    if args.model_name == "LogBERT":
        train_dataset = MaskedDataset(sequentials=sequentials, vocab=vocab, seq_len=32, idx=idxs)
    else:
        logger.info(f"Train dataset: {0 if sequentials is None else len(sequentials)}")

#        logger.info(f"Train dataset: {len(sequentials)}")
        train_dataset = LogDataset(sequentials=sequentials, quantitatives=quantitatives, semantics=semantics,
                                   is_unsupervised=is_unsupervised, labels=labels, idxs=idxs,
                                   remove_duplicates=args.remove_duplicates)
        logger.info(f"Train dataset: {len(train_dataset)}")

    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        valid_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    if args.model_name == "LogBERT":
        valid_dataset = MaskedDataset(sequentials=sequentials, vocab=vocab, seq_len=32, idx=sequence_idxs)
    else:
        valid_dataset = LogDataset(sequentials=sequentials, quantitatives=quantitatives, semantics=semantics,
                                   labels=labels, idxs=sequence_idxs, is_unsupervised=is_unsupervised,
                                   remove_duplicates=args.remove_duplicates)

    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Valid dataset: {len(valid_dataset)}")
    optimizer = get_optimizer(args, model.parameters())

    device = accelerator.device
    model = model.to(device)

    logger.info(f"Start training {args.model_name} model on {device} device")
    # pdb.set_trace()
    trainer = Trainer(
        model,
        train_dataset,
        valid_dataset=valid_dataset,
        is_train=True,
        optimizer=optimizer,
        no_epochs=args.max_epoch,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        warmup_rate=args.warmup_rate,
        accumulation_step=args.accumulation_step,
        logger=logger,
        accelerator=accelerator,
        num_classes=len(vocab) if is_unsupervised else args.n_class,
    )
    if args.resume and os.path.exists(f"{args.output_dir}/models/{args.model_name}.pt"):
        logger.info(f"Loading model from {args.output_dir}/models/{args.model_name}.pt...")
        trainer.load_model(f"{args.output_dir}/models/{args.model_name}.pt")
    mode = getattr(args, "eval_mode", "train")
    if mode not in ("train", "eval"):
        raise ValueError("eval_mode must be train or eval")
    ckpt = f"{args.output_dir}/models/{args.model_name}.pt"
    topk_path = os.path.join(args.output_dir, 'deeplog_selected_topk.json')
    if mode == "eval":
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Model checkpoint missing: {ckpt}; train first")
        if not os.path.isfile(topk_path):
            raise FileNotFoundError(f"Validation-selected top-k metadata missing: {topk_path}")
        import json
        with open(topk_path, encoding='utf-8') as f:
            args.topk = int(json.load(f)['topk'])
        trainer.load_model(ckpt)
    start_train_time = time.time()
    if mode == "train":
        train_loss, val_loss, val_acc, args.topk = trainer.train(device=device,
                                                                 save_dir=f"{args.output_dir}/models",
                                                                 model_name=args.model_name,
                                                                 topk=1 if not is_unsupervised else args.topk)


        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Trainer did not save a checkpoint: {ckpt}")
        trainer.load_model(ckpt)
        import json
        with open(topk_path,'w',encoding='utf-8') as f:
            json.dump({'topk':int(args.topk),'selected_on':'validation'},f,indent=2)

    train_time_min = (time.time() - start_train_time) / 60
    logger.info(f"Training completed in {train_time_min:.2f} minutes")

    if args.model_name == "LogBERT":
        # print("len" ,len(valid_dataset_abnormal))
        # loss_normal , loss_abnormal = trainer.predict_logbert(valid_dataset_normal , valid_dataset_abnormal ,device = device)
        # print(f"loss_normal: {loss_normal} , loss_abnormal : {loss_abnormal}")
        print("compute valid")
        acc, pre, rec, f1 = predict(trainer.model.to(device), abnormal_dataset=valid_dataset_abnormal,
                                    normal_dataset=valid_dataset_normal, device=device)
        print(f"Validation Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    elif is_unsupervised and mode == "train":
        acc, recommend_topk = trainer.predict_unsupervised(valid_dataset,
                                                           session_labels,
                                                           topk=args.topk,
                                                           device=device,
                                                           is_valid=True)
        logger.info(
            f"Validation Result:: Acc: {acc:.4f}, Top-{args.topk} Recommendation: {recommend_topk}")
    else:
        acc, f1, pre, rec = trainer.predict_supervised(valid_dataset,
                                                       session_labels,
                                                       device=device)
        logger.info(f"Validation Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Loading test dataset\n")
    data, stat = load_features(test_path,
                               is_unsupervised=is_unsupervised,
                               is_train=False)
    logger.info(f"Test data statistics: {stat}")
    # Keep all test sequences, including duplicates, as separate labelled records.
    # The previous dictionary-based deduplication could overwrite conflicting labels.
    num_sessions = [1] * len(data)

    start_test_time = time.time()
    counts_std, metrics_std, rows_std = _dl_evaluate_original_standard(
        trainer, data, vocab, args, device)
    tp_on, tn_on, fp_on, fn_on, metrics_on = _dl_evaluate_sequential_early(
        trainer=trainer, test_data=data, num_sessions=num_sessions,
        vocab=vocab, args=args, device=device,
    )
    counts_on = (tp_on, tn_on, fp_on, fn_on)
    rows_on = args._dl_audit_rows
    test_time_min = (time.time() - start_test_time) / 60
    _dl_save_audit_artifacts(args, 'DeepLog-Standard', *counts_std,
                             metrics_std, rows_std, {'original_complete_sequence': True})
    _dl_save_audit_artifacts(args, 'DeepLog-Online', *counts_on,
                             metrics_on, rows_on, args._dl_audit_diagnostics)
    _dl_final_two_column_report(args, counts_std, metrics_std,
                                counts_on, metrics_on)
    m = _dl_metrics_from_counts(*counts_on)
    return train_time_min, test_time_min, m['accuracy'], m['f1'], m['precision'], m['recall']



def run(args):
    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    os.makedirs(args.output_dir, exist_ok=True)

    #if args.grouping == "sliding":
    #    args.output_dir = f"{args.output_dir}/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"
    #else:
    #    args.output_dir = f"{args.output_dir}/{args.dataset_name}/session/train{args.train_size}"
    if args.grouping == "sliding":
        output_subdir = f"{args.output_dir}/DeepLog-Standard-Online/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
    else:
        output_subdir = f"{args.output_dir}/DeepLog-Standard-Online/{args.dataset_name}/session"

    os.makedirs(output_subdir, exist_ok=True)
    args.output_dir = output_subdir

    # first paper
    #file_path_train = 'dataset/HDFS/1_HDFS_Splitted_Datasets/train_df.pkl'
    #file_path_test = 'dataset/HDFS/1_HDFS_Splitted_Datasets/test_df.pkl'
    #file_path_val = 'dataset/HDFS/1_HDFS_Splitted_Datasets/val_df.pkl'

    # second paper
    #file_path_train = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl'
    #file_path_test = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl'
    #file_path_val = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl'

    split_dir = _dl_resolve_splits(args.data_dir, args.dataset_name)
    file_path_train = os.path.join(split_dir, 'train_df.pkl')
    file_path_test = os.path.join(split_dir, 'test_df.pkl')
    file_path_val = os.path.join(split_dir, 'val_df.pkl')
    df_train = pd.read_pickle(file_path_train)
    df_val = pd.read_pickle(file_path_val)
    df_test = pd.read_pickle(file_path_test)
    _dl_verify_splits(df_train, df_val, df_test, args.output_dir)
    for split_name, frame in [('train', df_train), ('val', df_val), ('test', df_test)]:
        if 'Label' not in frame.columns:
            raise ValueError(f"{split_name}: Label is required")
        if 'EventTemplate' not in frame.columns:
            raise ValueError(f"{split_name}: EventTemplate is required")
    # Normalize the original PKL ground-truth labels BEFORE generating DeepLog
    # sessions. The upstream feature loader treats 0 as normal; passing
    # strings such as "Normal" can make all sessions appear anomalous.
    # No labels are used as inference features.
    for split_name, frame in [('train', df_train), ('val', df_val), ('test', df_test)]:
        raw = frame['Label']
        numeric = raw.astype('string').str.strip().str.lower().map({
            'normal': 0, 'anomaly': 1, '0': 0, '1': 1,
            'false': 0, 'true': 1,
        })
        if numeric.isna().any():
            unknown = raw[numeric.isna()].value_counts(dropna=False).head(10).to_dict()
            raise ValueError(
                f"{split_name}: unsupported Label encodings {unknown}; "
                "expected Normal/Anomaly or 0/1. No label values were guessed."
            )
        frame['Label'] = numeric.astype('int64')
        print(f"{split_name} input event labels (0=Normal, 1=Anomaly): "
              f"{frame['Label'].value_counts().sort_index().to_dict()}")

    # Use the user's existing Label column. Original_Label is not assumed.
    #output_dir = "/storage/home/roqaya/Exper_LOAGAD/output" #output_dir = "../../dataset/BGL/"
    train_path, valid_path, test_path = process_dataset_from_df(logger=logger, df_train=df_train, df_valid=df_val,
        df_test=df_test, output_dir=args.output_dir, grouping=args.grouping, window_size=args.window_size,
        step_size=args.step_size, session_type=args.session_level, dataset_name=args.dataset_name,
        data_dir=args.data_dir)
    # ------- until here is OK..........
    #output_dir = "/storage/home/roqaya/Exper_LOAGAD/output" #output_dir = "../../dataset/BGL/"
    #train_path, test_path = process_dataset_from_df(logger=logger, df_train=df_train, df_test=df_test, output_dir=output_dir,
    #    grouping="session",  # or "session for HDFS"
    #    window_size=120, step_size=120, session_type="entry",  # or "time"
    #    dataset_name="HDFS",  # or "BGL"
    #    data_dir="../../dataset/"  # needed only for session mode (HDFS)
    #)
    #train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
    #                                        log_file=args.log_file,
    #                                        dataset_name=args.dataset_name, grouping=args.grouping,
    #                                        window_size=args.window_size, step_size=args.step_size,
    #                                        train_size=args.train_size, is_chronological=args.is_chronological,
    #                                        session_type=args.session_level)

    # pdb.set_trace()

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"
    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog", "LogBERT"]
    log_vocab = build_vocab(vocab_path,
                            args.data_dir,
                            train_path,
                            args.embeddings,
                            embedding_dim=args.embedding_dim,
                            is_unsupervised=is_unsupervised,
                            logger=logger)
    print("DEBUG: DeepLog vocabulary size:", len(log_vocab))

    if len(log_vocab) <= 2:
        raise RuntimeError("DeepLog vocabulary contains only 2 classes. "
                           "Check vocabulary construction and the saved vocabulary cache.")

    model = build_model(args, vocab_size=len(log_vocab))
    train_time, test_time, acc, f1, precision, recall= train_and_eval(args,
                   train_path,
                   test_path,
                   valid_path,  # <-- add this
                   log_vocab,
                   model,
                   is_unsupervised=is_unsupervised,
                   logger=logger)
    print(f"Training time: {train_time:.2f} min")
    print(f"Testing time: {test_time:.2f} min")
    print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


if __name__ == "__main__":

    # Output directories are configured by YAML; do not create a
    # machine-specific directory before parsing the configuration.
    RESET = colorama.Fore.RESET

    # ---------------- Device setup (CPU ONLY) ----------------
    #device = torch.device("cpu")
    #torch.backends.cudnn.enabled = False
    #torch.backends.cuda.enabled = False
    #print(f"Using device: CPU only{RESET}")

    # Automatically select GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable cuDNN for GPU acceleration
    torch.backends.cudnn.enabled = True
    # Print the device being used
    print(f"Using device-------------------xxxxxxxxxxxxxx****-------------------: {device}")

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--dataset', default=None)
    extra_parser.add_argument('--data-root', default=None)
    extra_parser.add_argument('--eval-mode', choices=['train','eval'],default='train')
    extra_args, remaining = extra_parser.parse_known_args()
    parser = arg_parser()
    args = parser.parse_args(remaining)
    if args.config_file is None or not os.path.exists(args.config_file):
        raise FileNotFoundError(
            "Pass the matching YAML file using --config_file <path>; "
            f"received: {args.config_file!r}")
    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
    if extra_args.dataset is not None:
        args.dataset_name = extra_args.dataset
    if extra_args.data_root is not None:
        args.data_dir = extra_args.data_root
    args.eval_mode = extra_args.eval_mode
    if args.model_name != 'DeepLog':
        raise ValueError('This runner is for DeepLog only')
    if int(args.history_size) != 120:
        raise ValueError('history_size must be 120 for the requested protocol')
    print(f"Loaded config from {config_file}!")
    print("========== ARGUMENTS LOADED ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================")
#    exit()

    run(args)
