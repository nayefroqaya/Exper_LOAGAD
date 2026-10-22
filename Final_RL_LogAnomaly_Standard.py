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
from logadempirical.data.data_loader import process_dataset_from_df  #process_dataset

#from logadempirical.data import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
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


def _rl_predict_one_loganomaly_prefix(trainer, prefix_sequence, true_label, vocab, args, device):
    """Use the existing trained LogAnomaly predictor on one anomalous prefix."""
    import contextlib
    import io

    quiet_logger = logging.getLogger("RL_LOGANOMALY_EARLY")
    quiet_logger.disabled = True

    # MINI FIX:
    # load_features() may return numpy scalar labels. sliding_window() checks
    # specifically for Python int/list, so normalize the session label here.
    prefix_label = int(_rl_binary_label(true_label))
    prefix_data = [(list(prefix_sequence), prefix_label)]

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            sequentials, quantitatives, semantics, labels, idxs, prefix_session_labels = sliding_window(
                prefix_data,
                vocab=vocab,
                window_size=args.history_size,
                is_train=False,
                semantic=args.semantic,
                quantitative=args.quantitative,
                sequential=args.sequential,
                is_unsupervised=True,
                logger=quiet_logger,
            )
    except Exception:
        return None

    lengths=[]
    if sequentials is not None: lengths.append(len(sequentials))
    if quantitatives is not None: lengths.append(len(quantitatives))
    if semantics is not None: lengths.append(len(semantics))
    if not lengths or max(lengths) == 0:
        return None

    prefix_dataset = LogDataset(
        sequentials=sequentials,
        quantitatives=quantitatives,
        semantics=semantics,
        labels=labels,
        idxs=idxs,
        is_unsupervised=True,
        remove_duplicates=False,
    )
    if len(prefix_dataset) == 0:
        return None

    try:
        acc_p, f1_p, pre_p, rec_p = trainer.predict_unsupervised(
            prefix_dataset,
            prefix_session_labels,
            topk=args.topk,
            device=device,
            is_valid=False,
            num_sessions=[1],
        )
    except Exception:
        return None

    # one positive/anomalous sequence: recall=1 means detected
    return 1 if _rl_as_fraction(rec_p) >= 0.5 else 0


def _rl_real_loganomaly_early_metrics(
    trainer,
    test_data,
    num_sessions,
    vocab,
    args,
    device,
    full_test_recall=None,
):
    """
    Real first-detection timing using native LogAnomaly next-event inference.
    Aligned with the RL evaluator:
      detection_step  = prefix_length - 1
      detection_ratio = detection_step / (sequence_length - 1)
      EDR@k            = detections by k / detected anomalies
    """
    stride = max(1, int(getattr(args, "early_eval_stride", 1)))
    history_size = int(args.history_size)

    total_anomalies = 0
    detected_anomalies = 0
    detection_steps = []
    detection_ratios = []
    unique_anomaly_count = 0
    evaluated_prefixes = 0
    failed_prefixes = 0

    print("\n" + "=" * 60)
    print("REAL LOGANOMALY EARLY-DETECTION EVALUATION")
    print("=" * 60)
    print("history_size          :", history_size)
    print("prefix stride         :", stride)
    print("first valid prefix    :", history_size + 1)
    print("training/model        : unchanged")

    for i, (sequence, true_label) in enumerate(test_data):
        if _rl_binary_label(true_label) != 1:
            continue

        unique_anomaly_count += 1
        multiplicity = int(num_sessions[i]) if i < len(num_sessions) else 1
        multiplicity = max(1, multiplicity)
        total_anomalies += multiplicity

        seq = list(sequence)
        seq_len = len(seq)
        first_prefix = history_size + 1
        if seq_len < first_prefix:
            continue

        prefixes = list(range(first_prefix, seq_len + 1, stride))
        if prefixes[-1] != seq_len:
            prefixes.append(seq_len)

        first_step = None
        for prefix_len in prefixes:
            pred = _rl_predict_one_loganomaly_prefix(
                trainer, seq[:prefix_len], true_label, vocab, args, device
            )

            if pred is None:
                failed_prefixes += 1
                continue

            evaluated_prefixes += 1

            if pred == 1:
                first_step = prefix_len - 1
                break

        if first_step is not None:
            detected_anomalies += multiplicity
            ratio = first_step / max(1, seq_len - 1)
            detection_steps.extend([float(first_step)] * multiplicity)
            detection_ratios.extend([float(ratio)] * multiplicity)

        if unique_anomaly_count % 100 == 0:
            print(
                f"Processed anomalous unique sequences: {unique_anomaly_count} | "
                f"detected(weighted): {detected_anomalies}/{total_anomalies}"
            )

    fallback_full_window = False

    # If no prefix could be evaluated at all, do NOT report ratio=0.
    # ratio=0 would mean immediate detection, which is false.
    # In that case, retain the original full-test detections and time them
    # at the complete sequence (ratio=1.0).
    if evaluated_prefixes == 0:
        fallback_full_window = True

        recall_value = (
            0.0 if full_test_recall is None
            else _rl_as_fraction(full_test_recall)
        )
        detected_anomalies = int(round(recall_value * total_anomalies))
        detected_anomalies = max(0, min(detected_anomalies, total_anomalies))

        if detected_anomalies > 0:
            detection_steps = [float(history_size)] * detected_anomalies
            detection_ratios = [1.0] * detected_anomalies

    if detection_steps:
        ratios = np.asarray(detection_ratios, dtype=float)
        avg_step = float(np.mean(detection_steps))
        avg_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))

        # Match the current RL evaluator:
        # EDR denominator = detected anomalous sequences.
        edr25 = float(np.sum(ratios <= 0.25) / max(1, detected_anomalies))
        edr50 = float(np.sum(ratios <= 0.50) / max(1, detected_anomalies))
        edr75 = float(np.sum(ratios <= 0.75) / max(1, detected_anomalies))
    else:
        avg_step = 0.0
        avg_ratio = 0.0
        median_ratio = 0.0
        edr25 = edr50 = edr75 = 0.0

    print("\\n[LogAnomaly early-evaluation diagnostics]")
    print(f"Evaluated prefixes      : {evaluated_prefixes}")
    print(f"Failed prefix calls     : {failed_prefixes}")
    print(f"Full-window fallback    : {fallback_full_window}")
    if fallback_full_window:
        print(
            "Reason                  : no native internal next-event prediction "
            "point was evaluable; full-test detections are timed at ratio 1.0"
        )

    return {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "coverage": _rl_safe_div(detected_anomalies, total_anomalies),
        "avg_step": avg_step,
        "avg_ratio": avg_ratio,
        "median_ratio": median_ratio,
        "edr25": edr25,
        "edr50": edr50,
        "edr75": edr75,
        "detection_ratios": detection_ratios,
        "evaluated_prefixes": int(evaluated_prefixes),
        "failed_prefixes": int(failed_prefixes),
        "fallback_full_window": bool(fallback_full_window),
    }


def _rl_print_full_report(args, session_labels, num_sessions, precision, recall, f1, accuracy, early_metrics=None):
    """Print/save the complete metric block requested for the paper."""
    total_normals, total_anomalies = _rl_weighted_supports(session_labels, num_sessions)
    total = total_normals + total_anomalies

    precision = _rl_as_fraction(precision)
    recall = _rl_as_fraction(recall)
    f1 = _rl_as_fraction(f1)
    accuracy = _rl_as_fraction(accuracy)

    # Original predict_unsupervised() returns summary metrics, not raw y_pred.
    # Reconstruct confusion counts from REAL weighted supports + returned P/R.
    tp = int(round(recall * total_anomalies))
    tp = max(0, min(tp, total_anomalies))
    fn = total_anomalies - tp
    fp = int(round(tp / precision - tp)) if precision > 0 else 0
    fp = max(0, min(fp, total_normals))
    tn = total_normals - fp

    an_p = _rl_safe_div(tp, tp + fp)
    an_r = _rl_safe_div(tp, tp + fn)
    an_f1 = _rl_f1(an_p, an_r)
    no_p = _rl_safe_div(tn, tn + fn)
    no_r = _rl_safe_div(tn, tn + fp)
    no_f1 = _rl_f1(no_p, no_r)
    rep_acc = _rl_safe_div(tp + tn, total)

    macro_p=(no_p+an_p)/2.0
    macro_r=(no_r+an_r)/2.0
    macro_f1=(no_f1+an_f1)/2.0
    weighted_p=_rl_safe_div(no_p*total_normals + an_p*total_anomalies, total)
    weighted_r=_rl_safe_div(no_r*total_normals + an_r*total_anomalies, total)
    weighted_f1=_rl_safe_div(no_f1*total_normals + an_f1*total_anomalies, total)

    if early_metrics is None:
        early_metrics={
            "total_anomalies": total_anomalies,
            "detected_anomalies": tp,
            "coverage": _rl_safe_div(tp,total_anomalies),
            "avg_step": 0.0,
            "avg_ratio": 0.0,
            "median_ratio": 0.0,
            "edr25": 0.0,
            "edr50": 0.0,
            "edr75": 0.0,
            "detection_ratios": [],
        }

    fp_unit=float(getattr(args,"fp_unit_cost",10.0))
    fn_unit=float(getattr(args,"fn_unit_cost",20.0))
    delay_unit=float(getattr(args,"delay_unit_cost",5.0))
    fp_cost=fp*fp_unit
    fn_cost=fn*fn_unit
    delay_cost=float(sum(delay_unit*r for r in early_metrics["detection_ratios"]))
    total_cost=fp_cost+fn_cost+delay_cost
    avg_cost=_rl_safe_div(total_cost,total)

    dataset_name=getattr(args,"dataset_name","dataset")
    cm=f"[[{tn:6d} {fp:6d}]\n [{fn:6d} {tp:6d}]]"

    lines=[]
    lines += ["", "="*60, "FINAL TEST METRICS", "="*60]
    lines.append(f"{dataset_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    lines += [cm, ""]
    lines += [f"[Classification Metrics - Anomaly Class]  # {dataset_name}",
              "Positive class        : 1 = anomaly",
              f"Accuracy              : {accuracy:.4f}",
              f"Precision             : {precision:.4f}",
              f"Recall / TPR          : {recall:.4f}",
              f"F1-score              : {f1:.4f}", ""]

    lines += ["[Classification Report - Class 0 and Class 1]",
              "Class 0               : Normal",
              "Class 1               : Anomaly", "",
              "              precision    recall  f1-score   support", "",
              f"  Normal (0)     {no_p:.4f}    {no_r:.4f}    {no_f1:.4f}  {total_normals:8d}",
              f" Anomaly (1)     {an_p:.4f}    {an_r:.4f}    {an_f1:.4f}  {total_anomalies:8d}", "",
              f"    accuracy                         {rep_acc:.4f}  {total:8d}",
              f"   macro avg     {macro_p:.4f}    {macro_r:.4f}    {macro_f1:.4f}  {total:8d}",
              f"weighted avg     {weighted_p:.4f}    {weighted_r:.4f}    {weighted_f1:.4f}  {total:8d}", ""]

    lines += ["[Confusion Matrix]", "Labels: 0=normal, 1=anomaly", cm,
              f"TP={tp} TN={tn} FP={fp} FN={fn}", ""]

    lines += ["[Early Detection Metrics]",
              (
                  "Measured by native LogAnomaly next-event prefix inference."
                  if not early_metrics.get("fallback_full_window", False)
                  else "No internal next-event point was evaluable; full-sequence detections are timed at ratio 1.0."
              ),
              f"Total anomalies       : {early_metrics['total_anomalies']}",
              f"Detected anomalies    : {early_metrics['detected_anomalies']}",
              f"Detection coverage    : {early_metrics['coverage']:.4f}",
              f"Avg detection step    : {early_metrics['avg_step']:.4f}",
              f"Avg detection ratio   : {early_metrics['avg_ratio']:.4f}",
              f"Median detect. ratio  : {early_metrics['median_ratio']:.4f}",
              f"EDR@25                : {early_metrics['edr25']:.4f}",
              f"EDR@50                : {early_metrics['edr50']:.4f}",
              f"EDR@75                : {early_metrics['edr75']:.4f}", ""]

    lines += ["[Cost-Sensitive Metrics]",
              f"FP unit cost          : {fp_unit:.4f}",
              f"FN unit cost          : {fn_unit:.4f}",
              f"Delay unit cost       : {delay_unit:.4f}",
              f"False-positive cost   : {fp_cost:.4f}",
              f"False-negative cost   : {fn_cost:.4f}",
              f"Delay cost            : {delay_cost:.4f}",
              f"Total cost            : {total_cost:.4f}",
              f"Avg cost per sequence : {avg_cost:.4f}", ""]

    lines.append("[Note] Class supports restore original sequence multiplicities. "
                 "TP/TN/FP/FN are reconstructed from real supports and the original "
                 "predict_unsupervised() Precision/Recall because that method does not "
                 "return raw per-session predictions.")

    report='\n'.join(lines)
    print(report)
    try:
        p=os.path.join(args.output_dir, f"RL_{args.model_name}_{dataset_name}_final_metrics.txt")
        with open(p,'w',encoding='utf-8') as f:
            f.write(report+'\n')
        print('Saved final metrics:', os.path.abspath(p))
    except Exception as exc:
        print('[WARNING] Could not save final metrics report:', exc)




def _la_f2(p, r):
    return (5.0 * p * r / (4.0 * p + r)) if (4.0 * p + r) else 0.0


def _la_mcc(tp, tn, fp, fn):
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return ((tp * tn - fp * fn) / den) if den else 0.0


def _la_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    precision = _rl_safe_div(tp, tp + fp)
    recall = _rl_safe_div(tp, tp + fn)
    specificity = _rl_safe_div(tn, tn + fp)
    f1 = _rl_f1(precision, recall)
    f2 = _la_f2(precision, recall)
    accuracy = _rl_safe_div(tp + tn, total)
    fpr = _rl_safe_div(fp, fp + tn)
    fnr = _rl_safe_div(fn, fn + tp)
    mcc = _la_mcc(tp, tn, fp, fn)
    alert_rate = _rl_safe_div(tp + fp, total)

    no_p = _rl_safe_div(tn, tn + fn)
    no_r = specificity
    no_f1 = _rl_f1(no_p, no_r)

    macro_p = (no_p + precision) / 2.0
    macro_r = (no_r + recall) / 2.0
    macro_f1 = (no_f1 + f1) / 2.0

    total_normals = tn + fp
    total_anomalies = tp + fn
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


def _la_print_complete_report(
    args,
    variant_name,
    tp,
    tn,
    fp,
    fn,
    early_metrics,
    note,
    evaluation_time_min,
    training_time_min,
    score_note="Raw anomaly scores are not exposed by the current predict_unsupervised() API.",
):
    """
    Print/save the common paper metric block for LogAnomaly-Standard/Early.

    AUROC/AUPRC are intentionally not fabricated from hard class decisions.
    They remain N/A until the underlying unsupervised predictor exposes a
    continuous anomaly score per sequence.
    """
    m = _la_metrics_from_counts(tp, tn, fp, fn)

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

    cm = f"[[{tn:6d} {fp:6d}]\\n [{fn:6d} {tp:6d}]]"
    dataset_name = getattr(args, "dataset_name", "dataset")

    lines = []
    lines += ["", "=" * 78]
    lines += [f"{variant_name} - FINAL TEST METRICS  # {dataset_name}"]
    lines += ["=" * 78, ""]

    lines += [
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
    ]

    lines += [
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
            f"{m['macro_recall']:.4f}    {m['macro_f1']:.4f}  "
            f"{m['total']:8d}"
        ),
        (
            f"weighted avg     {m['weighted_precision']:.4f}    "
            f"{m['weighted_recall']:.4f}    {m['weighted_f1']:.4f}  "
            f"{m['total']:8d}"
        ),
        "",
    ]

    lines += [
        "[Confusion Matrix]",
        "Labels: 0=normal, 1=anomaly",
        cm,
        f"TP={tp} TN={tn} FP={fp} FN={fn}",
        "",
    ]

    lines += [
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
    ]

    lines += [
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
    ]

    lines += [
        "[Score-based metrics note]",
        score_note,
        "AUROC/AUPRC are therefore reported as N/A rather than computed from hard labels.",
        "",
        "[Protocol note]",
        note,
    ]

    report = "\\n".join(lines)
    print(report)

    safe_variant = variant_name.replace(" ", "_").replace("/", "_")
    try:
        out_path = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_all_metrics.txt",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\\n")
        print("Saved final metrics:", os.path.abspath(out_path))
    except Exception as exc:
        print("[WARNING] Could not save final metrics report:", exc)

    # ============================================================
    # Compact paper-table summary
    # ============================================================
    compact_lines = [
        "",
        "=" * 80,
        f"{variant_name} - COMPACT PAPER RESULTS  # {dataset_name}",
        "=" * 80,
        f"Precision           : {m['precision']:.4f}",
        f"Recall              : {m['recall']:.4f}",
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

    # Save compact TXT
    try:
        compact_txt = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_compact_results.txt",
        )
        with open(compact_txt, "w", encoding="utf-8") as f:
            f.write(compact_report + "\n")
        print("Saved compact results:", os.path.abspath(compact_txt))
    except Exception as exc:
        print("[WARNING] Could not save compact TXT:", exc)

    # Save one-row CSV for direct paper-table use.
    try:
        compact_csv = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_compact_results.csv",
        )
        pd.DataFrame([{
            "Method": variant_name,
            "Dataset": dataset_name,
            "Precision": m["precision"],
            "Recall": m["recall"],
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
        print("Saved compact CSV:", os.path.abspath(compact_csv))
    except Exception as exc:
        print("[WARNING] Could not save compact CSV:", exc)

    return m



def _la_standard_full_window_metrics(test_data, num_sessions, tp):
    """
    LogAnomaly-Standard timing convention:
    detected anomalies are credited at the end of their complete sequence.

    For variable-length HDFS sessions, Avg detection step is the weighted
    average full length of anomalous sessions. Avg/median ratio are 1.0.
    """
    total_anomalies = 0
    anomaly_lengths = []

    for i, (sequence, true_label) in enumerate(test_data):
        if _rl_binary_label(true_label) != 1:
            continue
        w = int(num_sessions[i]) if i < len(num_sessions) else 1
        w = max(1, w)
        total_anomalies += w
        anomaly_lengths.extend([float(len(sequence))] * w)

    detected = max(0, min(int(tp), total_anomalies))

    if detected > 0:
        # We do not know exactly which sessions were the TPs because the
        # original predictor exposes only aggregate P/R. Use the weighted
        # anomalous-session mean length for Avg step; ratio is exactly 1.
        avg_step = float(np.mean(anomaly_lengths)) if anomaly_lengths else 0.0
        ratios = [1.0] * detected
        avg_ratio = 1.0
        median_ratio = 1.0
    else:
        avg_step = 0.0
        ratios = []
        avg_ratio = 0.0
        median_ratio = 0.0

    return {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected),
        "coverage": _rl_safe_div(detected, total_anomalies),
        "avg_step": avg_step,
        "avg_ratio": avg_ratio,
        "median_ratio": median_ratio,
        "edr25": 0.0,
        "edr50": 0.0,
        "edr75": 0.0,
        "detection_ratios": ratios,
    }

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
    train_data, train_stat = load_features(train_path, is_unsupervised=is_unsupervised, is_train=True)
    valid_data, valid_stat = load_features(valid_path, is_unsupervised=is_unsupervised, is_train=False)
    logger.info(f"Train data statistics: {train_stat}")
    logger.info(f"Valid data statistics: {valid_stat}")

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
    args.train = True

    start_train_time = time.time()  # Start timer
    if args.train:
        train_loss, val_loss, val_acc, args.topk = trainer.train(device=device,
                                                                 save_dir=f"{args.output_dir}/models",
                                                                 model_name=args.model_name,
                                                                 topk=1 if not is_unsupervised else args.topk)


        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

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

    elif is_unsupervised:
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
    label_dict = {}
    counter = {}
    for (s, l) in data:
        label_dict[tuple(s)] = l
        try:
            counter[tuple(s)] += 1
        except Exception:
            counter[tuple(s)] = 1
    data = [(list(k), v) for k, v in label_dict.items()]

    num_sessions = [counter[tuple(k)] for k, _ in data]
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        data,
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
        sequentials_normal = []
        sequentials_abnormal = []
        for i in range(len(labels)):
            if labels[i] == 0 or labels[i] == "0":
                sequentials_normal.append(sequentials[i])
            else:
                sequentials_abnormal.append(sequentials[i])
        test_dataset_normal = MaskedDataset(sequentials=sequentials_normal, vocab=vocab, seq_len=32, idx=sequence_idxs)
        test_dataset_abnormal = MaskedDataset(sequentials=sequentials_abnormal, vocab=vocab, seq_len=32,
                                              idx=sequence_idxs)
        logger.info(f"Test dataset: {len(test_dataset_normal) + len(test_dataset_abnormal)}")
        print("compute valid")
        acc, pre, rec, f1 = predict(trainer.model.to(device), abnormal_dataset=test_dataset_abnormal,
                                    normal_dataset=test_dataset_normal, device=device)
        print(f"Train Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        return 0

    test_dataset = LogDataset(sequentials=sequentials, quantitatives=quantitatives, semantics=semantics,
                              labels=labels, idxs=sequence_idxs, is_unsupervised=is_unsupervised)
    logger.info(f"Test dataset: {len(test_dataset)}")
    start_test_time = time.time()
    if is_unsupervised:
        logger.info(f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")
        acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                         session_labels,
                                                         topk=args.topk,
                                                         device=device,
                                                         is_valid=False,
                                                         num_sessions=num_sessions)
    else:
        start_test_time = time.time()
        acc, f1, pre, rec = trainer.predict_supervised(test_dataset,
                                                       session_labels,
                                                       device=device)
    test_time_min = (time.time() - start_test_time) / 60  # in minutes
    logger.info(f"Testing completed in {test_time_min:.2f} minutes")
    logger.info(f"Training completed in {train_time_min:.2f} minutes")

    logger.info(f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # ============================================================
    # LogAnomaly-Standard: original full-sequence prediction only.
    # ============================================================
    total_normals, total_anomalies = _rl_weighted_supports(
        session_labels, num_sessions
    )

    p = _rl_as_fraction(pre)
    r = _rl_as_fraction(rec)

    tp = int(round(r * total_anomalies))
    tp = max(0, min(tp, total_anomalies))
    fn = total_anomalies - tp

    fp = int(round(tp / p - tp)) if p > 0 else 0
    fp = max(0, min(fp, total_normals))
    tn = total_normals - fp

    standard_early = _la_standard_full_window_metrics(
        test_data=data,
        num_sessions=num_sessions,
        tp=tp,
    )

    _la_print_complete_report(
        args=args,
        variant_name="LogAnomaly-Standard",
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        early_metrics=standard_early,
        note=(
            "Original LogAnomaly training and original full-sequence inference "
            "are unchanged. Correctly detected anomalous sequences are assigned "
            "to the end of the sequence (detection ratio=1.0), therefore "
            "EDR@25/50/75=0."
        ),
        evaluation_time_min=test_time_min,
        training_time_min=train_time_min,
    )

    return train_time_min, test_time_min, acc, f1, pre, rec


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
        output_subdir = f"{args.output_dir}/LogAnomaly-Standard/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
    else:
        output_subdir = f"{args.output_dir}/LogAnomaly-Standard/{args.dataset_name}/session"

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

    # Third paper - same split structure, driven by YAML dataset
    split_dir = os.path.join(args.data_dir, f"1_{args.dataset_name}_Splitted_Datasets")
    file_path_train = os.path.join(split_dir, "train_df.pkl")
    file_path_test = os.path.join(split_dir, "test_df.pkl")
    file_path_val = os.path.join(split_dir, "val_df.pkl")

    print("[RL] Train split:", file_path_train)
    print("[RL] Val split  :", file_path_val)
    print("[RL] Test split :", file_path_test)


    # Read pickle file
    df_train = pd.read_pickle(file_path_train)
    df_test = pd.read_pickle(file_path_test)
    df_val = pd.read_pickle(file_path_val)

  #  df_train = df_train.rename(columns={'processed_EventTemplate': 'EventTemplate'})
  #  df_test = df_test.rename(columns={'processed_EventTemplate': 'EventTemplate'})
    print(' In run function ......')
    df_train.info()
    df_test.info()
    df_val.info()
#    exit()

    df_train = df_train.drop(columns=['Label'])
    df_test = df_test.drop(columns=['Label'])
    df_val = df_val.drop(columns=['Label'])

    df_train = df_train.rename(columns={'Original_Label': 'Label'})
    df_test = df_test.rename(columns={'Original_Label': 'Label'})
    df_val = df_val.rename(columns={'Original_Label': 'Label'})

    df_train.info()
    df_test.info()
    df_val.info()
    print(df_train['Label'].unique())
    print(df_test['Label'].unique())
    print(df_val['Label'].unique())

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

    output_dir = "/storage/home/roqaya/Exper_LOAGAD/output"

    # Keep results from Standard and Early runs side-by-side.
    # Do NOT delete the global output directory here.
    os.makedirs(output_dir, exist_ok=True)
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

    parser = arg_parser()
    args = parser.parse_args()
    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
    print(f"Loaded config from {config_file}!")
    print("========== ARGUMENTS LOADED ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================")
#    exit()

    run(args)
