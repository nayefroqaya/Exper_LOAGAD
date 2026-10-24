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
# RL-compatible reporting for supervised full-window baselines
# Minimal addition: LogRobust training/model are unchanged.
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


def _rl_session_supports(session_labels):
    """
    Real supervised test supports.
    No test-sequence deduplication is used for LogRobust.
    """
    normal = 0
    anomaly = 0

    for _, label in session_labels.items():
        if _rl_binary_label(label) == 1:
            anomaly += 1
        else:
            normal += 1

    return normal, anomaly


def _rl_logrobust_full_window_metrics(args, total_anomalies, recall):
    """
    Native LogRobust timing.

    LogRobust is a supervised sequence/window classifier.
    Under history_size=120 with a 120-event sequence, the valid decision is
    at the complete window, so detected anomalies have ratio 1.0.

    These values are not artificially hard-coded as model performance:
    the number detected comes from the original test recall.
    Timing is assigned to the native full-window decision point.
    """
    recall = _rl_as_fraction(recall)
    detected = int(round(recall * total_anomalies))
    detected = max(0, min(detected, total_anomalies))

    history_size = int(
        getattr(args, "history_size", getattr(args, "window_size", 120))
    )

    ratios = [1.0] * detected
    steps = [float(history_size)] * detected

    return {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected),
        "coverage": _rl_safe_div(detected, total_anomalies),
        "avg_step": float(np.mean(steps)) if steps else 0.0,
        "avg_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "median_ratio": float(np.median(ratios)) if ratios else 0.0,
        "edr25": 0.0,
        "edr50": 0.0,
        "edr75": 0.0,
        "detection_ratios": ratios,
    }


def _rl_print_logrobust_report(
    args,
    session_labels,
    precision,
    recall,
    f1,
    accuracy,
):
    """
    Complete classification + timing + cost report for LogRobust.

    predict_supervised() returns summary metrics rather than raw y_pred,
    therefore TP/FP/TN/FN are reconstructed from the real supervised test
    supports and returned Precision/Recall.
    """
    total_normals, total_anomalies = _rl_session_supports(session_labels)
    total = total_normals + total_anomalies

    precision = _rl_as_fraction(precision)
    recall = _rl_as_fraction(recall)
    f1 = _rl_as_fraction(f1)
    accuracy = _rl_as_fraction(accuracy)

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

    macro_p = (no_p + an_p) / 2.0
    macro_r = (no_r + an_r) / 2.0
    macro_f1 = (no_f1 + an_f1) / 2.0

    weighted_p = _rl_safe_div(
        no_p * total_normals + an_p * total_anomalies, total
    )
    weighted_r = _rl_safe_div(
        no_r * total_normals + an_r * total_anomalies, total
    )
    weighted_f1 = _rl_safe_div(
        no_f1 * total_normals + an_f1 * total_anomalies, total
    )

    early = _rl_logrobust_full_window_metrics(
        args=args,
        total_anomalies=total_anomalies,
        recall=recall,
    )

    fp_unit = float(getattr(args, "fp_unit_cost", 10.0))
    fn_unit = float(getattr(args, "fn_unit_cost", 20.0))
    delay_unit = float(getattr(args, "delay_unit_cost", 5.0))

    fp_cost = fp * fp_unit
    fn_cost = fn * fn_unit
    delay_cost = float(
        sum(delay_unit * r for r in early["detection_ratios"])
    )
    total_cost = fp_cost + fn_cost + delay_cost
    avg_cost = _rl_safe_div(total_cost, total)

    dataset_name = getattr(args, "dataset_name", "dataset")
    cm = f"[[{tn:6d} {fp:6d}]\n [{fn:6d} {tp:6d}]]"

    lines = []
    lines += ["", "=" * 60, "FINAL TEST METRICS", "=" * 60]
    lines.append(
        f"{dataset_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    )
    lines += [cm, ""]

    lines += [
        f"[Classification Metrics - Anomaly Class]  # {dataset_name}",
        "Positive class        : 1 = anomaly",
        f"Accuracy              : {accuracy:.4f}",
        f"Precision             : {precision:.4f}",
        f"Recall / TPR          : {recall:.4f}",
        f"F1-score              : {f1:.4f}",
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
            f"  Normal (0)     {no_p:.4f}    {no_r:.4f}    "
            f"{no_f1:.4f}  {total_normals:8d}"
        ),
        (
            f" Anomaly (1)     {an_p:.4f}    {an_r:.4f}    "
            f"{an_f1:.4f}  {total_anomalies:8d}"
        ),
        "",
        (
            f"    accuracy                         {rep_acc:.4f}  "
            f"{total:8d}"
        ),
        (
            f"   macro avg     {macro_p:.4f}    {macro_r:.4f}    "
            f"{macro_f1:.4f}  {total:8d}"
        ),
        (
            f"weighted avg     {weighted_p:.4f}    {weighted_r:.4f}    "
            f"{weighted_f1:.4f}  {total:8d}"
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
        "Native LogRobust full-window sequence classification.",
        f"Total anomalies       : {early['total_anomalies']}",
        f"Detected anomalies    : {early['detected_anomalies']}",
        f"Detection coverage    : {early['coverage']:.4f}",
        f"Avg detection step    : {early['avg_step']:.4f}",
        f"Avg detection ratio   : {early['avg_ratio']:.4f}",
        f"Median detect. ratio  : {early['median_ratio']:.4f}",
        f"EDR@25                : {early['edr25']:.4f}",
        f"EDR@50                : {early['edr50']:.4f}",
        f"EDR@75                : {early['edr75']:.4f}",
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

    lines.append(
        "[Note] LogRobust is evaluated as its native supervised full-window "
        "classifier. The test set is not deduplicated. TP/TN/FP/FN are "
        "reconstructed from the real test supports and the Precision/Recall "
        "returned by predict_supervised(), because that method does not "
        "expose raw per-sequence predictions."
    )

    report = "\n".join(lines)
    print(report)

    try:
        report_path = os.path.join(
            args.output_dir,
            f"RL_{args.model_name}_{dataset_name}_final_metrics.txt",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print("Saved final metrics:", os.path.abspath(report_path))
    except Exception as exc:
        print("[WARNING] Could not save final metrics report:", exc)





# ============================================================
# LogRobust Standard/Early paper metrics
# ============================================================

def _lr_f2(p, r):
    return (5.0 * p * r / (4.0 * p + r)) if (4.0 * p + r) else 0.0


def _lr_mcc(tp, tn, fp, fn):
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return ((tp * tn - fp * fn) / den) if den else 0.0


def _lr_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    precision = _rl_safe_div(tp, tp + fp)
    recall = _rl_safe_div(tp, tp + fn)
    specificity = _rl_safe_div(tn, tn + fp)
    f1 = _rl_f1(precision, recall)
    f2 = _lr_f2(precision, recall)
    accuracy = _rl_safe_div(tp + tn, total)
    fpr = _rl_safe_div(fp, fp + tn)
    fnr = _rl_safe_div(fn, fn + tp)
    mcc = _lr_mcc(tp, tn, fp, fn)
    alert_rate = _rl_safe_div(tp + fp, total)

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
        "total": total,
    }


def _lr_print_compact(
    args,
    variant_name,
    tp,
    tn,
    fp,
    fn,
    early_metrics,
    evaluation_time_min,
    training_time_min,
):
    m = _lr_metrics_from_counts(tp, tn, fp, fn)

    fp_unit = float(getattr(args, "fp_unit_cost", 10.0))
    fn_unit = float(getattr(args, "fn_unit_cost", 20.0))
    delay_unit = float(getattr(args, "delay_unit_cost", 5.0))

    fp_cost = fp * fp_unit
    fn_cost = fn * fn_unit
    delay_cost = float(
        sum(delay_unit * float(r) for r in early_metrics["detection_ratios"])
    )
    total_cost = fp_cost + fn_cost + delay_cost

    dataset_name = getattr(args, "dataset_name", "dataset")

    compact_lines = [
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

    safe_variant = variant_name.replace(" ", "_").replace("/", "_")

    try:
        compact_txt = os.path.join(
            args.output_dir,
            f"RL_{safe_variant}_{dataset_name}_compact_results.txt",
        )
        with open(compact_txt, "w", encoding="utf-8") as f:
            f.write(compact_report + "\n")

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

        print("Saved compact TXT:", os.path.abspath(compact_txt))
        print("Saved compact CSV:", os.path.abspath(compact_csv))
    except Exception as exc:
        print("[WARNING] Could not save compact results:", exc)

    return m



def _lr_event_level_labels(label, seq_len):
    """
    Return event-level binary labels only when they are genuinely available
    and aligned with the sequence length. Otherwise return None.
    """
    if isinstance(label, np.ndarray):
        arr = label.tolist()
    elif isinstance(label, (list, tuple)):
        arr = list(label)
    else:
        return None

    if len(arr) != seq_len:
        return None

    return [_rl_binary_label(x) for x in arr]


def _lr_prefix_aware_supervised_training_data(train_data, args):
    """
    Keep every original full training sequence.

    Add 25/50/75% prefixes:
      - normal sequence -> prefix label 0
      - anomalous sequence -> use TRUE event-level onset if available
      - if anomalous onset cannot be determined -> add no anomalous prefixes

    No onset label is fabricated.
    """
    fractions = getattr(args, "early_train_prefix_fractions", [0.25, 0.50, 0.75])

    augmented = list(train_data)

    normal_prefixes = 0
    anomalous_prefixes = 0
    anomalous_without_onset = 0

    for sequence, seq_label in train_data:
        seq = list(sequence)
        T = len(seq)

        if T < 1:
            continue

        seq_binary = _rl_binary_label(seq_label)
        event_labels = _lr_event_level_labels(seq_label, T)

        onset = None
        if seq_binary == 1 and event_labels is not None:
            for idx, value in enumerate(event_labels):
                if value == 1:
                    onset = idx + 1  # 1-based
                    break

        used_ends = set()

        for frac in fractions:
            try:
                frac = float(frac)
            except Exception:
                continue

            end_pos = int(np.ceil(frac * T))
            end_pos = max(1, min(T, end_pos))

            if end_pos in used_ends:
                continue
            used_ends.add(end_pos)

            prefix = seq[:end_pos]

            if seq_binary == 0:
                augmented.append((prefix, 0))
                normal_prefixes += 1
                continue

            # Anomalous source sequence:
            # only generate prefix labels when true onset is available.
            if onset is None:
                continue

            prefix_label = 1 if end_pos >= onset else 0
            augmented.append((prefix, prefix_label))
            anomalous_prefixes += 1

        if seq_binary == 1 and onset is None:
            anomalous_without_onset += 1

    print("\n" + "=" * 72)
    print("LOGROBUST-EARLY PREFIX-AWARE SUPERVISED TRAINING")
    print("=" * 72)
    print("Original training items            :", len(train_data))
    print("Added normal prefixes              :", normal_prefixes)
    print("Added onset-grounded anomaly prefs :", anomalous_prefixes)
    print("Anomalous seqs without onset       :", anomalous_without_onset)
    print("Final training items               :", len(augmented))
    print("Prefix fractions                   :", fractions)
    print("Fabricated anomaly prefix labels   : False")

    return augmented


def _lr_predict_one_prefix(
    trainer,
    prefix_sequence,
    true_label,
    vocab,
    args,
    device,
):
    """
    Apply the trained supervised LogRobust classifier to one prefix.

    sliding_window(..., is_unsupervised=False) keeps the prefix and pads it
    to history_size according to the existing preprocessing implementation.
    """
    import contextlib
    import io

    prefix_label = int(_rl_binary_label(true_label))
    prefix_data = [(list(prefix_sequence), prefix_label)]

    quiet_logger = logging.getLogger("LOGROBUST_EARLY_PREFIX")
    quiet_logger.disabled = True

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            sequentials, quantitatives, semantics, labels, idxs, session_labels = sliding_window(
                prefix_data,
                vocab=vocab,
                window_size=args.history_size,
                is_train=False,
                semantic=args.semantic,
                quantitative=args.quantitative,
                sequential=args.sequential,
                is_unsupervised=False,
                logger=quiet_logger,
            )
    except Exception:
        return None

    if sequentials is None or len(sequentials) == 0:
        return None

    prefix_dataset = LogDataset(
        sequentials=sequentials,
        quantitatives=quantitatives,
        semantics=semantics,
        labels=labels,
        idxs=idxs,
        is_unsupervised=False,
        remove_duplicates=False,
    )

    if len(prefix_dataset) == 0:
        return None

    try:
        acc_p, f1_p, pre_p, rec_p = trainer.predict_supervised(
            prefix_dataset,
            session_labels,
            device=device,
        )
    except Exception:
        return None

    # For a one-sample evaluation, accuracy tells us whether the prediction
    # equals the supplied true sequence label.
    true_binary = _rl_binary_label(true_label)
    correct = _rl_as_fraction(acc_p) >= 0.5

    if correct:
        return true_binary
    return 1 - true_binary


def _lr_evaluate_sequential_early(
    trainer,
    test_data,
    vocab,
    args,
    device,
):
    """
    Final sequential Test evaluation for LogRobust-Early.

    Test is not used for training, prefix generation, or model selection.
    The first anomalous prefix triggers the alert.
    """
    stride = max(1, int(getattr(args, "early_eval_stride", 1)))
    min_prefix_len = max(1, int(getattr(args, "early_prefix_min_events", 1)))

    y_true = []
    y_pred = []
    tp_steps = []
    tp_ratios = []

    evaluated_prefixes = 0
    failed_prefixes = 0

    print("\n" + "=" * 72)
    print("LOGROBUST-EARLY SEQUENTIAL TEST INFERENCE")
    print("=" * 72)
    print("prefix stride          :", stride)
    print("first prefix length    :", min_prefix_len)
    print("first anomaly pred.    : alert")
    print("Test used for training : False")

    for i, (sequence, true_label) in enumerate(test_data):
        seq = list(sequence)
        T = len(seq)
        true_binary = _rl_binary_label(true_label)

        first_alert = None

        if T >= min_prefix_len:
            endpoints = list(range(min_prefix_len, T + 1, stride))
            if endpoints and endpoints[-1] != T:
                endpoints.append(T)

            for end_pos in endpoints:
                pred = _lr_predict_one_prefix(
                    trainer=trainer,
                    prefix_sequence=seq[:end_pos],
                    true_label=true_label,
                    vocab=vocab,
                    args=args,
                    device=device,
                )

                if pred is None:
                    failed_prefixes += 1
                    continue

                evaluated_prefixes += 1

                if pred == 1:
                    first_alert = end_pos
                    break

        seq_pred = 1 if first_alert is not None else 0

        y_true.append(true_binary)
        y_pred.append(seq_pred)

        if true_binary == 1 and seq_pred == 1:
            ratio = first_alert / max(1, T)
            tp_steps.append(float(first_alert))
            tp_ratios.append(float(ratio))

        if (i + 1) % 100 == 0:
            print(
                f"Processed test sequences: {i + 1} | "
                f"evaluated prefixes: {evaluated_prefixes}"
            )

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

        # Strict paper definition: denominator = ALL anomalies.
        edr25 = float(np.sum(ratios <= 0.25) / max(1, total_anomalies))
        edr50 = float(np.sum(ratios <= 0.50) / max(1, total_anomalies))
        edr75 = float(np.sum(ratios <= 0.75) / max(1, total_anomalies))
    else:
        avg_step = 0.0
        avg_ratio = 0.0
        median_ratio = 0.0
        edr25 = edr50 = edr75 = 0.0

    print("\n[LogRobust-Early diagnostics]")
    print("Evaluated prefixes            :", evaluated_prefixes)
    print("Failed prefix calls           :", failed_prefixes)
    print("Fabricated anomaly train labs :", False)
    print("Test used before final eval   :", False)

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

    # LogRobust-Early: TRAIN-only prefix-aware supervised augmentation.
    # DEV remains separate and TEST is untouched until final evaluation.
    if args.model_name == "LogRobust":
        train_data = _lr_prefix_aware_supervised_training_data(
            train_data=train_data,
            args=args,
        )

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
    # ============================================================
    # FINAL TEST: sequential prefix inference ONLY.
    # No preceding full-sequence test evaluation.
    # ============================================================
    start_test_time = time.time()

    tp, tn, fp, fn, early_metrics = _lr_evaluate_sequential_early(
        trainer=trainer,
        test_data=data,
        vocab=vocab,
        args=args,
        device=device,
    )

    test_time_min = (time.time() - start_test_time) / 60.0

    m = _lr_print_compact(
        args=args,
        variant_name="LogRobust-Early",
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        early_metrics=early_metrics,
        evaluation_time_min=test_time_min,
        training_time_min=train_time_min,
    )

    acc = m["accuracy"]
    pre = m["precision"]
    rec = m["recall"]
    f1 = m["f1"]

    logger.info(
        f"LogRobust-Early Test Result:: Acc: {acc:.4f}, "
        f"Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
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
        output_subdir = f"{args.output_dir}/LogRobust-Early/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
    else:
        output_subdir = f"{args.output_dir}/LogRobust-Early/{args.dataset_name}/session"

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
    split_dir = os.path.join(
        args.data_dir,
        f"1_{args.dataset_name}_Splitted_Datasets"
    )
    file_path_train = os.path.join(split_dir, "train_df.pkl")
    file_path_test = os.path.join(split_dir, "test_df.pkl")
    file_path_val = os.path.join(split_dir, "val_df.pkl")


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
    # Preserve Standard/Early outputs side-by-side.
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
