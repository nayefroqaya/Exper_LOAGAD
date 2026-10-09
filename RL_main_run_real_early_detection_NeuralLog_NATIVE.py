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
# RL-style test reporting for static/non-early-detection baselines
# Minimal addition: does NOT change model training or prediction.
# ============================================================

def _rl_as_fraction(x):
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def _rl_binary_labels(labels):
    out = []
    for label in labels:
        if isinstance(label, (list, tuple, np.ndarray)):
            label = np.max(label)

        if isinstance(label, (bool, np.bool_)):
            out.append(1 if bool(label) else 0)
            continue

        if isinstance(label, (int, float, np.integer, np.floating)):
            try:
                out.append(0 if float(label) == 0.0 else 1)
                continue
            except Exception:
                pass

        s = str(label).strip().lower()
        if s in {"0", "0.0", "-", "normal", "n", "false", "benign"}:
            out.append(0)
        else:
            out.append(1)

    return np.asarray(out, dtype=int)


def _rl_safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _rl_f1(p, r):
    return (2.0 * p * r / (p + r)) if (p + r) else 0.0


def _rl_predict_single_unsupervised_prefix(
    trainer,
    prefix_sequence,
    true_label,
    vocab,
    args,
    device,
    logger,
):
    """
    Run the EXISTING unsupervised prediction rule on one sequence prefix.

    Returns:
        1 -> predicted anomaly
        0 -> predicted normal
        None -> prefix is too short / cannot be evaluated
    """
    prefix_data = [(list(prefix_sequence), true_label)]

    try:
        sequentials, quantitatives, semantics, labels, idxs, session_labels = sliding_window(
            prefix_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=False,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            is_unsupervised=True,
            logger=logger,
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
        is_unsupervised=True,
        remove_duplicates=False,
    )

    if len(prefix_dataset) == 0:
        return None

    # One original sequence is being evaluated here.
    try:
        acc_p, f1_p, pre_p, rec_p = trainer.predict_unsupervised(
            prefix_dataset,
            session_labels,
            topk=args.topk,
            device=device,
            is_valid=False,
            num_sessions=[1],
        )
    except Exception:
        return None

    true_binary = int(_rl_binary_labels([true_label])[0])

    # For a single anomalous sequence:
    # recall = 1 iff it was detected as anomaly.
    if true_binary == 1:
        return 1 if _rl_as_fraction(rec_p) >= 0.5 else 0

    # For a single normal sequence:
    # accuracy = 1 iff it was predicted normal.
    return 0 if _rl_as_fraction(acc_p) >= 0.5 else 1



def _rl_predict_single_supervised_native_prefix(
    trainer,
    prefix_sequence,
    true_label,
    vocab,
    args,
    device,
    logger,
):
    """
    Supervised native-window baseline (LogRobust / NeuralLog):
    evaluate one complete native window with the existing supervised
    prediction path. Training/model parameters are unchanged.
    """
    prefix_data = [(list(prefix_sequence), true_label)]

    try:
        sequentials, quantitatives, semantics, labels, idxs, session_labels = sliding_window(
            prefix_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=False,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            is_unsupervised=False,
            logger=logger,
        )
    except Exception:
        return None

    enabled_feature_lengths = []
    if getattr(args, "sequential", False) and sequentials is not None:
        enabled_feature_lengths.append(len(sequentials))
    if getattr(args, "quantitative", False) and quantitatives is not None:
        enabled_feature_lengths.append(len(quantitatives))
    if getattr(args, "semantic", False) and semantics is not None:
        enabled_feature_lengths.append(len(semantics))

    if not enabled_feature_lengths or max(enabled_feature_lengths) == 0:
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

    # Prefixes are evaluated only for truly anomalous test sequences here.
    # With one positive sample, recall=1 means the supervised model predicted anomaly.
    return 1 if _rl_as_fraction(rec_p) >= 0.5 else 0


def compute_real_early_detection_metrics(
    trainer,
    test_data,
    num_sessions,
    vocab,
    args,
    device,
    logger,
):
    """
    REAL prefix-based early detection.

    DeepLog / LogAnomaly:
      - use the existing unsupervised next-event prediction path.

    LogRobust / NeuralLog:
      - use the existing supervised prediction path only on complete native windows.

    For every anomalous test sequence, record the first prefix that the
    trained baseline predicts as anomalous. No retraining is performed.

    Multiplicity from num_sessions is preserved so duplicate test sessions
    receive the same weighting as the original full-sequence evaluation.
    """
    stride = max(1, int(getattr(args, "early_eval_stride", 1)))

    total_anomalies = 0
    detected_anomalies = 0
    weighted_detection_steps = []
    weighted_detection_ratios = []

    anomaly_unique = 0

    print("\n" + "=" * 60)
    print("REAL PREFIX-BASED EARLY DETECTION EVALUATION")
    print("=" * 60)
    print("Model               :", args.model_name)
    print("Prefix stride        :", stride)
    if args.model_name in {"LogRobust", "NeuralLog"}:
        print("Method               : first COMPLETE native window flagged anomalous")
    else:
        print("Method               : first prefix flagged anomalous")
    if args.model_name in {"LogRobust", "NeuralLog"}:
        print(f"Prediction mode       : supervised {args.model_name}")
    else:
        print("Prediction mode       : unsupervised next-event")
    print("This uses the trained baseline model; no retraining is performed.")

    for i, (sequence, true_label) in enumerate(test_data):
        true_binary = int(_rl_binary_labels([true_label])[0])
        if true_binary != 1:
            continue

        anomaly_unique += 1
        multiplicity = int(num_sessions[i]) if i < len(num_sessions) else 1
        multiplicity = max(1, multiplicity)
        total_anomalies += multiplicity

        seq = list(sequence)
        seq_len = len(seq)
        if seq_len == 0:
            continue

        if args.model_name in {"LogRobust", "NeuralLog"}:
            # IMPORTANT:
            # Evaluate supervised sequence classifiers only when a COMPLETE native input window exists.
            # Do not create artificial early decisions by padding 1..119-event prefixes.
            # Therefore, with history_size=120 and a 120-event sequence,
            # the first valid LogRobust/NeuralLog decision is at event 120.
            first_evaluable = int(getattr(args, "history_size", 120))
            if seq_len < first_evaluable:
                continue
        else:
            # DeepLog/LogAnomaly need history_size context events plus
            # one next event for next-event prediction.
            first_evaluable = int(getattr(args, "history_size", 1)) + 1
            if seq_len < first_evaluable:
                continue

        first_detection_step = None

        # Exact scan when stride=1.
        candidate_steps = list(range(first_evaluable, seq_len + 1, stride))
        if not candidate_steps or candidate_steps[-1] != seq_len:
            candidate_steps.append(seq_len)

        for prefix_len in candidate_steps:
            if args.model_name in {"LogRobust", "NeuralLog"}:
                pred = _rl_predict_single_supervised_native_prefix(
                    trainer=trainer,
                    prefix_sequence=seq[:prefix_len],
                    true_label=true_label,
                    vocab=vocab,
                    args=args,
                    device=device,
                    logger=logger,
                )
            else:
                pred = _rl_predict_single_unsupervised_prefix(
                    trainer=trainer,
                    prefix_sequence=seq[:prefix_len],
                    true_label=true_label,
                    vocab=vocab,
                    args=args,
                    device=device,
                    logger=logger,
                )

            if pred == 1:
                first_detection_step = prefix_len
                break

        if first_detection_step is not None:
            detected_anomalies += multiplicity
            ratio = float(first_detection_step) / float(seq_len)

            weighted_detection_steps.extend(
                [float(first_detection_step)] * multiplicity
            )
            weighted_detection_ratios.extend(
                [ratio] * multiplicity
            )

        if anomaly_unique % 100 == 0:
            print(
                f"Processed anomalous unique sequences: {anomaly_unique} | "
                f"weighted detected: {detected_anomalies}/{total_anomalies}"
            )

    if weighted_detection_steps:
        avg_step = float(np.mean(weighted_detection_steps))
        avg_ratio = float(np.mean(weighted_detection_ratios))
        detected_ratios = np.asarray(weighted_detection_ratios, dtype=float)
        edr25 = float(np.sum(detected_ratios <= 0.25) / max(1, total_anomalies))
        edr50 = float(np.sum(detected_ratios <= 0.50) / max(1, total_anomalies))
        edr75 = float(np.sum(detected_ratios <= 0.75) / max(1, total_anomalies))
    else:
        avg_step = 0.0
        avg_ratio = 0.0
        edr25 = 0.0
        edr50 = 0.0
        edr75 = 0.0

    coverage = _rl_safe_div(detected_anomalies, total_anomalies)

    result = {
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "detection_coverage": float(coverage),
        "avg_detection_step": float(avg_step),
        "avg_detection_ratio": float(avg_ratio),
        "edr25": float(edr25),
        "edr50": float(edr50),
        "edr75": float(edr75),
        "stride": int(stride),
    }

    print("\nREAL early-detection result:")
    print(f"Total anomalies       : {result['total_anomalies']}")
    print(f"Detected anomalies    : {result['detected_anomalies']}")
    print(f"Detection coverage    : {result['detection_coverage']:.4f}")
    print(f"Avg detection step    : {result['avg_detection_step']:.4f}")
    print(f"Avg detection ratio   : {result['avg_detection_ratio']:.4f}")
    print(f"EDR@25                : {result['edr25']:.4f}")
    print(f"EDR@50                : {result['edr50']:.4f}")
    print(f"EDR@75                : {result['edr75']:.4f}")

    return result


def print_rl_style_test_metrics(
    args,
    session_labels,
    precision,
    recall,
    f1,
    real_early=None,
):
    """
    Print the same RL-style summary, using REAL prefix-based early metrics
    when real_early is provided.
    """
    y_true = _rl_binary_labels(session_labels)

    total = int(len(y_true))
    total_anomalies = int(np.sum(y_true == 1))
    total_normals = int(np.sum(y_true == 0))

    precision = _rl_as_fraction(precision)
    recall = _rl_as_fraction(recall)
    f1 = _rl_as_fraction(f1)

    tp = int(round(recall * total_anomalies))
    tp = max(0, min(tp, total_anomalies))
    fn = total_anomalies - tp

    if precision > 0:
        fp = int(round(tp / precision - tp))
    else:
        fp = 0

    fp = max(0, min(fp, total_normals))
    tn = total_normals - fp

    normal_precision = _rl_safe_div(tn, tn + fn)
    normal_recall = _rl_safe_div(tn, tn + fp)
    normal_f1 = _rl_f1(normal_precision, normal_recall)

    anomaly_precision = _rl_safe_div(tp, tp + fp)
    anomaly_recall = _rl_safe_div(tp, tp + fn)
    anomaly_f1 = _rl_f1(anomaly_precision, anomaly_recall)

    accuracy = _rl_safe_div(tp + tn, total)

    macro_p = 0.5 * (normal_precision + anomaly_precision)
    macro_r = 0.5 * (normal_recall + anomaly_recall)
    macro_f1 = 0.5 * (normal_f1 + anomaly_f1)

    weighted_p = _rl_safe_div(
        normal_precision * total_normals + anomaly_precision * total_anomalies,
        total,
    )
    weighted_r = _rl_safe_div(
        normal_recall * total_normals + anomaly_recall * total_anomalies,
        total,
    )
    weighted_f1 = _rl_safe_div(
        normal_f1 * total_normals + anomaly_f1 * total_anomalies,
        total,
    )

    dataset_name = getattr(
        args,
        "target_dataset",
        getattr(args, "dataset_name", "dataset")
    )

    # Use measured prefix metrics if available.
    if real_early is not None:
        early_total = real_early["total_anomalies"]
        detected_anomalies = real_early["detected_anomalies"]
        detection_coverage = real_early["detection_coverage"]
        avg_detection_step = real_early["avg_detection_step"]
        avg_detection_ratio = real_early["avg_detection_ratio"]
        edr25 = real_early["edr25"]
        edr50 = real_early["edr50"]
        edr75 = real_early["edr75"]
        early_method = (
            f"REAL prefix evaluation for {args.model_name}: "
            f"first prefix flagged anomalous (stride={real_early['stride']})."
        )
    else:
        early_total = total_anomalies
        detected_anomalies = tp
        detection_coverage = _rl_safe_div(detected_anomalies, total_anomalies)
        avg_detection_step = 0.0
        avg_detection_ratio = 0.0
        edr25 = edr50 = edr75 = 0.0
        early_method = "Early metrics were not evaluated."

    fp_unit_cost = float(getattr(args, "fp_unit_cost", 10.0))
    fn_unit_cost = float(getattr(args, "fn_unit_cost", 20.0))
    delay_unit_cost = float(getattr(args, "delay_unit_cost", 5.0))

    fp_cost = fp * fp_unit_cost
    fn_cost = fn * fn_unit_cost
    # Keep the same cost convention used in your RL comparison.
    delay_cost = detected_anomalies * delay_unit_cost

    cm_text = f"[[{tn:5d} {fp:5d}]\n [{fn:5d} {tp:5d}]]"

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("FINAL TEST METRICS")
    lines.append("=" * 60)
    lines.append(
        f"{dataset_name}: P={anomaly_precision:.4f}, "
        f"R={anomaly_recall:.4f}, F1={anomaly_f1:.4f}"
    )
    lines.append(cm_text)
    lines.append("")

    lines.append(f"[Classification Metrics - Anomaly Class]  # {dataset_name}")
    lines.append("Positive class        : 1 = anomaly")
    lines.append(f"Precision             : {anomaly_precision:.4f}")
    lines.append(f"Recall / TPR          : {anomaly_recall:.4f}")
    lines.append(f"F1-score              : {anomaly_f1:.4f}")
    lines.append("")

    lines.append("[Classification Report - Class 0 and Class 1]")
    lines.append("Class 0               : Normal")
    lines.append("Class 1               : Anomaly")
    lines.append("")
    lines.append("              precision    recall  f1-score   support")
    lines.append("")
    lines.append(
        f"  Normal (0)     {normal_precision:.4f}    {normal_recall:.4f}    "
        f"{normal_f1:.4f}  {total_normals:8d}"
    )
    lines.append(
        f" Anomaly (1)     {anomaly_precision:.4f}    {anomaly_recall:.4f}    "
        f"{anomaly_f1:.4f}  {total_anomalies:8d}"
    )
    lines.append("")
    lines.append(
        f"    accuracy                         {accuracy:.4f}  {total:8d}"
    )
    lines.append(
        f"   macro avg     {macro_p:.4f}    {macro_r:.4f}    "
        f"{macro_f1:.4f}  {total:8d}"
    )
    lines.append(
        f"weighted avg     {weighted_p:.4f}    {weighted_r:.4f}    "
        f"{weighted_f1:.4f}  {total:8d}"
    )
    lines.append("")

    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(cm_text)
    lines.append(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    lines.append("")

    lines.append("[Early Detection Metrics]")
    lines.append(early_method)
    lines.append(f"Total anomalies       : {early_total}")
    lines.append(f"Detected anomalies    : {detected_anomalies}")
    lines.append(f"Detection coverage    : {detection_coverage:.4f}")
    lines.append(f"Avg detection step    : {avg_detection_step:.4f}")
    lines.append(f"Avg detection ratio   : {avg_detection_ratio:.4f}")
    lines.append(f"EDR@25                : {edr25:.4f}")
    lines.append(f"EDR@50                : {edr50:.4f}")
    lines.append(f"EDR@75                : {edr75:.4f}")
    lines.append("")

    lines.append("[Cost-Sensitive Metrics]")
    lines.append(f"False-positive cost   : {fp_cost:.4f}")
    lines.append(f"False-negative cost   : {fn_cost:.4f}")
    lines.append(f"Delay cost            : {delay_cost:.4f}")
    lines.append("")
    lines.append(
        "[Note] Early-detection values above are measured from actual prefix "
        "inference, not fixed static assumptions."
    )

    report = "\n".join(lines)
    print(report)

    try:
        report_path = os.path.join(
            args.output_dir,
            f"RL_{getattr(args, 'model_name', 'model')}_{dataset_name}_test_metrics.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
        print("Saved RL-style metrics:", os.path.abspath(report_path))
    except Exception as exc:
        print("[WARNING] Could not save RL-style metrics file:", exc)


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

    # ------------------------------------------------------------
    # REAL early-detection evaluation.
    # DeepLog / LogAnomaly -> existing unsupervised prediction path
    # LogRobust / NeuralLog -> existing supervised prediction path
    # No retraining or model change.
    # ------------------------------------------------------------
    real_early = None
    if (
        (is_unsupervised or args.model_name in {"LogRobust", "NeuralLog"})
        and bool(getattr(args, "real_early_detection", True))
    ):
        real_early = compute_real_early_detection_metrics(
            trainer=trainer,
            test_data=data,
            num_sessions=num_sessions,
            vocab=vocab,
            args=args,
            device=device,
            logger=logger,
        )

    print_rl_style_test_metrics(
        args=args,
        session_labels=session_labels,
        precision=pre,
        recall=rec,
        f1=f1,
        real_early=real_early,
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
        output_subdir = f"{args.output_dir}/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
    else:
        output_subdir = f"{args.output_dir}/{args.dataset_name}/session"

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

    # In-domain dataset paths come from the YAML:
    #   data_dir: ../LWADLS/datasets/SP_150MB_ratio
    #   dataset_name: SP_150MB_ratio
    split_dir = os.path.join(
        args.data_dir,
        f"1_{args.dataset_name}_Splitted_Datasets"
    )

    file_path_train = os.path.join(split_dir, "train_df.pkl")
    file_path_val   = os.path.join(split_dir, "val_df.pkl")
    file_path_test  = os.path.join(split_dir, "test_df.pkl")

    print(f"[LOAD] {args.dataset_name} train: {file_path_train}")
    print(f"[LOAD] {args.dataset_name} val  : {file_path_val}")
    print(f"[LOAD] {args.dataset_name} test : {file_path_test}")


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

    if os.path.isdir(output_dir):
        for item in os.listdir(output_dir):
            path = os.path.join(output_dir, item)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
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
