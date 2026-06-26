import os
import os
import time   # ✅ ADD THIS
import pickle
from collections import Counter
import pickle
from collections import Counter
import colorama
import shutil
import json

#<<<<<<< HEAD
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
#=======
import pandas as pd
#>>>>>>> 6198c6c (update dataset portion)
from random import seed

import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
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
    metrics = None

    if is_unsupervised:
        logger.info(f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")
        acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                         session_labels,
                                                         topk=args.topk,
                                                         device=device,
                                                         is_valid=False,
                                                         num_sessions=num_sessions)
    else:
        # ------------------------------------------------------------
        # Supervised models: calculate AUROC/AUPRC from REAL model
        # probabilities instead of reporting N/A.
        # ------------------------------------------------------------
        try:
            y_true, y_pred, y_score = predict_supervised_with_real_probabilities(
                trainer=trainer,
                test_dataset=test_dataset,
                session_labels=session_labels,
                device=device,
                batch_size=args.batch_size,
            )

            metrics = compute_static_metrics_from_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                history_size=args.history_size,
                fp_unit_cost=10.0,
                fn_unit_cost=20.0,
                delay_unit_cost=5.0,
            )

            acc = metrics["accuracy"]
            f1 = metrics["f1_score"]
            pre = metrics["precision"]
            rec = metrics["recall"]

        except Exception as e:
            logger.warning(
                "Could not extract real probabilities for AUROC/AUPRC. "
                "Falling back to trainer.predict_supervised() summary metrics. "
                f"Reason: {e}"
            )
            acc, f1, pre, rec = trainer.predict_supervised(test_dataset,
                                                           session_labels,
                                                           device=device)

    test_time_min = (time.time() - start_test_time) / 60  # in minutes
    logger.info(f"Testing completed in {test_time_min:.2f} minutes")
    logger.info(f"Training completed in {train_time_min:.2f} minutes")

    logger.info(f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # If the model is unsupervised, or if probability extraction failed,
    # we still save all available metrics. AUROC/AUPRC stay N/A in this fallback.
    if metrics is None:
        metrics = compute_static_metrics_from_summary(
            session_labels=session_labels,
            accuracy=acc,
            precision=pre,
            recall=rec,
            f1=f1,
            history_size=args.history_size,
            fp_unit_cost=10.0,
            fn_unit_cost=20.0,
            delay_unit_cost=5.0,
        )

    save_static_metrics_report(
        metrics=metrics,
        output_dir=args.output_dir,
        model_name=args.model_name
    )

    return train_time_min, test_time_min, acc, f1, pre, rec




# ============================================================
# Full static-baseline metrics for normal/static approaches
# ============================================================

def _as_fraction(x):
    """Convert 85.0-style percentages to 0.85-style fractions when needed."""
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def _binary_label_array(labels):
    """Convert label values to 0=normal, 1=anomaly."""
    y = []
    for label in labels:
        if isinstance(label, (list, tuple, np.ndarray)):
            label = np.max(label)
        if label in [1, "1", "Anomaly", "anomaly", "Abnormal", "abnormal"]:
            y.append(1)
        else:
            y.append(0)
    return np.asarray(y, dtype=int)




def predict_supervised_with_real_probabilities(
    trainer,
    test_dataset,
    session_labels,
    device,
    batch_size=1024,
):
    """
    Run the trained supervised model on the test set and return real anomaly
    probabilities for AUROC/AUPRC.

    Returns:
        y_true  : true labels, 0=normal, 1=anomaly
        y_pred  : predicted labels, 0=normal, 1=anomaly
        y_score : real probability/score of anomaly class
    """
    model = trainer.model
    model.eval()
    model.to(device)

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true = _binary_label_array(session_labels)
    all_scores = []
    all_preds = []

    def move_to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(v) for v in obj)
        return obj

    def remove_label_fields(batch_dict):
        label_keys = {
            "label", "labels", "Label", "Labels",
            "y", "target", "targets",
            "idx", "idxs", "index", "indices",
        }
        return {k: v for k, v in batch_dict.items() if k not in label_keys}

    def forward_model(batch):
        """
        Supports common LogADEmpirical/PyTorch batch styles:
        - dict batch: model(batch) or model(**inputs)
        - tuple/list batch: model(*inputs)
        - tensor batch: model(batch)
        """
        batch = move_to_device(batch)

        if isinstance(batch, dict):
            inputs_no_labels = remove_label_fields(batch)

            # Many LogADEmpirical models expect one dictionary argument.
            try:
                return model(inputs_no_labels)
            except Exception:
                pass

            # Some PyTorch models expect keyword inputs.
            try:
                return model(**inputs_no_labels)
            except Exception:
                pass

            # Last fallback: pass the original batch including labels/idxs.
            return model(batch)

        if isinstance(batch, (list, tuple)):
            items = list(batch)

            # If the last item looks like labels/ids, remove it for forward pass.
            inputs = items
            if len(items) > 1 and torch.is_tensor(items[-1]) and items[-1].dim() <= 1:
                inputs = items[:-1]

            try:
                return model(*inputs)
            except Exception:
                if len(inputs) == 1:
                    return model(inputs[0])
                return model(inputs)

        return model(batch)

    def extract_logits(output):
        """Extract logits from tensor, dict, tuple, or list model outputs."""
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ["logits", "output", "outputs", "scores", "pred", "prediction"]:
                value = output.get(key)
                if torch.is_tensor(value):
                    return value

        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value) and value.dim() >= 2:
                    return value
            for value in output:
                if torch.is_tensor(value):
                    return value

        raise RuntimeError("Could not extract logits from model output.")

    def logits_to_anomaly_score_and_pred(logits):
        """
        Convert logits to anomaly scores and predicted labels.
        Supports:
        - [batch, 2] or [batch, n_class] classification logits
        - [batch, 1] binary logits
        - [batch] binary logits
        - [batch, seq_len, n_class] by taking the last time step
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if logits.dim() == 2 and logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)
            score = probs[:, 1]
            pred = torch.argmax(probs, dim=1)
            return score, pred

        if logits.dim() == 2 and logits.shape[1] == 1:
            score = torch.sigmoid(logits[:, 0])
            pred = (score >= 0.5).long()
            return score, pred

        if logits.dim() == 1:
            score = torch.sigmoid(logits)
            pred = (score >= 0.5).long()
            return score, pred

        raise RuntimeError(f"Unsupported logits shape for probability extraction: {tuple(logits.shape)}")

    with torch.no_grad():
        for batch in loader:
            output = forward_model(batch)
            logits = extract_logits(output)
            score, pred = logits_to_anomaly_score_and_pred(logits)

            all_scores.append(score.detach().cpu().numpy())
            all_preds.append(pred.detach().cpu().numpy())

    y_score = np.concatenate(all_scores, axis=0).astype(float)
    y_pred = np.concatenate(all_preds, axis=0).astype(int)

    # Safety alignment in case the dataset removes duplicates or changes length.
    min_len = min(len(y_true), len(y_pred), len(y_score))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    y_score = y_score[:min_len]

    return y_true, y_pred, y_score


def compute_static_metrics_from_predictions(
    y_true,
    y_pred,
    y_score,
    history_size=120,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """
    Compute metrics from real model predictions and real anomaly probabilities.

    AUROC and AUPRC are calculated from y_score, not from hard labels.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    total = int(len(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_anomalies = int(tp + fn)
    detected_anomalies = int(tp)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    # Real AUROC/AUPRC from anomaly probabilities/scores.
    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    else:
        auroc = None
        auprc = None

    detection_coverage = recall

    if detected_anomalies > 0:
        average_detection_step = float(history_size)
        average_detection_ratio = 1.0
        median_detection_ratio = 1.0
    else:
        average_detection_step = 0.0
        average_detection_ratio = 0.0
        median_detection_ratio = 0.0

    edr_25 = 0.0
    edr_50 = 0.0
    edr_75 = 0.0
    alert_rate = float(np.mean(y_pred == 1))

    fp_total_cost = float(fp * fp_unit_cost)
    fn_total_cost = float(fn * fn_unit_cost)
    delay_total_cost = float(tp * delay_unit_cost)
    total_cost = fp_total_cost + fn_total_cost + delay_total_cost
    average_cost_per_sequence = total_cost / max(1, total)

    # Static baselines do not have a true RL environment reward.
    # This avoids N/A by reporting a cost-based reward approximation.
    average_reward = -average_cost_per_sequence

    return {
        "num_sequences": int(total),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity_tnr": float(specificity),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(mcc),
        "auroc": None if auroc is None else float(auroc),
        "auprc": None if auprc is None else float(auprc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "anomaly_detection_coverage": float(detection_coverage),
        "average_detection_step": float(average_detection_step),
        "average_detection_ratio": float(average_detection_ratio),
        "median_detection_ratio": float(median_detection_ratio),
        "edr_25": float(edr_25),
        "edr_50": float(edr_50),
        "edr_75": float(edr_75),
        "average_reward": float(average_reward),
        "alert_rate": float(alert_rate),
        "false_positive_unit_cost": float(fp_unit_cost),
        "false_negative_unit_cost": float(fn_unit_cost),
        "delay_unit_cost": float(delay_unit_cost),
        "false_positive_total_cost": float(fp_total_cost),
        "false_negative_total_cost": float(fn_total_cost),
        "delay_total_cost": float(delay_total_cost),
        "total_cost": float(total_cost),
        "average_cost_per_sequence": float(average_cost_per_sequence),
        "auroc_auprc_source": "real_model_probability_or_score",
        "average_reward_source": "negative_average_cost_per_sequence_static_approximation",
    }

def compute_static_metrics_from_summary(
    session_labels,
    accuracy,
    precision,
    recall,
    f1,
    history_size=120,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """
    Compute all available metrics for a static/full-sequence baseline.

    The model does not output step-wise alert timing. Therefore:
      Avg detection ratio = 1.0 for detected anomalies
      EDR@25/50/75 = 0.0
      Delay cost = TP * delay_unit_cost
    """
    y_true = _binary_label_array(session_labels)
    total = int(len(y_true))
    total_anomalies = int(np.sum(y_true == 1))
    total_normals = int(np.sum(y_true == 0))

    accuracy = _as_fraction(accuracy)
    precision = _as_fraction(precision)
    recall = _as_fraction(recall)
    f1 = _as_fraction(f1)

    # Infer confusion matrix from recall and precision.
    tp = int(round(recall * total_anomalies))
    tp = max(0, min(tp, total_anomalies))
    fn = total_anomalies - tp

    if precision > 0:
        fp = int(round((tp / precision) - tp))
    else:
        fp = 0

    fp = max(0, min(fp, total_normals))
    tn = total_normals - fp

    cm = np.asarray([[tn, fp], [fn, tp]], dtype=int)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    detected_anomalies = tp
    detection_coverage = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0

    if detected_anomalies > 0:
        average_detection_step = float(history_size)
        average_detection_ratio = 1.0
        median_detection_ratio = 1.0
    else:
        average_detection_step = None
        average_detection_ratio = None
        median_detection_ratio = None

    edr_25 = 0.0
    edr_50 = 0.0
    edr_75 = 0.0

    alert_rate = (tp + fp) / max(1, total)

    fp_total_cost = float(fp * fp_unit_cost)
    fn_total_cost = float(fn * fn_unit_cost)
    delay_total_cost = float(tp * delay_unit_cost)
    total_cost = fp_total_cost + fn_total_cost + delay_total_cost
    average_cost_per_sequence = total_cost / max(1, total)

    # Static baselines do not have a true RL environment reward.
    # This is a cost-based approximation so the report does not show N/A.
    average_reward = -average_cost_per_sequence

    return {
        "num_sequences": total,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity_tnr": float(specificity),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(mcc),
        "auroc": None,
        "auprc": None,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "anomaly_detection_coverage": float(detection_coverage),
        "average_detection_step": average_detection_step,
        "average_detection_ratio": average_detection_ratio,
        "median_detection_ratio": median_detection_ratio,
        "edr_25": float(edr_25),
        "edr_50": float(edr_50),
        "edr_75": float(edr_75),
        "average_reward": float(average_reward),
        "alert_rate": float(alert_rate),
        "false_positive_unit_cost": float(fp_unit_cost),
        "false_negative_unit_cost": float(fn_unit_cost),
        "delay_unit_cost": float(delay_unit_cost),
        "false_positive_total_cost": float(fp_total_cost),
        "false_negative_total_cost": float(fn_total_cost),
        "delay_total_cost": float(delay_total_cost),
        "total_cost": float(total_cost),
        "average_cost_per_sequence": float(average_cost_per_sequence),
        "auroc_auprc_source": "not_available_no_real_model_probability",
        "average_reward_source": "negative_average_cost_per_sequence_static_approximation",
    }


def format_static_metrics_report(metrics, model_name):
    auroc_text = "N/A" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
    auprc_text = "N/A" if metrics["auprc"] is None else f"{metrics['auprc']:.4f}"
    avg_reward_text = "N/A" if metrics["average_reward"] is None else f"{metrics['average_reward']:.4f}"
    avg_step_text = "N/A" if metrics["average_detection_step"] is None else f"{metrics['average_detection_step']:.4f}"
    avg_ratio_text = "N/A" if metrics["average_detection_ratio"] is None else f"{metrics['average_detection_ratio']:.4f}"
    median_ratio_text = "N/A" if metrics["median_detection_ratio"] is None else f"{metrics['median_detection_ratio']:.4f}"

    lines = []
    lines.append("#" * 80)
    lines.append(f"{model_name}: FINAL TEST METRICS")
    lines.append("#" * 80)
    lines.append(f"Number of sequences   : {metrics['num_sequences']}")
    lines.append("")
    lines.append("[Classification Metrics]")
    lines.append(f"Accuracy              : {metrics['accuracy']:.4f}")
    lines.append(f"Balanced Accuracy     : {metrics['balanced_accuracy']:.4f}")
    lines.append(f"Precision             : {metrics['precision']:.4f}")
    lines.append(f"Recall / TPR          : {metrics['recall']:.4f}")
    lines.append(f"Specificity / TNR     : {metrics['specificity_tnr']:.4f}")
    lines.append(f"F1-score              : {metrics['f1_score']:.4f}")
    lines.append(f"F2-score              : {metrics['f2_score']:.4f}")
    lines.append(f"FPR                   : {metrics['fpr']:.4f}")
    lines.append(f"FNR                   : {metrics['fnr']:.4f}")
    lines.append(f"MCC                   : {metrics['mcc']:.4f}")
    lines.append(f"AUROC                 : {auroc_text}")
    lines.append(f"AUPRC                 : {auprc_text}")
    lines.append(f"AUROC/AUPRC source    : {metrics.get('auroc_auprc_source', 'unknown')}")
    lines.append("")
    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(str(np.asarray(metrics["confusion_matrix"])))
    lines.append(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    lines.append("")
    lines.append("[Early Detection Metrics]")
    lines.append("This model is treated as a full-sequence classifier.")
    lines.append("Default assumption: detected anomalies are detected at the end of the sequence.")
    lines.append(f"Total anomalies       : {metrics['total_anomalies']}")
    lines.append(f"Detected anomalies    : {metrics['detected_anomalies']}")
    lines.append(f"Detection coverage    : {metrics['anomaly_detection_coverage']:.4f}")
    lines.append(f"Avg detection step    : {avg_step_text}")
    lines.append(f"Avg detection ratio   : {avg_ratio_text}")
    lines.append(f"Median detect. ratio  : {median_ratio_text}")
    lines.append(f"EDR@25%               : {metrics['edr_25']:.4f}")
    lines.append(f"EDR@50%               : {metrics['edr_50']:.4f}")
    lines.append(f"EDR@75%               : {metrics['edr_75']:.4f}")
    lines.append("")
    lines.append("[RL Metrics]")
    lines.append(f"Average reward        : {avg_reward_text}")
    lines.append(f"AvgReward source      : {metrics.get('average_reward_source', 'unknown')}")
    lines.append(f"Alert rate            : {metrics['alert_rate']:.4f}")
    lines.append("")
    lines.append("[Cost Metrics]")
    lines.append(f"FP unit cost          : {metrics['false_positive_unit_cost']:.4f}")
    lines.append(f"FN unit cost          : {metrics['false_negative_unit_cost']:.4f}")
    lines.append(f"Delay unit cost       : {metrics['delay_unit_cost']:.4f}")
    lines.append(f"FP total cost         : {metrics['false_positive_total_cost']:.4f}")
    lines.append(f"FN total cost         : {metrics['false_negative_total_cost']:.4f}")
    lines.append(f"Delay total cost      : {metrics['delay_total_cost']:.4f}")
    lines.append(f"Total cost            : {metrics['total_cost']:.4f}")
    lines.append(f"Avg cost / sequence   : {metrics['average_cost_per_sequence']:.4f}")
    lines.append("#" * 80)

    return "\n".join(lines)


def save_static_metrics_report(metrics, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model_name.replace(" ", "_")
    json_path = os.path.join(output_dir, f"{safe_model_name}_test_metrics.json")
    txt_path = os.path.join(output_dir, f"{safe_model_name}_test_metrics.txt")

    report = format_static_metrics_report(metrics, model_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    print(report)
    print("Saved metrics JSON:", os.path.abspath(json_path))
    print("Saved metrics TXT :", os.path.abspath(txt_path))

    return json_path, txt_path


def find_project_root():
    """
    Find the current project root by looking for the datasets folder.
    Outputs are saved under:
        <project_root>/datasets/<dataset>/<model_name>_results/
    """
    candidates = []
    cwd = os.path.abspath(os.getcwd())
    candidates.append(cwd)

    parent = cwd
    for _ in range(8):
        parent = os.path.dirname(parent)
        candidates.append(parent)

    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidates.append(script_dir)

    parent = script_dir
    for _ in range(8):
        parent = os.path.dirname(parent)
        candidates.append(parent)

    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "datasets")):
            return cand

    return cwd


def find_lwadls_data_root():
    """
    Find LWADLS/datasets. Preferred structure:
        ../../LWADLS/datasets
    You can also pass --data_root in the config/args if your framework supports it.
    """
    candidates = [
        os.path.abspath("../../LWADLS/datasets"),
        os.path.abspath("../LWADLS/datasets"),
        os.path.abspath("LWADLS/datasets"),
        os.path.abspath("/storage/home/roqaya/LWADLS/datasets"),
    ]

    cwd = os.path.abspath(os.getcwd())
    parent = cwd
    for _ in range(6):
        candidates.append(os.path.join(parent, "LWADLS", "datasets"))
        parent = os.path.dirname(parent)

    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    # Return preferred path even if not found, so missing-file errors are explicit.
    return os.path.abspath("../../LWADLS/datasets")

# ============================================================
# Cross-dataset helper functions
# These functions are used ONLY when CASE = "cross_dataset_with_fraction".
# The original in-domain code path is kept unchanged.
# ============================================================

DATASET_BASE_DIR = find_lwadls_data_root()


def get_dataset_paths(dataset_name, base_dir=DATASET_BASE_DIR):
    """
    Build train / val / test PKL paths using the LWADLS 3_* structure.

    Example:
        ../../LWADLS/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_train_df.pkl
        ../../LWADLS/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_val_df.pkl
        ../../LWADLS/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_test_df.pkl
    """
    split_dir = os.path.join(
        base_dir,
        dataset_name,
        f"1_{dataset_name}_Splitted_Datasets"
    )

    return {
        "train": os.path.join(split_dir, f"train_df.pkl"),
        "val": os.path.join(split_dir, f"val_df.pkl"),
        "test": os.path.join(split_dir, f"test_df.pkl"),
        "data_dir": os.path.join(base_dir, dataset_name),
    }

def fix_label_column_for_cross(df):
    """
    Same label logic as the in-domain code:
        drop Label
        rename Original_Label to Label

    This is safer for cross-dataset because some files may already be fixed.
    """
    df = df.copy()

    if "Original_Label" in df.columns:
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])
        df = df.rename(columns={"Original_Label": "Label"})

    return df


def load_dataset_for_cross(dataset_name, base_dir=DATASET_BASE_DIR):
    """
    Load one dataset for cross-dataset experiments.
    """
    paths = get_dataset_paths(dataset_name, base_dir=base_dir)

    df_train = pd.read_pickle(paths["train"])
    df_val = pd.read_pickle(paths["val"])
    df_test = pd.read_pickle(paths["test"])

    df_train = fix_label_column_for_cross(df_train)
    df_val = fix_label_column_for_cross(df_val)
    df_test = fix_label_column_for_cross(df_test)

    return df_train, df_val, df_test


def get_normal_target_rows(df):
    """
    Select only normal rows from target train.
    Supports numeric and string labels.
    """
    return df[
        (df["Label"] == 0) |
        (df["Label"] == "0") |
        (df["Label"] == "Normal") |
        (df["Label"] == "normal")
    ].copy()


def prepare_cross_dataset_data(source_datasets,
                               target_dataset,
                               target_normal_fraction=0.20,
                               random_seed=42,
                               base_dir=DATASET_BASE_DIR):
    """
    Cross-dataset setting:
        train = source train + fraction of target normal train
        val   = target val
        test  = target test untouched
    """
    if not source_datasets:
        raise ValueError("SOURCE_DATASETS cannot be empty for cross_dataset_with_fraction")

    source_train_list = []

    print("\n==============================")
    print("Cross-dataset experiment")
    print("==============================")
    print("Source datasets:", source_datasets)
    print("Target dataset:", target_dataset)
    print("Target normal fraction:", target_normal_fraction)

    for source_name in source_datasets:
        src_train, src_val, src_test = load_dataset_for_cross(source_name, base_dir=base_dir)
        source_train_list.append(src_train)

        print("\nLoaded source dataset:", source_name)
        print("Source train used:", len(src_train))
        print("Source val not used:", len(src_val))
        print("Source test not used:", len(src_test))
        print("Source train labels:")
        print(src_train["Label"].value_counts())

    source_train = pd.concat(source_train_list, ignore_index=True)

    target_train, target_val, target_test = load_dataset_for_cross(target_dataset, base_dir=base_dir)

    target_normal_train = get_normal_target_rows(target_train)
    number_to_take = int(len(target_normal_train) * target_normal_fraction)

    if number_to_take > 0:
        target_normal_sample = target_normal_train.sample(
            n=number_to_take,
            random_state=random_seed
        )
    else:
        target_normal_sample = target_normal_train.iloc[0:0].copy()

    df_train = pd.concat([source_train, target_normal_sample], ignore_index=True)
    df_val = target_val
    df_test = target_test

    print("\n==============================")
    print("Final cross-dataset splits")
    print("==============================")
    print("Final train = source train + target normal fraction:", len(df_train))
    print("Final val = target val:", len(df_val))
    print("Final test = target test untouched:", len(df_test))
    print("Target normal available:", len(target_normal_train))
    print("Target normal used:", len(target_normal_sample))

    print("\nFinal train labels:")
    print(df_train["Label"].value_counts())
    print("\nTarget val labels:")
    print(df_val["Label"].value_counts())
    print("\nTarget test labels:")
    print(df_test["Label"].value_counts())

    return df_train, df_val, df_test


def merge_cross_dataset_embeddings(source_datasets,
                                   target_dataset,
                                   embeddings_name,
                                   output_dir,
                                   base_dir=DATASET_BASE_DIR):
    """
    Merge embedding JSON files from all source datasets and the target dataset.

    The original build_vocab() reads only one file:
        os.path.join(data_dir, embeddings)

    Therefore, for cross-dataset experiments we create one merged embedding JSON
    and pass its folder/file name to build_vocab().
    """
    dataset_names = list(source_datasets) + [target_dataset]
    merged_embeddings = {}

    print("\n==============================")
    print("Merging embeddings for cross-dataset")
    print("==============================")

    for dataset_name in dataset_names:
        emb_path = os.path.join(base_dir, dataset_name, embeddings_name)

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")

        with open(emb_path, "r") as f:
            one_embedding = json.load(f)

        before = len(merged_embeddings)
        merged_embeddings.update(one_embedding)
        after = len(merged_embeddings)

        print(f"Loaded embeddings from {dataset_name}: {len(one_embedding)} entries")
        print(f"Merged size: {before} -> {after}")

    merged_dir = os.path.join(output_dir, "cross_merged_embeddings")
    os.makedirs(merged_dir, exist_ok=True)

    merged_file = "merged_" + "_".join(dataset_names) + "_" + embeddings_name
    merged_path = os.path.join(merged_dir, merged_file)

    with open(merged_path, "w") as f:
        json.dump(merged_embeddings, f)

    print("Merged embedding file saved to:", merged_path)

    return merged_dir, merged_file

def run(args):
    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Experiment settings
    # For your current working one-dataset run, keep CASE = "in_domain".
    # For cross-dataset, change CASE and SOURCE_DATASETS only.
    # ============================================================

    # You can set these in YAML if needed:
    #   case: in_domain
    #   source_datasets: ["BGL", "TH_1G", "SP_150MB_ratio"]
    #   target_normal_fraction: 0.20
    CASE = getattr(args, "case", "in_domain")
    TARGET_DATASET = args.dataset_name
    SOURCE_DATASETS = getattr(args, "source_datasets", ["BGL", "TH_1G", "SP_150MB_ratio"])
    TARGET_NORMAL_FRACTION = float(getattr(args, "target_normal_fraction", 0.20))
    RANDOM_SEED = int(getattr(args, "random_seed", 42))

    PROJECT_ROOT = find_project_root()
    LOCAL_DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
    local_dataset_output_root = os.path.join(
        LOCAL_DATASETS_ROOT,
        TARGET_DATASET,
        f"{args.model_name}_results"
    )
    os.makedirs(local_dataset_output_root, exist_ok=True)

    print("\n==============================")
    print("Resolved paths")
    print("==============================")
    print("Project root:", PROJECT_ROOT)
    print("LWADLS data root:", DATASET_BASE_DIR)
    print("Target dataset:", TARGET_DATASET)
    print("Local output root:", os.path.abspath(local_dataset_output_root))
    print("==============================\n")

    if CASE == "cross_dataset_with_fraction":
        source_name = "_".join(SOURCE_DATASETS)
        fraction_name = str(TARGET_NORMAL_FRACTION).replace(".", "p")
        experiment_name = f"cross_source_{source_name}_target_{TARGET_DATASET}_frac_{fraction_name}"

        if args.grouping == "sliding":
            output_subdir = os.path.join(local_dataset_output_root, experiment_name, TARGET_DATASET, f"sliding_W{args.window_size}_S{args.step_size}_C{args.is_chronological}")
        else:
            output_subdir = os.path.join(local_dataset_output_root, experiment_name, TARGET_DATASET, "session")
    else:
        # ============================================================
        # Original in-domain output directory code: unchanged
        # ============================================================
        #if args.grouping == "sliding":
        #    args.output_dir = f"{args.output_dir}/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"
        #else:
        #    args.output_dir = f"{args.output_dir}/{args.dataset_name}/session/train{args.train_size}"
        if args.grouping == "sliding":
            output_subdir = os.path.join(local_dataset_output_root, f"sliding_W{args.window_size}_S{args.step_size}_C{args.is_chronological}")
        else:
            output_subdir = os.path.join(local_dataset_output_root, "session")

    os.makedirs(output_subdir, exist_ok=True)
    args.output_dir = output_subdir

    # ============================================================
    # Data loading
    # in_domain: original code unchanged
    # cross_dataset_with_fraction: new separate code path
    # ============================================================

    if CASE == "cross_dataset_with_fraction":
        df_train, df_val, df_test = prepare_cross_dataset_data(
            source_datasets=SOURCE_DATASETS,
            target_dataset=TARGET_DATASET,
            target_normal_fraction=TARGET_NORMAL_FRACTION,
            random_seed=RANDOM_SEED,
            base_dir=DATASET_BASE_DIR
        )

        vocab_data_dir, vocab_embeddings = merge_cross_dataset_embeddings(
            source_datasets=SOURCE_DATASETS,
            target_dataset=TARGET_DATASET,
            embeddings_name=args.embeddings,
            output_dir=args.output_dir,
            base_dir=DATASET_BASE_DIR
        )

    else:   # in domain
        # first paper
        #file_path_train = 'dataset/HDFS/1_HDFS_Splitted_Datasets/train_df.pkl'
        #file_path_test = 'dataset/HDFS/1_HDFS_Splitted_Datasets/test_df.pkl'
        #file_path_val = 'dataset/HDFS/1_HDFS_Splitted_Datasets/val_df.pkl'

        # second paper
        #file_path_train = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl'
        #file_path_test = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl'
        #file_path_val = '../NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl'

        # Third paper
        #file_path_train = '../LWADLS/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/train_df.pkl'
        #file_path_test = '../LWADLS/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/test_df.pkl'
        #file_path_val = '../LWADLS/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/val_df.pkl'


        paths = get_dataset_paths(TARGET_DATASET, base_dir=DATASET_BASE_DIR)

        print("\n==============================")
        print("In-domain PKL paths")
        print("==============================")
        for split_name in ["train", "val", "test"]:
            print(f"{split_name}: {os.path.abspath(paths[split_name])} | exists={os.path.exists(paths[split_name])}")
        print("==============================\n")

        missing_paths = [
            paths[split_name]
            for split_name in ["train", "val", "test"]
            if not os.path.exists(paths[split_name])
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Missing required PKL files:\n" + "\n".join(missing_paths)
            )

        df_train = pd.read_pickle(paths["train"])
        df_val = pd.read_pickle(paths["val"])
        df_test = pd.read_pickle(paths["test"])

        df_train = fix_label_column_for_cross(df_train)
        df_val = fix_label_column_for_cross(df_val)
        df_test = fix_label_column_for_cross(df_test)

        print("In-domain data loaded")
        df_train.info()
        df_val.info()
        df_test.info()

        vocab_data_dir = paths["data_dir"]
        vocab_embeddings = args.embeddings

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
                            vocab_data_dir,
                            train_path,
                            vocab_embeddings,
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

    # Do not delete previous outputs automatically.
    # Metrics and model files are saved under:
    #   datasets/<dataset>/<model_name>_results/
    output_dir = None
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
