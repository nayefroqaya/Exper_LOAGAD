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
import yaml
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
    return train_time_min, test_time_min, acc, f1, pre, rec



# ============================================================
# Cross-dataset helper functions
# These functions are used ONLY when CASE = "cross_dataset_with_fraction".
# The original in-domain code path is kept unchanged.
# ============================================================

DATASET_BASE_DIR = "../LWADLS/datasets"


def get_dataset_paths(dataset_name, base_dir=DATASET_BASE_DIR):
    """
    Build train / val / test paths for one dataset.

    Expected structure:
        base_dir / dataset_name / 1_dataset_name_Splitted_Datasets / train_df.pkl
        base_dir / dataset_name / 1_dataset_name_Splitted_Datasets / val_df.pkl
        base_dir / dataset_name / 1_dataset_name_Splitted_Datasets / test_df.pkl
    """
    split_dir = os.path.join(base_dir, dataset_name, f"1_{dataset_name}_Splitted_Datasets")

    return {
        "train": os.path.join(split_dir, "train_df.pkl"),
        "val": os.path.join(split_dir, "val_df.pkl"),
        "test": os.path.join(split_dir, "test_df.pkl"),
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

    # Print the ACTUAL files that will be opened.
    # These lines are the best verification that no wrong/hard-coded dataset is used.
    print(f"[LOAD] {dataset_name} train: {paths['train']}")
    print(f"[LOAD] {dataset_name} val  : {paths['val']}")
    print(f"[LOAD] {dataset_name} test : {paths['test']}")

    df_train = pd.read_pickle(paths["train"])
    df_val = pd.read_pickle(paths["val"])
    df_test = pd.read_pickle(paths["test"])

    df_train = fix_label_column_for_cross(df_train)
    df_val = fix_label_column_for_cross(df_val)
    df_test = fix_label_column_for_cross(df_test)

    return df_train, df_val, df_test


def get_normal_mask(df):
    """
    Identify NORMAL rows robustly across common log-label conventions.
    """
    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column, but none was found.")

    def is_normal_label(value):
        if value is False:
            return True

        if isinstance(value, (int, np.integer, float, np.floating)):
            try:
                if not pd.isna(value) and float(value) == 0.0:
                    return True
            except Exception:
                pass

        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {
                "0", "0.0", "normal", "n", "-", "false", "benign"
            }

        return False

    mask = df["Label"].map(is_normal_label)

    if int(mask.sum()) == 0:
        print("\n[DIAGNOSTIC] No normal rows were recognized.")
        print("[DIAGNOSTIC] Raw Label values and counts:")
        print(df["Label"].value_counts(dropna=False).head(30))

    return mask


def get_normal_target_rows(df):
    return df[get_normal_mask(df)].copy()


def _print_label_counts(title, df):
    print(f"\n{title}")
    if "Label" not in df.columns:
        print("No Label column found.")
        return
    print(df["Label"].value_counts(dropna=False))


def sample_target_fraction(target_train,
                           target_fraction=0.20,
                           fraction_mode="normal_only",
                           random_seed=42):
    """
    Select a fixed budget from TARGET TRAINING DATA.

    The budget is:
        int(len(target_train) * target_fraction)

    fraction_mode:
        "normal_only"
            Take that complete budget from NORMAL target-training rows only.

        "normal_anomaly"
            Take that complete budget from target training using stratified
            sampling, so both normal and anomaly rows are represented while
            approximately preserving the original class distribution.

    IMPORTANT:
        For both modes, target_fraction=0.20 means 20% of ALL target_train,
        not 20% of only the normal subset.
    """
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be in (0, 1].")

    if len(target_train) == 0:
        raise ValueError("Target training dataframe is empty.")

    n_take = int(len(target_train) * target_fraction)
    if n_take < 1:
        raise ValueError(
            f"target_fraction={target_fraction} gives 0 rows for "
            f"target_train size={len(target_train)}."
        )

    if fraction_mode == "normal_only":
        normal_rows = get_normal_target_rows(target_train)

        if len(normal_rows) < n_take:
            print("\nRaw target Label values and counts:")
            print(target_train["Label"].value_counts(dropna=False).head(30))
            raise ValueError(
                "Not enough recognized normal rows to create a normal-only "
                f"target fraction equal to {target_fraction:.2%}. "
                f"Need {n_take}, but only {len(normal_rows)} were recognized."
            )

        sampled = normal_rows.sample(
            n=n_take,
            random_state=random_seed
        ).copy()

    elif fraction_mode == "normal_anomaly":
        normal_mask = get_normal_mask(target_train)
        binary_strata = normal_mask.map(
            lambda is_normal: "normal" if is_normal else "anomaly"
        )

        print("\nTarget binary class counts before sampling:")
        print(binary_strata.value_counts(dropna=False))

        if binary_strata.nunique(dropna=False) < 2:
            raise ValueError(
                "normal_anomaly mode requires both normal and anomaly rows "
                "in target_train. Check the raw Label values."
            )

        sampled, _ = train_test_split(
            target_train,
            train_size=n_take,
            random_state=random_seed,
            shuffle=True,
            stratify=binary_strata
        )
        sampled = sampled.copy()

        sampled_binary = get_normal_mask(sampled).map(
            lambda is_normal: "normal" if is_normal else "anomaly"
        )

        print("\nSelected target fraction binary class counts:")
        print(sampled_binary.value_counts(dropna=False))

        if sampled_binary.nunique(dropna=False) < 2:
            raise ValueError(
                "The sampled target fraction does not contain both classes."
            )

    else:
        raise ValueError(
            "fraction_mode must be one of: 'normal_only', 'normal_anomaly'."
        )

    return sampled


def prepare_cross_dataset_data(source_datasets,
                               target_dataset,
                               target_fraction=0.20,
                               fraction_mode="normal_only",
                               random_seed=42,
                               base_dir=DATASET_BASE_DIR):
    """
    Cross-domain DeepLog experiment.

    Final splits:
        train = all SOURCE train rows + selected TARGET train fraction
        val   = TARGET validation split
        test  = TARGET test split, untouched

    fraction_mode:
        normal_only:
            target fraction budget = 20% of ALL target_train,
            but every selected target row must be normal.

        normal_anomaly:
            target fraction budget = 20% of ALL target_train,
            selected using stratified sampling from normal + anomaly rows.
    """
    if not source_datasets:
        raise ValueError("source_datasets cannot be empty for cross-domain runs.")

    if target_dataset in source_datasets:
        raise ValueError(
            f"Target dataset '{target_dataset}' is also present in source_datasets. "
            "Remove it from the source list for a true cross-domain experiment."
        )

    source_train_list = []

    for source_name in source_datasets:
        src_train, src_val, src_test = load_dataset_for_cross(
            source_name,
            base_dir=base_dir
        )
        source_train_list.append(src_train)

        print("\nLoaded source dataset:", source_name)
        print("Source train used:", len(src_train))
        print("Source val not used:", len(src_val))
        print("Source test not used:", len(src_test))
        _print_label_counts("Source train labels:", src_train)

    source_train = pd.concat(source_train_list, ignore_index=True)

    target_train, target_val, target_test = load_dataset_for_cross(
        target_dataset,
        base_dir=base_dir
    )

    target_sample = sample_target_fraction(
        target_train=target_train,
        target_fraction=target_fraction,
        fraction_mode=fraction_mode,
        random_seed=random_seed
    )

    df_train = pd.concat(
        [source_train, target_sample],
        ignore_index=True
    )
    df_val = target_val.copy()
    df_test = target_test.copy()

    print("\n==============================")
    print("Final cross-domain splits")
    print("==============================")
    print("Whole target train:", len(target_train))
    print(
        f"Requested target budget: {target_fraction:.2%} "
        f"= {int(len(target_train) * target_fraction)} rows"
    )
    print("Target rows actually used:", len(target_sample))
    print("Final train = source train + target fraction:", len(df_train))
    print("Final val = target val:", len(df_val))
    print("Final test = target test untouched:", len(df_test))

    _print_label_counts("Selected target fraction labels:", target_sample)
    _print_label_counts("Final train labels:", df_train)
    _print_label_counts("Target val labels:", df_val)
    _print_label_counts("Target test labels:", df_test)

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

    # ============================================================
    # experiment_case:
    #   in_domain
    #   cross_normal_only
    #   cross_normal_anomaly
    # ============================================================
    experiment_case = getattr(args, "experiment_case", "in_domain")
    target_dataset = getattr(args, "target_dataset", args.dataset_name)
    source_datasets = getattr(args, "source_datasets", [])
    target_fraction = float(getattr(args, "target_fraction", 0.20))
    random_seed = int(getattr(args, "random_seed", 42))
    dataset_base_dir = getattr(args, "dataset_base_dir", DATASET_BASE_DIR)

    valid_cases = {
        "in_domain",
        "cross_normal_only",
        "cross_normal_anomaly",
    }
    if experiment_case not in valid_cases:
        raise ValueError(
            f"Unknown experiment_case='{experiment_case}'. "
            f"Choose one of: {sorted(valid_cases)}"
        )

    # Downstream preprocessing should always see the target dataset name.
    args.dataset_name = target_dataset

    target_paths = get_dataset_paths(
        target_dataset,
        base_dir=dataset_base_dir
    )
    target_data_dir = target_paths["data_dir"]

    # ============================================================
    # Output directory
    # ============================================================
    base_output_dir = args.output_dir

    if experiment_case == "in_domain":
        experiment_name = f"in_domain_target_{target_dataset}"
    else:
        if not source_datasets:
            raise ValueError(
                "source_datasets must be provided for cross-domain experiments."
            )

        source_name = "_".join(source_datasets)
        fraction_name = str(target_fraction).replace(".", "p")
        experiment_name = (
            f"{experiment_case}_source_{source_name}_"
            f"target_{target_dataset}_frac_{fraction_name}"
        )

    if args.grouping == "sliding":
        output_subdir = (
            f"{base_output_dir}/{experiment_name}/{target_dataset}/sliding/"
            f"W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
        )
    else:
        output_subdir = (
            f"{base_output_dir}/{experiment_name}/{target_dataset}/session"
        )

    os.makedirs(output_subdir, exist_ok=True)
    args.output_dir = output_subdir

    # ============================================================
    # Data loading
    # ============================================================
    if experiment_case == "in_domain":
        # target train -> target validation -> target test
        df_train, df_val, df_test = load_dataset_for_cross(
            target_dataset,
            base_dir=dataset_base_dir
        )

        vocab_data_dir = target_data_dir
        vocab_embeddings = args.embeddings

        print("\nIn-domain split sizes")
        print("Train:", len(df_train))
        print("Validation:", len(df_val))
        print("Test:", len(df_test))
        _print_label_counts("In-domain train labels:", df_train)
        _print_label_counts("In-domain validation labels:", df_val)
        _print_label_counts("In-domain test labels:", df_test)

    else:
        fraction_mode = (
            "normal_only"
            if experiment_case == "cross_normal_only"
            else "normal_anomaly"
        )

        df_train, df_val, df_test = prepare_cross_dataset_data(
            source_datasets=source_datasets,
            target_dataset=target_dataset,
            target_fraction=target_fraction,
            fraction_mode=fraction_mode,
            random_seed=random_seed,
            base_dir=dataset_base_dir
        )

        vocab_data_dir, vocab_embeddings = merge_cross_dataset_embeddings(
            source_datasets=source_datasets,
            target_dataset=target_dataset,
            embeddings_name=args.embeddings,
            output_dir=args.output_dir,
            base_dir=dataset_base_dir
        )

    # ============================================================
    # Existing preprocessing/training pipeline
    # ============================================================
    df_train.info()
    df_test.info()
    df_val.info()

    print("Train labels:", df_train["Label"].unique())
    print("Test labels:", df_test["Label"].unique())
    print("Validation labels:", df_val["Label"].unique())

    train_path, valid_path, test_path = process_dataset_from_df(
        logger=logger,
        df_train=df_train,
        df_valid=df_val,
        df_test=df_test,
        output_dir=args.output_dir,
        grouping=args.grouping,
        window_size=args.window_size,
        step_size=args.step_size,
        session_type=args.session_level,
        dataset_name=target_dataset,
        data_dir=target_data_dir
    )

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"

    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog", "LogBERT"]

    log_vocab = build_vocab(
        vocab_path,
        vocab_data_dir,
        train_path,
        vocab_embeddings,
        embedding_dim=args.embedding_dim,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    model = build_model(args, vocab_size=len(log_vocab))

    train_time, test_time, acc, f1, precision, recall = train_and_eval(
        args,
        train_path,
        test_path,
        valid_path,
        log_vocab,
        model,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    print(f"Training time: {train_time:.2f} min")
    print(f"Testing time: {test_time:.2f} min")
    print(
        f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}"
    )


if __name__ == "__main__":

    # Do not delete the global output directory here.
    # Each experiment writes to its own case-specific subdirectory.
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
