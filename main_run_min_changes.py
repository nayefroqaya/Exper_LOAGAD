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

    # MINI CHANGE: debug label distribution after load_features
    print("\nDEBUG train label distribution after load_features:")
    print(Counter([str(l) for _, l in train_data]))

    print("\nDEBUG valid label distribution after load_features:")
    print(Counter([str(l) for _, l in valid_data]))

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

    # MINI CHANGE: debug label distribution after load_features
    print("\nDEBUG test label distribution after load_features:")
    print(Counter([str(l) for _, l in data]))
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
# Dataset paths and domain experiment helpers
# Minimal changes: only replace fixed dataframe loading in run(args)
# ============================================================

DATASETS = {
    "BGL": {
        "train": "/storage/home/roqaya/NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl",
        "val":   "/storage/home/roqaya/NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl",
        "test":  "/storage/home/roqaya/NovaAD_Plus/datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl",
    },
    "HDFS": {
        "train": "/storage/home/roqaya/Exper_LogForm/datasets/HDFS/1_HDFS_Splitted_Datasets/train_df.pkl",
        "val":   "/storage/home/roqaya/Exper_LogForm/datasets/HDFS/1_HDFS_Splitted_Datasets/val_df.pkl",
        "test":  "/storage/home/roqaya/Exper_LogForm/datasets/HDFS/1_HDFS_Splitted_Datasets/test_df.pkl",
    },
    "TH_1G": {
        "train": "/storage/home/roqaya/Exper_LogForm/datasets/TH_1G/1_TH_1G_Splitted_Datasets/train_df.pkl",
        "val":   "/storage/home/roqaya/Exper_LogForm/datasets/TH_1G/1_TH_1G_Splitted_Datasets/val_df.pkl",
        "test":  "/storage/home/roqaya/Exper_LogForm/datasets/TH_1G/1_TH_1G_Splitted_Datasets/test_df.pkl",
    },
    "SP_150MB_ratio": {
        "train": "/storage/home/roqaya/Exper_LogForm/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/train_df.pkl",
        "val":   "/storage/home/roqaya/Exper_LogForm/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/val_df.pkl",
        "test":  "/storage/home/roqaya/Exper_LogForm/datasets/SP_150MB_ratio/1_SP_150MB_ratio_Splitted_Datasets/test_df.pkl",
    },
}


def fix_label_column(df):
    """
    Keep one clean label column named Label.
    If Original_Label exists, use it as the final Label column.
    """
    df = df.copy()

    if "Original_Label" in df.columns:
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])
        df = df.rename(columns={"Original_Label": "Label"})

    return df


def load_dataset_df(dataset_name):
    """
    Load train / val / test dataframe for one dataset.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASETS.keys())}"
        )

    paths = DATASETS[dataset_name]

    df_train = pd.read_pickle(paths["train"])
    df_val = pd.read_pickle(paths["val"])
    df_test = pd.read_pickle(paths["test"])

    df_train = fix_label_column(df_train)
    df_val = fix_label_column(df_val)
    df_test = fix_label_column(df_test)

    return df_train, df_val, df_test


def get_target_normal_df(df):
    """
    Select only Normal samples from target train.
    Supports numeric and string labels.
    """
    return df[
        (df["Label"] == 0) |
        (df["Label"] == "0") |
        (df["Label"] == "Normal") |
        (df["Label"] == "normal")
    ].copy()


def prepare_domain_case(case,
                        target_dataset,
                        source_datasets=None,
                        target_normal_fraction=0.20,
                        random_seed=42):
    """
    Supported cases:

    1) in_domain:
       train = target train
       val   = target val
       test  = target test

    2) cross_dataset_with_fraction:
       train = source train + fraction of target normal train
       val   = target val
       test  = target test untouched

    Note:
    Using target validation is not test leakage, because target test is untouched.
    But it means model selection is tuned on target-domain validation data.
    """
    if source_datasets is None:
        source_datasets = []

    np.random.seed(random_seed)

    if case == "in_domain":
        df_train, df_val, df_test = load_dataset_df(target_dataset)

        print("\n==============================")
        print("In-domain experiment")
        print("==============================")
        print("Target dataset:", target_dataset)
        print("Train = target train")
        print("Val   = target val")
        print("Test  = target test")
        print("Train size:", len(df_train))
        print("Val size:", len(df_val))
        print("Test size:", len(df_test))
        print("\nTrain label distribution:")
        print(df_train["Label"].value_counts())
        print("\nVal label distribution:")
        print(df_val["Label"].value_counts())
        print("\nTest label distribution:")
        print(df_test["Label"].value_counts())

        return df_train, df_val, df_test

    elif case == "cross_dataset_with_fraction":
        if len(source_datasets) == 0:
            raise ValueError("For cross_dataset_with_fraction, source_datasets must not be empty.")

        source_train_list = []

        print("\n==============================")
        print("Cross-dataset with target normal fraction")
        print("==============================")
        print("Sources:", source_datasets)
        print("Target:", target_dataset)
        print("Target normal fraction:", target_normal_fraction)
        print("Train = source train + fraction of target normal train")
        print("Val   = target val")
        print("Test  = target test untouched")

        for source_name in source_datasets:
            src_train, src_val, src_test = load_dataset_df(source_name)
            source_train_list.append(src_train)

            print("\nLoaded source dataset:", source_name)
            print("Source train used:", len(src_train))
            print("Source val NOT used:", len(src_val))
            print("Source test NOT used:", len(src_test))
            print("\nSource train label distribution:")
            print(src_train["Label"].value_counts())

        source_train = pd.concat(source_train_list, ignore_index=True)

        target_train, target_val, target_test = load_dataset_df(target_dataset)

        target_normal_train = get_target_normal_df(target_train)
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
        print("Final cross-dataset data")
        print("==============================")
        print("Source train:", len(source_train))
        print("Target train:", len(target_train))
        print("Target normal train:", len(target_normal_train))
        print("Target normal used:", len(target_normal_sample))
        print("Target val used:", len(df_val))
        print("Target test untouched:", len(df_test))
        print("Final train:", len(df_train))
        print("Final val:", len(df_val))
        print("Final test:", len(df_test))
        print("\nFinal train label distribution:")
        print(df_train["Label"].value_counts())
        print("\nTarget val label distribution:")
        print(df_val["Label"].value_counts())
        print("\nTarget test label distribution:")
        print(df_test["Label"].value_counts())

        return df_train, df_val, df_test

    else:
        raise ValueError("case must be either 'in_domain' or 'cross_dataset_with_fraction'")


def run(args):
    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Experiment settings
    # Change ONLY this block when you want another experiment
    # ============================================================

    # -----------------------------
    # Option 1: In-domain
    # train = target train
    # val   = target val
    # test  = target test
    # -----------------------------
    CASE = "in_domain"
    TARGET_DATASET = args.dataset_name
    SOURCE_DATASETS = []          # not used for in-domain
    TARGET_NORMAL_FRACTION = 0.20 # not used for in-domain
    RANDOM_SEED = 42

    # -----------------------------
    # Option 2: Cross-dataset with fraction
    # train = source train + fraction of target normal train
    # val   = target val
    # test  = target test untouched
    # -----------------------------
    #CASE = "cross_dataset_with_fraction"
    #TARGET_DATASET = args.dataset_name
    #SOURCE_DATASETS = ["BGL"]
    #TARGET_NORMAL_FRACTION = 0.20
    #RANDOM_SEED = 42

    # MINI CHANGE: make sure args.history_size exists.
    # train_and_eval() uses args.history_size, while preprocessing uses args.window_size.
    if not hasattr(args, "history_size") or args.history_size is None:
        args.history_size = args.window_size
        print(f"history_size was missing; set history_size = window_size = {args.history_size}")

    # ============================================================
    # Output directory
    # ============================================================

    if CASE == "in_domain":
        experiment_name = f"in_domain_target_{TARGET_DATASET}"

    elif CASE == "cross_dataset_with_fraction":
        if len(SOURCE_DATASETS) == 0:
            raise ValueError("SOURCE_DATASETS cannot be empty for cross_dataset_with_fraction")

        source_name = "_".join(SOURCE_DATASETS)
        fraction_name = str(TARGET_NORMAL_FRACTION).replace(".", "p")
        experiment_name = (
            f"cross_dataset_source_{source_name}"
            f"_target_{TARGET_DATASET}"
            f"_targetval_frac_{fraction_name}"
        )

    else:
        raise ValueError("CASE must be either 'in_domain' or 'cross_dataset_with_fraction'")

    if args.grouping == "sliding":
        output_subdir = (
            f"{args.output_dir}/{experiment_name}/{TARGET_DATASET}/sliding/"
            f"W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
        )
    else:
        output_subdir = f"{args.output_dir}/{experiment_name}/{TARGET_DATASET}/session"

    os.makedirs(output_subdir, exist_ok=True)
    args.output_dir = output_subdir

    print("\n==============================")
    print("Experiment configuration")
    print("==============================")
    print("CASE:", CASE)
    print("TARGET_DATASET:", TARGET_DATASET)
    print("SOURCE_DATASETS:", SOURCE_DATASETS)
    print("TARGET_NORMAL_FRACTION:", TARGET_NORMAL_FRACTION)
    print("RANDOM_SEED:", RANDOM_SEED)
    print("Output dir:", args.output_dir)

    # ============================================================
    # Prepare dataframe splits
    # ============================================================

    df_train, df_val, df_test = prepare_domain_case(
        case=CASE,
        target_dataset=TARGET_DATASET,
        source_datasets=SOURCE_DATASETS,
        target_normal_fraction=TARGET_NORMAL_FRACTION,
        random_seed=RANDOM_SEED
    )

    print("\n==============================")
    print("Dataframe information")
    print("==============================")

    print("\nTrain dataframe:")
    df_train.info()

    print("\nValidation dataframe:")
    df_val.info()

    print("\nTest dataframe:")
    df_test.info()

    print("\nTrain labels:")
    print(df_train["Label"].unique())

    print("\nValidation labels:")
    print(df_val["Label"].unique())

    print("\nTest labels:")
    print(df_test["Label"].unique())

    # ============================================================
    # Convert dataframe to LOAD format
    # This keeps your existing pipeline unchanged
    # ============================================================

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
        dataset_name=TARGET_DATASET,
        data_dir=args.data_dir
    )

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)

    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"

    # MINI CHANGE: force rebuild vocab every run to avoid using stale vocab
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
        print(f"Removed old vocab: {vocab_path}")

    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog", "LogBERT"]

    log_vocab = build_vocab(
        vocab_path,
        args.data_dir,
        train_path,
        args.embeddings,
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

    print("\n==============================")
    print("Final Result")
    print("==============================")
    print(f"Training time: {train_time:.2f} min")
    print(f"Testing time: {test_time:.2f} min")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":

    # MINI CHANGE: clean the same output folder used in the config by default
    output_dir = "./output"

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
