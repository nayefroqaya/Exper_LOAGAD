import os
import pandas as pd
from logadempirical.data.grouping import time_sliding_window, session_window, session_window_bgl, fixed_window
import pickle
from sklearn.utils import shuffle
from logging import Logger
from typing import List, Tuple
import pdb

# ---------------------------
# 1. Load train_df and test_df
# ---------------------------
with open("train_df.pkl", "rb") as f:
    df_train = pickle.load(f)

with open("test_df.pkl", "rb") as f:
    df_test = pickle.load(f)

# ---------------------------
# 2. Define logger (simple example)
# ---------------------------
import logging
logger = logging.getLogger("process_dataset")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------------------
# 3. Define your sliding/session functions
# ---------------------------
# You must already have these from your project:
# fixed_window, time_sliding_window, session_window, session_window_bgl
# For example:
# from your_module import fixed_window, time_sliding_window, session_window, session_window_bgl

# ---------------------------
# 4. Define process_dataset_from_df
# ---------------------------
def process_dataset_from_df(
        logger,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        output_dir: str,
        grouping: str,
        window_size: int,
        step_size: int,
        dataset_name: str = "BGL",
        session_type: str = "entry",
        data_dir: str = None):
    """
    Same as original process_dataset, but uses df_train and df_test directly.
    NO train/test splitting is done here.
    Produces SAME output format (train.pkl, test.pkl).
    """

    train_pkl = os.path.join(output_dir, "train.pkl")
    test_pkl  = os.path.join(output_dir, "test.pkl")

    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        logger.info("Loading existing train.pkl and test.pkl")
        return train_pkl, test_pkl

    # -----------------------------
    # SLIDING WINDOW MODE
    # -----------------------------
    if grouping == "sliding":
        df_train = df_train.copy()
        df_test  = df_test.copy()

        df_train["Label"] = df_train["Label"].apply(lambda x: int(x != "-"))
        df_test["Label"]  = df_test["Label"].apply(lambda x: int(x != "-"))

        if session_type == "entry":
            sliding_fn = fixed_window
        elif session_type == "time":
            sliding_fn = time_sliding_window
        else:
            raise ValueError("session_type must be 'entry' or 'time'")

        train_window = sliding_fn(
            df_train[["Label", "processed_EventTemplate", "Content"]],
            window_size=window_size,
            step_size=step_size
        )

        test_window = sliding_fn(
            df_test[["Label", "processed_EventTemplate", "Content"]],
            window_size=window_size,
            step_size=step_size
        )

    # -----------------------------
    # SESSION MODE
    # -----------------------------
    elif grouping == "session":
        if dataset_name == "HDFS":
            id_regex = r'(blk_-?\d+)'
            blk_df = pd.read_csv(os.path.join(data_dir, "anomaly_label.csv"))
            label_dict = {row["BlockId"]: (1 if row["Label"]=="Anomaly" else 0)
                          for _, row in blk_df.iterrows()}

            train_window = session_window(df_train, id_regex, label_dict, window_size)
            test_window  = session_window(df_test,  id_regex, label_dict, window_size)

        elif dataset_name == "BGL":
            train_window = session_window_bgl(df_train)
            test_window  = session_window_bgl(df_test)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    else:
        raise NotImplementedError(f"grouping '{grouping}' not supported.")

    # -----------------------------
    # Save outputs
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    with open(train_pkl, "wb") as f:
        pickle.dump(train_window, f)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_window, f)

    logger.info(f"Saved: {train_pkl}, {test_pkl}")
    return train_pkl, test_pkl




'''
def process_dataset(logger: Logger,
                    data_dir: str,
                    output_dir: str,
                    log_file: str,
                    dataset_name: str,
                    grouping: str,
                    window_size: int,
                    step_size: int,
                    train_size: float,
                    is_chronological: bool = False,
                    session_type: str = "entry") -> Tuple[str, str]:
    """
    creating log sequences by sliding window
    :param logger:
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param dataset_name:
    :param grouping:
    :param window_size:
    :param step_size:
    :param train_size:
    :param is_chronological:
    :param session_type:
    :return:
    """

    if os.path.exists(os.path.join(output_dir, "train.pkl")) and os.path.exists(os.path.join(output_dir, "test.pkl")):
        logger.info(f"Loading {output_dir}/train.pkl and {output_dir}/test.pkl")
        return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")

    logger.info(f"Loading {data_dir}/{log_file}_structured.csv")
    df = pd.read_csv(f'{data_dir}/{log_file}_structured.csv')

    # build log sequences
    if grouping == "sliding":
        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        n_train = int(len(df) * train_size)
        if session_type == "entry":
            sliding = fixed_window
        elif session_type == "time":
            sliding = time_sliding_window
            window_size = window_size
            step_size = step_size
        if not is_chronological:
            window_df = sliding(
                df[["Label", "EventId", "EventTemplate", "Content"]],
                window_size=window_size,
                step_size=step_size
            )
            window_df = shuffle(window_df)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        else:
            train_window = sliding(
                df[["Timestamp", "Label", "EventId", "EventTemplate", "Content"]].iloc[:n_train, :],
                window_size=window_size,
                step_size=step_size
            )
            test_window = sliding(
                df[["Timestamp", "Label", "EventId", "EventTemplate", "Content"]].iloc[n_train:, :].reset_index(
                    drop=True),
                window_size=window_size,
                step_size=step_size
            )
            pdb.set_trace()

    elif grouping == "session":
        if dataset_name == "HDFS":
            # get first 10% of df as training data
            # train_df = df.iloc[:int(len(df) * train_size), :]
            # test_df = df.iloc[int(len(df) * train_size):, :]
            id_regex = r'(blk_-?\d+)'
            label_dict = {}
            blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
            blk_df = pd.read_csv(blk_label_file)
            for _, row in enumerate(blk_df.to_dict("records")):
                label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0
            logger.info("label dict size: {}".format(len(label_dict)))
            window_df = session_window(df, id_regex, label_dict, window_size=int(window_size))
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
            # train_window = session_window(train_df, id_regex, label_dict, window_size=int(window_size))
            # test_window = session_window(df, id_regex, label_dict, window_size=int(window_size))
        elif dataset_name == "BGL":
            # df["NodeId"] = df["Node"].apply(lambda x: str(x).split(":")[0])
            window_df = session_window_bgl(df)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        else:
            raise NotImplementedError(f"{dataset_name} with {grouping} is not implemented")
    else:
        raise NotImplementedError(f"{grouping} is not implemented")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")
    '''


if __name__ == '__main__':
    #process_dataset(Logger("BGL"),
    #                data_dir="../../dataset/", output_dir="../../dataset/", log_file="BGL.log",
    #                dataset_name="bgl",
    #                grouping="sliding", window_size=10, step_size=10, train_size=0.8, is_chronological=True,
    #                session_type="entry")∂ƒ√
    output_dir = "../../dataset/BGL/"  # change to your preferred output folder
    process_dataset_from_df(logger=logger, df_train=df_train, df_test=df_test, output_dir=output_dir,
        grouping="sliding",  # or "session for HDFS"
        window_size=120, step_size=120, session_type="entry",  # or "time"
        dataset_name="BGL",  # or "BGL"
        data_dir="../../dataset/"  # needed only for session mode (HDFS)
    )
