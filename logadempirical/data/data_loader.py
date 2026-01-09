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
#with open("train_df.pkl", "rb") as f:
#    df_train = pickle.load(f)

#with open("test_df.pkl", "rb") as f:
#    df_test = pickle.load(f)

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
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str,
    grouping: str,
    window_size: int,
    step_size: int,
    dataset_name: str = "SP_150MB",
    session_type: str = "entry",
    data_dir: str = None
):
    """
    Processes train, validation, and test datasets from DataFrames.
    Produces SAME output format (train.pkl, valid.pkl, test.pkl).
    """

    train_pkl = os.path.join(output_dir, "train.pkl")
    valid_pkl = os.path.join(output_dir, "valid.pkl")
    test_pkl  = os.path.join(output_dir, "test.pkl")

    print('Final data before starting benchmark')

    # Keep only relevant columns
    cols = ["Timestamp", "Label", "EventId", "EventTemplate", "Content", "processed_EventTemplate"]
    df_train = df_train[cols].copy()
    df_valid = df_valid[cols].copy()
    df_test  = df_test[cols].copy()

    print(dataset_name)
    df_train.info()
    df_valid.info()
    df_test.info()

    # -----------------------------
    # SLIDING WINDOW MODE
    # -----------------------------
    if grouping == "sliding":
        df_train["Label"] = df_train["Label"].apply(lambda x: int(x != "-"))
        df_valid["Label"] = df_valid["Label"].apply(lambda x: int(x != "-"))
        df_test["Label"]  = df_test["Label"].apply(lambda x: int(x != "-"))

        if session_type == "entry":
            sliding_fn = fixed_window
        else:
            raise ValueError("session_type must be 'entry' or 'time'")

        train_window = sliding_fn(df_train, window_size=window_size, step_size=step_size)
        valid_window = sliding_fn(df_valid, window_size=window_size, step_size=step_size)
        test_window  = sliding_fn(df_test,  window_size=window_size, step_size=step_size)

    # -----------------------------
    # SESSION MODE
    # -----------------------------
    elif grouping == "session":
        if dataset_name == "HDFS":
            id_regex = r'(blk_-?\d+)'
            blk_df = pd.read_csv("/app/home/roqaya/Exper_LOAGAD/dataset/HDFS/anomaly_label.csv")
            label_dict = {row["BlockId"]: (1 if row["Label"]=="Anomaly" else 0)
                          for _, row in blk_df.iterrows()}

            train_window = session_window(df_train, id_regex, label_dict, window_size)
            valid_window = session_window(df_valid, id_regex, label_dict, window_size)
            test_window  = session_window(df_test,  id_regex, label_dict, window_size)
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
    with open(valid_pkl, "wb") as f:
        pickle.dump(valid_window, f)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_window, f)

    logger.info(f"Saved: {train_pkl}, {valid_pkl}, {test_pkl}")
    return train_pkl, valid_pkl, test_pkl



if __name__ == '__main__':
    #process_dataset(Logger("BGL"),
    #                data_dir="../../dataset/", output_dir="../../dataset/", log_file="BGL.log",
    #                dataset_name="bgl",
    #                grouping="sliding", window_size=10, step_size=10, train_size=0.8, is_chronological=True,
#<<<<<<< HEAD
    #                session_type="entry")∂ƒ√
#    output_dir = "/storage/home/roqaya/Exper_LOAGAD/output"  # change to your preferred output folder
#=======
    #                session_type="entry")
    output_dir = "/storage/home/roqaya/Exper_LOAGAD/output"  #output_dir = "output"  # change to your preferred output folder
#>>>>>>> 608a4fb (update dataset portion)

    process_dataset_from_df(logger=logger, df_train=df_train, df_valid=df_valid,
        # make sure you have a validation dataframe
        df_test=df_test, output_dir=output_dir, grouping="sliding",  # "session" for HDFS/BGL datasets
        window_size=120, step_size=120, session_type="entry",  # or "time"
        dataset_name="SP_150MB",  # "HDFS" or "BGL"
        data_dir="../../dataset/SP_150MB/"  # needed only for session mode
    )
    #process_dataset_from_df(logger=logger, df_train=df_train, df_test=df_test, output_dir=output_dir,
    #    grouping="session",  # or "session for HDFS"
    #    window_size=120, step_size=120, session_type="entry",  # or "time"
    #    dataset_name="BGL",  # or "BGL"
    #    data_dir="../../dataset/BGL/"  # needed only for session mode (HDFS)
    #)
