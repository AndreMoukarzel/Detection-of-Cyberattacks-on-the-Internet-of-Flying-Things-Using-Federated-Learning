# -*- coding: utf-8 -*-
"""Utility functions used in both Centralized and Federated training """

import json
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split



def get_treated_df() -> list:
    """Returns the treated Dataframe, the feature list and both the binary and
    multiclass target variables' column names
    """
    df = pd.read_csv("ECU-IoFT.csv")

    # Renames columns to remove ambiguity and be more usable in Python
    df = df.rename(columns={"Type.of.Attack": "type_of_attack", "Attack.Scenario": "attack_scenario"})
    df.columns = df.columns.str.lower()

    # Divides info field into multiple fields for better variable comprehension
    df["info_flags"] = np.where(len(df["info"].str.split("Flags=")) > 1, df["info"].str.split("Flags=").str[1].str.split(",").str[0], "")
    df["info_flags"] = df["info_flags"].str.strip(".").str.split(".").str.join("")
    df["info_sn"] = np.where(len(df["info"].str.split("SN=")) > 1, df["info"].str.split("SN=").str[1].str.split(",").str[0], "0")
    df["info_fn"] = np.where(len(df["info"].str.split("FN=")) > 1, df["info"].str.split("FN=").str[1].str.split(",").str[0], "0")
    df["info_bi"] = np.where(len(df["info"].str.split("BI=")) > 1, df["info"].str.split("BI=").str[1].str.split(",").str[0], "0")
    df["info_ssid"] = np.where(len(df["info"].str.split("SSID=")) > 1, df["info"].str.split("SSID=").str[1].str.split(",").str[0], "")
    df["info_len"] = np.where(len(df["info"].str.split("Len=")) > 1, df["info"].str.split("Len=").str[1].str.split(",").str[0], "0")
    df["info_base"] = df["info"].str.split(",").str[0]

    # Create classes for classification
    # Binary classification
    df["attack_binary"] = np.where(df["type"] == "Normal", 0, 1)

    # Multi-class classification
    df["attack_class"] = np.where(df["type_of_attack"] == "No Attack", 0,
                                np.where(df["type_of_attack"] == "WPA2-PSK WIFI Cracking Attack", 1,
                                        np.where(df["type_of_attack"] == "Wifi Deauthentication Attack", 2, 3)))
    
    # Normalize protocol length
    df["length"] = df["length"].astype("float32")/df["length"].max()

    # Apply minmax to all features
    # Categorize string columns which objective values are irrelevant
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype("category").cat.codes
        # Categorized columns are then normalized for future used in the neural network
        min = df[col].min()
        max = df[col].max()
        df[col] = (df[col] - min)/(max - min)

    # Keep only the final features
    feature_columns = ["protocol", "length", "info_base", "info_flags", "info_sn", "info_fn", "info_bi", "info_ssid", "info_len"]
    return [df[feature_columns + ["attack_binary", "attack_class"]], feature_columns, "attack_binary", "attack_class"]


def preprocess(
    dataset: tf.data.Dataset,
    data_shape: int):
    """Preprocesses federated learning data into adequate format for training.

    Parameters
    ----------
    client_data : tf.data.Dataset
        TensorSliceDataset batch
    data_shape : int
        Number of features in the client_data
    
    Returns
    -------
    batch dataset
        Single Batch dataset readeable by a tff.Computation
    """
    NUM_EPOCHS = 3
    BATCH_SIZE = 512
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10

    def batch_format_fn(batch) -> collections.OrderedDict:
        """Reshapes a batch and returns the features and target labels as an
        OrderedDict
        """
        return collections.OrderedDict(
            x=tf.reshape(batch['data'], [-1, data_shape]),
            y=tf.reshape(batch['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(
    client_data: tff.simulation.datasets.ClientData,
    client_ids: list,
    data_shape: int) -> list:
    """Transforms client_data into a format readeable by TensorFlow Federated's
    federated averaging process

    Parameters
    ----------
    client_data : tff.simulation.datasets.ClientData
        -
    client_ids : list
        List of ids of clients from client_data
    data_shape : int
        Number of features in the client_data

    Returns
    -------
    list
        List of single batch datasets accepted for training in a tff.Computation
    """
    return [preprocess(
        client_data.create_tf_dataset_for_client(x), data_shape
        ) for x in client_ids]


def get_split_data(
    df: pd.DataFrame,
    features: list,
    target: str,
    client_num: int,
    test_size: float=0.2) -> list:
    """Returns Client Data for all clients' training in Federated Learning by
    splitting the full training DataFrame into equal parts for each client.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing (at least) all features and target variable
    features : List[str]
        List with all features' column names
    target : str
        Column name of target variable
    client_num : int
        Number of clients expected in training the Federated algorithm
    test_size: float, optional
        Test size's percentage of the total DataFrame

    Returns
    -------
    list
        List of single batch datasets accepted for training in a tff.Computation
    """
    # Train/Test split
    x = df[features]
    y = df[[target]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    # Client_train_dataset must be an OrderedDict with the workers data
    client_train_dataset = collections.OrderedDict()

    # Split the data equally for each worker and put it on client_train_dataset
    size = x_train.shape[0]//client_num
    for i in range(client_num):
        x = x_train[i * size: (i + 1) * size]
        y = y_train[i * size: (i + 1) * size]

        data = collections.OrderedDict((('label', y), ('data', x)))
        client_train_dataset["client_" + str(i)] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    data_shape = x_train.shape[1]

    return make_federated_data(train_dataset, train_dataset.client_ids, data_shape)


def get_bagged_data(
    df: pd.DataFrame,
    features: list,
    target: str,
    client_num: int,
    client_df_size: float=0.5,
    test_size: float=0.2) -> list:
    """Returns Client Data for all clients' training in Federated Learning by
    getting a random sample of the full training DataFrame (with repetition) for
    each client.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing (at least) all features and target variable
    features : List[str]
        List with all features' column names
    target : str
        Column name of target variable
    client_num : int
        Number of clients expected in training the Federated algorithm
    client_df_size: float, optional
        Size of each client's dataset relative to total size of training data
    test_size: float, optional
        Test size's percentage of the total DataFrame

    Returns
    -------
    list
        List of single batch datasets accepted for training in a tff.Computation
    """
    # Train/Test split
    x = df[features]
    y = df[[target]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    # Client_train_dataset must be an OrderedDict with the workers data
    client_train_dataset = collections.OrderedDict()

    for i in range(client_num):
        x, _, y, _ = train_test_split(x_train, y_train, test_size=(1 - client_df_size), stratify=y_train)

        data = collections.OrderedDict((('label', y), ('data', x)))
        client_train_dataset["client_" + str(i)] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    data_shape = x_train.shape[1]

    return make_federated_data(train_dataset, train_dataset.client_ids, data_shape)


def train_federated_model(
    train_data : list,
    num_rounds : int,
    target_type : str,
    model_fn,
    verbose : int=1,
    logdir : str="/tmp/logs/scalars/training/") -> list:
    """Transforms client_data into a format readeable by TensorFlow Federated's
    federated averaging process

    Parameters
    ----------
    train_data : list
        List of single batch datasets accepted for training in a tff.Computation
    num_rounds : int
        Number of training rounds
    target_type : str("binary" | "categorical")
        Type of target variable (present in the train data) the model will be
        trained to predict
    model_fn : tff Model function
        tff model call function
    verbose : int, optional
        If not zero, prints training metrics each round
    logdir : str, optional
        Log directory where training summary will be saved. Useful for plotting
        in tensorboard

    Returns
    -------
    List, List, state
        Accuracy and Loss during training (in list format), and the current
        state of the averaging iterative process after training
    
    Raises
    ------
    NameError
        when an unspecified target_type variable is used
    """    
    if target_type == "binary":
        acc_var = "binary_accuracy"
    elif target_type == "categorical":
        acc_var = "sparse_categorical_accuracy"
    else:
        raise NameError("Unknown target_type used. Please use binary | categorical.")

    averaging_iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.0005),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    summary_writer = tf.summary.create_file_writer(logdir)
    state = averaging_iterative_process.initialize()
    accs = []
    losses = []

    with summary_writer.as_default():
        for round_num in range(num_rounds):
            state, metrics = averaging_iterative_process.next(state, train_data)
            accs.append(metrics['train'][acc_var])
            losses.append(metrics['train']['loss'])
            if verbose:
                print('round {:2d}, metrics={}'.format(round_num, metrics['train']))

            for name, value in metrics['train'].items():
                tf.summary.scalar(name, value, step=round_num)
    
    return [accs, losses, state]


def get_all_results(path: str) -> dict:
    """Returns dictionary with all result from specified path"""
    models_data = {}
    model_names = ["binary_fed", "binary_fed_bagged", "multiclass_fed", "multiclass_fed_bagged"]

    for i in [3, 5, 10]:
        for name in model_names:
            try:
                with open(path + '/' + name + str(i) + ".json", "r") as f:
                    models_data[name + str(i)] = json.load(f)
            except FileNotFoundError:
                pass
    
    with open(path + '/' + "multiclass_fed_bagged10_smallmodel.json", "r") as f:
        models_data["multiclass_fed_bagged10_smallmodel"] = json.load(f)
    with open(path + '/' + "binary_cen.json", "r") as f:
        models_data["binary_cen"] = json.load(f)
    with open(path + '/' + "binary_cen_small.json", "r") as f:
        models_data["binary_cen_small"] = json.load(f)
    with open(path + '/' + "classes_cen.json", "r") as f:
        models_data["multiclass_cen"] = json.load(f)
    with open(path + '/' + "classes_cen_small.json", "r") as f:
        models_data["multiclass_cen_small"] = json.load(f)

    return models_data