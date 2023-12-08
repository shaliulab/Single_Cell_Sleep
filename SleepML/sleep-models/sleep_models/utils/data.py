import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_label_mapping(label_mapping):
    """
    Arguments:
        label_mapping (str): Path to a label mapping in .yaml
    Returns:
        label_mapping (dict): Keys are values in one column of the AnnData.obs dataframe and values are pseudo-labels
    
    """
    if label_mapping is None:
        label_mapping = None
    else:
        with open(label_mapping, "r") as filehandle:
            label_mapping = yaml.load(filehandle, yaml.SafeLoader)

    return label_mapping


def backup_dataset(
    cluster,
    X_train,
    y_train,
    X_test,
    y_test,
    original_data=None,
    encoding=None,
    output=".",
):
    """
    Saves provided dataset spilt to .csv files in the desired output folder
    Optionally encoding and original data are also saved

    Arguments

        cluster (str): Cell type from which the dataset is produced
        X_train (pd.DataFrame): (nxm) Feature dataset used for training
        y_train (pd.DataFrame): (nxc) Label dataset used for training
        X_test (pd.DataFrame): (nxm) Feature dataset used for testing
        y_test (pd.DataFrame): (nxc) Label dataset used for testing

    Returns:
        None

    """

    X_train.to_csv(os.path.join(output, f"{cluster}_X-train.csv"), index=True)
    X_test.to_csv(os.path.join(output, f"{cluster}_X-test.csv"), index=True)
    y_train.to_csv(os.path.join(output, f"{cluster}_y-train.csv"), index=True)
    y_test.to_csv(os.path.join(output, f"{cluster}_y-test.csv"), index=True)
    if encoding is not None:
        with open(os.path.join(output, f"{cluster}_encoding.csv"), "w") as filehandle:
            filehandle.write("label,code\n")
            for code, label in encoding.items():
                filehandle.write(f"{label},{code}\n")

    if original_data is not None:
        original_data.to_csv(
            os.path.join(output, f"{cluster}_original_data.csv"), index=True
        )


def restore_dataset(input, cluster):
    """
    Restore previously cached dataset

    Arguments:
        input (str): Path to folder where data is saved
        cluster (str): Cell type which should be restored

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict)

    """
    X_train = pd.read_csv(os.path.join(input, f"{cluster}_X-train.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(input, f"{cluster}_y-train.csv"), index_col=0)
    X_test = pd.read_csv(os.path.join(input, f"{cluster}_X-test.csv"), index_col=0)
    y_test = pd.read_csv(os.path.join(input, f"{cluster}_y-test.csv"), index_col=0)

    encoding_path = os.path.join(input, f"{cluster}_encoding.csv")

    if os.path.exists(encoding_path):
        encoding_df = pd.read_csv(encoding_path)
        encoding = {row["code"]: row["label"] for _, row in encoding_df.iterrows()}

    else:
        encoding = None

    return X_train, y_train, X_test, y_test, encoding


def split_dataset(X, y, train_size, random_state, stratify=None):

    # split the dataframe into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=stratify, random_state=random_state, train_size=train_size,
    )

    return X_train, y_train, X_test, y_test
