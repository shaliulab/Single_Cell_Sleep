from cProfile import label
import logging
import os
import os.path

import anndata
import scanpy as sc

import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from sleep_models.constants import (
    HIGHLY_VARIABLE_GENES,
    MEAN_SCALE,
    MIN_MEAN_HVG,
    MAX_MEAN_HVG,
    MIN_DISP_HVG,
    TRAIN_SIZE,
)

from sleep_models.utils.logging import setup_logging
import sleep_models.utils.data as data_utils
from sleep_models.utils.data import split_dataset

logger = logging.getLogger("sleep_models.preprocessing")
logger = logging.getLogger(__name__)

# Creating Test DataSets using sklearn.datasets.make_blobs


def read_h5ad(
    file,
    *args,
    highly_variable_genes=False,
    raw=False,
    exclude_genes_file=None,
    **kwargs,
):

    adata = _read_h5ad(file, *args, **kwargs)
    if highly_variable_genes:
        if not "highly_variable" in adata.var.columns:
            highly_variable_genes(
                adata,
                min_mean=MIN_MEAN_HVG,
                max_mean=MAX_MEAN_HVG,
                min_disp=MIN_DISP_HVG,
            )

        adata = adata[:, adata.var.index[adata.var["highly_variable"]]]

    if raw:
        adata = _use_raw_anndata(adata)

    if exclude_genes_file is not None:
        logger.info(f"Removing genes listed in {exclude_genes_file}")
        adata = remove_genes(adata, exclude_genes_file)

    return adata


def load_adata(
    h5ad_input,
    cluster,
    exclude_genes_file=None,
    highly_variable_genes=True,
    logger=None,
):
    """
    Loads an anndata.AnnData cached to an h5ad file

    * Optionally remove undesired genes
    * Optionally keeps highly variable genes only
    * Keeps cells belonging to passed cluster

    Arguments:

        h5ad_input (str): Path to a cached anndata.Anndata in an h5ad file
        cluster (str): Cell type matching to one of the celltypes in the CellType column of the restored Anndata's obs attr
        exclude_genes_file (str): Path to a plain text file with genes that should be excluded from the analysis
        highly_variable_genes (bool): If True, only hvg are kept in the resulting adata

    Returns:
        adata (anndata.Anndata)
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Reading {h5ad_input} to memory")
    print(f"Reading {h5ad_input} to memory")
    adata = read_h5ad(
        h5ad_input,
        highly_variable_genes=highly_variable_genes,
        exclude_genes_file=exclude_genes_file,
    )
    if cluster is not None:
        adata = adata[adata.obs["CellType"] == cluster]
    return adata


class Pipeline:
    """
    A toolset to transform the data in an anndata.Anndata into a
    dataset amenable for sleep_models
    """

    _encoding = None

    def __init__(self, target, model_properties=None, random_state=1000):
        self.original_data = None
        if model_properties is not None:
            self._encoding = model_properties.encoding
            self._target = target
            self._estimator_type = model_properties.estimator_type

        self.random_state = random_state

    def encode_y(self, y):
        """
        Encode y using the method stored in the _encoding attribute
        Only One Hot supported, so this method just calls the one_hot_encoder

        Arguments:
            y (np.array): (n,1) or (n, ) array of labels where n is the number of samples (single cells)
            and the ith value corresponds to the label of the ith sample

        Returns:
            y (np.array): nxc array of 0/1 where the i,j cell is 1 if the ith sample has class j and 0 otherwise
        """

        if self._encoding == "ONE_HOT":
            y = self.one_hot_encoder(y)

        else:
            raise Exception("Please define a valid encoding")

        return y

    def one_hot_encoder(self, y):
        """
        Perform one hot encoding using
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        Arguments:
            y (np.array): List of labels

        Returns:
            y (np.array): nxc array of 0/1 where the i,j cell is 1 if the ith sample has class j and 0 otherwise
        """
        # reshape to 2D as expected by the OneHotEncoder
        if len(y.shape) == 1:
            n_points = y.shape[0]
            n_targets = 1
            y = y.reshape((n_points, n_targets))

        self._encoder = OneHotEncoder()  # drop="first")
        self._encoder.fit(y)
        logger.info(f"One-Hot encoding categories: {self._encoder.categories_}")
        y = self._encoder.transform(y).toarray()
        return y

    @staticmethod
    def simplify_y(y, mapping):
        """
        Rewrite labels using some heuristic stored in mapping
        This is useful if the user wants to group labels into broader categories

        Arguments:

            y (np.array): Array of non-encoded labels
            mapping (dict): Dictionary mapping labels (keys) to pseudolabels (values)

        Returns:
           y (np.array): Array of non-encoded pseudolabels
        """

        new_y = []
        for value in y:
            new_y.append(mapping.get(value, None))

        new_y = np.array(new_y)

        assert len(new_y) == len(y)
        return new_y

    def process_adata(self, adata, target, label_mapping=None, trim=False):
        """
        Shuffle the adata to avoid biases due to cell sorting
        Keeps a record of the original dataset for future reference
        Optionally, simplifies the labels according to the passed label_mapping, if any
        Optionally, downsamples (trims) the classes so all are equally populated

        Encodes y according to the encoder

        Arguments:
            adata (anndata.AnnData): Instance of AnnData containing results of a SingleCellSequencing experiment
            label_mapping (dict): Keys are values in one column of the AnnData.obs dataframe and values are pseudo-labels
            trim (bool): If True, all classes are downsampled to their size matches that of the smallest one

        Returns:
            Tuple of 2 dataframes containing X and y
        """
        self._adata = adata

        # shuffle the adata
        barcodes = adata.obs.index.values.copy()
        np.random.shuffle(barcodes)
        adata = adata[barcodes]

        # fetch X (features) and y (labels or outcome to be predicted)
        X = pd.DataFrame(adata.X, index=adata.obs.index)

        # NOTE
        # The original line breaks when target is CellType
        # but the new one works in either case

        # y = adata.obs[target].values.flatten()
        y = np.array(adata.obs[target].tolist()).flatten()

        # keep original data record
        original_data = X.copy()
        original_data = pd.concat([original_data, adata.obs], axis=1)
        original_data["label"] = y
        self.original_data = original_data

        # simplify y if a mapping is pased:
        # useful to group values of the target into broader categories
        if label_mapping is not None:
            y = self.simplify_y(y, mapping=label_mapping)

        keep = [label is not None for label in y]
        X = X.iloc[keep]
        y = y[keep]
        adata = adata[keep, :]

        # downsample (trim) the categories
        if trim:

            keep_all_cells = []
            pseudo_labels, counts = np.unique(y, return_counts=True)
            least_freq = counts.min()

            for pseudo_label in pseudo_labels:
                labels = [k for k, v in label_mapping.items() if v == pseudo_label]
                keep_cells_with_label = np.where(adata.obs.isin(labels))[0][:least_freq]
                keep_all_cells.extend(keep_cells_with_label)

        else:
            keep_all_cells = np.arange(X.shape[0])

        # encode y as fit for ML
        y = pd.DataFrame(self.encode_y(y))
        X.columns = adata.var.index
        y.index = X.index

        # NOTE: Should I select the kept cells before the encoding?
        y = y.iloc[keep_all_cells]
        X = X.iloc[keep_all_cells]

        return X, y

    @staticmethod
    def remove_mean(X_train, y_train, X_test, y_test):
        feat_mean = X_train.mean(axis=1)
        X_train -= feat_mean
        X_test -= feat_mean
        return X_train, y_train, X_test, y_test


def scale_x(X_train, X_test):
    """
    
    """
    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std

    X_test = X_test - mean / std

    X_train, X_test


def remove_genes(adata, exclude_genes_file):
    if (
        not exclude_genes_file is None
        and not exclude_genes_file == ""
        and not exclude_genes_file == "None"
    ):
        with open(exclude_genes_file, "r") as fh:
            exclude_genes = [gene.strip("\n") for gene in fh.readlines()]

        adata = remove_genes_from_list(adata, exclude_genes)

    return adata


def keep_cells_from_this_background(adata, background):
    """
    Filter the single cells in the input anndata.AnnData
    so only cells belonging to the passed background are kept

    Arguments:
        adata (anndata.AnnData): Input anndata to be filtered
        background (pd.DataFrame):

    """

    assert "louvain_res" in background.columns
    assert "idx" in background.columns

    clusters = [(row["louvain_res"], row["idx"]) for i, row in background.iterrows()]
    index = _find_barcodes_belonging_to_clusters(adata, clusters)
    adata = adata[index, :]
    return adata


def shuffle_adata(adata, filename, pinned_columns=[]):
    """
    The result of the shuffling is a new adata where the rows of the counts matrix (adata.X)
    are shuffled, so the reads for cell 1 are assigned to any other random cell
    This is useful when simulating null hypotheses
    If pinned_columns is passed, this column in the obs table are NOT shuffled, i.e. cells that had a given value
    in the column will be shuffled only with cells with the same value
    """

    adatas = []
    if pinned_columns:
        pins = []
        for i in range(adata.shape[0]):
            pin = []
            for column in pinned_columns:
                pin.append(adata.obs.iloc[i][column])

            pins.append(tuple(pin))

        pins = set(pins)

        for pin in pins:
            selector = []
            for i, k in enumerate(pin):
                selector.append((adata.obs[pinned_columns[i]].values == k))

            selector = np.stack(selector, axis=1).all(axis=1)

            adata_shuffled = adata[selector, :].copy()
            adata_shuffled.obs = adata_shuffled.obs.sample(frac=1)
            adatas.append(adata_shuffled)

        adata_shuffled = anndata.concat(adatas)

    else:
        adata_shuffled = adata.copy()
        adata_shuffled.obs = adata_shuffled.obs.sample(frac=1)

    # adata_shuffled.obs.reset_index(inplace=True)
    # adata_shuffled.obs["original_index"]=adata_shuffled.obs["index"]
    # adata_shuffled.obs.drop("index", axis=1, inplace=True)

    logger.info(f"Saving shuffled h5ad to disk at {filename}")
    adata_shuffled.write_h5ad(filename)
    return adata_shuffled


def get_bad_genes(batch_genes_file, exclude_genes_file):
    """
    Read the genes contained in
    *  a batch_genes file (excel file with a sheet called all and a column on it called gene)
    * a exclude genes file (plain text with a gene per line)

    Combine both and return them
    """
    if batch_genes_file is not None:
        batch_genes = pd.read_excel(batch_genes_file, sheet_name="all")[
            "gene"
        ].values.tolist()
    else:
        batch_genes = []
    if exclude_genes_file is not None:
        with open(exclude_genes_file, "r") as fh:
            exclude_genes = [gene.split("\t")[0].strip() for gene in fh.readlines()]
    else:
        exclude_genes = []

    bad_genes = np.unique(batch_genes + exclude_genes).tolist()

    return bad_genes


def _template_matching(data, template):
    """
    Replace the values in data with the values specified in the template

    template = {
        "value-1": x,
        "value-2: y,
        ...
    }
    """

    return [template[val] for val in data]


def template_matching(adata, template):
    """
    Use the provided template to perform template matching in the
    passed adata
    """

    column_name = template[
        "target"
    ]  # a string with the name of a column in the adata.obs table
    template = template["template"]  # a dictionary of condition: float pairs

    # remove the cells whose condition is NOT in the template
    keep_cells = [val in template for val in adata.obs[column_name]]
    adata = adata[keep_cells, :]

    data = adata.obs[column_name]
    adata.obs["Template"] = _template_matching(data, template)
    # adata_obs = adata.obs.copy().sort_values("Template")
    # np.unique(["-".join([str(x) for x in e[0]]) for  e in zip(obs_sorted[["Template", "Condition"]].values.tolist())])

    return adata


def encode_y(y, encoding):
    """
    Encode y using OneHot encoding with the encoding provided

    Arguments

        y (np.array): Array of non encoded labels
        encoding (dict): Map of labels to columns

    Returns
        y (np.array): Array of one hot encoded labels
    """

    categories = [encoding[i] for i in range(len(encoding))]
    encoder = OneHotEncoder(categories=categories)
    encoder.fit(y)
    y_encoded = encoder.transform(y).toarray()
    return y_encoded


def simulate_data(
    h5ad_input,
    cluster,
    model_properties,
    random_state=1000,
    mean_scale=MEAN_SCALE,
    highly_variable_genes=HIGHLY_VARIABLE_GENES,
    label_mapping=None,
    stratify=True,
):
    """
    Simulate an easy dataset with shape similar to what would be loaded from the cached Anndata

    Arguments:
        TODO


    Returns: (np.ndarray, np.ndarray, np.ndarray, np.ndarray), i.e.
    a tuple with the X and y of the training and test sets respectively
    """

    label_mapping = data_utils.load_label_mapping(label_mapping)

    adata = load_adata(
        h5ad_input, cluster=cluster, highly_variable_genes=highly_variable_genes,
    )

    n_obs, n_features = adata.shape

    data_preprocessor = Pipeline(
        model_properties=model_properties, random_state=random_state
    )

    if label_mapping is None:
        y = adata.obs[model_properties.target].to_numpy()
        y = data_preprocessor.simplify_y(y, mapping=label_mapping)
        n_classes = len(set(y))
    else:
        n_classes = len(set(list(label_mapping.values())))

    X, y = make_blobs(
        n_samples=n_obs, centers=n_classes, cluster_std=1, n_features=n_features
    )

    if len(y.shape) == 1:
        y = np.stack([y, 1 - y], axis=1)

    y = np.float64(y)
    X = np.float32(X)

    if stratify:
        X_train, y_train, X_test, y_test = split_dataset(
            X, y, stratify=y, random_state=random_state, train_size=0.75
        )
    else:
        X_train, y_train, X_test, y_test = split_dataset(
            X, y, random_state=random_state, train_size=0.75
        )

    if mean_scale:
        X_train, X_test = scale_x(X_train, X_test)

    return {"datasets": (X_train, y_train, X_test, y_test), "encoding": None}


def sort_encoding(label_mapping, encoding):
    """
    Sets the order of the labels in the encoding
    to the same order of the label_mapping
    (independent of the code assigned in the encoding)
    This has the effect of setting the order of the labels
    in the axes of plots downstream
    """

    if label_mapping is None:
        return encoding

    labels = list(label_mapping.values())
    target_order = []
    for l in labels:
        if l in target_order:
            pass
        else:
            target_order.append(l)

    sorted_encoding = {}
    for tag in target_order:
        for i, tag2 in encoding.items():
            if tag == tag2:
                sorted_encoding[i] = tag2
                break

    encoding = sorted_encoding

    return encoding


def load_data(
    h5ad_input,
    output,
    cluster,
    model_properties,
    target,
    random_state=1000,
    exclude_genes_file=None,
    highly_variable_genes=HIGHLY_VARIABLE_GENES,
    verbose=logging.WARNING,
    logfile=None,
    label_mapping=None,
    trim=False,
    fraction=1.0,
    stratify=True,
):

    """
    Load the anndata.Anndata object stored in a cache h5ad file

    * Pick the cells that belong to cluster cluster (as per the CellType column)
    * Process the data according to
        * model_properties: one hot encoding
        * optionally pas a label_mapping to implement pseudo-label i.e. broader categories of labels
        * trim: optional downsampling of categories

    Arguments:

        h5ad_input (str): Path to an h5ad file containing an anndata.AnnData
            This AnnData must fulfil the following requirements
                * .obs["CellType"] is defined
                * .obs["Condition"] or .obs["Treatment"] are defined (depending on the selected target)
                * .X is a numpy array with shape ncellsxngenes
        output (str): Path to a folder where the results will be saved.
            On this folder, a new folder will be created with name random-state-{random_state}
        cluster (str): CellType to filter by, so only cells whose CellType matches are used
        random_state (int): random random_state for reproducibility
        exclude_genes_file (str): Path to a plain text file where every line contains the name of a gene to be ignored in the analysis
        highly_variable_genes (bool): If True, only the genes labeled as highly variable are used, otherwise all are used
            A gene is highly variable if it is found so in `sleep_models.preprocessing.read_h5ad`

        
        trim (boolean): If True, all labels are downsampled to the least frequent label
        fraction (float): Fraction of the dataset to be used

    Returns: (np.ndarray, np.ndarray, np.ndarray, np.ndarray), i.e.
    a tuple with the X and y of the training and test sets respectively
    """

    label_mapping = data_utils.load_label_mapping(label_mapping)

    if logfile is None:
        logfile = os.path.join(output, "logs", f"train_model_{cluster}.log")

    logger, _ = setup_logging(verbose, logfile)
    logger.info(f"Training on cell type {cluster} starting!")

    data_preprocessor = Pipeline(
        model_properties=model_properties, target=target, random_state=random_state
    )

    adata = load_adata(
        h5ad_input,
        cluster=cluster,
        exclude_genes_file=exclude_genes_file,
        highly_variable_genes=highly_variable_genes,
        logger=logger,
    )

    if fraction != 1.0:
        keep_n_cells = int(adata.shape[0] * fraction)
        adata = adata[:keep_n_cells, :]

    X, y = data_preprocessor.process_adata(
        adata, target=target, label_mapping=label_mapping, trim=trim
    )

    if stratify:
        X_train, y_train, X_test, y_test = split_dataset(
            X, y, stratify=y, random_state=random_state, train_size=0.75
        )
    else:
        X_train, y_train, X_test, y_test = split_dataset(
            X, y, random_state=random_state, train_size=0.75
        )

    encoding = list(enumerate(data_preprocessor._encoder.categories_[0]))
    encoding = {v[0]: v[1] for v in encoding}

    encoding = sort_encoding(label_mapping, encoding)

    data_utils.backup_dataset(
        cluster,
        X_train,
        y_train,
        X_test,
        y_test,
        data_preprocessor.original_data,
        encoding=encoding,
        output=output,
    )
    return {
        "datasets": (X_train.values, y_train.values, X_test.values, y_test.values),
        "encoding": encoding,
    }


def make_confusion_table(model, ground_truth, prediction):
    return pd.DataFrame(
        {
            "truth": [model._label_code[v] for v in ground_truth],
            "prediction": [model._label_code[v] for v in prediction],
        }
    )


def make_confusion_long_to_square(confusion_table_long, order=None):
    """
    Transform a confusion table in long format to wide format

    Example:

    truth,prediction
    A,A
    B,B
    A,B
    B,A
    A,A

    becomes
      A B
    A 2 1
    B 1 1
    """

    if order is None:
        label_set = set(confusion_table_long["truth"])
    else:
        label_set = order

    labels = confusion_table_long["truth"]
    predictions = confusion_table_long["prediction"]

    counts = np.zeros((len(label_set), len(label_set)))
    confusion_table = pd.DataFrame(counts)
    confusion_table.index = label_set
    confusion_table.columns = label_set

    for i in range(confusion_table_long.shape[0]):
        row = labels[i]
        col = predictions[i]
        confusion_table.loc[row][col] += 1

    return confusion_table


def remove_genes_from_list(adata, discarded_genes):
    """
    Remove the genes provided in discarded genes from the input adata

    Arguments:
        adata (anndata.AnnData): Input adata
        discarded_genes (list): A list of genes available in the input adata

    Returns:
        adata (anndata.AnnData)
    """
    logger.info("Discarding bad genes")

    keep_genes = [gene not in discarded_genes for gene in adata.var.index]
    adata = adata[:, keep_genes]
    logger.info(f"adata has now {adata.shape[1]} genes")
    return adata


def _read_h5ad(file, *args, **kwargs):
    """
    Read a cached anndata.AnnData using scanpy
    """
    return sc.read_h5ad(file, *args, **kwargs)


def _highly_variable_genes(*args, **kwargs):
    return sc.pp.highly_variable_genes(*args, **kwargs)


def _use_raw_anndata(adata):
    """
    Make the single cell dataset stored in .raw the actual dataset
    """

    X = adata.raw.X.toarray()
    adata = anndata.AnnData(X=X, obs=adata.obs, var=adata.raw.var)
    return adata


def _concatenate_adatas(adatas):
    return anndata.concat(adatas)


def _get_cluster(adata, louvain_resolution, idx):
    """
    set a single cell dataset
    by keeping only the cells that got assigned
    a particular index at a particular louvain resolution

    Arguments:

        adata (anndata.AnnData):
        louvain_resolution (str): Column available in the .obs table
            that stores the indices of each single cell at the desired louvain resolution
            This column must already exist in the Anndata object

    Returns:
        barcodes (list): Strings representing the barcodes (single cell identifiers) of the cells
        that match the passed identifiers

    """
    assert louvain_resolution in adata.obs.columns

    idx = str(idx)
    barcodes = adata.obs.loc[adata.obs[louvain_resolution] == idx].index.tolist()
    logger.info(f"n cells = {len(barcodes)}")
    logger.info(f"n genes = {adata.var.shape[0]}")
    return barcodes


def _find_barcodes_belonging_to_clusters(adata, identifiers):
    """
    Given an adata and a list of clusters, provided as a list of tuples (louvain_res, idx),
    return the barcodes of the cells that belong to these clusters

    Arguments:
        adata (anndata.AnnData):
        identifiers (list): List of identifiers, where each element is a tuple containing
            1. the name of a column in the .obs table,
                which stores the result of a louvain clustering at a particular resolution
            2. the index at the said resolution

    Returns:
        barcodes (list): Strings representing the barcodes (single cell identifiers) of the cells
        that match the passed identifiers
    """

    matched_barcodes = []
    for louvain_resolution, idx in identifiers:
        barcodes = _get_cluster(adata, louvain_resolution, idx)
        matched_barcodes.extend(barcodes)

    n_repeated_cells = len(matched_barcodes) - len(set(matched_barcodes))
    if n_repeated_cells != 0:
        logger.warning(f"{n_repeated_cells} cells appear in more than one cluster")
        matched_barcodes = list(set(matched_barcodes))

    return matched_barcodes


def assign_cell_type(adata, background):
    """
    Annotate on the CellType column of adata.obs

    Arguments:

        adata (anndata.AnnData): An Anndata object annotated with louvain_res and idx for each cell
        background (pd.DataFrame): A data frame with columns [cluster,louvain_res,idx]

    Returns:
        adata with new column called CellType containing the cell type (cluster) annotation of each cell
    """

    clusters = [
        (row["louvain_res"], str(row["idx"])) for i, row in background.iterrows()
    ]

    barcodes = [
        _find_barcodes_belonging_to_clusters(adata, [clusters[i]])
        for i in range(len(clusters))
    ]

    cluster_barcodes = list(zip(background["cluster"].values, barcodes))

    cell_type = [None,] * adata.obs.shape[0]

    for i in range(len(cluster_barcodes)):

        cell_type_name, cell_type_barcodes = cluster_barcodes[i]
        cell_type_pos = [barcode in cell_type_barcodes for barcode in adata.obs.index]
        for j, status in enumerate(cell_type_pos):
            if status:
                cell_type[j] = cell_type_name

    adata.obs["CellType"] = cell_type
    return adata
