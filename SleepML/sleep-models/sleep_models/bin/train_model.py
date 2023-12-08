import argparse
import os.path
import logging
import logging

from sleep_models.models import MODELS
import sleep_models.preprocessing as pp
from sleep_models.utils.logging import setup_logging
from sleep_models.utils.data import backup_dataset
from sleep_models.constants import MEAN_SCALE, HIGHLY_VARIABLE_GENES
from sleep_models.models.train import train_model
import sleep_models.models.utils.config as config_utils
from sleep_models.models.variables import AllConfig
import sleep_models.models.utils.torch as torch_utils


def base_parser(ap=None, models=MODELS):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("cluster", type=str, help="Which cluster should I model?")
    ap.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Folder on which to save the results",
        required=True,
    )
    ap.add_argument(
        "--random-state",
        "--seed",
        type=int,
        default=1000,
        help="The following stochastic procedures occur in this script: 1) train-test split 2) EBM initialization 3) Sample off All_Combined. Setting the value of this argument makes them deterministic. Pass a different value to generate a different technical replicate",
    )
    ap.add_argument(
        "--architecture",
        dest="arch",
        type=str,
        help="Name of model. One of the sleep models in the models.py module",
        choices=models,
    )

    return ap


def get_parser(ap=None, *args, **kwargs):

    ap = base_parser(ap, *args, **kwargs)
    # TODO This should be on the generate-dataset script
    ap.add_argument(
        "--highly-variable-genes-only",
        dest="highly_variable_genes",
        action="store_true",
        default=HIGHLY_VARIABLE_GENES,
    )
    ap.add_argument(
        "--h5ad-input",
        type=str,
        dest="h5ad_input",
        help=".h5ad with input adata",
        required=True,
    )
    ap.add_argument("--label-mapping", dest="label_mapping", default=None)

    ap.add_argument(
        "--exclude-genes-file",
        type=str,
        dest="exclude_genes_file",
        help="Genes contained in this file will not be used in the analysis. The program expects one gene per line (separated by \n)",
    )
    ap.add_argument(
        "--verbose",
        type=int,
        default=20,
        help="Logging severity level. 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR. A lower value is more verbose and a higher value less",
    )
    ap.add_argument(
        "--logfile", type=str, default="train_model.log", help="Logging output",
    )

    return ap


def pick_cluster(adata, cluster, background_mapping):
    adata = pp.get_cluster(
        adata,
        background_mapping.loc[background_mapping["cluster"] == cluster][
            "louvain_res"
        ].values.tolist()[0],
        str(
            background_mapping.loc[background_mapping["cluster"] == cluster][
                "idx"
            ].values.tolist()[0]
        ),
    )
    return adata


def setup_model_and_datasets(
    adata, arch, cluster, output, random_state, logger, **model_kwargs
):

    logger.info("Initializing model")
    ModelClass = MODELS[arch]
    model = ModelClass(
        name=cluster, output=output, random_state=random_state, **model_kwargs,
    )

    logger.info(f"Preparing dataset {cluster} and model {arch}")
    X_train, y_train, X_test, y_test = model.split_dataset(adata=adata)
    logger.info("Backing up datasets")

    backup_dataset(cluster, X_train, y_train, X_test, y_test, output=output)

    return (model, (X_train, y_train, X_test, y_test))


def train(
    h5ad_input,
    arch,
    output,
    cluster,
    target,
    random_state=1000,
    highly_variable_genes=True,
    verbose=logging.WARNING,
    label_mapping=None,
    logfile=None,
    fraction=1.0,
):
    """
    Train a sleep_models API model using a single cell dataset contained in an h5ad file

    Arguments:

        h5ad_input (str): Path to an h5ad file containing an anndata.AnnData
            This AnnData must fulfil the following requirements
                * .obs["CellType"] is defined
                * .obs["Condition"] or .obs["Treatment"] are defined (depending on the selected  target)
                * .X is a numpy array with shape ncellsxngenes
        arch (str): One of the classes available under sleep_models.models.MODELS.
            The selected target attribute must be one of the columns in the adata.obs pd.DataFrame
        output (str): Path to a folder where the results will be saved.
            On this folder, a new folder will be created with name random-state-{random_state}
        cluster (str): CellType to filter by, so only cells whose CellType matches are used
        random_state (int): random random_state for reproducibility
        exclude_genes_file (str): Path to a plain text file where every line contains the name of a gene to be ignored in the analysis
        highly_variable_genes (bool): If True, only the genes labeled as highly variable are used, otherwise all are used
            A gene is highly variable if it is found so in `sleep_models.preprocessing.read_h5ad`

    Returns: None
    """

    if logfile is None:
        logfile = os.path.join("logs", f"train_model_{cluster}.log")

    logger, _ = setup_logging(verbose, logfile)

    logger.info(f"Training on cell type {cluster} starting!")
    output = os.path.join(output, arch, f"random-state_{random_state}")
    os.makedirs(output, exist_ok=True)

    ModelClass = MODELS[arch]

    data = pp.load_data(
        h5ad_input,
        output=output,
        cluster=cluster,
        random_state=random_state,
        highly_variable_genes=highly_variable_genes,
        label_mapping=label_mapping,
        model_properties=ModelClass.model_properties(),
        target=target,
        fraction=fraction,
    )

    training_config = config_utils.setup_config(arch)
    device = torch_utils.get_device()

    config = AllConfig(
        model_name=arch,
        training_config=training_config,
        cluster=cluster,
        output=output,
        device=device,
        random_state=random_state,
        target=target,
    )

    model = ModelClass.new_model(
        config,
        # these last three are ignored in the EBM, KNN and MLP
        X_train=data["datasets"][0],
        y_train=data["datasets"][1],
        encoding=data["encoding"],
    )

    train_model(model, data["datasets"])

    logger.info(f"Training on cell type {cluster} finished successfully")


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    return train(
        h5ad_input=args.h5ad_input,
        arch=args.arch,
        output=args.output,
        cluster=args.cluster,
        exclude_genes_file=args.exclude_genes_file,
        highly_variable_genes=args.highly_variable_genes,
        verbose=args.verbose,
        logfile=args.logfile,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
