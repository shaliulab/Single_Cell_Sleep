import argparse
import os.path
import logging
import json
import pandas as pd
import numpy as np
import sleep_models.preprocessing as pp


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--h5ad-input",
            type=str,
            dest="h5ad_input",
            help=".h5ad with input adata",
            required=True,
        )
        ap.add_argument(
            "--h5ad-output",
            type=str,
            dest="h5ad_output",
            help="Path to generated h5ad file",
            required=True,
        )
        ap.add_argument(
            "--random-state",
            "--seed",
            type=int,
            help="The following stochastic procedures occur in this script: 1) train-test split 2) EBM initialization 3) Sample off All_Combined. Setting the value of this argument makes them deterministic. Pass a different value to generate a different technical replicate",
            required=True,
        )
        ap.add_argument(
            "--raw",
            dest="raw",
            action="store_true",
            help="If passed, the raw anndata.AnnData (andata.AnnData.raw) will be used",
            default=True,
        )
        ap.add_argument(
            "--not-raw",
            dest="raw",
            action="store_false",
            help="If passed, the default anndata.AnnData will be used",
            default=True,
        )
        ap.add_argument(
            "--background",
            type=str,
            help="This file tells the program which clusters should the current cluster be compared to",
            required=True,
        )
        ap.add_argument(
            "--exclude-genes-file",
            type=str,
            dest="exclude_genes_file",
            help="Genes contained in this file will not be used in the analysis. The program expects one gene per line (separated by \n)",
        )
        ap.add_argument(
            "--batch-genes-file",
            type=str,
            dest="batch_genes_file",
            help="Excel sheet of batch effect genes",
        )
        ap.add_argument(
            "--template-file",
            type=str,
            dest="template_file",
            help="If passed, a template is matched to the dataset",
            required=False,
        )
        ap.add_argument(
            "--shuffles",
            type=int,
            default=0,
            dest="shuffles",
            help="Number of random shuffles of the input dataset",
        )
        ap.add_argument(
            "--verbose",
            type=int,
            default=30,
            help="Logging severity level. 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR. A lower value is more verbose and a higher value less",
        )
        ap.add_argument(
            "--logfile", type=str, default="make_dataset.log", help="Logging output",
        )

    return ap


def make_dataset(
    h5ad_input,
    h5ad_output,
    background,
    random_state,
    batch_genes_file=None,
    exclude_genes_file=None,
    template_file=None,
    shuffles=0,
    logfile="make_dataset.log",
    raw=True,
    verbose=30,
    pinned_columns=[],
):

    """
    Arguments:

    h5ad (str): Path to an h5ad file containing a cached anndata.AnnData object
    raw (bool): If True, the .raw.X slot of the AnnData is used to read the counts matrix, instead of .X

    """

    logging.basicConfig(
        filename=logfile,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=verbose,
    )

    if template_file is None:
        template_filename = "notemplate"
    else:
        template_filename = os.path.basename(template_file)

    filename = (
        os.path.basename(background.strip(".csv")) + "-" + template_filename + ".h5ad"
    )

    np.random.seed(random_state)

    logging.info("Loading anndata to memory")
    adata = pp.read_h5ad(h5ad_input, raw=raw)
    background_file = background
    background = pd.read_csv(background, index_col=False, comment="#")

    adata = pp.assign_cell_type(adata, background)

    if template_file:
        with open(template_file, "r") as fh:
            template = json.load(fh)

        adata = pp.template_matching(adata, template)

    assert "CellType" in adata.obs.columns

    # preprocess
    bad_genes = pp.get_bad_genes(batch_genes_file, exclude_genes_file)
    adata = pp.remove_genes_from_list(adata, bad_genes)

    adata._uns.update({"background": background_file})

    adata = pp.keep_cells_from_this_background(adata, background)
    logging.info(f"Saving h5ad to disk at {h5ad_output}")
    adata.write_h5ad(h5ad_output)

    for i in range(shuffles):
        filename = h5ad_output.replace(".h5ad", f"_shuffled_{i}.h5ad")
        adata_shuffled = pp.shuffle_adata(
            adata, filename, pinned_columns=pinned_columns
        )


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    make_dataset(
        h5ad_input=args.h5ad_input,
        h5ad_output=args.h5ad_output,
        background=args.background,
        random_state=args.random_state,
        batch_genes_file=args.batch_genes_file,
        exclude_genes_file=args.exclude_genes_file,
        template_file=args.template_file,
        shuffles=args.shuffles,
        logfile=args.logfile,
        raw=args.raw,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
