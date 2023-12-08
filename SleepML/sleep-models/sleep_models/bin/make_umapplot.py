import argparse
import os.path
import pickle
import numpy as np
from sleep_models.plotting import make_umap_plot
from sleep_models import preprocessing as pp


def get_embedding(path):

    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist")

    with open(path, "rb") as filehandle:
        embedding = pickle.load(filehandle)

    return embedding


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    try:
        threshold = float(args.threshold)

    except Exception as error:
        if args.threshold == "inf":
            threshold = np.inf
        else:
            raise error

    embedding = get_embedding(
        os.path.join(
            args.input, f"threshold-{threshold}", f"{args.algorithm}_embedding.pkl"
        )
    )

    adata = pp.read_h5ad(args.h5ad_input)

    marker_genes_file = os.path.join(
        args.input, f"threshold-{threshold}", "marker_genes.txt"
    )
    with open(marker_genes_file, "r") as filehandle:
        marker_genes = filehandle.readlines()
        marker_genes = [e.strip("\n") for e in marker_genes]
        marker_genes = list(set(marker_genes))

    os.makedirs(args.output, exist_ok=True)
    _, _, (centers, distances, silhouette) = make_umap_plot(
        args.output,
        threshold,
        embedding,
        adata,
        title=None,
        marker_genes=marker_genes,
        ignore_cell_types=args.ignore_cell_types,
    )


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
        "--input",
        type=str,
        help="Folder to look for threshold-X.X folders",
        required=True,
    )
    ap.add_argument(
        "--output",
        type=str,
        help="Folder to which other output is saved",
        required=True,
    )
    ap.add_argument(
        "--threshold",
        type=str,
        help="""
        absolute logFC threshold to use
        when defining marker genes.
        Genes with a higher abs_logFC
        will be considered marker genes
        """,
    )
    ap.add_argument(
        "--ignore-cell-types",
        dest="ignore_cell_types",
        type=str,
        nargs="+",
        default=None,
        help="""
        Cell types to ignore
        """,
    )

    return ap


if __name__ == "__main__":
    main()
