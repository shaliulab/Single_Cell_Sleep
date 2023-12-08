import argparse
import os.path
import pickle
import sys

import numpy as np
import pandas as pd

import sleep_models.dimensionality_reduction as dr
import sleep_models.preprocessing as pp


def get_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--h5ad-input",
        type=str,
        dest="h5ad_input",
        help=".h5ad with input adata",
        required=True,
    )
    ap.add_argument(
        "--marker-gene-file",
        type=str,
        dest="marker_gene_file",
        help="text file with a marker gene on each line",
        required=True,
    )
    ap.add_argument(
        "--h5ad-output",
        type=str,
        dest="h5ad_output",
        help=".h5ad with marker genes removed",
        required=True,
    )

    return ap


def remove_marker_genes(
    h5ad_input, h5ad_output, marker_gene_file,
):

    adata = pp.read_h5ad(h5ad_input)
    adata_wo_marker_genes = adata.copy()

    with open(marker_gene_file, "r") as fh:
        markers = [gene.strip("\n") for gene in fh.readlines()]

    non_marker_genes = [gene not in markers for gene in adata.var.index]
    adata_wo_marker_genes = adata_wo_marker_genes[:, non_marker_genes]

    adata_wo_marker_genes._uns.update({"removed_marker_genes": marker_gene_file})
    adata_wo_marker_genes.write_h5ad(h5ad_output)


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    remove_marker_genes(
        h5ad_input=args.h5ad_input,
        h5ad_output=args.h5ad_output,
        marker_gene_file=args.marker_gene_file,
    )


if __name__ == "__main__":
    main()
