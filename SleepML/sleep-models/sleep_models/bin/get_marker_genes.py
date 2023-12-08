import argparse
import os
import os.path
import sys
import glob

import sleep_models.dimensionality_reduction as dr
import sleep_models.preprocessing as pp
import sleep_models.plotting as plotting
from sleep_models.dimensionality_reduction.algorithms import ALGORITHMS


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
        "--output",
        type=str,
        help="Folder to which other output is saved",
        required=True,
    )
    ap.add_argument(
        "--algorithm",
        type=str,
        help="DR algoritm",
        choices=ALGORITHMS.keys(),
        required=True,
    )
    ap.add_argument(
        "--max-clusters",
        dest="max_clusters",
        help="""
        Genes are considered marker gene for the analysis
        if they are a marker gene of less than this number of clusters.
        For KC, 3 is a sensible default (3 different cell types)
        But for glia, which are many more cell types, it makes
        sense to choose a value under the number of cell types, like 5.
        This is not a typo. If a gene is a marker of many clusters of the background
        it is not differentiating between them! It is a marker of the background
        but not of the separate cell types
        """,
    )
    ap.add_argument("--ncores", default=1, type=int)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="""
        absolute logFC thresholds to use
        when defining marker genes.
        Genes with a higher abs_logFC
        will be considered marker genes
        """,
    )

    return ap


def get_marker_genes(
    h5ad_input, marker_database, output, max_clusters, thresholds, algorithm, ncores=1, cache=False
):

    os.makedirs(output, exist_ok=True)

    # load the data
    adata = pp.read_h5ad(h5ad_input)
    cell_types = list(set(adata.obs["CellType"].values.tolist()))
    markers = dr.get_markers(cell_types, marker_database)

    background = os.path.basename(h5ad_input).rstrip("h5ad")
    name = background + " logFC < %s"

    # generate the initial DR
    embedding = dr.SingleCellEmbedding.analysis(
        adata=adata,
        root_fs=output,
        markers=markers,
        algorithm=algorithm,
        max_clusters=max_clusters,
        threshold=None,
        reducer=None,
        normalize=True,
        name=name,
        limits=None,
    )

    # for each threshold
    # compute a new DR and
    # compute the distances
    # and the silhouette
    dr.homogenize(
        output=output,
        adata=adata,
        reducer=embedding.reducer,
        algorithm=embedding.algorithm,  # just the name of the reducer
        markers=markers,
        thresholds=thresholds,
        max_clusters=max_clusters,
        ncores=ncores,
        name=name,
        cache=cache,
        limits=embedding._limits,
    )
    paths = sorted(
        glob.glob(
            os.path.join(
                output, "threshold-*", "png", f"{embedding.algorithm}_threshold-*.png"
            )
        ),
        reverse=True,
    )
    plotting.make_gif(
        paths, os.path.join(output, f"{embedding.algorithm}_homogenization.gif")
    )

    return 0


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    get_marker_genes(
        args.h5ad_input,
        args.output,
        args.max_number_of_clusters_per_marker_gene,
        args.thresholds,
        algorithm=args.algorithm,
        ncores=1,
    )


if __name__ == "__main__":
    main()
