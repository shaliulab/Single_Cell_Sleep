import time
import warnings
import os.path
import logging
import pickle
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score


from sleep_models.plotting import make_dr_plot, plot_homogenization

from .algorithms import ALGORITHMS

logger = logging.getLogger(__name__)

class SingleCellEmbedding:
    """
    Select the marker genes for a given threshold
    and run a DR algorithm on the dataset resulting
    from setting the counts of these genes to 0
    
    Arguments:

        * output (str): In this folder, a new folder called threshold-X will be created
            with the results of this function
        * adata (anndata.AnnData): Single cell dataset
        * markers (pd.DataFrame): Marker genes of the background
            These are collected by reading .csv files from Scope and filtering relevant marker genes
            i.e. shared among some cell types (but not all or most of them)

        * max_clusters (int): A marker gene shared among this number of clusters or more is not considered
          a marker gene anymore
        * threshold (float): A logFC threshold to consider actual marker genes
        * normalize (bool): If True, the embedding is centered around 0
    
    Returns:
    """

    def __init__(
        self,
        adata,
        root_fs,
        markers,
        max_clusters,
        threshold,
        name,
        algorithm,
        limits=None,
        cache=False,
        reducer=None,
        normalize=True,
    ):
        self.adata = adata
        self.root_fs = root_fs
        self.algorithm = algorithm
        if reducer is None:
            reducer = get_dr_algorithm(algorithm)
            self._fit = True
        else:
            self._fit = False

        self.reducer = reducer
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.name = name
        self.normalize = normalize
        self.output_folder = os.path.join(self.root_fs, f"threshold-{self.threshold}")
        self._marker_genes_raw = markers

        self._count_matrix = None
        self._marker_genes = None
        self._embedding = None
        self._centers = None
        self._center_pairs = None
        self._pair_distance = None
        self._cache = cache
        self._limits = limits

        self._embedding_file = os.path.join(
            self.output_folder, f"{algorithm}_embedding.pkl"
        )
        self._genes_file = os.path.join(self.output_folder, "marker_genes.txt")

    @classmethod
    def analysis(cls, algorithm, *args, **kwargs):
        embedding = cls(*args, algorithm=algorithm, **kwargs)
        embedding.compute()
        fig, ax = embedding.draw_dr_plot(limits=embedding._limits)

        filenames = [
            os.path.join(
                embedding.output_folder,
                "png",
                f"{embedding.algorithm}_threshold-{embedding.threshold}.png",
            ),
            os.path.join(
                embedding.output_folder,
                "svg",
                f"{embedding.algorithm}_threshold-{embedding.threshold}.svg",
            ),
        ]

        for filename in filenames:
            fig.savefig(filename)

        if embedding._limits is None:
            embedding._limits = (ax.get_xlim(), ax.get_ylim())

        # limits None means the limits are computed on the spot
        # based on the coordinates of the embedding
        # i.e. minimize the white area
        fig, ax = embedding.draw_dr_plot(limits=None)

        filenames = [
            os.path.join(
                embedding.output_folder,
                "png",
                f"{algorithm}_zoomin_threshold-{embedding.threshold}.png",
            ),
            os.path.join(
                embedding.output_folder,
                "svg",
                f"{algorithm}_zoomin_threshold-{embedding.threshold}.svg",
            ),
        ]

        for filename in filenames:
            fig.savefig(filename)

        return embedding

    @property
    def embedding(self):
        return self._embedding

    @property
    def count_matrix(self):
        if self._count_matrix is None:
            count_matrix = non_marker_genes_data(self.adata, self.marker_genes, value=0)
            self._count_matrix = count_matrix

        return self._count_matrix

    @property
    def marker_genes(self):
        return self._marker_genes

    @marker_genes.getter
    def marker_genes(self):
        if self._marker_genes is None:
            self._marker_genes = self.select_marker_genes(
                self._marker_genes_raw,
                max_clusters=self.max_clusters,
                threshold=self.threshold,
            )
        return self._marker_genes

    @marker_genes.setter
    def marker_genes(self, value):
        self._marker_genes = value

    # @property
    # def marker_genes(self):
    #     if self._marker_genes is None:
    #         self._marker_genes = self.select_marker_genes(
    #             self._marker_genes_raw,
    #             max_clusters=self.max_clusters,
    #             threshold=self.threshold
    #         )

    #     return self._marker_genes

    @staticmethod
    def select_marker_genes(markers, threshold, max_clusters):
        """
        Arguments:
            * markers (pd.DataFrame): table of marker genes
            * threshold (float): marker genes with a logFC under this value will not be treated as marker genes
        Returns:
            * genes (list): each entry is the name of a marker gene with a logFC above the passed threshold
        """

        counts = pd.DataFrame(markers["gene"].value_counts())
        counts.columns = ["count"]

        # NOTE This line of code is critical
        # This is where we decide which genes are considered marker genes
        # and which ones are not

        # A marker gene is any gene appearing in less than max_clusters
        # If it appears on max_clusters or more, it means it is a marker for many clusters
        # so it does not "differentiate so much between them"
        # It could actually be that it's a marker of each cluster separately
        # and it still looks different across them too
        # However, for most marker genes it is a fair assumption
        # and in the case of the assumption being wrong, it is only making our analysis
        # more stringent

        counts["marker"] = counts["count"] < max_clusters
        ##########################################################################
        marker_genes = counts.index[counts["marker"]].tolist()
        keep = [gene in marker_genes for gene in markers["gene"]]
        markers = markers.loc[keep]
        markers = markers.sort_values("abs_logFC", ascending=False)  # decreasing

        if threshold is None:
            # assume None means infinite i.e. no marker gene actually
            genes = []
        else:
            try:
                markers = markers.loc[markers["abs_logFC"] > threshold]
            except Exception as error:
                print(error)
                raise error
            genes = markers["gene"].tolist()
        return genes

    @property
    def pair_distance(self):
        return self._pair_distance

    @property
    def silhouette(self):
        return silhouette_score(self._embedding, self.adata.obs["CellType"])

    def restore_from_cache(self):

        with open(self._embedding_file, "rb") as fh:
            embedding = pickle.load(fh)

        with open(self._genes_file, "r") as fh:
            marker_genes = [fh.readline()]

        return embedding, marker_genes

    def compute_dimensionality_reduction(self):
        """
        Produce an embedding for the object's count matrix
        using the algorithms in ALGORITHMS
        and an existing projection (stored in the self.reducer object)
        """
        self._embedding = run_reducer(self.reducer, self.count_matrix, fit=self._fit)

        # center the embedding
        if self.normalize:
            self._embedding -= self._embedding.mean(axis=0)

        self.compute_centers()

    def draw_dr_plot(self, limits):

        return make_dr_plot(
            embedding=self._embedding,
            adata=self.adata,
            threshold=self.threshold,
            centers=self._centers,
            center_pairs=self._center_pairs,
            distances=self.pair_distance,
            output=self.output_folder,
            title=self.name % self.threshold,
            marker_genes=self.marker_genes,
            limits=limits,
        )

    def compute(self):
        cache = self.check_cache()
        if self._cache and cache is not None:
            self._embedding = cache[0]
            self.compute_centers()
        else:
            os.makedirs(self.output_folder, exist_ok=True)
            self.compute_dimensionality_reduction()
            self.update_cache()

        self.compute_center_pairs()
        self.compute_distances()

    def compute_centers(self):
        self._centers = {}
        cell_types = self.adata.obs["CellType"].unique()

        for cell_type in cell_types:

            x = self._embedding[np.where(self.adata.obs["CellType"] == cell_type), 0]
            y = self._embedding[np.where(self.adata.obs["CellType"] == cell_type), 1]

            c = (x.mean(), y.mean())
            self._centers[cell_type] = np.array(c)

    def compute_center_pairs(self):

        assert self._centers is not None
        self._center_pairs = list(itertools.combinations(self._centers, 2))

    def compute_distances(self):

        self._pair_distance = {}

        for c1, c2 in self._center_pairs:
            dist = round(
                np.sqrt(np.sum((self._centers[c1] - self._centers[c2]) ** 2)), ndigits=2
            )
            self._pair_distance[(c1, c2)] = dist

    def check_cache(self):

        if (
            os.path.exists(self.output_folder)
            and os.path.exists(self._embedding_file)
            and os.path.exists(self._genes_file)
        ):
            logger.info(f"threshold {self.threshold} already done")
            return self.restore_from_cache()
        else:
            return None

    def update_cache(self):
        with open(self._embedding_file, "wb") as fh:
            pickle.dump(self._embedding, fh)

        with open(self._genes_file, "w") as fh:
            for gene in self.marker_genes:
                fh.write(gene + "\n")


def get_markers(cell_types, marker_database=None):
    """
    Arguments:
        * cell_types (list): cell types whose markers should be loaded
    Returns:
       * markers (pd.DataFrame): table with columns
           gene avg_logFC pval abs_logFC cluster
        where every row belongs to a gene that is a marker of up to max_clusters-1 clusters

    Detail:
       No filtering is applied in this step
    """

    markers = {}

    # concatenate all markers in the background
    for cell_type in cell_types:
        cluster_markers = pd.read_csv(
            os.path.join(marker_database, f"{cell_type}.tsv"), sep="\t"
        )
        cluster_markers["abs_logFC"] = cluster_markers["avg_logFC"].abs()
        # cluster_markers.sort_values("abs_logFC", inplace=True)
        cluster_markers["cluster"] = cell_type
        markers[cell_type] = cluster_markers
    markers = pd.concat(markers.values())
    return markers


def non_marker_genes_data(adata, marker_genes, value=0):
    """
    Arguments:
        * adata (anndata.AnnData): Single cell dataset where .X is np.ndarray
        * marker_genes (list): marker gene names
        * value: should always be zero

    Returns
        * gene_data (np.array) the matrix of counts
        where the marker gene count is set to 0,
        (so it's like they are not observed)
    """
    assert isinstance(adata.X, np.ndarray)

    gene_data = adata.X.copy()
    index = [g in marker_genes for g in adata.var.index]

    if len(index) != 0:
        gene_data[:, index] = value
    return gene_data


def get_dr_algorithm(algorithm):
    """
    Return a DR algoritm
    """
    assert algorithm in ALGORITHMS
    reducer = ALGORITHMS[algorithm]()
    return reducer


def run_reducer(reducer, data, fit=True):

    """
    Arguments:
        * reducer (instance of one of the classes in ALGORITHMS).
            An instance of a DR algorithm with methods transform and fit_transform
        * data (np.ndarray): shape cells x genes storing gene counts
    
    Returns:
        * embedding (np.ndarray): Projection of the cell data onto a dimensionally reduced space

    Calls either transform or fit_transform method of the DR algorithm instance with the passed dataset
    This is useful if you want to project different datasets on to the same space (fit=False)
    or not i.e. separate datasets get different projection spaces
    A projection space is generated by calling fit
    """

    start_time = time.time()
    if fit:
        embedding = reducer.fit_transform(data)
    else:
        f = getattr(reducer, "transform", None)
        if f is None:
            warnings.warn(
                f"{reducer} has no transform method available. Using fit_transform"
            )
            f = getattr(reducer, "fit_transform")
        embedding = f(data)

    elapsed = round(time.time() - start_time, ndigits=2)
    print(f"Done in {elapsed} seconds")
    return embedding


def get_embeddings(
    output,
    adata,
    algorithm,
    reducer,
    markers,
    max_clusters,
    thresholds,
    limits,
    name="DR",
    ncores=1,
    cache=False,
):
    """
    Compute the embedding resulting from removing all marker genes
    defined at each threshold t
    """

    pbar = tqdm(thresholds)

    if ncores == 1:
        results = {}
        for threshold in pbar:
            # https://stackoverflow.com/a/45519268/3541756
            pbar.set_description(f"Computing DR at threshold = {threshold}")
            results[threshold] = SingleCellEmbedding.analysis(
                root_fs=output,
                adata=adata,
                reducer=reducer,
                markers=markers,
                max_clusters=max_clusters,
                threshold=threshold,
                name=name,
                cache=cache,
                limits=limits,
                algorithm=algorithm,
            )

    else:
        parallel_output = Parallel(n_jobs=ncores)(
            delayed(SingleCellEmbedding).analysis(
                root_fs=output,
                adata=adata,
                reducer=reducer,
                markers=markers,
                max_clusters=max_clusters,
                threshold=threshold,
                name=name,
                cache=cache,
                limits=limits,
                algorithm=algorithm,
            )
            for threshold in thresholds
        )
        results = {
            threshold: parallel_output[i] for i, threshold in enumerate(thresholds)
        }

    return results


def homogenize(adata, output, **kwargs):

    single_cell_embeddings = get_embeddings(output=output, adata=adata, **kwargs)
    benchmark = benchmark_homogenization(output, single_cell_embeddings)
    algorithm = list(single_cell_embeddings.values())[0].algorithm
    plot_homogenization(
        output, adata, single_cell_embeddings, benchmark, algorithm=algorithm
    )


def benchmark_homogenization(output, single_cell_embeddings):

    algorithm = list(single_cell_embeddings.values())[0].algorithm

    distances = {
        threshold: single_cell_embeddings[threshold].pair_distance
        for threshold in single_cell_embeddings
    }

    dfs = []
    i = 0
    for threshold, dists in distances.items():

        d = {"-".join(k): v for k, v in dists.items()}
        df = pd.DataFrame(d, index=[i])
        df["silhouette"] = single_cell_embeddings[threshold].silhouette
        df["logFC"] = threshold

        dfs.append(df)
        i += 1

    benchmark = pd.concat(dfs).sort_values("logFC").loc[::-1, :]
    benchmark.to_csv(os.path.join(output, f"{algorithm}_benchmark.csv"))
    return benchmark
