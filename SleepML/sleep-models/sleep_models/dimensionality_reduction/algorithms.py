from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from umap import UMAP  # takes a while to load

ALGORITHMS = {
    "TruncatedSVD": TruncatedSVD,
    "UMAP": UMAP,
    "TSNE": TSNE,
}
