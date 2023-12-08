import os.path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

STATIC_DATA = "tests/static_data/"


def make_data():

    X_y = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    X_ab = np.array([[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    X_abp = np.array([[0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]])

    obs = pd.DataFrame(
        {
            "CellType": ["y", "y", "y", "ab", "ab", "ab", "abp", "abp", "abp"],
            "Condition": [
                "SD++",
                "sleep",
                "SD",
                "SD++",
                "sleep",
                "SD",
                "SD++",
                "sleep",
                "SD",
            ],
            "Treatment": [
                "wake",
                "sleep",
                "wake",
                "wake",
                "sleep",
                "wake",
                "wake",
                "sleep",
                "wake",
            ],
        },
        index=[str(i) for i in range(9)],
    )

    var = pd.DataFrame(
        {
            "gene_ids": [f"gene{i}" for i in range(12)],
        },
        index=[f"gene{i}" for i in range(12)],
    )

    X = np.vstack(
        [
            X_y + np.random.normal(loc=0.0, scale=1.0, size=(3, 12)),
            X_ab + np.random.normal(loc=0.0, scale=1.0, size=(3, 12)),
            X_abp + np.random.normal(loc=0.0, scale=1.0, size=(3, 12)),
        ]
    )

    return X, obs, var


def make_simple_adata():

    X, obs, var = make_data()

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    print(adata)

    adata.write_h5ad(os.path.join(STATIC_DATA, "test_adata.h5ad"))
    adata_without_gene = anndata.AnnData(X=X[:, :11], obs=obs, var=var.iloc[:11, :])
    adata_without_gene.write_h5ad(
        os.path.join(STATIC_DATA, "test_adata_without_gene.h5ad")
    )


def make_complex_adata(n=100):

    X, obs, var = make_data()
    for i in range(n - 1):
        X_i, obs_i, _ = make_data()
        X = np.vstack([X, X_i])
        obs = pd.concat([obs, obs_i])

    obs.index = np.arange(obs.shape[0])

    print(X)

    # create a fake adata_raw
    adata_raw = anndata.AnnData(X=sparse.csr.csr_matrix(X), obs=obs, var=var)
    adata = anndata.AnnData(X=X, obs=obs, var=var, raw=adata_raw)
    adata.write_h5ad(os.path.join(STATIC_DATA, "test_complex_adata.h5ad"))


def main():
    make_simple_adata()
    make_complex_adata()


if __name__ == "__main__":
    main()
