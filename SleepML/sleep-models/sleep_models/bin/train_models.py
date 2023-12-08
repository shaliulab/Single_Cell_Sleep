import argparse
import os.path
import logging
import itertools

from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

from sleep_models.bin.train_model import train_model
from sleep_models.bin.train_model import get_parser as train_parser


def get_parser(ap=None):

    if ap is None:
        ap = train_parser(ap)

    ap.add_argument("--clusters", nargs="+", default=None)
    ap.add_argument("--ncores", type=int, default=1)
    return ap


def main(args=None):
    """
    Train an EBM
    """

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    train_models(
        args, background=args.background, clusters=args.clusters, ncores=args.ncores,
    )


def train_models(
    *args, ncores=-2, background=None, random_states=[1000], clusters=None, **kwargs
):

    assert not (background is None and clusters is None)

    if clusters is None:
        background = pd.read_csv(background, index_col=0, comment="#")
        clusters = background.index.tolist()

    if ncores == 1:
        for cluster in tqdm(clusters):
            for random_state in random_states:
                train_model(*args, cluster=cluster, random_state=random_state, **kwargs)
    else:

        combinations = list(itertools.product(clusters, random_states))
        Parallel(n_jobs=ncores)(
            delayed(train_model)(
                *args,
                cluster=combinations[i][0],
                random_state=combinations[i][1],
                **kwargs
            )
            for i in range(len(combinations))
        )


if __name__ == "__main__":
    main()
