import argparse
import os.path

import pandas as pd
import numpy as np

from sleep_models.plotting import make_matrix_from_array, make_matrixplot


def get_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--prediction-results",
        "--input",
        type=str,
        dest="prediction_results",
        help="Folder where the replicate data splits+train have been saved",
        required=True,
    )
    ap.add_argument(
        "--barlimits",
        type=int,
        nargs="+",
        help="Minimum and maximum percent value of accuracy encompassed by the colorbar. Example: 50 70 will assign the lowest color to 50 percent acc and the highest to 70 percent. Everything above 70 or below 50 will look like 70 and 50, respectively.",
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


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    make_matrixplot_main(
        args.prediction_results,
        barlimits=args.barlimits,
        ignore_cell_types=args.ignore_cell_types,
    )


def make_matrixplot_main(
    prediction_results, barlimits=None, ignore_cell_types=None, **plotting_kwargs
):

    replicate_folders = os.listdir(prediction_results)
    replicate_folders = [f for f in replicate_folders if f.startswith("random-state")]

    results = {
        f: (
            pd.read_csv(
                os.path.join(prediction_results, f, "accuracy.csv"), index_col=0,
            ),
            None,
        )
        for f in replicate_folders
    }
    for key in results:
        dataframe = results[key][0]
        if ignore_cell_types:
            for cell_type in ignore_cell_types:
                dataframe = dataframe.drop(cell_type, axis=0).drop(cell_type, axis=1)

        results[key] = (dataframe, *results[key][1:])

    clusters = results[list(results.keys())[0]][0].columns

    # make matrixplot
    replicates = np.array([tables[0].values for tables in results.values()])

    accuracies = {}
    for statistic in ["mean", "std"]:
        accuracy = pd.DataFrame(getattr(replicates, statistic)(axis=0))
        accuracy.index = list(results.values())[0][0].index
        accuracy.columns = list(results.values())[0][0].columns
        accuracy.to_csv(os.path.join(prediction_results, f"matrixplot_{statistic}.csv"))
        accuracies[statistic] = accuracy

    img = make_matrix_from_array(accuracies["mean"].values)

    return make_matrixplot(
        img,
        clusters,
        filenames=tuple(
            os.path.join(prediction_results, f"matrixplot.{ext}")
            for ext in ["svg", "png"]
        ),
        barlimits=barlimits,
        **plotting_kwargs,
    )


if __name__ == "__main__":
    main()
