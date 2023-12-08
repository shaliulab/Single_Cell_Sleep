import os.path
from typing import Iterable
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.figsize"] = (15, 15)
plt.tight_layout()
plt.rc("font", family="monospace", weight="bold", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=80)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=60)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=60)  # fontsize of the y tick labels
plt.rc("legend", fontsize=20)  # fontsize of the legend
plt.set_cmap("seismic")
plt.rcParams["figure.figsize"] = (30, 15)
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.size"] = 22

from PIL import Image
import cv2

ON_COLOR = tuple([e / 255 for e in [0, 100, 0]])
OFF_COLOR = tuple([e / 255 for e in [144, 238, 144]])


def make_gif(paths, output):

    images = [Image.fromarray(cv2.imread(p)) for p in paths]

    images[0].save(
        fp=output,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=1000,  # ms per frame
        loop=0,
    )


def make_matrix_from_array(array):
    """
    Given a table wide format where cell i,j contains the accuracy
    of model i on cluster j,
    give me the numpy array resulting from taking the accuracy as a gray color (0-255)
    """

    assert array.max() <= 1
    assert array.min() >= 0

    # acc goes from 0 to 1, images from 0 to 255
    array *= 255
    # make it into an image where comparison is a pixel with a grayscale color
    matrix_uint8 = np.uint8(np.round(array))
    return matrix_uint8


def plot_training_and_test(y_train_flat, y_pred_train, y_test_flat, y_pred):
    fig = plt.figure(figsize=[10, 7])
    ax0 = make_scatterplot(fig, y_train_flat, y_pred_train, 1, "Training")
    ax1 = make_scatterplot(fig, y_test_flat, y_pred, 2, "Test")
    return fig, (ax0, ax1)


def make_scatterplot(fig, y, y_pred, i, label):

    ax = fig.add_subplot(1, 2, i)
    ax.scatter(y, y_pred)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="red")
    ax.set_title(label)
    return ax


def make_matrixplot(
    matrix: np.uint8,
    clusters: Iterable,
    filenames: Iterable[str],
    dpi=600,
    rotation=(0, 0),
    barlimits=None,
):
    """
    Produce a matrixplot where the color of the i,j square maps to the number stored in the i,j cell of matrix
    Labels for x and y axes are taken from clusters
    Matrixplot is saved to filename
    A colorbar is produced
    """

    # customise appearance (font and color)
    plt.tight_layout()
    plt.rc(
        "font", family="monospace", weight="bold", size=10
    )  # controls default text size
    plt.rc("axes", titlesize=10)  # fontsize of the title
    plt.rc("axes", labelsize=80)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=60)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=60)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=20)  # fontsize of the legend
    plt.set_cmap("seismic")

    # plot the data
    fig = plt.figure()
    ax = plt.gca()
    matrix = np.uint8(100 * np.float64(matrix) / 255)
    RES = 5

    if barlimits is None:
        min_bar = matrix.max()
        max_bar = matrix.min()
        img = ax.imshow(matrix.T)
        mult = 1

    else:
        min_bar = barlimits[0]
        max_bar = barlimits[1]
        matrix = cv2.resize(
            matrix,
            (matrix.shape[0] * 100, matrix.shape[1] * 100),
            interpolation=cv2.INTER_AREA,
        )
        matrix[0, 0] = min_bar
        matrix[-1, -1] = max_bar
        img = ax.imshow(matrix.T)
        mult = 100

    # customise axes
    ax.set_xticks(
        np.arange(0, len(clusters) * mult, mult) + mult // 2
    )  # my_datasets, rotation=80)
    ax.set_xticklabels(clusters, rotation=rotation[0])
    ax.set_yticks(
        np.arange(0, len(clusters) * mult, mult) + mult // 2
    )  # my_datasets, rotation=80)
    ax.set_yticklabels(clusters, rotation=rotation[1])
    ax.set_ylabel("Trained on")
    ax.set_xlabel("Predicts  on")

    # configure colorbar
    ## create an axes on the right side of ax. The width of cax will be 5%
    ## of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ticks = np.arange(min_bar, max_bar + 1, RES)
    plt.colorbar(img, cax=cax, ticks=ticks)

    # save
    for f in filenames:
        fig.savefig(f, transparent=True, dpi=dpi, bbox_inches="tight")

    return fig


def make_dr_plot(
    embedding,
    adata,
    threshold,
    centers,
    center_pairs,
    distances,
    output,
    title,
    marker_genes=None,
    limits=None,
    style="default",
):

    os.makedirs(os.path.join(output, f"png"), exist_ok=True)
    os.makedirs(os.path.join(output, f"svg"), exist_ok=True)

    if marker_genes is None:
        fraction_of_genes_that_are_marker_genes = 0
    else:
        fraction_of_genes_that_are_marker_genes = round(
            len(marker_genes) / adata.var.shape[0], 3
        )

    fraction_of_genes_that_are_not_marker_genes = (
        1 - fraction_of_genes_that_are_marker_genes
    )

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.rcParams["legend.loc"] = "upper right"
    plt.rcParams["font.size"] = 22

    cell_types = list(centers.keys())
    color_idxs = {c: i for i, c in enumerate(cell_types)}
    palette = sns.color_palette()

    with plt.style.context(style):

        fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [0.05, 0.95]})
        ax = axs[1]

        for cell_name in cell_types:

            x = embedding[np.where(adata.obs["CellType"] == cell_name), 0]
            y = embedding[np.where(adata.obs["CellType"] == cell_name), 1]

            color_idx = color_idxs[cell_name]
            color = [palette[color_idx]]

            ax.scatter(x, y, c=color, s=0.1, label=cell_name)
            ax.scatter(*centers[cell_name], c=color, s=1000, marker="*")

        ax.legend(markerscale=20)

        for c1, c2 in center_pairs:

            dist = distances[(c1, c2)]

            pts = list(zip(centers[c1], centers[c2]))
            pts = [np.array(e) for e in pts]
            midline = [e.mean() for e in pts]

            ax.plot(*pts, c="black")
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.annotate(str(dist), midline, fontsize=20)

            if limits is not None:
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])

        axs[0].barh([""], [1], color=OFF_COLOR)
        axs[0].barh([""], [fraction_of_genes_that_are_not_marker_genes], color=ON_COLOR)
        axs[0].set_yticks([])
        axs[0].set_yticklabels([])
        axs[0].set_xlim([0, 1])
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[0].annotate(
            str(round(fraction_of_genes_that_are_not_marker_genes * 100, 3)) + " %",
            xy=(fraction_of_genes_that_are_not_marker_genes + 0.025, 0),
            textcoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
            size=22,
        )
        # axs[0].barh([""], [fraction_of_genes_that_are_marker_genes], color=OFF_COLOR)

    plt.gca().set_aspect("equal", "datalim")
    plt.title(title, fontsize=24)
    # https://stackoverflow.com/a/24707567/3541756
    # lgnd = plt.legend(scatterpoints=1, fontsize=40)
    # for h in lgnd.legendHandles:
    #     h._sizes = [100]
    return fig, ax


def plot_accuracy_by_label(acc, output):

    accuracy_by_class = {k: acc[k][1] / (acc[k][0] + acc[k][1]) for k in acc}

    y = list(accuracy_by_class.values())
    x = list(accuracy_by_class.keys())

    plt.bar(x, y)
    plt.savefig(output)
    plt.clf()


def plot_confusion_table(confusion_table, output):
    """ """

    values = confusion_table.values

    for i in range(values.shape[0]):
        values[i, :] = values[i, :] / values.sum(axis=1)[i]

    values *= 255
    # confusion_table = pd.DataFrame(values, index=confusion_table.index, columns=confusion_table.columns)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xticks(np.arange(len(confusion_table.columns)))
    ax.set_yticks(np.arange(len(confusion_table.index)))
    ax.set_xticklabels(confusion_table.columns)
    ax.set_yticklabels(confusion_table.index)
    _ = ax.imshow(values.T, cmap="gray")

    fig.savefig(output)
    plt.clf()
    fig.clear()


def plot_homogenization(output, adata, single_cell_embeddings, df, algorithm):

    nclusters = len(np.unique(adata.obs["CellType"]))
    palette = sns.color_palette()
    embeddings = {
        threshold: single_cell_embeddings[threshold].embedding
        for threshold in single_cell_embeddings
    }

    fig, ax = plt.subplots()

    for i, col in enumerate(df.columns[:nclusters]):
        comb_df = df[[col, "logFC", "silhouette"]]
        x, y = (comb_df["logFC"].tolist(), comb_df[col].tolist())
        ax.plot(x, y, label=col, c=palette[i])

    t_p = [
        (k, len(single_cell_embeddings[k].marker_genes) / adata.shape[1])
        for k in list(sorted(embeddings))
    ]

    x = [e[0] for e in t_p]
    y = [e[1] for e in t_p]
    secax = ax.twinx()
    secax.set_ylabel("%")

    secax.plot(
        x,
        np.array(y) * 100,
        label="genes out",
        linestyle="dotted",
        c=sns.color_palette()[i + 1],
    )
    secax.plot(
        df["logFC"],
        df["silhouette"] * 100,
        label="Silhouette",
        linestyle="dotted",
        c=sns.color_palette()[i + 2],
    )
    ax.legend()
    secax.legend(loc=2)

    ax.invert_xaxis()
    fig.savefig(os.path.join(output, f"{algorithm}_homogenization.png"))
