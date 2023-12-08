import os.path
import pandas as pd
from sleep_models.plotting import plot_confusion_table
from sleep_models.preprocessing import make_confusion_long_to_square


def load_and_plot_confusion_table(folder, cluster_data):
    confusion_table_long = pd.read_csv(
        os.path.join(folder, f"{cluster_data}_confusion_table.csv"), index_col=0
    )

    confusion_table_square = make_confusion_long_to_square(confusion_table_long)

    print(confusion_table_square)

    plot_confusion_table(
        confusion_table_square,
        os.path.join(folder, f"{cluster_data}_confusion_table.png"),
    )
