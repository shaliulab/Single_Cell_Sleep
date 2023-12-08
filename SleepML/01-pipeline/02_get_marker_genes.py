# Progressively remove marker genes starting from those with highest logFC
# Use a list of logFC thresholds, given in the config
# Only consider a marker gene if it's not shared by most celltypes in the background


import os.path
import yaml
from sleep_models.bin.get_marker_genes import get_marker_genes

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

RESULTS_DIR = config["results_dir"]
cache = config["cache"]
TEMP_DATA_DIR = config["temp_data_dir"]
DATA_DIR = config["data_dir"]

for background in config["background"]:
    max_clusters_per_marker = config["max_clusters"][background]
    thresholds = config["log2FC_thresholds"][background]

    for algorithm in config["DR_algorithm"]:
        get_marker_genes(
            h5ad_input=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}.h5ad"),
            marker_database=os.path.join(DATA_DIR, config["marker_database"]),
            output=os.path.join(RESULTS_DIR, f"{background}_get-marker-genes/"),
            max_clusters=max_clusters_per_marker,
            thresholds=thresholds,
            ncores=1,
            cache=cache,
            algorithm=algorithm,
        )
        # for i in range(config["shuffles"]):
        #     get_marker_genes(
        #         h5ad_input=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}_shuffled_{i}.h5ad"),
        #         output=os.path.join(RESULTS_DIR, f"{background}_shuffled_{i}_get-marker-genes/"),
        #         max_clusters=max_clusters_per_marker,
        #         thresholds=thresholds,
        #         ncores=1,
        #         cache=cache,
        #         algorithm=algorithm,
        #     )