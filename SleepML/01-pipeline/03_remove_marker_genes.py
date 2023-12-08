import os.path
import yaml

from sleep_models.bin.remove_marker_genes import remove_marker_genes

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

TEMP_DATA_DIR = config["temp_data_dir"]
RESULTS_DIR = config["results_dir"]

for background in config["background"]:

    threshold = str(float(config["user_defined_log2FC_threshold"][background]))

    marker_gene_file=os.path.join(
        RESULTS_DIR,
        f"{background}_get-marker-genes",
        f"threshold-{threshold}",
        "marker_genes.txt"
    )
    remove_marker_genes(
        h5ad_input=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}.h5ad"),
        h5ad_output=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}-no-marker-genes.h5ad"),
        marker_gene_file=marker_gene_file,
    )

    for i in range(config["shuffles"]):
        remove_marker_genes(
            h5ad_input=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}_shuffled_{i}.h5ad"),
            h5ad_output=os.path.join(TEMP_DATA_DIR, f"h5ad/{background}_shuffled_{i}-no-marker-genes.h5ad"),
            marker_gene_file=marker_gene_file,
        )
