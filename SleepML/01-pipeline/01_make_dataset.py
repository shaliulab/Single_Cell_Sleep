# Create a dataset of single cells sharing a background
# e.g. KC only or glia only
# Optionally removes manually curated batch effect genes

import yaml
import os.path
from sleep_models.bin.make_dataset import make_dataset


with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

DATA_DIR = config["data_dir"]
TEMP_DATA_DIR = config["temp_data_dir"]
H5AD_INPUT = config["h5ad_input"]
SHUFFLES = config["shuffles"]

os.makedirs(
    os.path.join(TEMP_DATA_DIR, 'h5ad'),
    exist_ok=True
)

for background in config["background"]:
    ## for KC
    make_dataset(
        h5ad_input=os.path.join(DATA_DIR, H5AD_INPUT),
        h5ad_output=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad"),
        random_state=1500,
        background=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
        batch_genes_file=os.path.join(DATA_DIR, config["batch_genes_file"]),
        shuffles=SHUFFLES,
        raw=config["raw"],
        pinned_columns=config["pinned_columns"]
    )
