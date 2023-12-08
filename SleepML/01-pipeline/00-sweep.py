# sleep-models-sweep ${CELLTYPE} --output results/${BACKGROUND}-sweeps/ --sweep-config data/sweeps/2022-02-25_sweep.conf --model-name ${MODEL} --input results/${BACKGROUND}-models/NeuralNetwork/${CELLTYPE}/random-state_1000_fraction_1.0/ #  > logs/sweep_${CELLTYPE}.log
import argparse
import os.path
import yaml
import pandas as pd
from sleep_models.bin.sweep import main

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

DATA_DIR = config["data_dir"]
RESULTS_DIR=config["results_dir"]
MODEL_NAME="NeuralNetwork"
SWEEP_CONFIG="2022-02-25_sweep.conf"


def get_celltypes(background):

    return pd.read_csv(
        os.path.join(
            DATA_DIR, "backgrounds", f"{background}.csv"
        ), index_col=0, comment="#"
    )["cluster"].tolist()

for background in config["background"]:
    celltypes = get_celltypes()
    input_h5ad=os.path.join(DATA_DIR, f"h5ad/{background}-no-marker-genes.h5ad")
    for celltype in celltypes:
        args = argparse.Namespace(
            cluster=celltype,
            output=os.path.join(RESULTS_DIR, f"{background}-sweeps"),
            model_name=MODEL_NAME,
            sweep_config=os.path.join(DATA_DIR, f"sweeps/{SWEEP_CONFIG}"),
            input=os.path.join(RESULTS_DIR, f"{background}-models/{MODEL_NAME}/{celltype}/random-state_1000")
        )
        print(f"Starting Weights & Bias sweep for celltype {celltype}")
        main(args=args)