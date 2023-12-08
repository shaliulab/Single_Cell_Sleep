# crosspredict --background data/backgrounds/KC_mapping.csv       --root-dir 2021-11-07_KC_mapping.csv  --ncores 3

import os.path
import yaml
from sleep_models.predict import replicate
from sleep_models.bin.make_matrixplot import make_matrixplot_main as make_matrixplot

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]

def predict(background_folder, architectures, random_states, **kwargs):
    for arch in architectures:
            
        arch_folder = os.path.join(background_folder, arch)

        for random_state in random_states:

            replicate_folder=os.path.join(arch_folder, f"random-state_{random_state}")

            if not os.path.exists(replicate_folder):
                raise Exception(f"Please make sure {replicate_folder} exists")

            accuracy_table, f_sleep_table = replicate(
                background=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
                replicate_folder=replicate_folder,
                ncores=3
            )

            accuracy_table.to_csv(
                os.path.join(replicate_folder, "accuracy.csv")
            )
            f_sleep_table.to_csv(
                os.path.join(replicate_folder, "f_sleep.csv")
            )


        make_matrixplot(
            prediction_results=os.path.dirname(
                replicate_folder
            ),
            **kwargs
        )

for background in config["background"]:
    
    background_folder = os.path.join(RESULTS_DIR, f"{background}-train")
    predict(background_folder, config["arch"], config["seeds"], **config["plotting_kwargs"])
   
    for i in range(config["shuffles"]):
        background_folder = os.path.join(RESULTS_DIR, f"{background}_shuffled_{i}-train")
        predict(background_folder, config["arch"], config["seeds"], **config["plotting_kwargs"])




    
