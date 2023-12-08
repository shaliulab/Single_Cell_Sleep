import argparse
import os.path
import traceback
import logging

from sleep_models.train.torch_grid_search import sweep
from sleep_models.models import MODELS

logger = logging.getLogger("sleep_models.sweep")
logger.setLevel(logging.DEBUG)


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("cluster", type=str)
    ap.add_argument("--sweep-config", dest="sweep_config", required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--h5ad-input", dest="input", type=str)
    group.add_argument("--input", dest="input", type=str)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--project", type=str, required=False, default="uncategorized")
    ap.add_argument("--model-name", dest="model_name", type=str, choices=MODELS.keys())
    ap.add_argument(
        "--sweep-count",
        dest="sweep_count",
        type=int,
        default=100,
        help="Number of models to be instantiated with a different combination of parameters",
    )
    return ap


def main(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = get_parser()
        args = ap.parse_args()

    try:

        assert os.path.exists(args.sweep_config), f"{args.sweep_config} does not exist"
        input_data = os.path.join(args.input, f"{args.cluster}_X-train.csv")
        assert os.path.exists(input_data), f"{input_data} does not exist"
        os.makedirs(args.output, exist_ok=True)

        sweep(
            input=args.input,
            model_name=args.model_name,
            cluster=args.cluster,
            output=args.output,
            config_file=args.sweep_config,
            sweeps=args.sweep_count,
            project=args.project,
        )

    except Exception as error:
        logging.error(error)
        logging.error(traceback.print_exc())


if __name__ == "__main__":
    main()
