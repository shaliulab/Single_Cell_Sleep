import logging
import os.path
import yaml

from sleep_models.bin.train_model import get_parser as base_parser
from sleep_models.models.torch.nn import MODELS
from sleep_models.constants import *
from sleep_models.models.core import train
from sleep_models.models.variables import AllConfig
from sleep_models import preprocessing as pp
import sleep_models.models.utils.torch as torch_utils
import sleep_models.models.utils.config as config_utils

HIGHLY_VARIABLE_GENES = False


def get_parser(ap=None, *args, **kwargs):
    ap = base_parser(ap, *args, **kwargs)
    ap.add_argument("--label-mapping", dest="label_mapping", default=None)
    ap.add_argument("--simulate", action="store_true", default=False)
    ap.add_argument(
        "--fractions", dest="fractions", nargs="+", type=float, default=[1.0]
    )
    return ap


def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap=ap, models=MODELS.values())
        args = ap.parse_args()

    ModelClass = MODELS[args.model_name]
    output = os.path.join(args.output, args.model_name, args.cluster)
    os.makedirs(output, exist_ok=True)
    output_prefix = os.path.join(output, f"random-state_{args.random_state}")

    for fraction in [1.0]:

        if not args.simulate:
            data = pp.load_data(
                args.h5ad_input,
                output=output,
                cluster=args.cluster,
                random_state=args.random_state,
                mean_scale=args.mean_scale,
                highly_variable_genes=args.highly_variable_genes,
                label_mapping=args.label_mapping,
                model_properties=ModelClass.model_properties(),
                target=args.target,
                fraction=fraction,
            )

        else:
            data = pp.simulate_data(
                args.h5ad_input,
                cluster=args.cluster,
                random_state=args.random_state,
                mean_scale=args.mean_scale,
                highly_variable_genes=args.highly_variable_genes,
                model_properties=ModelClass.model_properties(),
                target=target,
                label_mapping=args.label_mapping,
            )

        X_train, y_train, X_test, y_test = data["datasets"]
        encoding = data["encoding"]

        device = torch_utils.get_device()

        training_config = config_utils.setup_config(args.model_name)

        config = AllConfig(
            model_name=args.model_name,
            training_config=training_config,
            cluster=args.cluster,
            output=output,
            device=device,
        )

        model, tolog = train(
            X_train, y_train, X_test, y_test, encoding=encoding, config=config
        )
        print(tolog)


if __name__ == "__main__":
    main()
