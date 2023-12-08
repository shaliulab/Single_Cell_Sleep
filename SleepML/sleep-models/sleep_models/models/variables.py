from collections import namedtuple
from sleep_models.constants import (
    KNNHyperParams,
    NeuralNetworkHyperParams,
    EBMHyperParams,
    RFHyperParams,
)

NeuralNetworkConfig = namedtuple("TrainingConfig", NeuralNetworkHyperParams,)

KNNConfig = namedtuple("TrainingConfig", KNNHyperParams)

EBMConfig = namedtuple("TrainingConfig", EBMHyperParams)

RFConfig = namedtuple("TrainingConfig", RFHyperParams)

AllConfig = namedtuple(
    "AllConfig",
    [
        "training_config",
        "cluster",
        "output",
        "device",
        "target",
        "model_name",
        "random_state",
    ],
)
ModelProperties = namedtuple("ModelProperties", ["encoding", "estimator_type"])

CONFIGS = {
    "NeuralNetwork": (NeuralNetworkConfig, NeuralNetworkHyperParams),
    "KNN": (KNNConfig, KNNHyperParams),
    "EBM": (EBMConfig, EBMHyperParams),
    "RF": (RFConfig, RFHyperParams),
}
