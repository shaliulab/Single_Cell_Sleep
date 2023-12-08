# Adata
MIN_MEAN_HVG = 0.0125
MAX_MEAN_HVG = 3
MIN_DISP_HVG = 0.5


# Training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
L2 = 0.002
PATIENCE = 50
EPOCHS = 1000
N_NEURONS = [100, 10]
DROPOUTS = []
OPTIMIZER = "SGD"
LOSS_FUNCTION = "CrossEntropyLoss"

OUTER_BAGS = 8
INNER_BAGS = 0
# Boosting
LEARNING_RATE = 0.01
VALIDATION_SIZE = 0.15
EARLY_STOPPING_TOLERANCE = 1e-4
MAX_ROUNDS = 5000
# Trees
MIN_SAMPLES_LEAF = 2
MAX_LEAVES = 3

# Paths
RUN_CONFIG = "run_config.yaml"
TRAINING_PARAMS = "training_params.yaml"


HIGHLY_VARIABLE_GENES = False
MEAN_SCALE = False
TRAIN_SIZE = 0.75

MAX_DEPTH = None
N_ESTIMATORS = 100

DEFAULT_DICT = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "patience": PATIENCE,
    "epochs": EPOCHS,
    "l2": L2,
    "n_neurons": N_NEURONS,
    "dropouts": DROPOUTS,
    "optimizer": OPTIMIZER,
    "loss_function": LOSS_FUNCTION,
    "outer_bags": OUTER_BAGS,
    "inner_bags": INNER_BAGS,
    # Boosting
    "learning_rate": LEARNING_RATE,
    "validation_size": VALIDATION_SIZE,
    "early_stopping_tolerance": EARLY_STOPPING_TOLERANCE,
    "max_rounds": MAX_ROUNDS,
    # Trees
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "max_leaves": MAX_LEAVES,
    "early_stopping": True,
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
}

NeuralNetworkHyperParams = [
    "learning_rate",
    "l2",
    "batch_size",
    "loss_function",
    "patience",
    "optimizer",
    "epochs",
    "early_stopping",
    "n_neurons",
    "dropouts",
]

KNNHyperParams = [
    "n_neighbors",
    "weights",
    "leaf_size",
    "p",
    "metric",
]


EBMHyperParams = [
    # "outer_bags",
    # "inner_bags",
    # # Boosting
    # "learning_rate",
    # "validation_size",
    # "early_stopping_rounds",
    # "early_stopping_tolerance",
    # "max_rounds",
    # Trees
    "min_samples_leaf",
    "max_leaves",
]
RFHyperParams = [
    "n_estimators",
    "max_depth",
]
