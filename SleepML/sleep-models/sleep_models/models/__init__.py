from .models import EBM, MLP, KNN, RF
from .torch.nn import NeuralNetwork

MODELS = {
    "EBM": EBM,
    "KNN": KNN,
    "MLP": MLP,
    "RF": RF,
    "NeuralNetwork": NeuralNetwork,
}
