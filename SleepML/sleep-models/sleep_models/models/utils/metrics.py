import numpy as np


def compute_labels_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log2(p))
    return entropy


def compute_expected_accuracy(entropy):
    return 1 / (2 ** (entropy))


def compute_trivial_accuracy(labels):
    counts = np.unique(labels.argmax(1), return_counts=True)[1]
    return round(counts.max() / counts.sum(), 3)


def calibrate_accuracy(labels):
    """
    """

    entropy = compute_labels_entropy(labels.argmax(axis=1))
    random_accuracy = compute_expected_accuracy(entropy)
    print(f"Random accuracy: {random_accuracy}")
    trivial_accuracy = compute_trivial_accuracy(labels)
    print(f"Trivial accuracy: {trivial_accuracy}")
