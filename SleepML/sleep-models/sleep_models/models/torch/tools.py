# https://raw.githubusercontent.com/Bjarten/early-stopping-pytorch/master/pytorchtools.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if metric doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        should_decrease=True,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            should_decrease (bool): If True, the lower the metric, the better. Otherwise, the higher the better
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self._should_decrease = should_decrease
        if should_decrease:
            self.best_metric = np.Inf
        else:
            self.best_metric = -np.Inf

        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, metric, model):

        if self._should_decrease:
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif self._score_has_improved(score):
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def _score_has_improved(self, score):
        if self._should_decrease:
            return score < self.best_score + self.delta
        else:
            return score > (self.best_score + self.delta)

    def save_checkpoint(self, metric, model):
        """Saves model when metric improves."""
        if self.verbose:
            self.trace_func(
                f"Metric improved ({self.best_metric:.6f} --> {metric:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric
