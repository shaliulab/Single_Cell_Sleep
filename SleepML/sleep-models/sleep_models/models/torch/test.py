import logging
import torch

# from sleep_models.models.utils.torch import get_device
from sleep_models.preprocessing import (
    make_confusion_table,
    make_confusion_long_to_square,
)


def test_loop(model, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0
    ground_truth_l = []
    prediction_l = []

    loss_function = model.loss_function

    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            prediction = logits.argmax(1)
            loss += loss_function(logits, y).item()

            ground_truth = y.argmax(1)
            correct += (prediction == ground_truth).type(torch.float).sum().item()
            ground_truth_l.extend(ground_truth.cpu().numpy())
            prediction_l.extend(prediction.cpu().numpy())

            logging.debug(f"Prediction: {prediction}")
            logging.debug(f"Ground truth: {ground_truth}")

    confusion_table = model.get_confusion_table(ground_truth_l, prediction_l)

    loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {accuracy*100:>0.1f}%, Avg loss: {loss:>8f} \n")
    return accuracy, loss, confusion_table


def test(model, dataloader):

    size = len(dataloader.dataset)
    label = {}

    # initialize a list to store the predicted label on each sample
    prediction = [None,] * size

    # initialize a list to store whether the ith prediction is right (True) or not (False)
    right = [None,] * size

    # initialize a tensor to store the probability of each class
    probability = torch.tensor([]).type(torch.float).to(model.device)

    # initialize a tensor to store the average loss
    loss = torch.tensor([0]).type(torch.float).to(model.device)

    with torch.no_grad():

        for batch, (X, y) in enumerate(dataloader):

            # make sure X and y are only for ONE sample
            assert X.shape[0] == 1
            assert y.shape[0] == 1

            # predict
            logits = model(X)
            probability_i = model.prob_layer(logits)
            probability = torch.cat((probability, probability_i), 0)
            code_hat_i = probability_i.argmax(1)

            label_i_hat = model._label_code[code_hat_i.item()]
            prediction[batch] = label_i_hat

            # compute loss
            probability_i = model.prob_layer(logits)
            probability = torch.cat((probability, probability_i), 0)
            loss += model.loss_function(probability_i, y)

            # evaluate
            code_i = y.argmax(1)
            label_i = model._label_code[code_i.item()]
            right_i = int((code_hat_i == code_i).item())
            if label_i not in label:
                label[label_i] = [0, 0]

            label[label_i][right_i] += 1
            right[batch] = right_i

    loss /= size
    loss = loss.cpu().numpy()
    accuracy = sum(right) / size

    print(f"Test Error: \n Accuracy: {accuracy*100:>0.1f}%\n")
    return accuracy, loss, prediction, right
