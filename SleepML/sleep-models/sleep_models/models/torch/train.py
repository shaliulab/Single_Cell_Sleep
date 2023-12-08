import torch
from torch.utils.data import DataLoader
from sleep_models.models.torch.test import test_loop


def train_loop(model, dataloader, optimizer):

    size = len(dataloader.dataset)
    loss, correct = 0, 0
    loss_function = model.loss_function

    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()

        # Compute prediction and loss
        logits = model(X)
        # probabilities = model.prob_layer(logits)
        # prediction = probabilities.argmax(1)
        prediction = logits.argmax(1)

        loss = loss_function(logits, y)

        with torch.no_grad():
            ground_truth = y.argmax(1)
            correct += (prediction == ground_truth).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    accuracy = 100 * correct
    print(f"Train Accuracy: {accuracy:>0.1f}")

    return accuracy, loss


def train_model(model, data):

    OptimizerClass = getattr(torch.optim, model.optimizer)
    optimizer = OptimizerClass(
        model.parameters(), lr=model.learning_rate, weight_decay=model.l2
    )

    training_data = data["training"]
    test_data = data["test"]

    train_dataloader = DataLoader(
        training_data, batch_size=model.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    best_train_acc = 0
    best_test_acc = 0

    for t in range(model.max_iter):
        print(f"Epoch {t+1}\n-------------------------------")

        train_acc, train_loss = train_loop(
            model=model, dataloader=train_dataloader, optimizer=optimizer
        )
        test_acc, test_loss, confusion_table = test_loop(
            model=model, dataloader=test_dataloader
        )

        if test_acc > best_test_acc:
            model.best_metrics["test-accuracy"] = test_acc
            best_test_acc = test_acc

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # early_stopping needs the test loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        model.early_stopping(test_acc, model)

        if model.early_stopping.early_stop:
            print("Early stopping")
            break

    print("Done training!")
    model.save()
    model.save_results(confusion_table=confusion_table)
    metrics = {
        "train-accuracy": train_acc,
        "test-accuracy": test_acc,
        "train-loss": train_loss,
        "test-loss": test_loss,
    }

    return t + 1, metrics
