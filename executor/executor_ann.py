import torch
from models import model_ann, loss
from utils import helpers
import numpy as np


class Training:
    def __init__(self, train_loader, test_loader):
        super().__init__()
        """ Load training configurations """
        config = helpers.Config()
        self.cfg = config.from_json("training")
        """ Load model configurations """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = model_ann.ANN()
        self.loss_function = loss.CustomLoss()
        """ Dynamic optimizer based on config """
        optimizer_function = getattr(torch.optim, self.cfg.optimizer)
        self.optimizer = optimizer_function(self.model.parameters(), lr=self.cfg.lr)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def exec(self):
        """Set model to training mode"""
        self.model.train()
        """variables"""
        epoch = self.cfg.epoch
        train_accuracy = torch.zeros(epoch)
        test_accuracy = torch.zeros(epoch)

        for epoch_i in range(epoch):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            train_accuracy[epoch_i] = train_acc
            test_accuracy[epoch_i] = test_acc
            if (epoch_i+1) % 5 == 0:
                print("--------------------------------------------------------")
                print(f"Epoch {epoch_i+1}: ")
                print(f"Train Loss: {train_loss}")
                print(f"Test Loss: {test_loss}")
                print(f"Train Accuracy: {train_acc}")
                print(f"Test Accuracy: {test_acc}")

        print("--------------------------------------------------------")
        print(f"Final Training Accuracy: epoch {torch.argmax(train_accuracy)+1} with accuracy: {train_accuracy[torch.argmax(train_accuracy)]}")
        print(f"Final Testing Accuracy: epoch {torch.argmax(test_accuracy)+1} with accuracy: {test_accuracy[torch.argmax(test_accuracy)]}")

        return train_accuracy, test_accuracy

    def train(self):
        """Set model to training mode"""
        self.model.train()
        """variables"""
        train_acc = 0.0
        train_loss = 0.0
        # loop over training data batches
        for X, y in self.train_loader:
            # forward pass and loss
            predictions = self.model(X)
            loss_value = self.loss_function(predictions, y)

            # backprop
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # compute loss
            train_loss += loss_value.item()
            # compute accuracy
            train_acc += 100 * Training.accuracy(predictions, y)
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)

        return train_loss, train_acc

    def test(self):
        """ Set model to evaluation mode """
        self.model.eval()
        """variables"""
        test_acc = 0.0
        test_loss = 0.0

        with torch.no_grad():  # don't calculate gradients
            # loop over training data batches
            for X, y in self.test_loader:
                # forward pass and loss
                predictions = self.model(X)
                loss_value = self.loss_function(predictions, y)
                # compute loss
                test_loss += loss_value.item()
                # compute accuracy
                test_acc += 100 * Training.accuracy(predictions, y)

        # now that we've trained through the batches, get their average training accuracy
        test_loss /= len(self.test_loader)
        test_acc /= len(self.test_loader)

        return test_loss, test_acc

    @staticmethod
    def accuracy(prediction, labels):
        """ R^2 score"""
        u = ((labels - prediction) ** 2).sum()
        v = ((labels - labels.mean()) ** 2).sum()
        return 1 - u / v
