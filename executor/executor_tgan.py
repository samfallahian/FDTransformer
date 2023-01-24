import torch
from models import model_tgan, loss
from utils import helpers
import numpy as np


class Training:
    def __init__(self, data_loader):
        super(Training, self).__init__()
        """ Load training configurations """
        config = helpers.Config()
        self.cfg = config.from_json("training")
        self.batch_size = config.from_json("data").batch_size

        """ Find if GPU is available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        """ Load model configurations """
        self.generator_model = model_tgan.Generator().to(self.device)
        self.discriminator_model = model_tgan.Discriminator().to(self.device)
        self.loss_function = loss.CustomLoss()

        """ Dynamic optimizer based on config """
        optimizer_function = getattr(torch.optim, self.cfg.optimizer)
        self.generator_optimizer = optimizer_function(self.generator_model.parameters(), lr=self.cfg.lr)
        self.discriminator_optimizer = optimizer_function(self.discriminator_model.parameters(), lr=self.cfg.lr)
        self.data_loader = data_loader

    def forward(self):
        """Set model to training mode"""
        # self.model.train()
        """variables"""
        epochs = self.cfg.epoch
        # train_accuracy = torch.zeros(epochs)
        # test_accuracy = torch.zeros(epochs)

        for epoch in range(epochs):
            # generator_loss, disc_loss = self.train()
            d_loss, g_loss = self.train()

            # train_accuracy[epoch] = train_acc
            # test_accuracy[epoch] = test_acc
            if (epoch + 1) % 2 == 0:
                print("--------------------------------------------------------")
                print(f"Epoch {epoch + 1}: ")
                print(f"Generator Loss: {g_loss}")
                print(f"Discriminator Loss : {d_loss}")
                # print(f"Train Accuracy: {train_acc}")
                # print(f"Test Accuracy: {test_acc}")

        print("--------------------------------------------------------")
        # print(
        #     f"Final Training Accuracy: epoch {torch.argmax(train_accuracy) + 1} with accuracy: {train_accuracy[torch.argmax(train_accuracy)]}")
        # print(
        #     f"Final Testing Accuracy: epoch {torch.argmax(test_accuracy) + 1} with accuracy: {test_accuracy[torch.argmax(test_accuracy)]}")
        train_accuracy = 0
        test_accuracy = 0
        return train_accuracy, test_accuracy

    def train(self):
        # """Set model to training mode"""
        # self.generator_model.train()
        # self.discriminator_model.train()
        """variables"""
        generator_similarity_score = 0.0
        discriminator_acc = 0.0
        discriminator_precision = 0.0
        discriminator_recall = 0.0
        discriminator_f1 = 0.0

        discriminator_loss = 0
        generator_loss = 0

        # loop over training data batches
        """ Set the range for label for producing fake_label"""

        for X, (data, labels) in enumerate(self.data_loader):
            # Train the discriminator
            real_output = self.discriminator_model(data)
            real_loss = self.loss_function(real_output, labels)


            self.discriminator_optimizer.zero_grad()
            # real_output = self.discriminator_model(labels)
            # real_loss = self.loss_function(real_output, torch.ones(real_output.size()))
            fake_output = self.discriminator_model(self.generator_model(data))
            fake_loss = self.loss_function(fake_output, torch.zeros(fake_output.size()))
            d_loss = (real_loss + fake_loss)/self.batch_size
            d_loss.backward()
            self.discriminator_optimizer.step()

            # Train the generator
            self.generator_optimizer.zero_grad()
            fake_output = self.discriminator_model(self.generator_model(data))
            g_loss = self.loss_function(fake_output, torch.ones(fake_output.size()))
            g_loss = g_loss/self.batch_size
            g_loss.backward()
            self.generator_optimizer.step()

            discriminator_loss += d_loss.item()
            generator_loss += g_loss.item()

            # # Train the discriminator
            # self.discriminator_model.zero_grad()
            # real_data = data[:self.batch_size]
            # real_labels = labels[:self.batch_size]
            # real_output = self.discriminator_model(real_data, real_labels)
            # real_loss = self.loss_function(real_output, torch.ones(self.batch_size, 1))
            # real_loss.backward()
            #
            # noise = torch.rand(self.batch_size, self.cfg.n_input) * 2 - 1
            # fake_labels = torch.rand(self.batch_size, self.cfg.n_classes) * 2 - 1
            #
            # fake_data = self.generator_model(noise, fake_labels)
            # fake_output = self.discriminator_model(fake_data, fake_labels)
            # fake_loss = self.loss_function(fake_output, torch.zeros(self.batch_size, 1))
            # fake_loss.backward()
            #
            # disc_loss = real_loss + fake_loss
            # self.discriminator_optimizer.step()
            #
            # # Train the generator
            # self.generator_model.zero_grad()
            # noise = torch.rand(self.batch_size, self.cfg.n_input) * 2 - 1
            # fake_labels = torch.rand(self.batch_size, self.cfg.n_classes) * 2 - 1
            # fake_data = self.generator_model(noise, fake_labels)
            # fake_output = self.discriminator_model(fake_data, fake_labels)
            # # The generator is trying to generate data that the discriminator will classify as real -> target is 1
            # generator_loss = self.loss_function(fake_output, torch.ones(self.batch_size, 1))
            # generator_loss.backward()
            # self.generator_optimizer.step()

            # compute evaluating metric
            # generator_similarity_score = + 100 * Training.jaccard_similarity(fake_data, real_data)
            # discriminator_acc, discriminator_precision, discriminator_recall, discriminator_f1 = + Training.discriminator_eval_metric(
            #     fake_output, torch.ones(self.batch_size, 1))

        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        # generator_similarity_score /= len(self.data_loader)
        # discriminator_acc /= len(self.data_loader)
        # discriminator_precision /= len(self.data_loader)
        # discriminator_recall /= len(self.data_loader)
        # discriminator_f1 /= len(self.data_loader)
        # print(generator_loss.item())
        # print(disc_loss.item())

        discriminator_loss /= len(self.data_loader)
        generator_loss /= len(self.data_loader)
        # return generator_loss.item(), disc_loss.item()
        return discriminator_loss, generator_loss

    # def test(self):
    #     """ Set model to evaluation mode """
    #     self.model.eval()
    #     """variables"""
    #     test_acc = 0.0
    #     test_loss = 0.0
    #
    #     with torch.no_grad():  # don't calculate gradients
    #         # loop over training data batches
    #         for X, y in self.test_loader:
    #             # forward pass and loss
    #             predictions = self.model(X)
    #             loss_value = self.loss_function(predictions, y)
    #             # compute loss
    #             test_loss += loss_value.item()
    #             # compute accuracy
    #             test_acc += 100 * Training.accuracy(predictions, y)
    #
    #     # now that we've trained through the batches, get their average training accuracy
    #     test_loss /= len(self.test_loader)
    #     test_acc /= len(self.test_loader)
    #
    #     return test_loss, test_acc

    @staticmethod
    def accuracy(prediction, labels):
        """ R^2 score"""
        u = ((labels - prediction) ** 2).sum()
        v = ((labels - labels.mean()) ** 2).sum()
        return 1 - u / v

    @staticmethod
    def jaccard_similarity(real, generated):
        intersection = (real * generated).sum()
        union = real.sum() + generated.sum() - intersection
        return intersection / union

    @staticmethod
    def discriminator_eval_metric(output, target):
        predictions = (output > 0.5).float()
        true_positives = (predictions * target).sum()
        false_positives = (predictions * (1 - target)).sum()
        true_negatives = ((1 - predictions) * (1 - target)).sum()
        false_negatives = ((1 - predictions) * target).sum()
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / (
                true_positives + true_negatives + false_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1
