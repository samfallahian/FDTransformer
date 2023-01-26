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
        self.generator_model = model_tgan.Generator()
        self.discriminator_model = model_tgan.Discriminator()
        self.loss_function = loss.CustomLoss()
        # self.loss_function = nn.BCELoss()

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
        """Set model to training mode"""
        self.generator_model.train()
        self.discriminator_model.train()
        """variables"""
        generator_similarity_score = 0.0
        discriminator_acc = 0.0
        discriminator_precision = 0.0
        discriminator_recall = 0.0
        discriminator_f1 = 0.0

        discriminator_loss = 0
        generator_loss = 0

        # loop over training data batches
        for i, (data, label) in enumerate(self.data_loader):
            # Check for the size of last batch
            if data.size()[0] < self.batch_size:
                continue
            """ create minibatches of fake data and labels """
            # noise between -1 , 1
            noise = torch.rand(self.batch_size, self.cfg.n_input) * 2 - 1
            fake_labels = torch.zeros(self.batch_size, 1)
            real_labels = torch.ones(self.batch_size, 1)
            fake_data = self.generator_model(
                torch.cat((noise, label), 1))  # concatenate noise and labels as input to generator

            """Train the discriminator"""
            self.discriminator_optimizer.zero_grad()
            """by putting optimizer.zero_grad() before computing the loss, you are starting with a clean slate for 
            each iteration and ensuring that the gradients being calculated are based solely on the current batch of 
            data. This can help prevent issues such as gradients accumulating over time and causing the model to 
            become unstable or diverge from the optimal solution."""

            # forward pass and loss for real data
            real_output = self.discriminator_model(data)
            real_loss = self.loss_function(real_output, real_labels)

            # forward pass and loss for fake data
            # with torch.no_grad():
            fake_output = self.discriminator_model(fake_data)
            fake_loss = self.loss_function(fake_output, fake_labels)
            disc_loss = (real_loss + fake_loss) / self.batch_size
            discriminator_loss += disc_loss.item()
            disc_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            """Train the generator"""
            self.generator_optimizer.zero_grad()
            pred_labels = self.discriminator_model(fake_data)
            gen_loss = self.loss_function(pred_labels, real_labels) / self.batch_size
            generator_loss += gen_loss.item()
            gen_loss.backward(retain_graph=True)
            """retain_graph tells the autograd engine to retain the intermediate values of the graph,
            instead of freeing them, so that they can be used in the next backward pass."""
            self.generator_optimizer.step()

            # compute evaluating metric
            generator_similarity_score += 100 * Training.jaccard_similarity(fake_data.detach(), data.detach())
            dis_acc, dis_precision, dis_recall, dis_f1 = Training.discriminator_eval_metric(
                fake_output.detach(), real_labels.detach())
            discriminator_acc += dis_acc
            discriminator_precision += dis_precision
            discriminator_recall += dis_recall
            discriminator_f1 += dis_f1

        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        generator_similarity_score /= len(self.data_loader)
        discriminator_acc /= len(self.data_loader)
        discriminator_precision /= len(self.data_loader)
        discriminator_recall /= len(self.data_loader)
        discriminator_f1 /= len(self.data_loader)
        # print(generator_similarity_score.item())
        # print(discriminator_acc.item())
        # print(discriminator_precision.item())
        # print(discriminator_recall.item())

        discriminator_loss /= len(self.data_loader)
        generator_loss /= len(self.data_loader)
        return discriminator_loss, generator_loss

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
