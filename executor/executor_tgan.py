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

        """ printing model details """
        # print(self.generator_model)
        # for i in self.generator_model.named_parameters():
        #     print(i[0], i[1].shape, i[1].numel())

        """ Dynamic optimizer """
        optimizer_function = getattr(torch.optim, self.cfg.optimizer)
        step_size = self.batch_size * len(data_loader) * 0.5
        self.generator_optimizer = optimizer_function(self.generator_model.parameters(), lr=self.cfg.lr,
                                                      weight_decay=self.cfg.weight_decay)
        self.discriminator_optimizer = optimizer_function(self.discriminator_model.parameters(), lr=self.cfg.lr,
                                                          weight_decay=self.cfg.weight_decay)
        """ lr decay scheduler """
        if self.cfg.has_lr_decay:
            self.generator_scheduler = torch.optim.lr_scheduler.StepLR(self.generator_optimizer, step_size=step_size,
                                                                       gamma=self.cfg.lr_decay_gamma)
            self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer,
                                                                           step_size=step_size,
                                                                           gamma=self.cfg.lr_decay_gamma)
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
        discriminator_loss = 0
        generator_loss = 0
        epoch_real_output = 0
        epoch_fake_output = 0
        epoch_real_data = 0
        epoch_fake_data = 0

        # loop over training data batches
        for i, (data, label) in enumerate(self.data_loader):
            # Check for the size of last batch
            data = data.to(self.device)
            label = label.to(self.device)
            if data.size()[0] < self.batch_size:
                continue
            """ create minibatches of fake data and labels """
            # noise between -1 , 1
            noise = torch.rand(self.batch_size, self.cfg.n_input).to(self.device) * 2 - 1
            fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
            real_labels = torch.ones(self.batch_size, 1).to(self.device)
            # concatenate noise and labels as input to generator
            fake_data = self.generator_model(torch.cat((noise, label), 1))

            """Train the discriminator"""
            self.discriminator_optimizer.zero_grad()
            """by putting optimizer.zero_grad() before computing the loss, you are starting with a clean slate for 
            each iteration and ensuring that the gradients being calculated are based solely on the current batch of 
            data. This can help prevent issues such as gradients accumulating over time and causing the model to 
            become unstable or diverge from the optimal solution."""

            # forward pass and loss for real data
            real_output = self.discriminator_model(data)
            real_loss = self.loss_function(real_output, real_labels, fake_data, data)

            # forward pass and loss for fake data
            # with torch.no_grad():
            fake_output = self.discriminator_model(fake_data)
            fake_loss = self.loss_function(fake_output, fake_labels, fake_data, data)
            disc_loss = (real_loss + fake_loss) / self.batch_size
            discriminator_loss += disc_loss.item()
            disc_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()
            if self.cfg.has_lr_decay:
                self.discriminator_scheduler.step()

            """Train the generator"""
            self.generator_optimizer.zero_grad()
            pred_labels = self.discriminator_model(fake_data)
            gen_loss = self.loss_function(pred_labels, real_labels, fake_data, data) / self.batch_size
            generator_loss += gen_loss.item()
            gen_loss.backward(retain_graph=True)
            """retain_graph tells the autograd engine to retain the intermediate values of the graph,
            instead of freeing them, so that they can be used in the next backward pass."""
            self.generator_optimizer.step()
            if self.cfg.has_lr_decay:
                self.generator_scheduler.step()

            epoch_real_output = real_output
            epoch_fake_output = fake_output
            epoch_real_data = data
            epoch_fake_data = fake_data
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        # print(self.generator_scheduler.get_last_lr()[0])

        total = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        MSE = 0
        for j in range(epoch_real_output.size()[0]):
            if epoch_real_output[j].item() > 0.5:
                if epoch_fake_output[j].item() < 0.5:
                    TP += 1
                else:
                    FN += 1
            elif epoch_real_output[j].item() <= 0.5:
                if epoch_fake_output[j].item() >= 0.5:
                    TP += 1
                else:
                    FP += 1
            else:
                TN+=1
            total += 1
            MSE += sum((epoch_real_data[j] - epoch_fake_data[j]) ** 2) / epoch_real_output.size()[0]

        if TP ==0:
            accuracy = 0.0
            recall = 0.0
            precision = 0.0
        else:
            accuracy = (TP+TN) / total
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            MSE = MSE / epoch_real_output.size()[0]
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Mean Squared Error: {:.2f}".format(MSE))
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
