import torch
from models import model_cae, loss
from utils import helpers
from datetime import datetime
import pandas as pd


class Training:
    def __init__(self, data_loader):
        super(Training, self).__init__()
        """ Load training configurations """
        config = helpers.Config()
        self.cfg = config.from_json("training")
        self.batch_size = config.from_json("data").batch_size
        self.logger = helpers.Log(self.cfg.model_file_name)

        """ Find if GPU is available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        """ Load model configurations """
        self.cae_model = model_cae.CAE().to(self.device)
        self.loss_function = loss.CustomLoss()

        """ printing model details """
        # print(self.cae)
        # for i in self.cae.named_parameters():
        #     print(i[0], i[1].shape, i[1].numel())

        """ Dynamic optimizer """
        optimizer_function = getattr(torch.optim, self.cfg.optimizer)
        self.cae_optimizer = optimizer_function(self.cae_model.parameters(), lr=self.cfg.lr,
                                                      weight_decay=self.cfg.weight_decay)
        """ Defining Data Loader"""
        self.data_loader = data_loader

        """ Creating dataframe fo saving logs"""
        self.df_result = pd.DataFrame(
            columns=["epoch", "loss", "gen_mse", "time"])

    def forward(self):
        """Set model to training mode"""
        # self.model.train()
        """variables"""
        epochs = self.cfg.epoch

        for epoch in range(epochs):
            loss, mse = self.train()
            self.df_result.loc[len(self.df_result.index)] = [epoch + 1, round(loss, 4),round(mse, 4),
                                                             datetime.now().strftime("%m/%d/%Y, %H:%M:%S")]

            if (epoch + 1) % 2 == 0:
                print("--------------------------------------------------------")
                print(f"Epoch {epoch + 1}: ")
                print(f"Time : {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
                print("Generator Loss : {:.4f}".format(loss))
                print("Genrator Mean Squared Error: {:.3f}".format(mse))

        print("--------------------------------------------------------")
        # Saving Results
        self.logger.save_result(self.df_result)

        return self.cae_model

    def train(self):
        """Set model to training mode"""
        self.cae_model.train()

        """variables"""
        loss = 0
        mse = 0

        # loop over training data batches
        for i, (data, label) in enumerate(self.data_loader):
            # Check for the size of last batch
            x = torch.cat((data, label), dim=1).to(self.device)
            if x.size()[0] < self.batch_size:
                continue
            """ create minibatches of fake data and labels """
            encoded, decoded = self.cae_model(x)

            loss = self.loss_function.contractive_loss(decoded, data_batch) + contractive_coef * model.contractive_loss(encoded, data_batch)

            """Train the discriminator"""
            self.cae_optimizer.zero_grad()

            # forward pass and loss for real data
            real_output = self.discriminator_model(data, label)

            if self.cfg.is_critic:
                real_loss = self.loss_function.cgan_loss(real_output, torch.ones_like(real_output), fake_data, data,
                                                         is_generator=False)
            else:
                real_loss = self.loss_function.cgan_loss(real_output, real_labels, fake_data, data, is_generator=False)

            # forward pass and loss for fake data
            # with torch.no_grad():
            fake_output = self.discriminator_model(fake_data, label)
            if self.cfg.is_critic:
                fake_loss = self.loss_function.cgan_loss(fake_output, torch.zeros_like(fake_output), fake_data, data,
                                                         is_generator=False)
            else:
                fake_loss = self.loss_function.cgan_loss(fake_output, fake_labels, fake_data, data,
                                                         is_generator=False)

            disc_loss = (real_loss + fake_loss) / (self.batch_size if self.cfg.scaled_loss == True else 1)
            discriminator_loss += disc_loss.item()
            disc_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()
            if self.cfg.has_lr_decay:
                self.discriminator_scheduler.step()

            """Train the generator"""
            self.generator_optimizer.zero_grad()
            trained_fake_output = self.discriminator_model(fake_data, label)
            if self.cfg.is_critic:
                gen_loss = self.loss_function.cgan_loss(trained_fake_output, torch.ones_like(trained_fake_output),
                                                        fake_data, data, is_generator=True) / (
                               self.batch_size if self.cfg.scaled_loss == True else 1)
            else:
                gen_loss = self.loss_function.cgan_loss(trained_fake_output, real_labels, fake_data,
                                                        data, is_generator=True) / (
                               self.batch_size if self.cfg.scaled_loss == True else 1)
            generator_loss += gen_loss.item()
            gen_loss.backward(retain_graph=True)
            """retain_graph tells the autograd engine to retain the intermediate values of the graph,
            instead of freeing them, so that they can be used in the next backward pass."""
            self.generator_optimizer.step()
            if self.cfg.has_lr_decay:
                self.generator_scheduler.step()

            accuracy_t, recall_t, precision_t, mse_t = Training.eval_metric(real_output, fake_output,
                                                                            data, fake_data)
            accuracy += accuracy_t
            recall += recall_t
            precision += precision_t
            mse += mse_t
            # end of batch loop...

        # tracking learning rate
        # print(self.generator_scheduler.get_last_lr()[0])
        # get their average training eval metrics and losses
        accuracy /= len(self.data_loader)
        recall /= len(self.data_loader)
        precision /= len(self.data_loader)
        mse /= len(self.data_loader)
        discriminator_loss /= len(self.data_loader)
        generator_loss /= len(self.data_loader)

        return discriminator_loss, generator_loss, accuracy, recall, precision, mse

    @staticmethod
    def eval_metric(real_output, fake_output, real_data, fake_data):
        total = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        mse = 0
        for j in range(real_output.size()[0]):
            if real_output[j].item() > 0.5:
                if fake_output[j].item() < 0.5:
                    tp += 1
                else:
                    fn += 1
            elif real_output[j].item() <= 0.5:
                if fake_output[j].item() >= 0.5:
                    tp += 1
                else:
                    fp += 1
            else:
                tn += 1
            total += 1
            mse += sum((real_data[j] - fake_data[j]) ** 2).item() / real_output.size()[0]

        if tp == 0:
            accuracy = 0.0
            recall = 0.0
            precision = 0.0
        else:
            accuracy = (tp + tn) / total
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
        mse = mse / real_output.size()[0]
        return accuracy, precision, recall, mse
