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
        self.cfg = config.from_json("training").cae
        self.batch_size = config.from_json("data").cae.batch_size
        self.logger = helpers.Log(self.cfg.model_file_name)

        """ Find if GPU is available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        """ Load model configurations """
        self.cae_model = model_cae.CAE().to(self.device)
        self.loss_function = loss.CustomLoss()

        """ printing model details """
        print("--------------------------------------------------------")
        print(f"Model Architecture: ")
        print("||||||||||||||||||||")
        print(self.cae_model)
        print("--------------------------------------------------------")
        print(f"Model Details: ")
        print("|||||||||||||||")
        for i in self.cae_model.named_parameters():
            print(i[0], i[1].shape, i[1].numel())
        print("--------------------------------------------------------")

        """ Dynamic optimizer """
        optimizer_function = getattr(torch.optim, self.cfg.optimizer)
        self.cae_optimizer = optimizer_function(self.cae_model.parameters(), lr=self.cfg.lr,
                                                weight_decay=self.cfg.weight_decay)
        """ Defining Data Loader"""
        self.data_loader = data_loader

        """ Creating dataframe fo saving logs"""
        self.df_result = pd.DataFrame(
            columns=["epoch", "total_loss", "mse", "time"])

    def forward(self):
        """Set model to training mode"""
        # self.model.train()
        """variables"""
        epochs = self.cfg.epoch

        for epoch in range(epochs):
            loss, mse_loss = self.train()
            self.df_result.loc[len(self.df_result.index)] = [epoch + 1, round(loss, 4), round(mse_loss, 4),
                                                             datetime.now().strftime("%m/%d/%Y, %H:%M:%S")]

            if (epoch + 1) % 2 == 0:
                print("--------------------------------------------------------")
                print(f"Epoch {epoch + 1}: ")
                print(f"Time : {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
                print("MSE : {:.4f}".format(mse_loss))
                print("TotalLoss : {:.4f}".format(loss))

        print("--------------------------------------------------------")
        # Saving Results
        self.logger.save_result(self.df_result)

        return self.cae_model

    def train(self):
        """Set model to training mode"""
        self.cae_model.train()

        """variables"""
        total_loss = 0
        mse_loss = 0

        # loop over training data batches
        for i, (data, label) in enumerate(self.data_loader):
            # Check for the size of last batch
            x = torch.cat((data, label), dim=1).to(self.device)
            if x.size()[0] < self.batch_size:
                continue
            """ create minibatches of fake data and labels """
            encoded, decoded = self.cae_model(x)

            W = self.cae_model.state_dict()['input.weight']
            loss, mse = self.loss_function.cae_loss(W, x, decoded, encoded)
            total_loss += loss.item()
            mse_loss += mse.item()

            """Train"""
            self.cae_optimizer.zero_grad()
            loss.backward()
            self.cae_optimizer.step()
            # end of batch loop...

        total_loss /= len(self.data_loader)
        mse_loss /= len(self.data_loader)

        return total_loss, mse_loss
