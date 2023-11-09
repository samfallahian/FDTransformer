import torch
from TransformerTrain import TrainTransformer
from TransformerModel import Seq2PointTransformer, TransformerModel
from TransformerDataReader import DataReader
from TransformerDataLoader import CustomDataset

# Define constants
feature_size = 48  # size of each velocity vector
d_model = 48 # size of each timestep
nhead = 6  # number of heads in multihead attention
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 2048  # size of feedforward network in transformer

# Data
batch_size = 240
beta = 0.02
source_len = 4
num_time_frame = 1200  # should be 1200 for this problem

# Training
epochs = 301
dropout = 0.1
lr = 0.001
scheduler_step = 220000
lr_gamma = 0.98

# Log
is_wandb = True
log_interval = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
# model = Seq2PointTransformer(nhead, num_encoder_layers, dim_feedforward, feature_size = feature_size, max_seq_len=5000, dropout=dropout).to(device)
# model = Seq2PointTransformer(d_model, nhead, num_encoder_layers, dim_feedforward, feature_size=feature_size, dropout=dropout).to(
#     device)

model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1)

# data_reader = DataReader("/mnt/d/Normalized/latent_representation_for_")
data_reader = DataReader("/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_")

data_pivot = data_reader.load_data(num_time_frame)

dataset = CustomDataset(df=data_pivot, source_len=source_len)

trainer = TrainTransformer(model, device, dataset=dataset,
                           lr=lr, epochs=epochs, log_interval=log_interval,
                           beta=beta, scheduler_step=scheduler_step,
                           batch_size=batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb)

loss = trainer.train()
