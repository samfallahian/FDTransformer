import torch
from TransformerTrain import TrainTransformer
from TransformerModel import TransformerModel
from TransformerDataReader import DataReader
from TransformerDataLoader import CustomDataset

# Define constants

# Model
d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2

# Data
batch_size = 512
source_size = 8
target_size = 2
source_len = 8
target_len = 2
num_time_frame = 200 # should be 1200 for this problem

# Training
learning_rate = 0.001
epochs = 301
dropout = 0.1
lr = 0.001
scheduler_step = 20000
lr_gamma = 0.97

# Log
is_wandb = True
log_interval = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout).to(device)

# data_reader = DataReader("/Users/mfallahi/Sources/cgan/dataset/3p6_time_{}.torch")
# data_reader = DataReader("/mnt/d/sources/cgan/playground/dataset/3p6_time_{}.torch")
data_reader = DataReader("/mnt/d/sources/cgan/dataset/3p6/{}_tensor_for_transformer.torch.gz")

data_by_coords = data_reader.load_data(num_time_frame)
dataset = CustomDataset(data_by_coords=data_by_coords, source_len=source_len, target_len=target_len)

trainer = TrainTransformer(model, device, dataset=dataset,
                           lr=lr, epochs=epochs, log_interval=log_interval,
                           source_size=source_size, target_size=target_size, scheduler_step=scheduler_step,
                           batch_size=batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb)

loss = trainer.train()
