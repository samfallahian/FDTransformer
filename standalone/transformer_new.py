import torch
from TransformerTrain import TrainTransformer
from TransformerModel import TransformerModel




# Define constants
d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2
learning_rate = 0.001
epochs = 301
batch_size = 48
dropout = 0.1
lr = 0.001
log_interval = 50
scheduler_step = 5000
lr_gamma = 0.97
source_size = 8
target_size = 2
data_path= "/mnt/d/sources/cgan/standalone/dataset/encoded_tensor_08082023.pickle"
# data_path= "../playground/convolutional/dataset/encoded_tensor.pickle"
is_wandb = True

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout).to(device)
trainer = TrainTransformer(model, device, data_path=data_path,
                 lr=lr, epochs=epochs, log_interval=log_interval,
                 source_size = source_size, target_size = target_size, scheduler_step=scheduler_step,
                 batch_size = batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb)


loss = trainer.train()
