import torch
from TransformerTrain import TrainTransformer
from TransformerModel import TransformerModel




# Define constants
seq_len = 48
epochs = 10 # 301
ninp = 48  # The dimension of your input feature
nhid = 128  # 200  # Dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # Number of heads in nn.MultiheadAttention models
dropout = 0.1
lr = 0.001
log_interval = 50
batch_src_seq = 9
batch_tgt_seq = 1
scheduler_step = 5000
lr_gamma = 0.97
# data_path= "/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle"
data_path= "../playground/convolutional/dataset/encoded_tensor.pickle"
is_wandb = False

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(ninp, nhead, nhid, nlayers, dropout).to(device)
trainer = TrainTransformer(model, device, data_path=data_path,
                 lr=lr, seq_len=seq_len, epochs=epochs, log_interval=log_interval,
                 batch_src_seq=9, batch_tgt_seq=batch_tgt_seq, scheduler_step=scheduler_step,
                 lr_gamma=lr_gamma, is_wandb=is_wandb)
loss = trainer.train()
