import torch
from TransformerTrain import TrainTransformer
from TransformerModel import TimeSeriesTransformer, TransformerModelTarget, Seq2PointPosTransformer, CustomTransformer
from TransformerDataReader import DataReader
from TransformerDataLoader import SpatioTemporalDataset

# Data
batch_size = 256
start_time_frame = 1
num_time_frame = 15 #1200

# Training
epochs = 50
scheduler_step = 220000
lr_gamma = 0.98

# Log
is_wandb = False
log_interval = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model1 = TimeSeriesTransformer().to(device)
model2 = TransformerModelTarget().to(device)
model3 = Seq2PointPosTransformer().to(device)
model4 = CustomTransformer().to(device)

# data_reader = DataReader("/mnt/d/Normalized/latent_representation_for_")
data_reader = DataReader("/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_")

data_pivot = data_reader.load_data(num_time_frame, start_time_frame)

dataset = SpatioTemporalDataset(dataframe=data_pivot, start_time_frame=start_time_frame,sequence_length=5)

trainer = TrainTransformer(model1, device, dataset=dataset,
                           epochs=epochs, log_interval=log_interval,
                           scheduler_step=scheduler_step,
                           batch_size=batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb,
                           kind=1)

loss = trainer.train()
