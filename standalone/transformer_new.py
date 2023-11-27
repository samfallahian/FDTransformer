import torch
import torch.nn as nn
from TransformerTrain import TrainTransformer
from TransformerModel import TimeSeriesTransformer, TransformerModelTarget, Seq2PointPosTransformer, CustomTransformer
from TransformerDataReader import DataReader
from TransformerDataLoader import SpatioTemporalDataset

# Data
batch_size = 512
start_time_frame = 1
end_time_frame = 50 #1200

# Training
epochs = 102
scheduler_step = 280000
lr_gamma = 0.98

# Log
is_wandb = True
log_interval = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model_no = 4

if model_no == 1:
    model = TimeSeriesTransformer().to(device)
elif model_no == 2:
    model = TransformerModelTarget().to(device)
elif model_no == 3:
    model = Seq2PointPosTransformer().to(device)
elif model_no == 4:
    model = CustomTransformer().to(device)
    model = nn.DataParallel(model)

data_reader = DataReader("/mnt/d/Normalized/latent_representation_for_")
# data_reader = DataReader("/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_")

data = data_reader.load_data(end_time_frame, start_time_frame)

dataset = SpatioTemporalDataset(dataframe=data, num_files=end_time_frame,window_size=5, start_time_frame=start_time_frame)

trainer = TrainTransformer(model, device, dataset=dataset, lr=0.001,
                           epochs=epochs, log_interval=log_interval,
                           scheduler_step=scheduler_step,
                           batch_size=batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb,
                           kind=model_no, start_time_frame=start_time_frame, end_time_frame=end_time_frame)

loss = trainer.train()
