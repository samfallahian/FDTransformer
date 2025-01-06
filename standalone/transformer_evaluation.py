import torch
import torch.nn as nn
from TransformerModel import TimeSeriesTransformer, TransformerModelTarget, Seq2PointPosTransformer, CustomTransformer
from TransformerDataReader import DataReader
from TransformerDataLoader import SpatioTemporalDataset
from torch.utils.data import Dataset, DataLoader
from TransformerEvaluation import Eval
from collections import defaultdict
import pandas as pd

# Data
batch_size = 1
start_time_frame = 1
end_time_frame = 10  # 1200


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

model_path = "/mnt/d/sources/cgan/saved_models/transformer_sequence_1101_to_1200_11302023.pth"

model.load_state_dict(torch.load(model_path)["model_state_dict"])

# Load Data
# data_reader = DataReader("/mnt/d/Normalized/4p6/latent_representation_for_")
data_reader = DataReader("/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_")

data = data_reader.load_data(end_time_frame, start_time_frame, step=10)

dataset = SpatioTemporalDataset(dataframe=data, num_files=end_time_frame, window_size=5,
                                start_time_frame=start_time_frame)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

evaluator = Eval(model, device)

mse, mae, r2, results = evaluator.evaluate(dataloader)

print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')

results_df = pd.DataFrame(results, columns=["coord", "time", "result"])
results_df.to_csv("/mnt/d/sources/cgan/standalone/predictions/result.csv", index=False)
print(results_df.head())
