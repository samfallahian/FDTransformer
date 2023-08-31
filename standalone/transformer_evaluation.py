import torch
from TransformerModel import TransformerModel
from TransformerDataReader import DataReader
from TransformerDataLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader
from TransformerEvaluation import Eval
from collections import defaultdict



# Model
d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2

# Data
batch_size = 256
source_size = 8
target_size = 2
source_len = 8
target_len = 2
num_time_frame = 1200 # should be 1200 for this problem

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

model.load_state_dict(torch.load('/mnt/d/sources/cgan/saved_models/transformer_final_saved_model_08302023.pth'))
# model.load_state_dict(torch.load('/Users/mfallahi/Sources/cgan/saved_models/transformer_Final_08212023.pth', map_location=torch.device('mps')))
model.eval()  # Set the model to evaluation mode

# Load Data
data_reader = DataReader("/mnt/d/sources/cgan/dataset/4p4/{}_tensor_for_transformer.torch.gz")
# data_reader = DataReader("/Users/mfallahi/Sources/cgan/playground/dataset/3p6_time_{}.torch")

data_by_coords = data_reader.load_data(num_time_frame)
dataset = CustomDataset(data_by_coords=data_by_coords, source_len=source_len, target_len=target_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

evaluator = Eval(model, device)

mse, mae, r2, predictions_by_coords = evaluator.evaluate(dataloader)
print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')
# print(predictions_by_coords[-117, -76, -25])