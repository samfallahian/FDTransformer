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
batch_size = 48
source_size = 8
target_size = 2
source_len = 8
target_len = 2
num_time_frame = 10 # should be 1200 for this problem

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

model.load_state_dict(torch.load('/mnt/d/sources/cgan/saved_models/transformer_Final_08212023.pth'))
# model.load_state_dict(torch.load('/Users/mfallahi/Sources/cgan/saved_models/transformer_Final_08212023.pth', map_location=torch.device('mps')))
model.eval()  # Set the model to evaluation mode

# Load Data
data_reader = DataReader("/mnt/d/sources/cgan/playground/dataset/3p6_time_{}.torch")
# data_reader = DataReader("/Users/mfallahi/Sources/cgan/playground/dataset/3p6_time_{}.torch")

data_by_coords = data_reader.load_data(num_time_frame)
dataset = CustomDataset(data_by_coords=data_by_coords, source_len=source_len, target_len=target_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

evaluator = Eval(model, device)

mse, mae, r2, all_preds, all_coords = evaluator.evaluate(dataloader)
print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')

predictions_by_coords = defaultdict(list)
#
# # This loop iterates over batches of coordinates and predictions
# for coords_batch, preds_batch in zip(all_coords, all_preds):
#     # This inner loop iterates over individual items within the batch
#     for coord, pred in zip(coords_batch, preds_batch):
#         # Convert tensor coordinate to a tuple and append prediction to the respective key in the dictionary
#         c_tuple = tuple(coord.cpu().numpy())
#         predictions_by_coords[c_tuple].append(pred.cpu().numpy())
#


for coords_batch, preds_batch in zip(all_coords, all_preds):
    # Assuming both coords_batch and preds_batch are 2D
    for i in range(coords_batch.shape[0]):
        c_tuple = tuple(coords_batch[i])
        predictions_by_coords[c_tuple].append(preds_batch[i])

print(predictions_by_coords[-117, -76, -25])