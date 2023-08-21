import torch
import torch.nn as nn
from TransformerTrain import TrainTransformer
from TransformerModel import TransformerModel
from TransformerDataReader import DataReader
from TransformerDataLoader import CustomDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F



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
model.eval()  # Set the model to evaluation mode

# Load Data
data_reader = DataReader("/mnt/d/sources/cgan/playground/dataset/3p6_time_{}.torch")

data_by_coords = data_reader.load_data(num_time_frame)
dataset = CustomDataset(data_by_coords=data_by_coords, source_len=source_len, target_len=target_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()  # Or whichever loss you used during training


# def evaluate(model, test_loader, criterion):
#     model.eval()
#     total_loss = 0.0
#
#     with torch.no_grad():
#         for source, target in test_loader:
#             output = model(source)
#             loss = criterion(output, target)
#             total_loss += loss.item()
#
#     avg_loss = total_loss / len(test_loader)
#     return avg_loss




def evaluate(model, data_loader):
    total_mse = 0.0
    total_mae = 0.0
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for coords_batch, source_batch, target_batch in data_loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            tgt = torch.zeros((source_batch.size(0), 2, source_batch.size(2)), device=device)
            output = model(source_batch, tgt)
            loss = criterion(output, target_batch)
            mse = F.mse_loss(output, target_batch, reduction='mean').item()
            mae = F.l1_loss(output, target_batch, reduction='mean').item()
            total_mse += mse
            total_mae += mae
            total_loss += loss.item()

            all_preds.extend(output.view(-1).tolist())
            all_targets.extend(target_batch.view(-1).tolist())


    r2 = r2_score(all_targets, all_preds)
    mean_mse = total_mse / len(dataloader)
    mean_mae = total_mae / len(dataloader)
    mean_loss = total_loss / len(dataloader)

    return mean_mse, mean_mae, r2, mean_loss


mse, mae, r2, loss = evaluate(model, dataloader)
print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}, LOSS: {loss}')



