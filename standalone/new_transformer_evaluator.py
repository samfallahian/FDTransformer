import torch
from TransformerModel import TransformerModel
import gzip
import io
import torch.nn.functional as F
import os
import pickle
from HybridAutoencoder import HybridAutoencoder

################ Loading Model ###########################################
d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
model.load_state_dict(torch.load('/mnt/d/sources/cgan/saved_models/transformer_final_saved_model_09162023.pth'))
model.eval()
############################################################

############# test data config #############################
source_len = 4
target_len = 1  # Don't change it
window_size = source_len + target_len
no_seq = 10  # 1200 sequences
coordinate_length = 2  # 33598 coordinate in each file
test_dataset = 'test'  # folder name for test dataset
##############################################################

result_directory = "prediction_results"
os.makedirs(result_directory, exist_ok=True)

######################## Prediction ##############################
# Given that the pattern is every `window_size` frames
for i in range(1, no_seq, window_size):
    results = []
    for coordinate_idx in range(5):
        all_tensors = []
        source_tensors = []
        target_tensors = []
        for j in range(i, i + window_size):
            with gzip.open(f"/mnt/d/sources/cgan/dataset/{test_dataset}/{j}_tensor_for_transformer.torch.gz",
                           'rb') as gz_file:
                buffer = io.BytesIO(gz_file.read())
            data = torch.load(buffer)
            data = data[coordinate_idx]
            coordinate = data['coordinates']
            all_tensors.append(data['answer'].squeeze(0))

        source_tensors = torch.stack(all_tensors[:source_len], dim=0).view(source_len, -1)
        target_tensors = torch.stack(all_tensors[source_len:], dim=0).view(target_len, -1)
        with torch.no_grad():
            source_tensors = source_tensors.to(device)
            target_tensors = target_tensors.to(device)
            output = model(source_tensors, target_tensors)
            mse = F.mse_loss(output, target_tensors, reduction='mean').item()
            mae = F.l1_loss(output, target_tensors, reduction='mean').item()

        results.append({'seq': source_len + i, 'coordinates': coordinate, 'answer': target_tensors.view(1, 8, 6),
                        'predicted_answer': output.view(1, 8, 6), 'mse': mse, 'mae': mae})
    with gzip.open(f"{result_directory}/{source_len + i}_{test_dataset}_transformer_result.torch.gz", 'wb') as f:
        pickle.dump(results, f)
    print(results)
