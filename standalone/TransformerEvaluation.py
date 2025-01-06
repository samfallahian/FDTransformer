import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from collections import defaultdict


# from DataDecoder import DecodeData

class Eval:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, data_loader):
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        all_preds = []
        all_preds_2d = []
        all_targets_2d = []
        results = []
        print(len(data_loader))

        with torch.no_grad():
            for src, tgt, coord, sequences in data_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                src = src.transpose(0, 1)
                tgt_input = tgt.unsqueeze(0)

                output = self.model(src, tgt_input)

                results.append({"coord": [tensor.item() for tensor in coord], "time": sequences.tolist()[0][-1], "result": output[-1][0].tolist()})

                mse = F.mse_loss(output, tgt_input, reduction='mean').item()
                mae = F.l1_loss(output, tgt_input, reduction='mean').item()

                total_mse += mse
                total_mae += mae

                all_preds_2d.extend(output.view(-1).tolist())
                all_targets_2d.extend(tgt_input.view(-1).tolist())

                all_preds.append(output.cpu())

        r2 = r2_score(all_targets_2d, all_preds_2d)
        mean_mse = total_mse / len(data_loader)
        mean_mae = total_mae / len(data_loader)

        return mean_mse, mean_mae, r2, results
