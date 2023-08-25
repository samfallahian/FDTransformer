import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from collections import defaultdict

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
        # all_targets = []
        all_targets_2d = []
        all_coords = []

        with torch.no_grad():
            for coords_batch, source_batch, target_batch in data_loader:
                source_batch = source_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                tgt = torch.zeros((source_batch.size(0), 2, source_batch.size(2)), device=self.device)
                output = self.model(source_batch, tgt)
                mse = F.mse_loss(output, target_batch, reduction='mean').item()
                mae = F.l1_loss(output, target_batch, reduction='mean').item()
                total_mse += mse
                total_mae += mae

                all_preds_2d.extend(output.view(-1).tolist())
                # all_coords.extend(coords_batch.view(-1).tolist())
                all_targets_2d.extend(target_batch.view(-1).tolist())

                all_preds.append(output.cpu())
                all_coords.append(coords_batch.cpu())
                # all_targets.append(target_batch.cpu())

            all_preds = torch.cat(all_preds, dim=0)
            all_coords = torch.cat(all_coords, dim=0)
            # all_targets = torch.cat(all_targets, dim=0)

        r2 = r2_score(all_targets_2d, all_preds_2d)
        mean_mse = total_mse / len(data_loader)
        mean_mae = total_mae / len(data_loader)

        predictions_by_coords = defaultdict(list)

        for i in range(all_coords.size(0)):
            c_tuple = tuple(all_coords[i].numpy())
            predictions_by_coords[c_tuple].append(all_preds[i].numpy())


        return mean_mse, mean_mae, r2, predictions_by_coords
