import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

class Eval:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, data_loader):
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        all_preds = []
        all_targets = []

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

                all_preds.extend(output.view(-1).tolist())
                all_targets.extend(target_batch.view(-1).tolist())

        r2 = r2_score(all_targets, all_preds)
        mean_mse = total_mse / len(data_loader)
        mean_mae = total_mae / len(data_loader)

        return mean_mse, mean_mae, r2
