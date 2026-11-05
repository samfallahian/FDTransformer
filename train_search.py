import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import sys
from model import OrderedTransformerNeurIPS

# Configuration
class Config:
    TRAIN_H5 = "transformer_neurIPS/data/train_40.h5"
    VAL_H5 = "transformer_neurIPS/data/val_40.h5"
    NUM_TIME = 40
    NUM_X = 26
    SEQ_LEN = NUM_TIME * NUM_X
    INPUT_DIM = 52
    LATENT_DIM = 47
    BIAS = True
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    DROPOUT = 0.1
    
    # Architecture search space
    SEARCH_SPACE = [
        {"EMBED_SIZE": 128, "N_HEADS": 4, "N_LAYERS": 4},
        {"EMBED_SIZE": 128, "N_HEADS": 8, "N_LAYERS": 4},
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 4},
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6}, # Similar to v2
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 4},
        {"EMBED_SIZE": 512, "N_HEADS": 16, "N_LAYERS": 4},
        {"EMBED_SIZE": 256, "N_HEADS": 4, "N_LAYERS": 8},
        {"EMBED_SIZE": 128, "N_HEADS": 4, "N_LAYERS": 8},
        {"EMBED_SIZE": 384, "N_HEADS": 12, "N_LAYERS": 6},
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 8},
    ]

class TransformerDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        data = self._file['data'][idx]
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        return torch.from_numpy(data).float()

def l2_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

def train_model(variant_idx):
    variant = Config.SEARCH_SPACE[variant_idx]
    # Update Config with variant
    Config.EMBED_SIZE = variant["EMBED_SIZE"]
    Config.N_HEADS = variant["N_HEADS"]
    Config.N_LAYERS = variant["N_LAYERS"]
    
    run_name = f"variant_{variant_idx}_E{Config.EMBED_SIZE}_H{Config.N_HEADS}_L{Config.N_LAYERS}"
    wandb.init(project="transformer_neurIPS_search", name=run_name, config=variant)
    
    train_dataset = TransformerDataset(Config.TRAIN_H5)
    val_dataset = TransformerDataset(Config.VAL_H5)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = OrderedTransformerNeurIPS(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
            batch = batch.to(Config.DEVICE)
            # Teacher forcing: input is [0:T-1], target is [1:T]
            # But wait, target is only latent dim
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :Config.LATENT_DIM]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l2_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"):
                batch = batch.to(Config.DEVICE)
                inputs = batch[:, :-1, :]
                targets = batch[:, 1:, :Config.LATENT_DIM]
                outputs = model(inputs)
                loss = l2_loss(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"transformer_neurIPS/saved_models/{run_name}_best.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        variant_idx = int(sys.argv[1])
        train_model(variant_idx)
    else:
        for i in range(len(Config.SEARCH_SPACE)):
            print(f"Starting search for variant {i}...")
            train_model(i)
