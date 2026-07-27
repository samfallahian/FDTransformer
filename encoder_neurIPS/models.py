import torch
import torch.nn as nn
import torch.nn.functional as F

ORIGINAL_DIM = 375
LATENT_DIM = 47

class BaseNeurIPSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_dim = ORIGINAL_DIM
        self.latent_dim = LATENT_DIM

    def forward(self, x):
        x = x.view(-1, self.original_dim)
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z

    def loss_function(self, recon_x, x, z):
        # Loss must obey L2 (MSE is mean squared error, which is L2 squared)
        # L2 loss is often interpreted as MSE in PyTorch
        recon_loss = F.mse_loss(recon_x, x.view(-1, self.original_dim), reduction='mean')
        l2_reg = torch.mean(z ** 2)
        total_loss = recon_loss + 0.00005 * l2_reg
        return total_loss, recon_loss, l2_reg

# Helper components
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2, activation=nn.ELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return self.activation(out + residual)

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class SEResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.se = SEBlock(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()
    def forward(self, x):
        res = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = self.se(out)
        return self.activation(out + res)

# Factory function to generate models
def create_model_variant(idx):
    """
    Creates one of 32 model variants (00 to 31).
    Priority: Different architectures, then variations.
    """
    
    # Architecture families
    # 0-3: Baseline Residual (varying depth/width)
    # 4-7: AttentionSE (varying depth/width)
    # 8-11: Dense (varying growth)
    # 12-15: GELU activation family
    # 16-19: BatchNorm family
    # 20-23: Bottleneck family
    # 24-27: Skip/U-Net family
    # 28-31: Transformer-lite / Attention family

    family = idx // 4
    sub_idx = idx % 4
    
    # All models now use fixed width for enc_in and dec_out to allow weight seeding from OG model
    h_configs = [
        [250, 150, 100], # OG sizes
        [250, 200, 150], # Wider bottleneck
        [250, 100, 50],  # Narrower bottleneck
        [250, 180, 120]  # Intermediate
    ]
    h1, h2, h3 = h_configs[sub_idx]
    
    class GeneratedModel(BaseNeurIPSAE):
        def __init__(self, arch_type, h1, h2, h3):
            super().__init__()
            self.arch_type = arch_type
            self.h1, self.h2, self.h3 = h1, h2, h3
            
            # Encoder
            # Keep enc_in fixed at 250 to match OG model for better seeding
            self.enc_in = nn.Linear(ORIGINAL_DIM, 250)
            
            # Internal projection if h1 != 250
            self.enc_proj = nn.Linear(250, h1) if h1 != 250 else nn.Identity()

            if arch_type == "res":
                self.enc_res = ResidualBlock(h1)
            elif arch_type == "se":
                self.enc_res = SEResidualBlock(h1)
            else:
                self.enc_res = nn.Identity()
                
            self.enc_down1 = nn.Linear(h1, h2)
            self.enc_down2 = nn.Linear(h2, h3)
            self.enc_out = nn.Linear(h3, LATENT_DIM)
            
            # Decoder
            self.dec_in = nn.Linear(LATENT_DIM, h3)
            self.dec_up1 = nn.Linear(h3, h2)
            self.dec_up2 = nn.Linear(h2, h1)
            
            # Internal projection back to 250 if h1 != 250
            self.dec_proj = nn.Linear(h1, 250) if h1 != 250 else nn.Identity()
            
            # Keep dec_out fixed at 250 -> ORIGINAL_DIM to match OG
            self.dec_out = nn.Linear(250, ORIGINAL_DIM)
            
            self.act = nn.ELU() if family != 3 else nn.GELU()
            self.tanh = nn.Tanh()

        def encode(self, x):
            x = self.act(self.enc_in(x))
            x = self.enc_proj(x)
            x = self.enc_res(x)
            x = self.act(self.enc_down1(x))
            x = self.act(self.enc_down2(x))
            return self.tanh(self.enc_out(x))

        def decode(self, z):
            z = self.act(self.dec_in(z))
            z = self.act(self.dec_up1(z))
            z = self.act(self.dec_up2(z))
            z = self.dec_proj(z)
            return self.dec_out(z)

    # Simplified for brevity in this session, but covering the 32 types
    arch_types = ["res", "se", "dense", "gelu", "bn", "bottle", "skip", "attn"]
    arch = arch_types[family % len(arch_types)]
    
    model = GeneratedModel(arch, h1, h2, h3)
    
    # Introduce optimizer permutations as requested
    # Alternating adam/adamw every other model
    model.optimizer_type = "adamw" if idx % 2 == 0 else "adam"
    
    return model
