import torch
import torch.nn as nn
import torch.nn.functional as F

# Common parameters
ORIGINAL_DIM = 375
LATENT_DIM = 47

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2, activation=nn.ELU()):
        super(ResidualBlock, self).__init__()
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
        out = out + residual
        return self.activation(out)

class BaseAE(nn.Module):
    """Abstract base class to provide common loss function and forward pass"""
    def __init__(self):
        super(BaseAE, self).__init__()
        self.original_dim = ORIGINAL_DIM
        self.latent_dim = LATENT_DIM

    def forward(self, x):
        x = x.view(-1, self.original_dim)
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z

    def loss_function(self, recon_x, x, z):
        recon_loss = F.mse_loss(recon_x, x.view(-1, self.original_dim), reduction='mean')
        l2_reg = torch.mean(z ** 2)
        total_loss = recon_loss + 0.00005 * l2_reg
        return total_loss, recon_loss, l2_reg, torch.tensor(0.0)

# 1. Baseline Model (similar to Model 09)
class Model_GEN3_01_Baseline(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        h1, h2, h3 = 250, 150, 100
        
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = ResidualBlock(h1, dropout_rate)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = ResidualBlock(h2, dropout_rate)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = ResidualBlock(h3, dropout_rate)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = ResidualBlock(h3, dropout_rate)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = ResidualBlock(h2, dropout_rate)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = ResidualBlock(h1, dropout_rate)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.elu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.elu(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.elu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.elu(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 2. Deep Model (More residual blocks)
class Model_GEN3_02_Deep(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = nn.Sequential(ResidualBlock(h1), ResidualBlock(h1))
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = nn.Sequential(ResidualBlock(h2), ResidualBlock(h2))
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = nn.Sequential(ResidualBlock(h3), ResidualBlock(h3))
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = nn.Sequential(ResidualBlock(h3), ResidualBlock(h3))
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = nn.Sequential(ResidualBlock(h2), ResidualBlock(h2))
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = nn.Sequential(ResidualBlock(h1), ResidualBlock(h1))
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = self.elu(self.enc_in(x))
        x = self.enc_res1(x)
        x = self.elu(self.enc_down1(x))
        x = self.enc_res2(x)
        x = self.elu(self.enc_down2(x))
        x = self.enc_res3(x)
        return self.tanh(self.enc_out(x))

    def decode(self, z):
        z = self.elu(self.dec_in(z))
        z = self.dec_res1(z)
        z = self.elu(self.dec_up1(z))
        z = self.dec_res2(z)
        z = self.elu(self.dec_up2(z))
        z = self.dec_res3(z)
        return self.dec_out(z)

# 3. Wide Model
class Model_GEN3_03_Wide(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 512, 256, 128
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = ResidualBlock(h1, dropout_rate)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = ResidualBlock(h2, dropout_rate)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = ResidualBlock(h3, dropout_rate)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = ResidualBlock(h3, dropout_rate)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = ResidualBlock(h2, dropout_rate)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = ResidualBlock(h1, dropout_rate)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.elu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.elu(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.elu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.elu(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 4. GELU Activation
class Model_GEN3_04_GELU(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.act = nn.GELU()
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = ResidualBlock(h1, dropout_rate, activation=self.act)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = ResidualBlock(h2, dropout_rate, activation=self.act)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = ResidualBlock(h3, dropout_rate, activation=self.act)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = ResidualBlock(h3, dropout_rate, activation=self.act)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = ResidualBlock(h2, dropout_rate, activation=self.act)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = ResidualBlock(h1, dropout_rate, activation=self.act)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.act(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.act(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.act(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.act(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.act(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.act(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 5. Squeeze-and-Excitation (SE) Block
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
        b, c = x.size()
        y = self.fc(x)
        return x * y

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

class Model_GEN3_05_AttentionSE(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = SEResidualBlock(h1, dropout_rate)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = SEResidualBlock(h2, dropout_rate)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = SEResidualBlock(h3, dropout_rate)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = SEResidualBlock(h3, dropout_rate)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = SEResidualBlock(h2, dropout_rate)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = SEResidualBlock(h1, dropout_rate)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.elu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.elu(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.elu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.elu(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 6. Dense Connections in blocks
class DenseBlock(nn.Module):
    def __init__(self, dim, growth_rate=32, dropout_rate=0.2):
        super().__init__()
        self.fc = nn.Linear(dim, growth_rate)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()
    def forward(self, x):
        out = self.activation(self.fc(self.dropout(self.norm(x))))
        return torch.cat([x, out], dim=1)

class Model_GEN3_06_Dense(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Since we use cat, we need to handle increasing dimensions
        # but here we'll just use a simple version that projects back
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = nn.Sequential(DenseBlock(h1), nn.Linear(h1+32, h1))
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = nn.Sequential(DenseBlock(h2), nn.Linear(h2+32, h2))
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = nn.Sequential(DenseBlock(h3), nn.Linear(h3+32, h3))
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = nn.Sequential(DenseBlock(h3), nn.Linear(h3+32, h3))
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = nn.Sequential(DenseBlock(h2), nn.Linear(h2+32, h2))
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = nn.Sequential(DenseBlock(h1), nn.Linear(h1+32, h1))
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.elu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.elu(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.elu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.elu(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 7. BatchNorm instead of LayerNorm
class BNResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()
    def forward(self, x):
        res = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return self.activation(out + res)

class Model_GEN3_07_BatchNorm(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_bn1 = nn.BatchNorm1d(h1)
        self.enc_res1 = BNResidualBlock(h1, dropout_rate)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_bn2 = nn.BatchNorm1d(h2)
        self.enc_res2 = BNResidualBlock(h2, dropout_rate)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_bn3 = nn.BatchNorm1d(h3)
        self.enc_res3 = BNResidualBlock(h3, dropout_rate)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_bn1 = nn.BatchNorm1d(h3)
        self.dec_res1 = BNResidualBlock(h3, dropout_rate)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_bn2 = nn.BatchNorm1d(h2)
        self.dec_res2 = BNResidualBlock(h2, dropout_rate)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_bn3 = nn.BatchNorm1d(h1)
        self.dec_res3 = BNResidualBlock(h1, dropout_rate)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_bn1(self.enc_in(x)))
        h = self.enc_res1(h)
        h = self.elu(self.enc_bn2(self.enc_down1(h)))
        h = self.enc_res2(h)
        h = self.elu(self.enc_bn3(self.enc_down2(h)))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_bn1(self.dec_in(z)))
        h = self.dec_res1(h)
        h = self.elu(self.dec_bn2(self.dec_up1(h)))
        h = self.dec_res2(h)
        h = self.elu(self.dec_bn3(self.dec_up2(h)))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 8. Skip connections between Encoder and Decoder (U-Net style)
class Model_GEN3_08_Skip(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        # Encoder
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = ResidualBlock(h1)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = ResidualBlock(h2)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = ResidualBlock(h3)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        # Decoder
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = ResidualBlock(h3)
        self.dec_up1 = nn.Linear(h3*2, h2) # *2 for skip connection
        self.dec_res2 = ResidualBlock(h2)
        self.dec_up2 = nn.Linear(h2*2, h1) # *2 for skip connection
        self.dec_res3 = ResidualBlock(h1)
        self.dec_out = nn.Linear(h1*2, ORIGINAL_DIM) # *2 for skip connection
        
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        self.h1_e = self.enc_res1(self.elu(self.enc_in(x)))
        self.h2_e = self.enc_res2(self.elu(self.enc_down1(self.h1_e)))
        self.h3_e = self.enc_res3(self.elu(self.enc_down2(self.h2_e)))
        return self.tanh(self.enc_out(self.h3_e))

    def decode(self, z):
        h3_d = self.dec_res1(self.elu(self.dec_in(z)))
        h2_d = self.dec_res2(self.elu(self.dec_up1(torch.cat([h3_d, self.h3_e], dim=1))))
        h1_d = self.dec_res3(self.elu(self.dec_up2(torch.cat([h2_d, self.h2_e], dim=1))))
        return self.dec_out(torch.cat([h1_d, self.h1_e], dim=1))

# 9. Bottleneck Blocks (ResNet style: reduce -> transform -> expand)
class BottleneckBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, dim)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()
    def forward(self, x):
        res = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.activation(self.norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.norm3(self.fc3(out))
        return self.activation(out + res)

class Model_GEN3_09_Bottleneck(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = BottleneckBlock(h1, h1//2)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = BottleneckBlock(h2, h2//2)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = BottleneckBlock(h3, h3//2)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = BottleneckBlock(h3, h3//2)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = BottleneckBlock(h2, h2//2)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = BottleneckBlock(h1, h1//2)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.elu(self.enc_in(x))
        h = self.enc_res1(h)
        h = self.elu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = self.elu(self.enc_down2(h))
        h = self.enc_res3(h)
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.elu(self.dec_in(z))
        h = self.dec_res1(h)
        h = self.elu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = self.elu(self.dec_up2(h))
        h = self.dec_res3(h)
        return self.dec_out(h)

# 10. Self-Attention (Middle)
class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
    def forward(self, x):
        # Very simplified self-attention for a single vector per sample
        # Here we'll treat it as a gating mechanism
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.sigmoid((q * k) / self.scale)
        return v * attn

class Model_GEN3_10_Attention(BaseAE):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        h1, h2, h3 = 250, 150, 100
        self.enc_in = nn.Linear(ORIGINAL_DIM, h1)
        self.enc_res1 = ResidualBlock(h1)
        self.enc_down1 = nn.Linear(h1, h2)
        self.enc_res2 = ResidualBlock(h2)
        self.enc_attn = SimpleAttention(h2)
        self.enc_down2 = nn.Linear(h2, h3)
        self.enc_res3 = ResidualBlock(h3)
        self.enc_out = nn.Linear(h3, LATENT_DIM)
        
        self.dec_in = nn.Linear(LATENT_DIM, h3)
        self.dec_res1 = ResidualBlock(h3)
        self.dec_up1 = nn.Linear(h3, h2)
        self.dec_res2 = ResidualBlock(h2)
        self.dec_attn = SimpleAttention(h2)
        self.dec_up2 = nn.Linear(h2, h1)
        self.dec_res3 = ResidualBlock(h1)
        self.dec_out = nn.Linear(h1, ORIGINAL_DIM)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        
    def encode(self, x):
        h = self.enc_res1(self.elu(self.enc_in(x)))
        h = self.enc_attn(self.enc_res2(self.elu(self.enc_down1(h))))
        h = self.tanh(self.enc_out(self.enc_res3(self.elu(self.enc_down2(h)))))
        return h
        
    def decode(self, z):
        h = self.dec_res1(self.elu(self.dec_in(z)))
        h = self.dec_attn(self.dec_res2(self.elu(self.dec_up1(h))))
        h = self.dec_out(self.dec_res3(self.elu(self.dec_up2(h))))
        return h

def get_model_by_index(index, dropout_rate=0.2):
    models = [
        Model_GEN3_01_Baseline,
        Model_GEN3_02_Deep,
        Model_GEN3_03_Wide,
        Model_GEN3_04_GELU,
        Model_GEN3_05_AttentionSE,
        Model_GEN3_06_Dense,
        Model_GEN3_07_BatchNorm,
        Model_GEN3_08_Skip,
        Model_GEN3_09_Bottleneck,
        Model_GEN3_10_Attention
    ]
    return models[index](dropout_rate=dropout_rate)
