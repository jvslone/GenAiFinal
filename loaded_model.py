import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONFIG
use_rho_time = 0
Input_File = 'ChangeDataset622v2.pickle'
latent_dim = 150  # Must match trained model

# LOAD DATA
def ReadPickle(filename: str):
    try:
        return pd.read_pickle(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f'Error reading {filename}')

Data = ReadPickle(Input_File)

# PREPROCESS
D_data = np.stack([np.tile(s['Diffusion'], (1000, 1)) for s in Data])
V_data = np.stack([np.tile(s['Convection'], (1000, 1)) for s in Data])
N_data = np.stack([s['Density'] for s in Data])
S_data = np.stack([s['Source'].T for s in Data])

if use_rho_time:
    R_data = np.stack([np.tile(s['Rho'], (1000, 1)) for s in Data])
    T_data = np.stack([np.tile(s['Time'], (50, 1)).T for s in Data])
    Database = np.stack([np.array([D_data[i], V_data[i], R_data[i], T_data[i], N_data[i], S_data[i]])
                         for i in range(len(Data))])
else:
    Database = np.stack([np.array([D_data[i], V_data[i], N_data[i], S_data[i]])
                         for i in range(len(Data))])

# STANDARDIZATION
num_channels = Database.shape[1]
means = Database.mean(axis=(0, 2, 3), keepdims=True)
stds = Database.std(axis=(0, 2, 3), keepdims=True)
Database_channel_standardized = (Database - means) / stds

# SAVE EACH CHANNEL SEPARATELY
np.save("Diffusion_channel.npy", D_data)
np.save("Convection_channel.npy", V_data)
np.save("Density_channel.npy", N_data)
np.save("Source_channel.npy", S_data)

# CUSTOM DATASET
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], 0

# MODEL DEFS
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=150):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels*2, 3, 2, 1),
            nn.SELU(),
            nn.Conv2d(num_channels*2, num_channels*4, 3, 2, 1),
            nn.SELU(),
            nn.Conv2d(num_channels*4, num_channels*8, 3, 2, 1),
            nn.SELU(),
            nn.Conv2d(num_channels*8, num_channels*16, 3, 2, 1),
            nn.SELU()
        )
        self.fc1 = nn.Linear(num_channels*16*63*4, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, num_channels*16*63*4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(num_channels*16, num_channels*8, 3, 2, 1, 0),
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*8, num_channels*4, 3, 2, 1, (1, 0)),
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*4, num_channels*2, 3, 2, 1, (1, 0)),
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*2, num_channels, 3, 2, 1, (1, 1))
        )

    def encode(self, x):
        x = self.encoder_conv(x).view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        x = self.decoder_fc(z).view(-1, num_channels*16, 63, 4)
        return self.decoder_conv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# LOAD VAE
vae = VariationalAutoencoder(latent_dim).to(device)
vae.load_state_dict(torch.load("vae_gan_best.pt", map_location=device))
vae.eval()

# DATALOADER
dataset = CustomDataset(Database_channel_standardized)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# GENERATE & SAVE SAMPLE
z = torch.randn(1, latent_dim, device=device)
generated = vae.decode(z).squeeze(0).cpu().numpy()
np.save("generated_sample.npy", generated)

# SAVE REAL SAMPLE
real_sample = next(iter(dataloader))[0][0]
np.save("real_sample.npy", real_sample.numpy())

print("Sample generation and saving completed.")
