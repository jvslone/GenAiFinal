import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 150
num_channels = 4
Input_File = 'ChangeDataset622v2.pickle'
model_path = 'vae_gan_best.pt'
channel_names = ['Diffusion', 'Convection', 'Density', 'Source']

# ---------------- Load & Standardize Data ----------------
def ReadPickle(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    return pd.read_pickle(filename)

Data = ReadPickle(Input_File)

D_data = np.stack([np.tile(s['Diffusion'], (1000, 1)) for s in Data])
V_data = np.stack([np.tile(s['Convection'], (1000, 1)) for s in Data])
N_data = np.stack([s['Density'] for s in Data])
S_data = np.stack([s['Source'].T for s in Data])

Database = np.stack([np.array([D_data[i], V_data[i], N_data[i], S_data[i]]) for i in range(len(Data))])

# Standardize per-channel
means = Database.mean(axis=(0, 2, 3), keepdims=True)
stds = Database.std(axis=(0, 2, 3), keepdims=True)
Database_standardized = (Database - means) / stds

# ---------------- VAE Architecture ----------------
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


# ---------------- Load VAE ----------------
vae = VariationalAutoencoder(latent_dim).to(device)
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

# ---------------- MMD Function ----------------
def MMD(x, y, kernel="rbf"):
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
    y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-8)

    def compute_distances(a, b):
        norm_a = (a ** 2).sum(dim=1).view(-1, 1)
        norm_b = (b ** 2).sum(dim=1).view(-1, 1)
        return norm_a + norm_b.t() - 2.0 * torch.mm(a, b.t())

    dxx = compute_distances(x, x)
    dyy = compute_distances(y, y)
    dxy = compute_distances(x, y)

    XX, YY, XY = torch.zeros_like(dxx), torch.zeros_like(dyy), torch.zeros_like(dxy)

    for a in [0.2, 0.5, 0.9, 1.3]:
        a2 = a ** 2
        XX += a2 * (a2 + dxx).reciprocal()
        YY += a2 * (a2 + dyy).reciprocal()
        XY += a2 * (a2 + dxy).reciprocal()
    return torch.mean(XX + YY - 2 * XY)

# ---------------- Plot Distribution ----------------
def plot_distribution():
    num_samples = len(Database_standardized)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Distribution Comparison: Real vs. Generated", fontsize=16)

    # Generate
    generated_all = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_dim, device=device)
            fake = vae.decode(z).squeeze(0).cpu().numpy()
            generated_all.append(fake)

    for c in range(4):
        ax = fig.add_subplot(2, 2, c + 1)
        real_flat = Database_standardized[:, c].flatten()
        gen_flat = np.concatenate([sample[c].flatten() for sample in generated_all])

        ax.hist(real_flat, bins=50, alpha=0.5, label='Real', density=True)
        ax.hist(gen_flat, bins=50, alpha=0.5, label='Generated', density=True)
        ax.set_title(f"{channel_names[c]} Distribution")
        ax.legend()

        # Stats
        real_mean, real_std = real_flat.mean(), real_flat.std()
        gen_mean, gen_std = gen_flat.mean(), gen_flat.std()

        try:
            real_tensor = torch.tensor(real_flat[:5000].reshape(-1, 100), dtype=torch.float32)
            gen_tensor = torch.tensor(gen_flat[:5000].reshape(-1, 100), dtype=torch.float32)
            mmd_val = MMD(real_tensor, gen_tensor, kernel="rbf").item()
        except Exception as e:
            mmd_val = float('nan')
            print(f"MMD failed on channel {channel_names[c]}: {e}")

        ax.text(0.02, 0.95,
                f"Real: μ={real_mean:.2f}, σ={real_std:.2f}\n"
                f"Gen: μ={gen_mean:.2f}, σ={gen_std:.2f}\n"
                f"MMD: {mmd_val:.4f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("Image", exist_ok=True)
    plt.savefig("Image/Distribution_MMD_Comparison.png", dpi=300)
    plt.show()

# ---------------- Main Call ----------------
plot_distribution()
