import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as timer
import pickle

use_rho_time = 0
Input_File = 'ChangeDataset622v2.pickle'

def ReadPickle(filename: str) -> dict:
  '''Reads in data from given pickle files, outputs a dictionary'''
  try:
    Data = pd.read_pickle(filename)
  except FileNotFoundError:
    raise FileNotFoundError(f'Error reading {filename}')
  return Data

print(f"Loading Dataset -- '{Input_File}'")
t_Load1a = timer.time()
Data = ReadPickle(Input_File)
t_Load2a = timer.time()
print(f"Dataset took {round(t_Load2a-t_Load1a,2)}s to load")

D_data = np.array([np.tile(sample['Diffusion'],(1000,1)) for sample in Data])
V_data = np.array([np.tile(sample['Convection'],(1000,1)) for sample in Data])
if use_rho_time:
  R_data = np.array([np.tile(sample['Rho'],(1000,1)) for sample in Data])
  T_data = np.array([np.tile(sample['Time'],(50, 1)).T for sample in Data])
N_data = np.array([Sample['Density'] for Sample in Data])
S_data = np.array([Sample['Source'].T for Sample in Data])
if use_rho_time:
  Database = np.array([np.array([D_data[i],V_data[i],R_data[i],T_data[i],N_data[i],S_data[i]]) for i in range(len(Data))])
else:
  Database = np.array([np.array([D_data[i],V_data[i],N_data[i],S_data[i]]) for i in range(len(Data))])
print(Database.shape)




#This i will code myself later, had gpt code up a basic scaler to use for testing now

#############################GPT CODE FOR SCALING##############################

import numpy as np

# Assume Database is loaded with shape (1000, 6, 1000, 50)
print("Original shape:", Database.shape)

# Prepare an empty array to hold the standardized data
Database_channel_standardized = np.empty_like(Database)

# Loop over each channel to standardize it independently.
# Here axis=(0, 2, 3) computes statistics across all samples and the 2D dimensions for the given channel.
num_channels = Database.shape[1]
for c in range(num_channels):
    channel_data = Database[:, c, :, :]
    channel_mean = np.mean(channel_data)
    channel_std = np.std(channel_data)
    print(f"Channel {c}: mean = {channel_mean:.4f}, std = {channel_std:.4f}")
    
    Database_channel_standardized[:, c, :, :] = (channel_data - channel_mean) / channel_std

# Optionally, verify the results for one channel:
print("\nChannel 3 after standardization: mean = {:.4f}, std = {:.4f}".format(
    np.mean(Database_channel_standardized[:, 3, :, :]),
    np.std(Database_channel_standardized[:, 3, :, :])
))

#############################GPT CODE FOR SCALING##############################





import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Convolutional Variational Autoencoder:
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=640):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder:
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=3, stride=2, padding=1),# -> (12, 500, 25)
            nn.SELU(),
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, stride=2, padding=1),# -> (24, 250, 13)
            nn.SELU(),
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3, stride=2, padding=1),# -> (48, 125, 7)
            nn.SELU(),
            nn.Conv2d(num_channels*8, num_channels*16, kernel_size=3, stride=2, padding=1),# -> (96, 63, 4)
            nn.SELU()
        )
        
        #Flattened conv feature size is 96*63*4 = 24192
        self.fc1 = nn.Linear(num_channels*16*63*4,128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        #Decoder:
        self.decoder_fc = nn.Linear(latent_dim, num_channels*16*63*4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(num_channels*16, num_channels*8, kernel_size=3, stride=2, padding=1, output_padding=0),# -> (48, 125, 7)
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*8, num_channels*4, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),# -> (24, 250, 13)
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*4, num_channels*2, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),# -> (12, 500, 25)
            nn.SELU(),
            nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))# -> (6, 1000, 50)
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0),-1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1,num_channels*16,63,4)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

#Updated Loss Function using Mean Squared Error (MSE)
def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

#Custom Dataset to handle the numerical array
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], 0
      
      
      

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 6, kernel_size=(15,3), stride=(100,10), padding=1),  # (6, 10, 5)
            nn.BatchNorm2d(6),
            nn.ReLU(),
            
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*10*5, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()  # Binary classification (real vs fake)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)





dataset = CustomDataset(Database_channel_standardized)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hyperparameters
latent_dim = 640
batch_size = 32
lr_vae = 1e-4
num_epochs_pretraining = 50#50

#Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VariationalAutoencoder(latent_dim=latent_dim).to(device)
opt_G = optim.Adam(vae.parameters(), lr=lr_vae)

#Training Loop
for epoch in range(num_epochs_pretraining):
    epoch_loss = 0
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        decoded_data, mu, logvar = vae(batch_data)
        
        loss = vae_loss(decoded_data, batch_data, mu, logvar)
        opt_G.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        opt_G.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs_pretraining}], Average Loss: {avg_loss:.4f}")
    
print("Training complete!")






torch.save(vae, 'VAE_Pretrained_NoRT_v1.pt')

import torch
import torch.nn as nn
import torch.optim as optim

vae = torch.load('VAE_Pretrained_NoRT_v1.pt', weights_only=False)
num_epochs = 25
lr_disc = 4e-5
lr_vae = 1e-2
disc = CNNDiscriminator().to(device)
opt_D = optim.Adam(disc.parameters(), lr=lr_disc)
opt_G = optim.Adam(vae.parameters(), lr=lr_vae)

# Loss functions
bce = nn.BCELoss()
mse_loss = nn.MSELoss(reduction='mean')

# Label funcs
def real_label(size):
    return torch.ones(size, 1, device=device)

def fake_label(size):
    return torch.zeros(size, 1, device=device)

# Training Loop
for epoch in range(1, num_epochs + 1):
    vae.train()
    disc.train()

    total_G_loss = 0.0
    total_D_loss = 0.0

    for x_real, _ in dataloader:
        x_real = x_real.to(device)
        bs = x_real.size(0)

        # Discriminator update
        with torch.no_grad():
            z = torch.randn(bs, latent_dim, device=device)
            x_sample = vae.decode(z) # Pure decoder samples
            bs, C, T, S = x_sample.shape

            # Time Average D & V
            time_mean = x_sample[:, 0:2, :, :].mean(dim=2, keepdim=True) 
            time_mean_expanded = time_mean.expand(bs, 2, T, S)
            x_sample[:, 0:2, :, :] = time_mean_expanded

        #Add Noise
        noise_real = 0.1*x_real.std(dim=[0,2,3], keepdim=True)*torch.randn_like(x_real)
        noise_fake = 0.1*x_sample.std(dim=[0,2,3], keepdim=True)*torch.randn_like(x_sample)

        x_real_noisy = x_real + noise_real
        x_sample_noisy = x_sample + noise_fake

        # 3) Discriminator forward
        d_out_real = disc(x_real_noisy)
        d_out_fake = disc(x_sample_noisy)

        loss_D_real = bce(d_out_real, real_label(bs))
        loss_D_fake = bce(d_out_fake, fake_label(bs))
        loss_D = 0.5*(loss_D_real + loss_D_fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Generator (VAE) update
        x_recon, mu, logvar = vae(x_real)
        d_out_fake = disc(x_recon)

        # Reconstruction + KL
        recon_loss = mse_loss(x_recon, x_real)
        kl_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Adversarial (fool discriminator)
        adv_loss = bce(d_out_fake, real_label(bs))

        # Weighted sum
        lambda_kl = 1e-4
        lambda_adv = 1e-4
        loss_G = recon_loss + lambda_kl*kl_loss + lambda_adv*adv_loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        total_D_loss += loss_D.item()
        total_G_loss += loss_G.item()

    avg_D = total_D_loss/len(dataloader)
    avg_G = total_G_loss/len(dataloader)
    print(f"Epoch [{epoch}/{num_epochs}]  Loss_D: {avg_D:.4f}  Loss_G: {avg_G:.4f}")
