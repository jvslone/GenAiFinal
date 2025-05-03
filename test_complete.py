import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as timer
import pickle

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable



# ==================================================================== Data Processing ====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data():
    use_rho_time = 0
    Input_File = 'ChangeDataset622v2.pickle'
    # Input_File = 'Data/3k.pickle'
    
    def ReadPickle(filename: str) -> dict:
        '''Reads in data from given pickle files, outputs a dictionary'''
        try:
            Data = pd.read_pickle(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f'Error reading {filename}')
        return Data
    
    logger.info(f"Loading Dataset -- '{Input_File}'")
    t_Load1a = time.time()
    Data = ReadPickle(Input_File)
    t_Load2a = time.time()
    logger.info(f"Dataset took {round(t_Load2a-t_Load1a,2)}s to load")
    
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
    
    logger.info(f"Shape: {Database.shape}")
    
    # Prepare an empty array to hold the standardized data
    Database_channel_standardized = np.empty_like(Database)
    
    # Loop over each channel to standardize it independently
    num_channels = Database.shape[1]
    for c in range(num_channels):
        channel_data = Database[:, c, :, :]
        channel_mean = np.mean(channel_data)
        channel_std = np.std(channel_data)
        
        print(f"Channel {c}: Original mean = {channel_mean:.4e}, std = {channel_std:.4e}")

        standardized = (channel_data - channel_mean) / channel_std
        Database_channel_standardized[:, c, :, :] = standardized
        
        print(f"Channel {c}: After standardization, mean = {np.mean(standardized):.4e}, std = {np.std(standardized):.4e}")

    return Database_channel_standardized


# ==================================================================== Configure logging ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s',
    handlers=[
        logging.FileHandler("vae_gan_opt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VAE-GAN-BayesOpt")

num_channels = 4 

# ==================================================================== Convolutional Variational Autoencoder ====================================================================
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

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar, beta = 0.1):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

    if not torch.isfinite(total_loss):
        logger.warning(f"Non-finite loss | MSE: {MSE.item()}, KL: {KLD.item()}")
        # Return a recoverable value when loss explodes
        return torch.tensor(10.0, device = x.device, requires_grad = True)
        
    return total_loss

# Custom Dataset to handle the numerical array
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], 0



# ==================================================================== CNN Discriminator ====================================================================
class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=(15,3), stride=(10,2), padding=1),  # (16, 100, 25)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, kernel_size=(15,3), stride=(10,2), padding=1),  # (32, 9, 13)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*9*13, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary classification (real vs fake)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)

# Label functions
def real_label(size, device, label_smoothing=0.0):
    return torch.ones(size, 1, device=device) * (1.0 - label_smoothing)
def fake_label(size, device, label_smoothing=0.0):
    return torch.zeros(size, 1, device=device) + label_smoothing



# ============================================== Function to train VAE in pretraining phase ====================================================================
def pretrain_vae(vae, dataloader, latent_dim, lr_vae, num_epochs, device, beta=1.0):
    """Pretrain the VAE model"""
    vae.to(device)
    opt_G = optim.Adam(vae.parameters(), lr=lr_vae)
    
    logger.info("Starting VAE pretraining...")
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            decoded_data, mu, logvar = vae(batch_data)
            
            loss = vae_loss(decoded_data, batch_data, mu, logvar, beta)
            opt_G.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_G.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    logger.info("VAE pretraining complete!")
    return vae, train_losses

# ============================================== Function to train VAE-GAN ====================================================================
def train_vae_gan(vae, disc, dataloader, latent_dim, lr_disc, lr_vae, num_epochs, device, 
                  lambda_kl=1e-4, lambda_adv=1e-4, noise_level=0.1, label_smoothing=0.0):
    """Train the VAE-GAN model"""
    vae.to(device)
    disc.to(device)
    
    opt_D = optim.Adam(disc.parameters(), lr=lr_disc)
    opt_G = optim.Adam(vae.parameters(), lr=lr_vae)
    
    # Loss functions
    bce = nn.BCELoss()
    mse_loss = nn.MSELoss(reduction='mean')
    
    logger.info("Starting VAE-GAN training...")
    d_losses = []
    g_losses = []
    disc_conf_history = []
    
    for epoch in range(1, num_epochs + 1):
        vae.train()
        disc.train()
        total_G_loss = 0.0
        total_D_loss = 0.0
        epoch_real_conf = []
        epoch_fake_conf = []
        
        total_real_conf = 0.0
        total_fake_conf = 0.0
        
        for x_real, _ in dataloader:
            x_real = x_real.to(device)
            bs = x_real.size(0)
            
            # Discriminator update
            with torch.no_grad():
                z = torch.randn(bs, latent_dim, device=device)
                x_sample = vae.decode(z)  # Pure decoder samples
                bs, C, T, S = x_sample.shape
                # Time Average D & V
                time_mean = x_sample[:, 0:2, :, :].mean(dim=2, keepdim=True) 
                time_mean_expanded = time_mean.expand(bs, 2, T, S)
                x_sample[:, 0:2, :, :] = time_mean_expanded
                
            # Add Noise
            noise_real = noise_level * x_real.std(dim=[0,2,3], keepdim=True) * torch.randn_like(x_real)
            noise_fake = noise_level * x_sample.std(dim=[0,2,3], keepdim=True) * torch.randn_like(x_sample)
            x_real_noisy = x_real + noise_real
            x_sample_noisy = x_sample + noise_fake
            
            # Discriminator forward
            d_out_real = disc(x_real_noisy)
            d_out_fake = disc(x_sample_noisy)
            
            epoch_real_conf.extend(d_out_real.detach().cpu().numpy().flatten())
            epoch_fake_conf.extend(d_out_fake.detach().cpu().numpy().flatten())
            
            loss_D_real = bce(d_out_real, real_label(bs, device, label_smoothing))
            loss_D_fake = bce(d_out_fake, fake_label(bs, device, label_smoothing))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            
            opt_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            opt_D.step()
            
            # Generator (VAE) update
            x_recon, mu, logvar = vae(x_real)
            d_out_fake = disc(x_recon)
            
            # Reconstruction + KL
            recon_loss = mse_loss(x_recon, x_real)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Adversarial (fool discriminator)
            adv_loss = bce(d_out_fake, real_label(bs, device, 0.0))  # No label smoothing for generator training
            
            # Weighted sum
            loss_G = recon_loss + lambda_kl * kl_loss + lambda_adv * adv_loss
            
            opt_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_G.step()
            
            total_D_loss += loss_D.item()
            total_G_loss += loss_G.item()
            total_real_conf += d_out_real.mean().item()
            total_fake_conf += d_out_fake.mean().item()
            # num_batches += 1
        
        avg_D = total_D_loss / len(dataloader)
        avg_G = total_G_loss / len(dataloader)
        avg_real_conf = np.mean(epoch_real_conf)
        avg_fake_conf = np.mean(epoch_fake_conf)
        
        d_losses.append(avg_D)
        g_losses.append(avg_G)
        disc_conf_history.append((avg_real_conf, avg_fake_conf))
        
        logger.info(f"Epoch [{epoch}/{num_epochs}]  Loss_D: {avg_D:.4f}  Loss_G: {avg_G:.4f}"
                    f"     Disc_Real_Conf: {avg_real_conf:.4f}  Disc_Fake_Conf: {avg_fake_conf:.4f}")
    
    logger.info("VAE-GAN training complete!")
        
    return vae, disc, d_losses, g_losses, disc_conf_history


# ============================================== Function to validate the model on validation data ====================================================================
def validate_model(vae, dataloader, device):
    """Validate the model and return reconstruction error"""
    vae.eval()
    total_recon_error = 0.0
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            recon_x, _, _ = vae(x)
            # Calculate MSE reconstruction error
            recon_error = F.mse_loss(recon_x, x, reduction='sum').item()
            total_recon_error += recon_error
    
    # Return average reconstruction error
    return total_recon_error / len(dataloader.dataset)


# ============================================== Bayesian Optimization objective function for VAE-GAN ====================================================================
def vae_gan_objective(latent_dim, learning_rate_vae, learning_rate_disc, 
                      lambda_kl, lambda_adv, noise_level, beta_vae, label_smoothing):
    """
    Objective function for Bayesian Optimization of VAE-GAN hyperparameters.
    Returns the negative reconstruction error (to maximize).
    """
    # Ensure latent_dim is int
    latent_dim = int(latent_dim)
    
    # New model for each evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VariationalAutoencoder(latent_dim=latent_dim).to(device) # Generator
    disc = CNNDiscriminator().to(device) # Discriminator

    logger.info(f"Hyperparameters: latent_dim={latent_dim}, lr_vae={learning_rate_vae:.6f}, "
                f"lr_disc={learning_rate_disc:.6f}, lambda_kl={lambda_kl:.6f}, "
                f"lambda_adv={lambda_adv:.6f}, noise={noise_level:.3f}, "
                f"beta={beta_vae:.3f}, label_smooth={label_smoothing:.3f}")
    
    # Pretrain the VAE
    vae, _ = pretrain_vae(
        vae=vae,
        dataloader=train_loader,
        latent_dim=latent_dim,
        lr_vae=learning_rate_vae,
        num_epochs=20,  # Reduced epochs for optimization
        device=device,
        beta=beta_vae
    )
    
    # Then train VAE-GAN
    vae, disc, _, _, _ = train_vae_gan(
        vae=vae,
        disc=disc,
        dataloader=train_loader,
        latent_dim=latent_dim,
        lr_disc=learning_rate_disc,
        lr_vae=learning_rate_vae,
        num_epochs=20,  # Reduced epochs for optimization
        device=device,
        lambda_kl=lambda_kl,
        lambda_adv=lambda_adv,
        noise_level=noise_level,
        label_smoothing=label_smoothing
    )
    
    # Validate the model
    recon_error = validate_model(vae, val_loader, device)

    logger.info(f"Validation reconstruction error: {recon_error:.4f}")
    
    # Return negative error since we want to maximize the objective
    return -recon_error



# ============================================== global ====================================================================
def main():
    global train_loader, val_loader  # global for optimization function
    
    # Set random seeds for reproducibility
    #torch.manual_seed(442)
    #np.random.seed(442)
    
    # Load the data
    Database_channel_standardized = load_data()
    
    # Convert data to torch dataset
    dataset = CustomDataset(Database_channel_standardized)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Dataset size: {len(dataset)}, Train size: {train_size}, Val size: {val_size}")
    
    # Search space for Bayesian Optimization
    pbounds = {
        'latent_dim': (150, 640),                  # Latent space dimension
        'learning_rate_vae': (0.00025, 0.001),     # Learning rate for VAE
        'learning_rate_disc': (0.000005, 0.00015), # Learning rate for discriminator
        'lambda_kl': (0.001, 2.0),                 # Weight for KL divergence loss
        'lambda_adv': (0.001, 1.0),                # Weight for adversarial loss
        'noise_level': (0.1, 0.4),                 # Noise level for discriminator input
        'beta_vae': (0.01, .1),                    # Weight for KL term in VAE pretraining
        'label_smoothing': (0.1, 0.45)             # Label smoothing for discriminator
    }
    
    # pbounds = {
    #     'latent_dim': (150, 150),                  # Latent space dimension
    #     'learning_rate_vae': (0.00025, 0.00025),     # Learning rate for VAE
    #     'learning_rate_disc': (0.000005, 0.000005), # Learning rate for discriminator
    #     'lambda_kl': (0.001, 0.001),                 # Weight for KL divergence loss
    #     'lambda_adv': (0.001, 0.001),                # Weight for adversarial loss
    #     'noise_level': (0.1, 0.1),                 # Noise level for discriminator input
    #     'beta_vae': (0.01, 0.01),                    # Weight for KL term in VAE pretraining
    #     'label_smoothing': (0.1, 0.1)             # Label smoothing for discriminator
    # }

    # Initialize Bayesian Optimization
    logger.info("Starting Bayesian Optimization")
    optimizer = BayesianOptimization(
        f = vae_gan_objective,
        pbounds = pbounds,
        #random_state = 442,
        verbose = 2
    )
    
    # Run optimization
    optimizer.maximize(
        init_points = 5,    # Number of random initial points
        n_iter = 15         # Number of optimization iterations
    )
    
    logger.info("Bayesian Optimization completed")
    logger.info(f"Best parameters: {optimizer.max['params']}")
    logger.info(f"Best reconstruction error: {-optimizer.max['target']:.4f}")
    
    # Train final model with best parameters
    best_params = optimizer.max['params']
    
    # Ensure latent dimension is an int 
    best_params['latent_dim'] = int(best_params['latent_dim'])
    
    logger.info("Training final model with best parameters...")
    
    # Create models with best parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_vae = VariationalAutoencoder(latent_dim=best_params['latent_dim']).to(device)
    final_disc = CNNDiscriminator().to(device)
    
    # Pretrain VAE with best parameters
    final_vae, vae_losses = pretrain_vae(
        vae=final_vae,
        dataloader=train_loader,
        latent_dim=best_params['latent_dim'],
        lr_vae=best_params['learning_rate_vae'],
        num_epochs=20,  # Full training run
        device=device,
        beta=best_params['beta_vae']
    )
    
    # Save pretrained VAE
    torch.save(final_vae.state_dict(), "vae_pretrained_best.pt")
    
    # Train VAE-GAN with best parameters
    final_vae, final_disc, d_losses, g_losses, disc_conf_history = train_vae_gan(
        vae=final_vae,
        disc=final_disc,
        dataloader=train_loader,
        latent_dim=best_params['latent_dim'],
        lr_disc=best_params['learning_rate_disc'],
        lr_vae=best_params['learning_rate_vae'],
        num_epochs=50,  # Full training run
        device=device,
        lambda_kl=best_params['lambda_kl'],
        lambda_adv=best_params['lambda_adv'],
        noise_level=best_params['noise_level'],
        label_smoothing=best_params['label_smoothing']
    )
    
    # Save trained models
    torch.save(final_vae.state_dict(), "vae_gan_best.pt")
    torch.save(final_disc.state_dict(), "discriminator_best.pt")
    

    # Final validation
    final_recon_error = validate_model(final_vae, val_loader, device)
    logger.info(f"Final validation reconstruction error: {final_recon_error:.4f}")
    

    # # Plot losses
    # plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(vae_losses)
    # plt.title('VAE Pretraining Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(d_losses, label='Discriminator Loss')
    # plt.plot(g_losses, label='Generator Loss')
    # plt.title('VAE-GAN Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    
    # # Discriminator confidence
    # real_confs = [x[0] for x in disc_conf_history]
    # fake_confs = [x[1] for x in disc_conf_history]
    # plt.subplot(1, 2, 2)
    # plt.plot(real_confs, label='Real Confidence')
    # plt.plot(fake_confs, label='Fake Confidence')
    # plt.title('Discriminator Confidence Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Confidence')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig("vae_gan_training_curves.png")


    # Plot losses in the first figure
    plt.figure(figsize=(10, 4))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('VAE-GAN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Image/VAEGAN_Losses.png", dpi=300, bbox_inches='tight')
    
    # Plot discriminator confidence in a separate figure
    real_confs = [x[0] for x in disc_conf_history]
    fake_confs = [x[1] for x in disc_conf_history]
    plt.figure(figsize=(10, 4))
    plt.plot(real_confs, label='Real Confidence')
    plt.plot(fake_confs, label='Fake Confidence')
    plt.title('Discriminator Confidence Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Confidence')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Image/Discriminator_Confidence.png", dpi=300, bbox_inches='tight')


    # Save hyperparameters
    with open('best_hyperparameters.txt', 'w') as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    return final_vae, final_disc, best_params, real_confs, fake_confs




# ============================================== MMD Function ====================================================================
def MMD(x, y, kernel="multiscale"):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches of samples.
    Args:
        x, y: tensors of shape [batch_size, features]
        kernel: type of kernel to use ("multiscale" or "rbf")
    Returns:
        Scalar MMD value
    """
    # Normalize features
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

    if kernel == "multiscale":
        bandwidths = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidths:
            a2 = a ** 2
            XX += a2 * (a2 + dxx).reciprocal()
            YY += a2 * (a2 + dyy).reciprocal()
            XY += a2 * (a2 + dxy).reciprocal()
    elif kernel == "rbf":
        bandwidths = [10, 15, 20, 50]
        for a in bandwidths:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2 * XY)







# ============================================== Plot VAE-GAN Comparison ====================================================================
# ==================================================== CMAP ======================================================================================
def plot_vaegan_comparison(vae, device, Database_channel_standardized, best_params, num_samples=3, save=False):
    # Set the VAE to evaluation mode
    vae.eval()
    
    # Ensure latent dim is an int
    latent_dim = int(best_params['latent_dim'])
    
    # Channel names
    channel_names = ['Diffusion', 'Convection', 'Density', 'Source']
    
    # Generate samples
    generated_samples = []
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim, device=device)
            generated = vae.decode(z)
            generated_samples.append(generated.squeeze(0).cpu().numpy())
    
    # Select random real samples for comparison
    real_indices = np.random.choice(len(Database_channel_standardized), num_samples, replace=False)
    real_samples = [Database_channel_standardized[idx] for idx in real_indices]
    
    # Figure shape and main title
    fig_all = plt.figure(figsize=(20, 18)) 
    fig_all.suptitle('Generated vs Real vs Difference (Channel-wise)', fontsize=18, y=0.98)
    
    # Plotting grid
    total_rows = 3 * num_samples
    total_cols = 4  # 4 channels
    
    # consistent color scaling across non-difference samples 
    channel_vmins = []
    channel_vmaxs = []
    
    for c in range(4):
        all_data = []
        for sample_idx in range(num_samples):
            all_data.append(generated_samples[sample_idx][c])
            all_data.append(real_samples[sample_idx][c])
        channel_vmins.append(np.min(all_data))
        channel_vmaxs.append(np.max(all_data))
    
    # cmaps
    data_cmap = 'viridis'
    diff_cmap = 'bwr'      
    
    # Find the max absolute difference for symmetric colorbar in difference plots
    max_diff = 0
    for sample_idx in range(num_samples):
        diff_sample = generated_samples[sample_idx] - real_samples[sample_idx]
        max_diff = max(max_diff, np.max(np.abs(diff_sample)))
    
    # Set up grid spec to have better control over spacing
    gs = fig_all.add_gridspec(total_rows + num_samples, total_cols, 
                             height_ratios=[0.5 if i % 4 == 3 else 1 for i in range(total_rows + num_samples)])
    

    for sample_idx in range(num_samples):
        # Calculate MSE for this sample
        mse = ((generated_samples[sample_idx] - real_samples[sample_idx])**2).mean()
        
        # Add sample title in its own row (with extra spacing)
        sample_title_y = sample_idx * 4  # Position for sample title (adjusted for spacing rows)
        sample_title_ax = fig_all.add_subplot(gs[sample_title_y, :])
        sample_title_ax.text(0.5, 0.5, f'Sample {sample_idx + 1}: MSE = {mse:.6f}', 
                           fontsize=14, ha='center', va='center')
        sample_title_ax.axis('off')
        
        # Get difference between current generated and real samples
        gen_sample = generated_samples[sample_idx]
        real_sample = real_samples[sample_idx]
        diff_sample = gen_sample - real_sample
        
        # Plot the channels for this sample
        for col in range(4):  # 4 channels
            # Row positions adjusted for title rows
            gen_row = sample_title_y + 1
            real_row = sample_title_y + 2
            diff_row = sample_title_y + 3
            
            # Plot generated data
            ax_gen = fig_all.add_subplot(gs[gen_row, col])
            im_gen = ax_gen.imshow(gen_sample[col], aspect='auto', cmap=data_cmap, 
                                  vmin=channel_vmins[col], vmax=channel_vmaxs[col])
            ax_gen.set_title(f'Generated {channel_names[col]}')
            ax_gen.axis('off')
            divider = make_axes_locatable(ax_gen)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_gen, cax=cax)
            
            # Plot real data
            ax_real = fig_all.add_subplot(gs[real_row, col])
            im_real = ax_real.imshow(real_sample[col], aspect='auto', cmap=data_cmap,
                                    vmin=channel_vmins[col], vmax=channel_vmaxs[col])
            ax_real.set_title(f'Real {channel_names[col]}')
            ax_real.axis('off')
            divider = make_axes_locatable(ax_real)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_real, cax=cax)
            
            # Plot difference
            ax_diff = fig_all.add_subplot(gs[diff_row, col])
            im_diff = ax_diff.imshow(diff_sample[col], aspect='auto', cmap=diff_cmap,
                                    vmin=-max_diff, vmax=max_diff)
            ax_diff.set_title(f'Diff {channel_names[col]}')
            ax_diff.axis('off')
            divider = make_axes_locatable(ax_diff)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_diff, cax=cax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Space for main title
    
    # Save
    if save:
        id_num = np.random.randint(10000)
        fig_all.savefig(f'Image/VAEGAN_AllSamples_v{id_num}.png', dpi=300, bbox_inches='tight')
    
        
    # Distribution comparison histogram
    fig_dist = plt.figure(figsize=(14, 10))
    fig_dist.suptitle('Distribution Comparison: Generated vs Real', fontsize=16)
    

    num_real_samples = len(Database_channel_standardized)
    print(f"Generating {num_real_samples} samples for distribution comparison...")
    




    # ==================================================== Distribution Analysis ======================================================================================
    # Generate additional samples for distribution analysis
    dist_generated_samples = []
    with torch.no_grad():
        for i in range(num_real_samples):
            z = torch.randn(1, latent_dim, device=device)
            generated = vae.decode(z)
            dist_generated_samples.append(generated.squeeze(0).cpu().numpy())

    for c in range(4):
        ax = fig_dist.add_subplot(2, 2, c+1)
        
        # Flatten ALL real data for consistent histograms
        real_data = Database_channel_standardized[:, c].flatten()
        
        # Flatten generated data from the samples
        gen_data = np.concatenate([sample[c].flatten() for sample in dist_generated_samples])
        
        # Plot histograms
        ax.hist(real_data, bins=50, alpha=0.5, label='Real', density=True)
        ax.hist(gen_data, bins=50, alpha=0.5, label='Generated', density=True)
        
        ax.set_title(f'{channel_names[c]} Distribution')
        ax.legend()
        
        # mean and std
        real_mean = np.mean(real_data)
        real_std = np.std(real_data)
        gen_mean = np.mean(gen_data)
        gen_std = np.std(gen_data)

        # Compute MMD (using small reshaped tensors to avoid memory issues)
        try:
            real_tensor = torch.tensor(real_data[:5000].reshape(-1, 100), dtype=torch.float32)
            gen_tensor = torch.tensor(gen_data[:5000].reshape(-1, 100), dtype=torch.float32)
            mmd_val = MMD(real_tensor, gen_tensor).item()
        except Exception as e:
            mmd_val = float('nan')
            print(f"Warning: MMD failed on channel {channel_names[c]}: {e}")

        
        # Compose and display stats
        stats_text = (f'Real: μ={real_mean:.2f}, σ={real_std:.2f}\n'
                      f'Gen: μ={gen_mean:.2f}, σ={gen_std:.2f}\n'
                      f'MMD: {mmd_val:.4f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # stats_text = f'Real: μ={real_mean:.2f}, σ={real_std:.2f}\nGen: μ={gen_mean:.2f}, σ={gen_std:.2f}'
        # ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Suptitle
    
    # Parameters summary figure
    fig_params = plt.figure(figsize=(12, 6))
    fig_params.suptitle('Best Hyperparameters from Bayesian Optimization', fontsize=16)
    
    # Text summary of parameters
    param_text = '\n'.join([f"{param}: {value}" for param, value in best_params.items()])
    fig_params.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
    
    # Save 
    if save:
        id_num = np.random.randint(10000)
        fig_dist.savefig(f'Image/VAEGAN_Distributions_v{id_num}.png', dpi=300, bbox_inches='tight')
        fig_params.savefig(f'Image/VAEGAN_BestParams_v{id_num}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":    
    final_vae, final_disc, best_params, real_confs, fake_confs = main()

    plot_vaegan_comparison(
        vae = final_vae,
        device = device,
        Database_channel_standardized = load_data(),
        best_params = best_params,
        num_samples = 3,
        save = True
    )


