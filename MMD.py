import numpy as np
import torch
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MMD Function ===
def MMD(x, y, kernel="multiscale"):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros_like(xx),
                  torch.zeros_like(yy),
                  torch.zeros_like(zz))

    if kernel == "multiscale":
        for a in [0.2, 0.5, 0.9, 1.3]:
            XX += a**2 * (a**2 + dxx).reciprocal()
            YY += a**2 * (a**2 + dyy).reciprocal()
            XY += a**2 * (a**2 + dxy).reciprocal()

    elif kernel == "rbf":
        for a in [10, 15, 20, 50]:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

# === Load Data ===
data_dir = ""  # Set your directory here if needed

real_path = os.path.join(data_dir, "real_sample.npy")
gen_path = os.path.join(data_dir, "generated_sample.npy")

real_sample = np.load(real_path)
gen_sample = np.load(gen_path)

# Check shape (Assuming data is in shape (channels, height, width))
assert real_sample.shape == gen_sample.shape, "Shape of real and generated samples must match"

# === Compute MMD per channel ===
num_channels = real_sample.shape[0]

mmd_results = []

for i in range(num_channels):
    real_channel = real_sample[i].reshape(1, -1)  # Flatten each channel
    gen_channel = gen_sample[i].reshape(1, -1)

    x = torch.tensor(real_channel, dtype=torch.float32).to(device)
    y = torch.tensor(gen_channel, dtype=torch.float32).to(device)

    mmd_result = MMD(x, y, kernel="multiscale")
    mmd_results.append(mmd_result.item())

    print(f"MMD result for channel {i+1}: {mmd_result.item():.6f}")

# If needed, print all results
print("\nMMD results per channel:")
for i, res in enumerate(mmd_results, 1):
    print(f"Channel {i}: {res:.6f}")
