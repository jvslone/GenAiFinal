import numpy as np
from scipy import stats

# Load the real and generated samples
real_sample = np.load("real_sample.npy")
gen_sample = np.load("generated_sample.npy")

# Print the shape of the samples to understand their structure
print(f"Shape of real sample: {real_sample.shape}")
print(f"Shape of generated sample: {gen_sample.shape}")

# If the data is 3D, the shape will be (batch_size, channels, features)
# We flatten the data along the last dimension (features) for comparison
real_flat = real_sample.reshape(real_sample.shape[0], -1)
gen_flat = gen_sample.reshape(gen_sample.shape[0], -1)

# Perform an independent two-sample t-test
t_stat, p_value = stats.ttest_ind(real_flat, gen_flat)

# Output the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the result
if np.any(p_value < 0.05):
    print("There is a significant difference between the real and generated samples.")
else:
    print("There is no significant difference between the real and generated samples.")

# If the data is 3D with channels, we loop over channels for t-tests per channel:
num_channels = real_sample.shape[0]

for channel in range(num_channels):
    real_channel = real_sample[:, channel, :].flatten()  # Flattening the features for each channel
    gen_channel = gen_sample[:, channel, :].flatten()

    t_stat, p_value = stats.ttest_ind(real_channel, gen_channel)
    
    print(f"Channel {channel + 1} - T-statistic: {t_stat}, P-value: {p_value}")
    
    if np.any(p_value < 0.05):
        print(f"Channel {channel + 1} has a significant difference.")
    else:
        print(f"Channel {channel + 1} has no significant difference.")

