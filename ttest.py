import numpy as np
from scipy import stats

# Load the real and generated samples
real_sample = np.load("real_sample.npy")
gen_sample = np.load("generated_sample.npy")

# Reshape them if necessary to compare the samples (e.g., along each channel)
# Assuming the data is 4D: (batch_size, channels, height, width)
# Flatten the samples into 1D arrays for comparison
real_flat = real_sample.flatten()
gen_flat = gen_sample.flatten()

# Perform an independent two-sample t-test
t_stat, p_value = stats.ttest_ind(real_flat, gen_flat)

# Output the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant difference between the real and generated samples.")
else:
    print("There is no significant difference between the real and generated samples.")

# Assuming `real_sample` and `gen_sample` have the shape (batch_size, channels, height, width)
num_channels = real_sample.shape[1]

# Loop through each channel and perform t-test
for channel in range(num_channels):
    real_channel = real_sample[:, channel, :, :].flatten()
    gen_channel = gen_sample[:, channel, :, :].flatten()
    
    t_stat, p_value = stats.ttest_ind(real_channel, gen_channel)
    
    print(f"Channel {channel + 1} - T-statistic: {t_stat}, P-value: {p_value}")
    
    if p_value < 0.05:
        print(f"Channel {channel + 1} has a significant difference.")
    else:
        print(f"Channel {channel + 1} has no significant difference.")