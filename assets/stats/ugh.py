import numpy as np

# Replace with your actual .npz file path
file_path = "cifar10_stats.npz"

# Load the .npz file
data = np.load(file_path)

# Print out all keys and the shape of each array
for key in data.files:
    print(f"{key}: shape = {data[key].shape}")

print(data["pool_3"].shape)
