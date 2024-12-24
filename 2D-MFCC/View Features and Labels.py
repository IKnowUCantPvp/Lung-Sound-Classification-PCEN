import numpy as np

# Load the data from the .npy files
features = np.load('features.npy')
labels = np.load('labels.npy')

# Display basic information about the data
print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

print("\nFirst few labels:", labels[:5])  # View the first few labels

# To inspect a specific feature set, you may print or plot it
print("\nFirst feature set (MFCC coefficients):")
print(features[0])  # This prints the first feature set (e.g., for a specific audio file)
