import speechbrain as sb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import speechbrain.nnet.normalization as sb_norm

# Create the PCEN layer from SpeechBrain
pcen = sb_norm.PCEN(
    input_size=128,  # This should match your number of mel bands
    alpha=0.98,      # Smoothing coefficient for AGC
    smooth_coef=0.04,  # This is the 's' parameter for temporal integration
    delta=2.0,      # Bias term for DRC
    root=0.5,       # Root compression coefficient (what we called 'r' before)
    floor=1e-12,    # Small constant for numerical stability (epsilon)
    trainable=True  # Whether parameters should be learned during training
)

# The layer expects input of shape (batch, time, n_mels)
# If your mel spectrogram is in log scale, you'll need to first convert it back to linear
mel_spec = torch.exp(mel_log_spec)  # Convert from log to linear scale
pcen_output = pcen(mel_spec)i dont