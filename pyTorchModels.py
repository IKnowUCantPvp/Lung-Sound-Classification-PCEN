import torch
import torch.nn as nn
import torchaudio.transforms as transforms
from speechbrain.nnet.normalization import PCEN


class Conv2DPCEN(nn.Module):
    def __init__(self, n_classes=10, sr=8000, dt=6.0):
        super(Conv2DPCEN, self).__init__()

        # Mel spectrogram layer
        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=128,
            power=2.0,
            normalized=False
        )

        # PCEN layer
        self.pcen = PCEN(
            input_size=128,  # n_mels (frequency bins)
            alpha=0.96,
            smooth_coef=0.04,
            delta=2.0,
            root=2.0,
            floor=1e-12,
            trainable=True,
            per_channel_smooth_coef=True,  # whether to learn independent smooth coefficients
            skip_transpose=False  # using batch x time x channel convention
        )

        # Layers
        self.batch_norm = nn.BatchNorm2d(1)

        # Conv blocks with proper activation functions
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), padding='same'),
            nn.Tanh()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding='same'),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

        # Calculate the flattened size
        with torch.no_grad():
            # Create dummy input to calculate flatten size
            dummy_input = torch.randn(1, 1, int(sr * dt))
            dummy_mel = self.mel_spectrogram(dummy_input)  # Shape: [batch, n_mels, time]
            # Remove channel dim if present
            if dummy_mel.dim() == 4:
                dummy_mel = dummy_mel.squeeze(1)
            # Transpose for PCEN
            dummy_mel = dummy_mel.transpose(1, 2)
            dummy_pcen = self.pcen(dummy_mel)  # Apply PCEN
            # Transpose back and add channel dim
            dummy_pcen = dummy_pcen.transpose(1, 2)
            dummy_pcen = dummy_pcen.unsqueeze(1)  # Add channel dim back
            dummy_out = self._forward_features(dummy_pcen)
            flatten_size = dummy_out.view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(flatten_size, 64)
        self.output = nn.Linear(64, n_classes)

    def _forward_features(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        # Compute mel spectrogram [batch, n_mels, time]
        x = self.mel_spectrogram(x)

        # Remove channel dimension if present
        if x.dim() == 4:
            x = x.squeeze(1)

        # Transpose for PCEN [batch, time, n_mels]
        x = x.transpose(1, 2)

        # Apply PCEN [batch, time, n_mels]
        x = self.pcen(x)

        # Transpose back [batch, n_mels, time]
        x = x.transpose(1, 2)

        # Add channel dimension for CNN [batch, channel, n_mels, time]
        x = x.unsqueeze(1)

        # Pass through CNN layers
        x = self._forward_features(x)

        # Final layers
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.output(x)

        return x