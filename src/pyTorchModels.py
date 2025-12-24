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
            normalized=True
        )

        # PCEN layer with improved parameters
        self.pcen = PCEN(
            input_size=128,  # n_mels (frequency bins)
            alpha=0.98,
            smooth_coef=0.025,
            delta=1.0,    # Reduced from 2.0
            root=0.5,     # Reduced from 2.0
            floor=1e-6,   # Increased from 1e-12
            trainable=True,
            per_channel_smooth_coef=False,
            skip_transpose=False
        )

        # Layers
        self.batch_norm = nn.BatchNorm2d(1)

        # Rest of the architecture remains exactly the same
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
        # Original forward pass
        x = self.mel_spectrogram(x)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)
        x = self.pcen(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.output(x)
        return x


from .TVaryingPCEN import TVaryingPCEN

class Conv2DPCEN_TVarying(nn.Module):
    def __init__(self, n_classes=10, sr=8000, dt=6.0, n_t_constants=8, trainable=True):
        super(Conv2DPCEN_TVarying, self).__init__()

        # Mel spectrogram layer
        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=128,
            power=2.0,
            normalized=True
        )

        self.n_t_constants = n_t_constants
        
        # Use separate TVaryingPCEN layer
        self.tvarying_pcen = TVaryingPCEN(n_t_constants=n_t_constants, trainable=trainable)

        # Batch Norm (input channels = n_t_constants)
        self.batch_norm = nn.BatchNorm2d(n_t_constants)

        # Convolutional layers
        # First conv layer now takes n_t_constants input channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_t_constants, 8, kernel_size=(7, 7), padding='same'),
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
            dummy_input = torch.randn(1, 1, int(sr * dt))
            # Mel Spectrogram
            dummy_mel = self.mel_spectrogram(dummy_input)
            if dummy_mel.dim() == 4:
                dummy_mel = dummy_mel.squeeze(1)
            # No transpose needed for new manual layer
            
            # Use layer
            dummy_stack = self.tvarying_pcen(dummy_mel)

            dummy_out = self._forward_features(dummy_stack)
            flatten_size = dummy_out.view(1, -1).size(1)


        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(flatten_size, 64)
        self.output = nn.Linear(64, n_classes)

    def _forward_features(self, x):
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
        # x: [batch, 1, time] or [batch, time]
        x = self.mel_spectrogram(x) # [batch, n_mels, time]
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # x is now [batch, n_mels, time].
        # Our custom PyTorchPCENLayer expects [batch, n_mels, time] or [batch, channel, n_mels, time].
        # No transpose needed here anymore.

        x = self.tvarying_pcen(x)
        # Output is [batch, n_t_constants, n_mels, time]

        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.output(x)
        return x