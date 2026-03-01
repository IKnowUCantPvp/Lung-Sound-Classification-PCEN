import torch
import torch.nn as nn
import torchaudio.transforms as transforms

class CustomPCEN(nn.Module):
    def __init__(self, n_mels=128, alpha=0.95, delta=1.0, root=0.5, init_smooth_coef=0.025, floor=1e-6):
        super(CustomPCEN, self).__init__()
        
        # Fixed parameters (not trainable)
        self.alpha = alpha
        self.delta = delta
        self.root = root
        self.floor = floor
        self.n_mels = n_mels
        
        # Trainable parameter: smooth_coef (initialized per channel or globally)
        # Using a scalar parameter that broadcasts, or a true per-channel parameter. 
        # Speechbrain usually has scalar smooth_coef unless specified. Let's make it per-channel for maximum adaptability.
        self.smooth_coef = nn.Parameter(torch.full((1, n_mels), init_smooth_coef))
        
    def forward(self, x):
        # x shape: (batch, time, n_mels) to match SpeechBrain
        
        # Calculate EMA recursively over time
        # smooth_coef must be bounded between 0 and 1
        s = torch.clamp(self.smooth_coef, 0.0, 1.0)
        
        # 1. Initialize EMA smoother state list
        smoothed_list = []
        current_smoothed = x[:, 0, :]
        smoothed_list.append(current_smoothed)
        
        for t in range(1, x.size(1)):
            current_smoothed = (1 - s) * current_smoothed + s * x[:, t, :]
            smoothed_list.append(current_smoothed)
            
        # Stack over time dimension to shape (batch, time, n_mels)
        smoothed = torch.stack(smoothed_list, dim=1)
            
        # 2. PCEN calculation: (E / (eps + M)^alpha + delta)^root - delta^root
        # Add floor to prevent division by zero or log(0) issues
        m = smoothed + self.floor
        
        # Apply PCEN
        pcen_output = (x / (m ** self.alpha) + self.delta) ** self.root - (self.delta ** self.root)
        
        return pcen_output


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

        # SpecAugment (Applied only during training on the mel spectrogram)
        self.freq_masking = transforms.FrequencyMasking(freq_mask_param=15)
        self.time_masking = transforms.TimeMasking(time_mask_param=35)

        # Custom PCEN layer (Fixed Alpha, Trainable Smooth)
        self.pcen = CustomPCEN(
            n_mels=128,
            alpha=0.95,
            delta=1.0,
            root=0.5,
            init_smooth_coef=0.025,
            floor=1e-6
        )

        # Scaled up Convolutional Architecture with Batch Normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 7), padding='same'),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, int(sr * dt))
            dummy_mel = self.mel_spectrogram(dummy_input)
            if dummy_mel.dim() == 4:
                dummy_mel = dummy_mel.squeeze(1)
            dummy_mel = dummy_mel.transpose(1, 2)
            dummy_pcen = self.pcen(dummy_mel)
            dummy_pcen = dummy_pcen.transpose(1, 2)
            dummy_pcen = dummy_pcen.unsqueeze(1)
            dummy_out = self._forward_features(dummy_pcen)
            flatten_size = dummy_out.view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4) # Increased dropout for larger model
        self.dense = nn.Linear(flatten_size, 128) # Scaled up dense layer
        self.dense_bn = nn.BatchNorm1d(128)
        self.dense_act = nn.ReLU()
        self.output = nn.Linear(128, n_classes)

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
        # Spec extraction
        x = self.mel_spectrogram(x)
        if x.dim() == 4:
            x = x.squeeze(1)
            
        # SpecAugment (only during training)
        if self.training:
            x = self.freq_masking(x)
            x = self.time_masking(x)
            
        x = x.transpose(1, 2)
        x = self.pcen(x)
        x = x.transpose(1, 2)
        
        # Add channel dimension back for Conv2d
        x = x.unsqueeze(1)
        
        # Features
        x = self._forward_features(x)
        
        # Classification head
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dense_bn(x)
        x = self.dense_act(x)
        x = self.output(x)
        return x


from TVaryingPCEN import TVaryingPCEN

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