from torch.utils.data import Dataset
import numpy as np
import torchaudio
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pyTorchModels import Conv2DPCEN
from glob import glob
import os
from sklearn.preprocessing import LabelEncoder
import warnings


class DataGenerator(Dataset):
    def __init__(self, wav_paths, labels, sr, dt, n_classes):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(self.wav_paths[idx])

            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != self.sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
                waveform = resampler(waveform)

            # Remove channel dimension
            waveform = waveform.squeeze(0)

            # Create one-hot encoded label
            label = torch.zeros(self.n_classes)
            label[self.labels[idx]] = 1.0

            return waveform, label

        except Exception as e:
            print(f"Error loading file {self.wav_paths[idx]}: {str(e)}")
            raise e


def get_dataloaders(src_root, sr=8000, dt=6.0, batch_size=32):
    # Get all wav files recursively
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

    # Extract labels from directory structure
    classes = sorted(os.listdir(src_root))
    n_classes = len(classes)

    # Encode labels
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    # Split data
    from sklearn.model_selection import train_test_split
    wav_train, wav_val, label_train, label_val = train_test_split(
        wav_paths, labels, test_size=0.2, random_state=42
    )

    # Verify we have enough samples
    assert len(label_train) >= batch_size, 'Number of train samples must be >= batch_size'

    # Check if all classes are represented
    if len(set(label_train)) != n_classes:
        warnings.warn(
            f'Found {len(set(label_train))}/{n_classes} classes in training data.')
    if len(set(label_val)) != n_classes:
        warnings.warn(
            f'Found {len(set(label_val))}/{n_classes} classes in validation data.')

    # Create datasets
    train_dataset = DataGenerator(wav_train, label_train, sr, dt, n_classes)
    val_dataset = DataGenerator(wav_val, label_val, sr, dt, n_classes)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, n_classes


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            )

    # Return average loss for the epoch
    return total_loss / len(train_loader)


def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    # Return average validation loss
    return total_loss / len(val_loader)


def main():
    # Training settings
    src_root = '/Users/samer/PycharmProjects/Lung-sounds-isef/CurrentDatasets/CleanDatasets (10 classes)/cleanTrainDataset (nonoise and COPD cut)'
    batch_size = 16
    delta_time = 6.0
    sample_rate = 8000
    num_epochs = 30
    learning_rate = 0.001

    # Get dataloaders
    train_loader, val_loader, n_classes = get_dataloaders(
        src_root,
        sample_rate,
        delta_time,
        batch_size
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = Conv2DPCEN(n_classes=n_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Save directory
    save_dir = 'newPCENModel'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)

        # Validate
        val_loss = validate(model, device, val_loader, criterion)

        print(f'Epoch: {epoch}')
        print(f'Average train loss: {train_loss:.6f}')
        print(f'Average validation loss: {val_loss:.6f}')

        # Save if this is the best model so far (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'n_classes': n_classes,
                'sample_rate': sample_rate,
                'delta_time': delta_time
            }, model_save_path)
            print(f'New best model saved with validation loss: {val_loss:.6f}')


if __name__ == '__main__':
    main()