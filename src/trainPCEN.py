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
import json
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tqdm import tqdm


class SoundDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate, delta_time):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.delta_time = delta_time

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, orig_sample_rate = torchaudio.load(file_path)
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        fixed_length = int(self.sample_rate * self.delta_time)
        if waveform.size(1) > fixed_length:
            waveform = waveform[:, :fixed_length]
        elif waveform.size(1) < fixed_length:
            padding = fixed_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, label


def get_dataloaders(src_root, sample_rate, delta_time, batch_size):
    wav_paths = glob(f'{src_root}/**/*.wav', recursive=True)
    labels = [os.path.basename(os.path.dirname(path)) for path in wav_paths]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Calculate n_classes and get class distribution
    n_classes = len(label_encoder.classes_)
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\nClass Distribution:")
    for label, count in zip(label_encoder.classes_, counts):
        print(f"Class {label}: {count} samples")
    print(f"Total number of classes: {n_classes}\n")

    total_files = len(wav_paths)
    split_idx = int(total_files * 0.8)
    train_files, val_files = wav_paths[:split_idx], wav_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = SoundDataset(train_files, train_labels, sample_rate, delta_time)
    val_dataset = SoundDataset(val_files, val_labels, sample_rate, delta_time)

    # Optimized DataLoaders
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

    return train_loader, val_loader, n_classes, label_encoder, counts


def save_pcen_parameters(model, epoch, save_dir):
    pcen_layer = model.pcen
    pcen_params = {
        'alpha': pcen_layer.alpha.data.cpu().numpy().tolist(),
        'delta': pcen_layer.delta.data.cpu().numpy().tolist(),
        'root': pcen_layer.root.data.cpu().numpy().tolist() if hasattr(pcen_layer.root, 'data') else float(
            pcen_layer.root),
        'smooth_coef': float(pcen_layer._smooth_coef) if isinstance(pcen_layer._smooth_coef, float)
                      else pcen_layer._smooth_coef.data.cpu().numpy().tolist(),
        'floor': float(pcen_layer._floor)
    }

    params_file = os.path.join(save_dir, f'pcen_params_epoch_{epoch}.json')
    with open(params_file, 'w') as f:
        json.dump(pcen_params, f, indent=4)

    return pcen_params


def plot_prediction_distribution(pred_dist, label_encoder, save_path):
    plt.figure(figsize=(12, 6))
    plt.bar(label_encoder.classes_, pred_dist.cpu().numpy())
    plt.title('Prediction Distribution Across Classes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Training settings
    src_root = '/Users/samer/PycharmProjects/Lung-sounds-isef/CurrentDatasets/CleanDatasets (10 classes)/cleanTrainDataset (nonoise and COPD cut)'
    save_dir = 'pcen_model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    pcen_params_dir = os.path.join(save_dir, 'pcen_parameters')
    os.makedirs(pcen_params_dir, exist_ok=True)

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    batch_size = 16
    delta_time = 6.0
    sample_rate = 8000
    num_epochs = 30
    learning_rate = 0.0001

    # Get data loaders and class information
    train_loader, val_loader, n_classes, label_encoder, class_counts = get_dataloaders(
        src_root,
        sample_rate,
        delta_time,
        batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = Conv2DPCEN(n_classes=n_classes).to(device)

    # Calculate class weights for balanced loss
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)

    print("\nClass weights for balanced loss:")
    for label, weight in zip(label_encoder.classes_, class_weights.cpu().numpy()):
        print(f"Class {label}: {weight:.4f}")
    print()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        # Training
        model.train()
        running_loss = 0.0
        train_pred_dist = torch.zeros(n_classes)

        train_pbar = tqdm(train_loader, desc='Training')
        for waveforms, labels in train_pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

            # Update training prediction distribution
            _, predicted = torch.max(outputs, 1)
            for i in range(n_classes):
                train_pred_dist[i] += (predicted == i).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        print("Training prediction distribution:", train_pred_dist)

        # Save PCEN parameters
        current_pcen_params = save_pcen_parameters(model, epoch, pcen_params_dir)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pred_dist = torch.zeros(n_classes)

        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for waveforms, labels in val_pbar:
                waveforms, labels = waveforms.to(device), labels.to(device)

                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update validation prediction distribution
                for i in range(n_classes):
                    val_pred_dist[i] += (predicted == i).sum().item()

                # Update progress bar
                current_acc = 100 * correct / total
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print("Validation prediction distribution:", val_pred_dist)

        # Plot prediction distributions
        plot_prediction_distribution(
            train_pred_dist,
            label_encoder,
            os.path.join(plots_dir, f'train_pred_dist_epoch_{epoch}.png')
        )
        plot_prediction_distribution(
            val_pred_dist,
            label_encoder,
            os.path.join(plots_dir, f'val_pred_dist_epoch_{epoch}.png')
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'n_classes': n_classes,
                'pcen_params': current_pcen_params,
                'class_weights': class_weights
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f'Saved new best model with validation accuracy: {val_acc:.2f}%')

        scheduler.step(val_acc)


if __name__ == '__main__':
    main()