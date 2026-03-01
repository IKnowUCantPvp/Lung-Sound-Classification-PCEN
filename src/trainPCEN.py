import torch_audio_backend
from torch.utils.data import Dataset, WeightedRandomSampler
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
import wandb


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate CE loss without reduction first
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        # Calculate probability pt = exp(-CE)
        pt = torch.exp(-ce_loss)
        # Calculate Focal Loss = (1-pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


def get_dataloaders(train_dir, val_dir, sample_rate, delta_time, batch_size):
    # Train Data
    train_paths = glob(f'{train_dir}/**/*.wav', recursive=True)
    train_labels_raw = [os.path.basename(os.path.dirname(path)) for path in train_paths]
    
    # Val Data
    val_paths = glob(f'{val_dir}/**/*.wav', recursive=True)
    val_labels_raw = [os.path.basename(os.path.dirname(path)) for path in val_paths]

    # Fit label encoder on unique classes from train directory to ensure consistency
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    
    train_labels = label_encoder.transform(train_labels_raw)
    val_labels = label_encoder.transform(val_labels_raw)

    # Calculate n_classes and get class distribution
    n_classes = len(label_encoder.classes_)
    unique_labels, counts = np.unique(train_labels, return_counts=True)

    print("\nClass Distribution (Training):")
    for label, count in zip(label_encoder.classes_, counts):
        print(f"Class {label}: {count} samples")
    print(f"Total number of classes: {n_classes}\n")
    
    train_dataset = SoundDataset(train_paths, train_labels, sample_rate, delta_time)
    val_dataset = SoundDataset(val_paths, val_labels, sample_rate, delta_time)

    # Calculate weights for WeightedRandomSampler (Train only)
    class_sample_counts = np.bincount(train_labels)
    weight = 1. / class_sample_counts
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    # Optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler, # Use sampler for balanced batches
        shuffle=False,   # Must be False when sampler is provided
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
    
    # Handle the new CustomPCEN layer where smooth_coef is trainable and clamped
    smooth_coef_data = torch.clamp(pcen_layer.smooth_coef, 0.0, 1.0).data.cpu().numpy().tolist()
    
    pcen_params = {
        'alpha': float(pcen_layer.alpha),
        'delta': float(pcen_layer.delta),
        'root': float(pcen_layer.root),
        'smooth_coef': smooth_coef_data,
        'floor': float(pcen_layer.floor)
    }

    params_file = os.path.join(save_dir, f'pcen_params_epoch_{epoch}.json')
    with open(params_file, 'w') as f:
        json.dump(pcen_params, f, indent=4)
        
    # Log PCEN params to wandb
    wandb.log(pcen_params, step=epoch)

    return pcen_params


def plot_prediction_distribution(pred_dist, label_encoder, save_path, title):
    plt.figure(figsize=(12, 6))
    plt.bar(label_encoder.classes_, pred_dist.cpu().numpy())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Log plot to wandb
    wandb.log({title: wandb.Image(save_path)})


import argparse

def main():
    parser = argparse.ArgumentParser(description='Train PCEN Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training data')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to validation data')
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="lung-sound-classification-pcen", name="pcen-training-run")
    
    # Training settings
    if args.data_dir:
        train_dir = args.data_dir
    else:
        # Fallback/Default logic
        train_dir = '/workspace/Lung-Sound-Classification-PCEN/Lung-Sound-Classification-PCEN/data/train'
    
    if args.val_dir:
        val_dir = args.val_dir
    else:
         val_dir = '/workspace/Lung-Sound-Classification-PCEN/Lung-Sound-Classification-PCEN/data/test'

    print(f"Using training data: {train_dir}")
    print(f"Using validation data: {val_dir}")

    save_dir = 'pcen_model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)


    pcen_params_dir = os.path.join(save_dir, 'pcen_parameters')
    os.makedirs(pcen_params_dir, exist_ok=True)

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    batch_size = 16
    delta_time = 6.0
    sample_rate = 8000
    num_epochs = args.epochs
    learning_rate = 0.0001
    
    print(f"Training for {num_epochs} epochs")
    
    # Log config
    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "sample_rate": sample_rate, 
        "delta_time": delta_time,
        "train_dir": train_dir,
        "val_dir": val_dir
    })

    # Get data loaders and class information
    train_loader, val_loader, n_classes, label_encoder, class_counts = get_dataloaders(
        train_dir,
        val_dir,
        sample_rate,
        delta_time,
        batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = Conv2DPCEN(n_classes=n_classes).to(device)
    
    # WandB watch model
    wandb.watch(model, log="all")

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
    # Loss and optimizer
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
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
            
            # Log batch loss
            wandb.log({"batch_train_loss": loss.item()})

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
        
        # Log epoch metrics
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        })

        # Plot prediction distributions
        plot_prediction_distribution(
            train_pred_dist,
            label_encoder,
            os.path.join(plots_dir, f'train_pred_dist_epoch_{epoch}.png'),
            f'Training Prediction Distribution Epoch {epoch}'
        )
        plot_prediction_distribution(
            val_pred_dist,
            label_encoder,
            os.path.join(plots_dir, f'val_pred_dist_epoch_{epoch}.png'),
            f'Validation Prediction Distribution Epoch {epoch}'
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
            wandb.run.summary["best_val_accuracy"] = val_acc

        scheduler.step(val_acc)


if __name__ == '__main__':
    main()