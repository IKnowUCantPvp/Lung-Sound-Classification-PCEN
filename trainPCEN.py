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


def save_pcen_parameters(model, epoch, save_dir):
    """
    Extract and save PCEN parameters from the SpeechBrain PCEN layer
    """
    pcen_layer = model.pcen
    pcen_params = {
        'alpha': pcen_layer.alpha.data.cpu().numpy().tolist(),
        'delta': pcen_layer.delta.data.cpu().numpy().tolist(),
        'root': pcen_layer.root.data.cpu().numpy().tolist() if hasattr(pcen_layer.root, 'data') else float(
            pcen_layer.root),
        'smooth_coef': pcen_layer.smooth_coef.data.cpu().numpy().tolist(),
        'floor': float(pcen_layer.floor),
        'trainable': pcen_layer.trainable,
        'per_channel_smooth_coef': pcen_layer.per_channel_smooth_coef
    }

    # Save parameters to JSON file
    params_file = os.path.join(save_dir, f'pcen_params_epoch_{epoch}.json')
    with open(params_file, 'w') as f:
        json.dump(pcen_params, f, indent=4)

    return pcen_params


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

    # Create a subdirectory for PCEN parameters
    pcen_params_dir = os.path.join(save_dir, 'pcen_parameters')
    os.makedirs(pcen_params_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_pcen_params = None

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)

        # Save PCEN parameters for this epoch
        current_pcen_params = save_pcen_parameters(model, epoch, pcen_params_dir)

        # Validate
        val_loss = validate(model, device, val_loader, criterion)

        print(f'Epoch: {epoch}')
        print(f'Average train loss: {train_loss:.6f}')
        print(f'Average validation loss: {val_loss:.6f}')

        # Save if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pcen_params = current_pcen_params

            # Save the best model
            model_save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'n_classes': n_classes,
                'sample_rate': sample_rate,
                'delta_time': delta_time,
                'pcen_params': best_pcen_params  # Include PCEN parameters in the best model checkpoint
            }, model_save_path)

            # Also save best PCEN parameters separately
            best_pcen_path = os.path.join(save_dir, 'best_pcen_params.json')
            with open(best_pcen_path, 'w') as f:
                json.dump(best_pcen_params, f, indent=4)

            print(f'New best model saved with validation loss: {val_loss:.6f}')

    # Save final PCEN parameters evolution plot
    try:
        import matplotlib.pyplot as plt

        # Load all saved PCEN parameters
        all_params = []
        for epoch in range(1, num_epochs + 1):
            with open(os.path.join(pcen_params_dir, f'pcen_params_epoch_{epoch}.json'), 'r') as f:
                all_params.append(json.load(f))

        # Plot evolution of trainable parameters
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, num_epochs + 1)

        # Alpha
        axes[0, 0].plot(epochs, [np.mean(p['alpha']) for p in all_params])
        axes[0, 0].set_title('Mean Alpha Evolution')

        # Delta
        axes[0, 1].plot(epochs, [np.mean(p['delta']) for p in all_params])
        axes[0, 1].set_title('Mean Delta Evolution')

        # Root
        if isinstance(all_params[0]['root'], list):
            axes[1, 0].plot(epochs, [np.mean(p['root']) for p in all_params])
        else:
            axes[1, 0].axhline(y=all_params[0]['root'], color='r')
        axes[1, 0].set_title('Root Evolution')

        # Smooth coefficient
        axes[1, 1].plot(epochs, [np.mean(p['smooth_coef']) for p in all_params])
        axes[1, 1].set_title('Mean Smooth Coefficient Evolution')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pcen_parameter_evolution.png'))

    except Exception as e:
        print(f"Could not create parameter evolution plot: {str(e)}")


if __name__ == '__main__':
    main()