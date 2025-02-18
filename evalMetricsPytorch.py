import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from glob import glob
import pandas as pd
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch audio classification models')
    parser.add_argument('--models_dir', type=str, default='newPCENModel',
                        help='directory containing .pt model files')
    parser.add_argument('--data_dir', type=str,
                        default=r'/Users/samer/PycharmProjects/Lung-sounds-isef/CurrentDatasets/CleanDatasets (10 classes)/cleanEvaluationDataset (noise and COPD cut)',
                        help='directory containing test audio files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='audio sample rate')
    parser.add_argument('--delta_time', type=float, default=6.0,
                        help='time duration of audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='model_evaluation.csv',
                        help='filename for saving metrics results')
    return parser.parse_args()


def get_dataloaders(src_root, sr=8000, dt=6.0, batch_size=16):
    """Helper function to create data loaders"""
    # Get list of WAV files recursively
    wav_paths = glob(f'{src_root}/**/*.wav', recursive=True)

    # Get class names from directory structure
    classes = sorted([d for d in os.listdir(src_root)
                      if not d.startswith('.') and os.path.isdir(os.path.join(src_root, d))])

    # Create label encoder
    le = LabelEncoder()
    le.fit(classes)

    # Extract labels from file paths
    labels = [os.path.basename(os.path.dirname(path)) for path in wav_paths
              if not os.path.basename(os.path.dirname(path)).startswith('.')]
    labels = le.transform(labels)

    return labels


class AudioDataset(Dataset):
    """Dataset for evaluating audio models"""

    def __init__(self, wav_paths, labels, sr=8000, dt=6.0, n_classes=10):
        # Filter out paths with hidden directories
        valid_indices = [i for i, path in enumerate(wav_paths)
                         if not any(part.startswith('.') for part in path.split(os.sep))]

        self.wav_paths = [wav_paths[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        print(f"Dataset initialized with {len(self.wav_paths)} valid files and {self.n_classes} classes")

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        try:
            # Load audio file
            rate, wav = wavfile.read(self.wav_paths[idx])
            target_length = int(self.sr * self.dt)

            # Trim or pad as needed
            if len(wav) > target_length:
                wav = wav[:target_length]
            elif len(wav) < target_length:
                wav = np.pad(wav, (0, target_length - len(wav)))

            # Convert to torch tensor
            x = torch.FloatTensor(wav)

            # Create label tensor
            label = torch.zeros(self.n_classes)
            label[self.labels[idx]] = 1

            return x, label

        except Exception as e:
            print(f"Error processing file {self.wav_paths[idx]}: {str(e)}")
            return torch.zeros(target_length), torch.zeros(self.n_classes)


def load_pt_model(model_path):
    """Load a PyTorch model saved as .pt
    Handles both full models and state dictionaries
    """
    try:
        checkpoint = torch.load(model_path)

        # Check if this is a state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # If it's a training checkpoint with multiple components
                state_dict = checkpoint['model_state_dict']
            else:
                # If it's just the state dict
                state_dict = checkpoint

            # Here you need to initialize your model architecture
            # This should match the architecture used during training
            # For example:
            from pyTorchModels import Conv2DPCEN  # Import your model class
            model = Conv2DPCEN()  # Initialize with correct parameters
            model.load_state_dict(state_dict)
        else:
            # If it's a full model
            model = checkpoint

        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    metrics = {}

    # Convert to numpy arrays if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_pred_proba):
        y_pred_proba = y_pred_proba.cpu().numpy()

    # Convert one-hot encoded to class labels
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    metrics['accuracy'] = accuracy_score(y_true_classes, y_pred_classes)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted',
        zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # AUC calculations
    n_classes = y_true.shape[1]
    metrics['auc_per_class'] = []

    try:
        metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba,
                                           multi_class='ovr',
                                           average='macro')
    except ValueError:
        metrics['auc_ovr'] = np.nan

    for i in range(n_classes):
        try:
            class_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            metrics['auc_per_class'].append(class_auc)
        except ValueError:
            metrics['auc_per_class'].append(np.nan)

    return metrics


def evaluate_model(model_path, data_dir, model_name, device, sr=8000, dt=6.0, batch_size=16):
    """Evaluate a single PyTorch model and return its metrics"""
    try:
        print(f"Processing model: {model_name} from path: {model_path}")

        # Load model
        model = load_pt_model(model_path)
        if model is None:
            return None

        model = model.to(device)

        # Prepare data
        wav_paths = glob(f'{data_dir}/**/*.wav', recursive=True)
        classes = sorted([d for d in os.listdir(data_dir)
                          if not d.startswith('.') and os.path.isdir(os.path.join(data_dir, d))])

        le = LabelEncoder()
        le.fit(classes)

        labels = [os.path.basename(os.path.dirname(path)) for path in wav_paths
                  if not os.path.basename(os.path.dirname(path)).startswith('.')]
        labels = le.transform(labels)

        # Create dataset and dataloader
        dataset = AudioDataset(wav_paths, labels, sr, dt, len(classes))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Evaluation
        criterion = nn.CrossEntropyLoss()
        all_losses = []
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Add batch dimension if missing
                if batch_x.dim() == 1:
                    batch_x = batch_x.unsqueeze(0)

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Store results
                all_losses.append(loss.item())
                all_y_true.append(batch_y.cpu())
                all_y_pred_proba.append(torch.softmax(outputs, dim=1).cpu())
                all_y_pred.append(torch.softmax(outputs, dim=1).cpu())

        # Concatenate results
        y_true = torch.cat(all_y_true, dim=0).numpy()
        y_pred_proba = torch.cat(all_y_pred_proba, dim=0).numpy()
        y_pred = y_pred_proba.copy()

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics['loss'] = np.mean(all_losses)

        # Generate plots
        try:
            # Generate confusion matrix plots
            y_true_classes = np.argmax(y_true, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            cm = confusion_matrix(y_true_classes, y_pred_classes)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'Confusion Matrix (Counts) - {model_name}')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            ax1.set_xticklabels(classes, rotation=45)
            ax1.set_yticklabels(classes, rotation=45)

            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
            ax2.set_title(f'Confusion Matrix (Normalized) - {model_name}')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            ax2.set_xticklabels(classes, rotation=45)
            ax2.set_yticklabels(classes, rotation=45)

            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name}.png')
            plt.close()

            # Generate ROC curves
            plt.figure(figsize=(10, 8))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {metrics["auc_per_class"][i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {model_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.savefig(f'roc_curves_{model_name}.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate plots: {str(e)}")

        return metrics

    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def find_model_files(base_dir):
    """Find all .pt model files"""
    model_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pt'):
                model_name = os.path.splitext(file)[0]
                model_files.append((model_name, os.path.join(root, file)))
    return model_files


if __name__ == "__main__":
    args = parse_args()
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find all model files
    model_pairs = find_model_files(args.models_dir)

    if not model_pairs:
        print(f"\nNo .pt model files found in {args.models_dir}")
        exit()

    print("\nFound models:")
    for model_name, model_path in model_pairs:
        print(f"- {model_name}")

    # Evaluate each model
    results = []
    for model_name, model_path in model_pairs:
        print(f"\nEvaluating model: {model_name}")

        metrics = evaluate_model(
            model_path,
            args.data_dir,
            model_name,
            device,
            args.sample_rate,
            args.delta_time,
            args.batch_size
        )

        if metrics:
            metrics['model'] = model_name
            results.append(metrics)

            print("\nMetrics:")
            print(f"Loss: {metrics['loss']:.4f}")
            for metric, value in metrics.items():
                if metric not in ['model', 'auc_per_class', 'loss']:
                    print(f"{metric}: {value:.4f}")
            print("\nPer-class AUC scores:")
            classes = sorted([d for d in os.listdir(args.data_dir)
                              if not d.startswith('.') and os.path.isdir(os.path.join(args.data_dir, d))])
            for cls, auc in zip(classes, metrics['auc_per_class']):
                print(f"{cls}: {auc:.4f}")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)