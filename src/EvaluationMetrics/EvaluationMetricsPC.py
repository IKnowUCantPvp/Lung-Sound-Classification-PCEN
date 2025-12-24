import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob
import pandas as pd
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
import kapre
from kapre.composed import get_melspectrogram_layer
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr=8000, dt=6.0, n_classes=10, batch_size=16):
        # Filter out any paths with hidden directories
        valid_indices = [i for i, path in enumerate(wav_paths)
                         if not any(part.startswith('.') for part in path.split(os.sep))]

        self.wav_paths = [wav_paths[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes  # Make sure this matches your actual number of classes
        self.batch_size = batch_size
        self.on_epoch_end()

        print(f"DataGenerator initialized with {len(self.wav_paths)} valid files and {self.n_classes} classes")

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = np.arange(index * self.batch_size, min((index + 1) * self.batch_size, len(self.wav_paths)))
        batch_paths = [self.wav_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        X = np.empty((len(indexes), int(self.sr * self.dt), 1), dtype=np.float32)
        Y = np.empty((len(indexes), self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            try:
                rate, wav = wavfile.read(path)
                if len(wav) > int(self.sr * self.dt):
                    wav = wav[:int(self.sr * self.dt)]
                elif len(wav) < int(self.sr * self.dt):
                    wav = np.pad(wav, (0, int(self.sr * self.dt) - len(wav)))
                X[i,] = wav.reshape(-1, 1)
                Y[i,] = to_categorical(label, num_classes=self.n_classes)
            except Exception as e:
                print(f"Error processing file {path}: {str(e)}")
                # Fill with zeros if there's an error
                X[i,] = np.zeros((int(self.sr * self.dt), 1))
                Y[i,] = np.zeros(self.n_classes)

        return X, Y

    def on_epoch_end(self):
        pass

def load_pb_model(model_dir):
    """Load a model saved in SavedModel (.pb) format"""
    print(f"\nAttempting to load model from: {model_dir}")
    print(f"Directory contents: {os.listdir(model_dir)}")

    try:
        print("Loading model...")
        model = tf.keras.models.load_model(model_dir)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model from {model_dir}: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics (excluding loss which is handled by model.evaluate)"""
    metrics = {}

    # Convert one-hot encoded to class labels for some metrics
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_classes, y_pred_classes)

    # Precision, Recall, F1 (weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted',
        zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # AUC calculations
    n_classes = y_true.shape[1]

    # Initialize AUC metrics
    metrics['auc_ovr'] = np.nan
    metrics['auc_per_class'] = []

    # Count number of samples per class
    class_samples = np.sum(y_true, axis=0)
    valid_classes = class_samples > 1

    # Only calculate AUC if we have at least two classes with more than one sample
    if np.sum(valid_classes) >= 2:
        try:
            # Overall AUC (one-vs-rest)
            metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba,
                                             multi_class='ovr',
                                             average='macro')
        except ValueError as e:
            print(f"Warning: Could not calculate overall AUC: {str(e)}")
            metrics['auc_ovr'] = np.nan

    # Per-class AUC scores
    for i in range(n_classes):
        try:
            # Only calculate AUC if we have more than one sample for this class
            if class_samples[i] > 1:
                class_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            else:
                class_auc = np.nan
            metrics['auc_per_class'].append(class_auc)
        except ValueError:
            metrics['auc_per_class'].append(np.nan)

    return metrics

def evaluate_model(model_dir, data_dir, model_name, sr=8000, dt=6.0, batch_size=16):
    """Evaluate a single model and return its metrics"""
    print(f"Processing model: {model_name} from directory: {model_dir}")

    if 'mfcc_cnn' in model_name:
        try:
            # Load MFCC model from SavedModel format
            model = load_pb_model(model_dir)
            if model is None:
                return None

            print(f"MFCC Model input shape: {model.input_shape}")

            # Load pre-computed MFCC features and labels
            features = np.load('../2D-MFCC/features.npy')
            label_array = np.load('../2D-MFCC/labels.npy')
            print(f"Loaded features shape: {features.shape}")

            # Reshape features if needed and add channel dimension
            if len(features.shape) == 3:
                features = features[..., np.newaxis]

            print(f"Features shape after preprocessing: {features.shape}")

            # Prepare labels
            classes = sorted([d for d in os.listdir(data_dir)
                              if not d.startswith('.') and os.path.isdir(os.path.join(data_dir, d))])
            le = LabelEncoder()
            le.fit(classes)
            labels = le.transform(label_array)

            # Convert labels to categorical
            y_true = to_categorical(labels, num_classes=len(classes))

            # Use model.evaluate for loss
            eval_results = model.evaluate(features, y_true, batch_size=batch_size, verbose=1)
            loss = eval_results[0]  # First metric is always loss

            # Get predictions for other metrics
            y_pred_proba = model.predict(features, batch_size=batch_size, verbose=1)
            y_pred = y_pred_proba.copy()

        except Exception as e:
            print(f"Error processing MFCC model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    elif any(x in model_name.lower() for x in ['conv2doldpcen', 'conv2dspec', 'conv2pcen']):
        try:
            print(f"\nProcessing Conv2D/PCEN model: {model_name}")
            print(f"Model directory path: {model_dir}")
            print(f"Directory exists: {os.path.exists(model_dir)}")

            # Load model from SavedModel format
            model = load_pb_model(model_dir)
            if model is None:
                return None

            print(f"Conv2D Model input shape: {model.input_shape}")

#THIS BLOCK IS UPDATED FOR WINDOWS!!
            # Prepare raw audio data
            wav_paths = glob(f'{data_dir}/**/*.wav', recursive=True)
            print(f"Found {len(wav_paths)} audio files")

            # Extract class names (subdirectories in `data_dir`)
            classes = sorted([
                d for d in os.listdir(data_dir)
                if not d.startswith('.') and os.path.isdir(os.path.join(data_dir, d))
            ])
            print(f"Found classes: {classes}")

            # Initialize LabelEncoder with valid class names
            le = LabelEncoder()
            le.fit(classes)

            # Correctly extract labels from file paths
            labels = [
                os.path.basename(os.path.dirname(path))  # Extract the immediate parent folder of each file
                for path in wav_paths
                if not os.path.basename(os.path.dirname(path)).startswith('.')  # Skip hidden folders
            ]
            print(f"Extracted labels: {set(labels)}")  # Debugging: check if labels are correct

            # Transform labels to integers using the encoder
            labels = le.transform(labels)



            # Print class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"Class {classes[label]}: {count} samples")

            # Create data generator
            batch_size = min(batch_size, 16)  # Reduce batch size if needed
            data_gen = DataGenerator(
                wav_paths, labels, sr, dt,
                n_classes=len(classes), batch_size=batch_size
            )

            # Use model.evaluate for loss
            eval_results = model.evaluate(data_gen, verbose=1)
            loss = eval_results[0]  # First metric is always loss

            # Get predictions for other metrics
            all_y_true = []
            all_y_pred = []

            total_batches = len(data_gen)
            print(f"\nProcessing {total_batches} batches...")

            for i in range(total_batches):
                if i % 10 == 0:  # Print progress every 10 batches
                    print(f"Processing batch {i}/{total_batches}")

                try:
                    x_batch, y_batch = data_gen[i]
                    pred_batch = model.predict(x_batch, verbose=0)
                    all_y_true.append(y_batch)
                    all_y_pred.append(pred_batch)
                except Exception as e:
                    print(f"Error processing batch {i}: {str(e)}")
                    continue

            y_true = np.vstack(all_y_true)
            y_pred_proba = np.vstack(all_y_pred)
            y_pred = y_pred_proba.copy()

        except Exception as e:
            print(f"Error processing Conv2D model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Unsupported model type: {model_name}")
        return None

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics['loss'] = loss  # Use the loss from model.evaluate

    # Get class predictions for confusion matrix
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'Confusion Matrix (Counts) - {model_name}')
    ax1.xaxis.set_ticklabels(classes, rotation=45)
    ax1.yaxis.set_ticklabels(classes, rotation=45)

    # Plot normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title(f'Confusion Matrix (Normalized) - {model_name}')
    ax2.xaxis.set_ticklabels(classes, rotation=45)
    ax2.yaxis.set_ticklabels(classes, rotation=45)

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # Generate ROC curves
    plt.figure(figsize=(10, 8))
    n_classes = len(metrics['auc_per_class'])
    for i in range(n_classes):
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

    return metrics

def find_model_dirs(base_dir):
    """Find all model directories containing saved_model.pb files"""
    model_dirs = []
    for model_type in os.listdir(base_dir):
        if model_type.startswith('.'):  # Skip hidden files
            continue
        model_path = os.path.join(base_dir, model_type)
        if os.path.isdir(model_path):
            if 'saved_model.pb' in os.listdir(model_path):
                model_dirs.append((model_type, model_path))
            else:
                for item in os.listdir(model_path):
                    if item.startswith('.'):  # Skip hidden files
                        continue
                    subpath = os.path.join(model_path, item)
                    if os.path.isdir(subpath):
                        if 'saved_model.pb' in os.listdir(subpath):
                            model_dirs.append((model_type, model_path))
                            break

    # Debug output
    print("\nFound model directories:")
    for model_type, path in model_dirs:
        print(f"{model_type}: {path}")
        try:
            print(f"Contents: {os.listdir(path)}")
        except Exception as e:
            print(f"Error listing contents: {str(e)}")

    return model_dirs


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument('--models_dir', type=str, default= 'TestingModels',
                        help='directory containing model directories')
    parser.add_argument('--data_dir', type=str, default= r'C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\CurrentDatasets\OLD AbnormalNormalDatasets (2 classes)\AbnormalNormalEvaluationDataset (noise+COPDcut)',
                        help='directory containing test audio files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='audio sample rate')
    parser.add_argument('--delta_time', type=float, default=6.0,
                        help='time duration of audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='model_evaluation.csv',
                        help='filename for saving metrics results')
    args = parser.parse_args()

    # Find all model directories
    model_pairs = find_model_dirs(args.models_dir)

    if not model_pairs:
        print(f"\nNo SavedModel directories found in {args.models_dir}")
        print("Looking for models in:", os.path.abspath(args.models_dir))
        print("\nAvailable directories:")
        try:
            if os.path.exists(args.models_dir):
                for item in os.listdir(args.models_dir):
                    full_path = os.path.join(args.models_dir, item)
                    if os.path.isdir(full_path):
                        print(f"Directory: {item}")
                        for subitem in os.listdir(full_path):
                            print(f"  - {subitem}")
            else:
                print(f"Directory {args.models_dir} does not exist!")
        except Exception as e:
            print(f"Error reading directory: {str(e)}")
        return

    # Print found models
    print("\nFound models:")
    for model_name, model_dir in model_pairs:
        print(f"- {model_name}")

    # Evaluate each model
    results = []
    for model_name, model_dir in model_pairs:
        print(f"\nEvaluating model: {model_name}")

        metrics = evaluate_model(
            model_dir,
            args.data_dir,
            model_name,
            args.sample_rate,
            args.delta_time,
            args.batch_size
        )

        if metrics:
            metrics['model'] = model_name
            results.append(metrics)

            print("\nMetrics:")
            print(f"Loss: {metrics['loss']:.4f}")
            if 'avg_batch_loss' in metrics:
                print(f"Average batch loss: {metrics['avg_batch_loss']:.4f}")
            for metric, value in metrics.items():
                if metric not in ['model', 'auc_per_class', 'loss', 'avg_batch_loss']:
                    print(f"{metric}: {value:.4f}")
            print("\nPer-class AUC scores:")
            classes = sorted(os.listdir(args.data_dir))
            for cls, auc in zip(classes, metrics['auc_per_class']):
                print(f"{cls}: {auc:.4f}")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == '__main__':
    main()