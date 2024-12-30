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
from sklearn.model_selection import train_test_split
import kapre
from kapre.composed import get_melspectrogram_layer


class DataGenerator(tf.keras.utils.Sequence):
    """DataGenerator for Conv2DSpec model"""

    def __init__(self, wav_paths, labels, sr=8000, dt=6.0, n_classes=6, batch_size=16):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        batch_paths = [self.wav_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]

        X = np.empty((self.batch_size, int(self.sr * self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        pass


def load_pb_model(model_dir):
    """Load a model saved in SavedModel (.pb) format"""
    try:
        model = tf.keras.models.load_model(model_dir)
        return model
    except Exception as e:
        print(f"Error loading model from {model_dir}: {str(e)}")
        return None


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics including loss"""
    metrics = {}

    # Initialize loss function
    loss_fn = CategoricalCrossentropy(from_logits=False)

    # Calculate categorical crossentropy loss
    metrics['loss'] = float(loss_fn(y_true, y_pred_proba).numpy())

    # Convert one-hot encoded to class labels for some metrics
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_classes, y_pred_classes)

    # Precision, Recall, F1 (weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # AUC calculations
    n_classes = y_true.shape[1]

    # Overall AUC (one-vs-rest)
    metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba,
                                       multi_class='ovr',
                                       average='macro')

    # Per-class AUC scores
    metrics['auc_per_class'] = []
    for i in range(n_classes):
        try:
            class_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            metrics['auc_per_class'].append(class_auc)
        except ValueError:
            metrics['auc_per_class'].append(np.nan)

    return metrics


def evaluate_model(model_dir, data_dir, model_name, sr=8000, dt=6.0, batch_size=16):
    """Evaluate a single model and return its metrics"""
    print(f"Processing model: {model_name} from directory: {model_dir}")

    if model_name == 'mfcc_cnn':
        try:
            # Load MFCC model from SavedModel format
            model = load_pb_model(model_dir)
            if model is None:
                return None

            print(f"MFCC Model input shape: {model.input_shape}")

            # Load pre-computed MFCC features and labels
            features = np.load('2D-MFCC/features.npy')
            label_array = np.load('2D-MFCC/labels.npy')
            print(f"Loaded features shape: {features.shape}")

            # Reshape features if needed and add channel dimension
            if len(features.shape) == 3:
                features = features[..., np.newaxis]

            print(f"Features shape after preprocessing: {features.shape}")

            # Prepare labels
            classes = sorted(os.listdir(data_dir))
            le = LabelEncoder()
            le.fit(classes)
            labels = le.transform(label_array)

            # Split data
            features_train, features_test, label_train, label_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            # Convert labels to categorical
            y_test = to_categorical(label_test, num_classes=len(classes))

            # Get predictions
            y_pred_proba = model.predict(features_test, batch_size=batch_size, verbose=1)
            y_pred = y_pred_proba.copy()

        except Exception as e:
            print(f"Error processing MFCC model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    elif any(x in model_name.lower() for x in ['conv2doldpcen', 'conv2dspec']):
        try:
            # Load model from SavedModel format
            model = load_pb_model(model_dir)
            if model is None:
                return None

            print(f"Conv2D Model input shape: {model.input_shape}")

            # Prepare raw audio data
            wav_paths = glob(f'{data_dir}/**/*.wav', recursive=True)
            classes = sorted(os.listdir(data_dir))
            le = LabelEncoder()
            le.fit(classes)
            labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
            labels = le.transform(labels)

            # Split data
            wav_train, wav_test, label_train, label_test = train_test_split(
                wav_paths, labels, test_size=0.2, random_state=42
            )

            # Create data generator
            test_gen = DataGenerator(
                wav_test, label_test, sr, dt,
                n_classes=len(classes), batch_size=batch_size
            )

            # Get predictions
            all_y_true = []
            all_y_pred = []
            batch_losses = []
            loss_fn = CategoricalCrossentropy(from_logits=False)

            for i in range(len(test_gen)):
                x_batch, y_batch = test_gen[i]
                pred_batch = model.predict(x_batch, verbose=0)

                # Calculate batch loss
                batch_loss = float(loss_fn(y_batch, pred_batch).numpy())
                batch_losses.append(batch_loss)

                all_y_true.append(y_batch)
                all_y_pred.append(pred_batch)

            y_test = np.vstack(all_y_true)
            y_pred_proba = np.vstack(all_y_pred)
            y_pred = y_pred_proba.copy()

            # Calculate average batch loss
            avg_batch_loss = np.mean(batch_losses)
            print(f"Average batch loss: {avg_batch_loss:.4f}")

        except Exception as e:
            print(f"Error processing Conv2D model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Unsupported model type: {model_name}")
        return None

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Add average batch loss for Conv2D models
    if 'batch_losses' in locals():
        metrics['avg_batch_loss'] = avg_batch_loss

    # Generate ROC curves
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    n_classes = len(metrics['auc_per_class'])
    classes = sorted(os.listdir(data_dir))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {metrics["auc_per_class"][i]:.2f})')

    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Save the plot
    plt.savefig(f'roc_curves_{model_name}.png', bbox_inches='tight')
    plt.close()

    return metrics


def find_model_dirs(base_dir):
    """Find all model directories containing saved_model.pb files"""
    model_dirs = []
    for model_type in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_type)
        if os.path.isdir(model_path):
            # Check if this directory contains saved_model.pb directly
            if 'saved_model.pb' in os.listdir(model_path):
                model_dirs.append((model_type, model_path))
            else:
                # Check subdirectories for saved_model.pb
                for root, dirs, files in os.walk(model_path):
                    if 'saved_model.pb' in files:
                        model_dirs.append((model_type, root))
    return model_dirs


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='directory containing model directories')
    parser.add_argument('--data_dir', type=str, default='clean',
                        help='directory containing test audio files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='audio sample rate')
    parser.add_argument('--delta_time', type=float, default=6.0,
                        help='time duration of audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
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
        output_file = 'model_evaluation_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == '__main__':
    main()