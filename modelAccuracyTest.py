import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, custom_object_scope
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
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


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    metrics = {}

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

    # Cross Entropy
    metrics['cross_entropy'] = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred_proba
    ).numpy().mean()

    return metrics


def evaluate_model(model_path, data_dir, sr=8000, dt=6.0, batch_size=16):
    """Evaluate a single model and return its metrics"""
    model_name = os.path.basename(model_path).replace('.h5', '')

    if model_name == 'mfcc_cnn':
        try:
            # Load MFCC model without custom objects
            model = load_model(model_path, compile=False)
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

    elif model_name == 'conv2dspec':
        try:
            # Custom objects for kapre melspectrogram
            from kapre.composed import get_melspectrogram_layer
            custom_objects = {
                'get_melspectrogram_layer': get_melspectrogram_layer
            }

            # Load model with custom objects
            with custom_object_scope(custom_objects):
                model = load_model(model_path, compile=False)
                print(f"Spec Model input shape: {model.input_shape}")

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
            for i in range(len(test_gen)):
                x_batch, y_batch = test_gen[i]
                pred_batch = model.predict(x_batch, verbose=0)
                all_y_true.append(y_batch)
                all_y_pred.append(pred_batch)

            y_test = np.vstack(all_y_true)
            y_pred_proba = np.vstack(all_y_pred)
            y_pred = y_pred_proba.copy()

        except Exception as e:
            print(f"Error processing Spec model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Unsupported model type: {model_name}")
        return None

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='directory containing .h5 model files')
    parser.add_argument('--data_dir', type=str, default='clean',
                        help='directory containing test audio files')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='audio sample rate')
    parser.add_argument('--delta_time', type=float, default=6.0,
                        help='time duration of audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
    args = parser.parse_args()

    # Get all model files
    model_files = glob(os.path.join(args.models_dir, '*.h5'))

    if not model_files:
        print(f"\nNo .h5 model files found in {args.models_dir}")
        print("Looking for models in:", os.path.abspath(args.models_dir))
        print("\nAvailable files in models directory:")
        try:
            if os.path.exists(args.models_dir):
                files = os.listdir(args.models_dir)
                for f in files:
                    print(f"- {f}")
            else:
                print(f"Directory {args.models_dir} does not exist!")
        except Exception as e:
            print(f"Error reading directory: {str(e)}")
        return

    # Print found models
    print("\nFound models:")
    for model_path in model_files:
        print(f"- {os.path.basename(model_path)}")

    # Evaluate each model
    results = []
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.h5', '')

        # Skip models that aren't conv2dspec or mfcc_cnn
        if model_name not in ['conv2dspec', 'mfcc_cnn']:
            print(f"\nSkipping {model_name} (not conv2dspec or mfcc_cnn)")
            continue

        print(f"\nEvaluating model: {model_name}")

        metrics = evaluate_model(
            model_path,
            args.data_dir,
            args.sample_rate,
            args.delta_time,
            args.batch_size
        )

        if metrics:
            metrics['model'] = model_name
            results.append(metrics)

            print("\nMetrics:")
            for metric, value in metrics.items():
                if metric != 'model' and metric != 'auc_per_class':
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