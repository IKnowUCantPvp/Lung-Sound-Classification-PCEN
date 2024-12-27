import tensorflow as tf
import numpy as np
import pandas as pd
import os
import wave
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from glob import glob
from tqdm import tqdm
import argparse
import sys

class AudioParameters:
    """
    Parameters matching your MFCC extraction configuration
    """

    def __init__(self):
        # Audio loading parameters
        self.sample_rate = 8000
        self.duration = 6
        self.mono = True
        self.res_type = 'kaiser_fast'

        # MFCC extraction parameters
        self.n_mfcc = 40
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 40
        self.fmin = 50
        self.fmax = 2000

        # Feature padding parameters
        self.max_pad_len = 188
        self.pad_mode = 'constant'
        self.pad_constant = 0


def verify_file_system(file_path):
    """Verify file system checks"""
    try:
        exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path)
        readable = os.access(file_path, os.R_OK)
        size = os.path.getsize(file_path) if exists and is_file else 0
        return exists and is_file and readable and size > 0
    except Exception:
        return False


def verify_wav_file(file_path):
    """Verify WAV file format"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            valid = (wav_file.getnchannels() > 0 and
                     wav_file.getsampwidth() > 0 and
                     wav_file.getframerate() > 0 and
                     wav_file.getnframes() > 0)
        return valid
    except Exception:
        return False


def extract_features(file_path, params):
    """Extract MFCC features using the same parameters as training"""
    if not verify_file_system(file_path) or not verify_wav_file(file_path):
        return None

    try:
        # Load audio with same parameters
        audio, _ = librosa.load(
            file_path,
            sr=params.sample_rate,
            duration=params.duration,
            mono=params.mono,
            res_type=params.res_type
        )

        # Extract MFCCs with matching parameters
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=params.sample_rate,
            n_mfcc=params.n_mfcc,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            n_mels=params.n_mels,
            fmin=params.fmin,
            fmax=params.fmax
        )

        # Pad or truncate to match expected length
        if mfccs.shape[1] > params.max_pad_len:
            mfccs = mfccs[:, :params.max_pad_len]
        else:
            pad_width = params.max_pad_len - mfccs.shape[1]
            mfccs = np.pad(
                mfccs,
                pad_width=((0, 0), (0, pad_width)),
                mode=params.pad_mode,
                constant_values=params.pad_constant
            )

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def load_models(models_dir):
    """
    Load TensorFlow SavedModel models from the specified directory.

    Args:
        models_dir (str): Directory containing the saved models.

    Returns:
        dict: A dictionary mapping model names to their loaded TensorFlow models.
    """
    models = {}
    for model_dir in glob(os.path.join(models_dir, '*')):
        model_name = os.path.basename(model_dir)
        try:
            models[model_name] = tf.keras.models.load_model(model_dir)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    return models


def test_models(args):
    """
    Test the loaded models on the new dataset.
    """
    params = AudioParameters()
    models = load_models(args.models_dir)

    for model_path in glob(os.path.join(args.models_dir, '*.h5')):
        model_name = os.path.basename(model_path).replace('.h5', '')
        try:
            models[model_name] = tf.keras.models.load_model(model_path)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            continue

    # Process files and collect results
    results = []

    # Get all WAV files recursively
    wav_paths = []
    labels = []
    print("\nScanning directory structure...")
    for diagnosis in os.listdir(args.data_dir):
        diagnosis_path = os.path.join(args.data_dir, diagnosis)
        if os.path.isdir(diagnosis_path):
            for root, _, files in os.walk(diagnosis_path):
                wav_files = [f for f in files if f.endswith('.wav')]
                for file in wav_files:
                    wav_paths.append(os.path.join(root, file))
                    labels.append(diagnosis)

    print(f"\nFound {len(wav_paths)} files to process")

    # Process each file
    for wav_path, true_label in tqdm(zip(wav_paths, labels), total=len(wav_paths)):
        # Extract features
        features = extract_features(wav_path, params)
        if features is None:
            continue

        file_results = {
            'file': wav_path,
            'true_class': true_label
        }

        # Test with each model
        for model_name, model in models.items():
            try:
                if 'mfcc' in model_name.lower():
                    # For MFCC model - add batch and channel dimensions
                    features_batch = features[np.newaxis, ..., np.newaxis]
                    pred = model.predict(features_batch, verbose=0)
                else:
                    # Load raw audio for spectrogram/PCEN models
                    audio, _ = librosa.load(wav_path, sr=params.sample_rate,
                                            duration=params.duration)
                    audio = audio[:int(params.sample_rate * params.duration)]
                    pred = model.predict(audio[np.newaxis, ..., np.newaxis], verbose=0)

                pred_class = np.argmax(pred[0])
                confidence = np.max(pred[0])

                file_results[f'{model_name}_pred'] = pred_class
                file_results[f'{model_name}_conf'] = confidence

            except Exception as e:
                print(f"Error with {model_name} on {wav_path}: {str(e)}")
                file_results[f'{model_name}_pred'] = None
                file_results[f'{model_name}_conf'] = None

        results.append(file_results)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)

    # Print metrics
    print("\nModel Performance Summary:")
    for model_name in models.keys():
        pred_col = f'{model_name}_pred'
        valid_preds = results_df[results_df[pred_col].notna()]
        if len(valid_preds) > 0:
            accuracy = (valid_preds['true_class'] == valid_preds[pred_col]).mean()
            print(f"{model_name}:")
            print(f"- Accuracy: {accuracy:.3f}")
            print(f"- Processed files: {len(valid_preds)}/{len(results_df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained models on new dataset')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='directory containing trained model files')
    parser.add_argument('--output_file', type=str, default='test_results.csv',
                        help='path to save results CSV')

    args = parser.parse_args()

    # Set up directory structure like in Lung-Sounds.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = os.path.join(current_dir, "newclean")

    if not os.path.exists(args.data_dir):
        print(f"\nERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    test_models(args)