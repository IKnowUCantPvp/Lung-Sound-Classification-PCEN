import numpy as np
import pandas as pd
import os
import sys
import wave
import librosa
import librosa.display
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


class AudioParameters:
    """
    Parameters optimized for 6-second lung sound classification at 8 kHz sampling rate.
    These parameters are specifically tuned to capture respiratory sounds while maintaining
    good time-frequency resolution for a 6-second analysis window.
    """

    def __init__(self):
        # Audio loading parameters
        self.sample_rate = 8000  # 8 kHz sampling rate for respiratory sounds
        self.duration = 6  # Matches your audio file length exactly
        self.mono = True  # Mono audio is sufficient for lung sounds
        self.res_type = 'kaiser_fast'  # Efficient resampling algorithm

        # MFCC extraction parameters
        self.n_mfcc = 13  # Standard for biomedical acoustic analysis
        self.n_fft = 1024  # ~128ms window for good frequency resolution
        self.hop_length = 256  # 75% overlap between frames
        self.n_mels = 40  # Focused on lower frequencies relevant to lung sounds
        self.fmin = 50  # Capture lowest frequency components
        self.fmax = 2000  # Upper limit for most lung sound components

        # Feature padding parameters
        self.max_pad_len = 188  # Calculated for 6-second files at 8kHz
        self.pad_mode = 'constant'
        self.pad_constant = 0

    def print_analysis_details(self):
        """Print detailed analysis parameters with explanations"""
        frame_duration_ms = (self.n_fft / self.sample_rate) * 1000
        hop_duration_ms = (self.hop_length / self.sample_rate) * 1000
        freq_resolution = self.sample_rate / self.n_fft
        time_steps = int((self.sample_rate * self.duration) / self.hop_length)

        print("\nAudio Analysis Configuration:")
        print(f"\nTemporal Parameters:")
        print(f"- Total duration: {self.duration} seconds")
        print(f"- Frame duration: {frame_duration_ms:.1f} ms")
        print(f"- Frame overlap: {(1 - self.hop_length / self.n_fft) * 100:.1f}%")
        print(f"- Time steps in feature matrix: {time_steps}")

        print(f"\nFrequency Parameters:")
        print(f"- Sample rate: {self.sample_rate} Hz")
        print(f"- Frequency resolution: {freq_resolution:.1f} Hz per bin")
        print(f"- Frequency range: {self.fmin}-{self.fmax} Hz")

        print(f"\nFeature Extraction:")
        print(f"- MFCC coefficients: {self.n_mfcc}")
        print(f"- Mel bands: {self.n_mels}")
        print(f"- Output shape: {self.n_mfcc} x {self.max_pad_len}")


def verify_file_system(file_path):
    """
    Perform basic file system checks on an audio file.
    Returns boolean indicating if all checks passed.
    """
    try:
        exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path)
        readable = os.access(file_path, os.R_OK)
        size = os.path.getsize(file_path) if exists and is_file else 0

        return exists and is_file and readable and size > 0
    except Exception:
        return False


def verify_wav_file(file_path):
    """
    Verify WAV file format and properties.
    Returns boolean indicating if verification passed.
    """
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
    """
    Extract MFCC features from an audio file using optimized parameters.
    Returns the features if successful, None if failed.
    """
    # Quick validation checks
    if not verify_file_system(file_path) or not verify_wav_file(file_path):
        return None

    try:
        # Load and process audio with explicit parameters
        audio, _ = librosa.load(
            file_path,
            sr=params.sample_rate,
            duration=params.duration,
            mono=params.mono,
            res_type=params.res_type
        )

        # Extract MFCCs with explicit parameters
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

        # Pad or truncate features
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
    except Exception:
        return None


def process_audio_files(data_dir, params):
    """
    Process all audio files in the data directory with progress bar and optimized parameters.
    """
    # Get files and labels
    all_files = []
    labels = []

    print("\nScanning directory structure...")
    diagnosis_folders = [d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))]

    print(f"Found diagnosis folders: {diagnosis_folders}")

    # Collect all files first
    for diagnosis in diagnosis_folders:
        diagnosis_path = os.path.join(data_dir, diagnosis)
        for root, _, files in os.walk(diagnosis_path):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                all_files.append(os.path.join(root, file))
                labels.append(diagnosis)

    print(f"\nFound {len(all_files)} total files across {len(diagnosis_folders)} conditions")

    # Process files with progress bar
    features = []
    successful_labels = []

    print("\nProcessing audio files...")
    for file_path, label in tqdm(zip(all_files, labels), total=len(all_files),
                                 desc="Extracting features", unit="file"):
        mfccs = extract_features(file_path, params)
        if mfccs is not None:
            features.append(mfccs)
            successful_labels.append(label)

    # Print summary
    print("\nProcessing Summary")
    print(f"Total files found: {len(all_files)}")
    print(f"Successfully processed: {len(features)}")
    print(f"Failed to process: {len(all_files) - len(features)}")
    print("\nSuccess rate by diagnosis:")

    for diagnosis in diagnosis_folders:
        total = labels.count(diagnosis)
        successful = successful_labels.count(diagnosis)
        print(f"  {diagnosis}: {successful}/{total} ({successful / total * 100:.1f}%)")

    return np.array(features), np.array(successful_labels)


# Main execution
if __name__ == "__main__":
    print("Starting audio processing...")

    # Initialize processing parameters
    params = AudioParameters()
    params.print_analysis_details()

    # Verify paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, "clean")

    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Process files with optimized parameters
    features, labels = process_audio_files(data_dir, params)

    if len(features) == 0:
        print("\nNo files were successfully processed.")
        sys.exit(1)

    print("\nProcessing completed successfully!")
    print(f"Final dataset shape: {features.shape}")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from librosa.display import specshow


def print_mfcc_numerical(mfcc, num_timesteps=5):
    """
    Print MFCC coefficients in a readable numerical format, showing first few timesteps
    """
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    df = pd.DataFrame(mfcc[:, :num_timesteps])
    df.index.name = 'MFCC'
    df.columns.name = 'Time Step'
    print("\nMFCC Coefficients (first {} time steps):".format(num_timesteps))
    print(df)


def plot_mfcc_heatmap(mfcc, title="MFCC Coefficients Heatmap"):
    """
    Create a heatmap visualization of MFCC coefficients
    """
    plt.figure(figsize=(10, 4))
    specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def summarize_mfcc_stats(mfcc):
    """
    Print summary statistics for each MFCC coefficient
    """
    stats = pd.DataFrame({
        'Mean': np.mean(mfcc, axis=1),
        'Std': np.std(mfcc, axis=1),
        'Min': np.min(mfcc, axis=1),
        'Max': np.max(mfcc, axis=1)
    })
    stats.index.name = 'MFCC'
    print("\nMFCC Statistics:")
    print(stats)


# Add these lines at the end of your main execution block:
if len(features) > 0:
    # Print numerical values for the first file
    print("\nShowing details for first audio file:")
    print_mfcc_numerical(features[0])

    # Show summary statistics
    summarize_mfcc_stats(features[0])

    # Create heatmap visualization
    plot_mfcc_heatmap(features[0], f"MFCC Coefficients - {labels[0]}")