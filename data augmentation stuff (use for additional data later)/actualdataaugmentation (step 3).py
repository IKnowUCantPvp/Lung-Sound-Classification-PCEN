import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm  # For progress bar

# Configuration
ORIGINAL_DIR = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\DiaphragmToNormal (NORMAL FILES)"
OUTPUT_DIR = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\Step 3 (NORMAL FILES)"
SAMPLE_RATE = 8000  # Target sample rate for all files

# Augmentation functions
def pitch_shift(audio, sr, n_steps):
    """Apply pitch shift to the audio."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate):
    """Apply time stretching to the audio."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def add_noise(audio, noise_level=0.01):
    """Add Gaussian noise to the audio."""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_level * noise
    return augmented / np.max(np.abs(augmented) + 1e-6)  # Normalize

def apply_gain(audio, gain_factor):
    """Apply volume gain to the audio."""
    return np.clip(audio * gain_factor, -1, 1)

# Helper functions
def save_audio(audio, sr, output_dir, file_name):
    """Save the augmented audio to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    sf.write(file_path, audio, sr)

def process_files(original_dir, output_dir, sample_rate):
    """Process all audio files by applying augmentations and saving them."""
    augmentations = {
        "pitch_shift_up": lambda audio, sr: pitch_shift(audio, sr, n_steps=0.5),
        "pitch_shift_down": lambda audio, sr: pitch_shift(audio, sr, n_steps=-0.5),
        "time_stretch_fast": lambda audio, sr: time_stretch(audio, rate=1.05),
        "time_stretch_slow": lambda audio, sr: time_stretch(audio, rate=0.95),
        "add_noise": lambda audio, sr: add_noise(audio, noise_level=0.005),
        "apply_gain": lambda audio, sr: apply_gain(audio, gain_factor=1.05),
    }

    # Walk through the original directory and process files
    for root, _, files in os.walk(original_dir):
        # Extract the class label from the folder structure
        label = Path(root).name

        # Initialize progress bar
        files = [f for f in files if f.endswith(".wav")]
        with tqdm(total=len(files), desc=f"Processing {label}") as pbar:
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Load the original audio file
                    audio, sr = librosa.load(file_path, sr=sample_rate)

                    # Save the original audio to the output directory under its class label
                    original_output_dir = os.path.join(output_dir, "original", label)
                    save_audio(audio, sr, original_output_dir, file)

                    # Apply each augmentation and save the augmented audio
                    for aug_name, aug_func in augmentations.items():
                        augmented_audio = aug_func(audio, sr)
                        aug_output_dir = os.path.join(output_dir, aug_name, label)
                        # Use the original file name without appending the augmentation type
                        save_audio(augmented_audio, sr, aug_output_dir, file)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

                # Update progress bar
                pbar.update(1)


# Run the process
if __name__ == "__main__":
    process_files(ORIGINAL_DIR, OUTPUT_DIR, SAMPLE_RATE)
