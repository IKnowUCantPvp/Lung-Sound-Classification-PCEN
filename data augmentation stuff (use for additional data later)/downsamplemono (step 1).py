import os
import argparse
import numpy as np
from scipy.io import wavfile
from librosa import resample as librosa_resample
from librosa.core import to_mono
import wavio


def downsample_mono(input_path, target_sr):
    """
    Convert audio to mono, downsample it to the target sample rate, and convert to np.float32.
    Args:
        input_path (str): Path to the input WAV file.
        target_sr (int): Target sample rate for resampling.
    Returns:
        target_sr (int): The target sample rate.
        wav (np.ndarray): The processed audio signal in mono and np.float32 format.
    """
    try:
        # Read WAV file
        obj = wavio.read(input_path)
        wav = obj.data.astype(np.float32)
        original_sr = obj.rate

        # Convert to mono if necessary
        if wav.ndim > 1:  # Stereo or multi-channel
            wav = to_mono(wav.T)  # Convert to mono

        # Resample to the target sample rate
        if original_sr != target_sr:
            wav = librosa_resample(y=wav, orig_sr=original_sr, target_sr=target_sr)

        # Normalize to -1.0 to 1.0
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val

        return target_sr, wav

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None, None


def process_audio_directory(input_dir, output_dir, target_sr):
    """
    Process all WAV files in a directory: downsample to mono and save as float32.
    Args:
        input_dir (str): Directory containing input WAV files.
        output_dir (str): Directory to save processed WAV files.
        target_sr (int): Target sample rate for downsampling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Calculate the relative path from the input directory
        relative_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(output_dir, relative_path)

        # Create any necessary output subdirectories
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_root, file)

                # Process the file
                target_sr, wav = downsample_mono(input_path, target_sr)

                if wav is not None:
                    # Save the processed file
                    wavfile.write(output_path, target_sr, wav.astype(np.float32))
                    print(f"Processed and saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample WAV files to mono and np.float32 format")
    parser.add_argument('--input_dir', type=str,
                        default="C:\\Users\\natha\\OneDrive\\Documents\\GitHub\\Audio-Classification\\wavfiles",
                        help='Directory containing input WAV files')
    parser.add_argument('--output_dir', type=str,
                        default="C:\\Users\\natha\\OneDrive\\Documents\\GitHub\\Audio-Classification\\cleantest",
                        help='Directory to save processed WAV files')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='Target sample rate for downsampling (default: 16000)')
    args = parser.parse_args()

    process_audio_directory(args.input_dir, args.output_dir, args.target_sr)

