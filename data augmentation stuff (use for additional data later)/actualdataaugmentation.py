import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import wavio
from scipy.sparse import bmat
import librosa
import librosa.display


def augment_audio(wav, rate, augmentations):
    """Apply audio augmentations and return augmented samples"""
    augmented_samples = []

    # Ensure the input wav is in floating-point format and normalized
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))

    for aug in augmentations:
        try:
            if aug == "pitch_shift":
                aug_sample = librosa.effects.pitch_shift(wav, sr=rate, n_steps=2)
            elif aug == "time_stretch":
                # Apply time stretching (ensure no invalid stretch rates)
                stretch_rate = 1.2  # Example: stretch by 20%
                if stretch_rate > 0:
                    aug_sample = librosa.effects.time_stretch(wav, stretch_rate)
                else:
                    raise ValueError(f"Invalid stretch rate: {stretch_rate}")
            elif aug == "add_noise":
                noise = np.random.normal(0, 0.005, wav.shape)
                aug_sample = wav + noise
            elif aug == "reverse":
                aug_sample = wav[::-1]

            # Normalize augmented sample
            if np.max(np.abs(aug_sample)) > 0:
                aug_sample = aug_sample / np.max(np.abs(aug_sample))

            augmented_samples.append(aug_sample)

        except Exception as e:
            print(f"Augmentation {aug} failed: {str(e)}")

    return augmented_samples


def downsample_mono(file_path, target_sr=16000):
    """Load an audio file, convert to mono and downsample to target sample rate."""
    try:
        wav, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return sr, wav
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def save_sample(sample, rate, target_dir, fn, ix):
    """Save audio segment with proper data type handling"""
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir, fn + '_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return

    # Ensure the data type is appropriate for WAV
    if sample.dtype == np.float32 or sample.dtype == np.float64:
        sample = np.int16(sample * 32767)

    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)


def augment_audio(wav, rate, augmentations):
    """Apply audio augmentations and return augmented samples"""
    augmented_samples = []

    # Ensure the input wav is in floating-point format and normalized
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))

    for aug in augmentations:
        try:
            if aug == "pitch_shift":
                aug_sample = librosa.effects.pitch_shift(wav, sr=rate, n_steps=2)
            elif aug == "time_stretch":
                aug_sample = librosa.effects.time_stretch(wav, 1.2)
            elif aug == "add_noise":
                noise = np.random.normal(0, 0.005, wav.shape)
                aug_sample = wav + noise
            elif aug == "reverse":
                aug_sample = wav[::-1]

            # Normalize augmented sample
            if np.max(np.abs(aug_sample)) > 0:
                aug_sample = aug_sample / np.max(np.abs(aug_sample))

            augmented_samples.append(aug_sample)

        except Exception as e:
            print(f"Augmentation {aug} failed: {str(e)}")

    return augmented_samples




def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time
    overlap_percent = args.overlap

    # Define augmentations
    augmentations = ["pitch_shift", "time_stretch", "add_noise", "reverse"]

    wav_paths = glob(os.path.join(src_root, '**', '*.wav'), recursive=True)
    check_dir(dst_root)

    # Get immediate subdirectories only
    classes = [d for d in os.listdir(src_root)
               if os.path.isdir(os.path.join(src_root, d))]

    for _cls in classes:
        src_dir = os.path.join(src_root, _cls)
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)

        wav_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]

        for fn in tqdm(wav_files):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn)

            if rate is None or wav is None:
                print(f"Skipping file {src_fn} due to processing error")
                continue

            # Calculate window and hop size with overlap
            window_size = int(dt * rate)
            hop_size = int(window_size * (1 - overlap_percent / 100))

            if wav.shape[0] < window_size:
                # If audio is shorter than window_size, pad with zeros
                sample = np.zeros(shape=(window_size,), dtype=wav.dtype)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)

                # Augment and save padded sample
                augmented_samples = augment_audio(sample, rate, augmentations)
                for aug_cnt, aug_sample in enumerate(augmented_samples):
                    aug_name = augmentations[aug_cnt]  # Use the augmentation name
                    save_sample(aug_sample, rate, target_dir, f"{fn}_aug_{aug_name}", 0)
            else:
                # Create overlapping windows
                for cnt, i in enumerate(range(0, wav.shape[0] - window_size + 1, hop_size)):
                    start = i
                    stop = i + window_size

                    if stop <= wav.shape[0]:
                        sample = wav[start:stop]
                        save_sample(sample, rate, target_dir, fn, cnt)

                        # Augment and save each segment
                        augmented_samples = augment_audio(sample, rate, augmentations)
                        for aug_cnt, aug_sample in enumerate(augmented_samples):
                            aug_name = augmentations[aug_cnt]
                            save_sample(aug_sample, rate, target_dir, f"{fn}_aug_{aug_name}_{cnt}", 0)

                # Handle the last window if there's remaining audio
                if stop < wav.shape[0]:
                    last_start = wav.shape[0] - window_size
                    last_sample = wav[last_start:]
                    save_sample(last_sample, rate, target_dir, fn, cnt + 1)

                    # Augment and save the last sample
                    augmented_samples = augment_audio(last_sample, rate, augmentations)
                    for aug_cnt, aug_sample in enumerate(augmented_samples):
                        aug_name = augmentations[aug_cnt]
                        save_sample(aug_sample, rate, target_dir, f"{fn}_aug_{aug_name}_{cnt + 1}", 0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting audio data with augmentation')
    parser.add_argument('--src_root', type=str, default='cleandownsamplemono',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='cleanaug',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=6.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--overlap', type=float, default=50.0,
                        help='percentage of overlap between windows (0-100)')
    args, _ = parser.parse_known_args()

    split_wavs(args)