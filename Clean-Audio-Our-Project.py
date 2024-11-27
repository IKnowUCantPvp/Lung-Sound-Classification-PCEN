import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import wavio
ok
from scipy.sparse import bmat




def downsample_mono(path):
    """
    Convert to mono while preserving original sample rate and data type.
    """
    try:
        obj = wavio.read(path)
        wav = obj.data
        rate = obj.rate

        # Convert to float32 for processing
        wav = wav.astype(np.float32)

        # Convert to mono if needed (average channels)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Normalize to prevent clipping
        if np.max(np.abs(wav)) > 0:
            wav = wav / np.max(np.abs(wav))

        # Convert back to original data type range
        if obj.sampwidth == 1:
            wav = (wav * 127 + 128).astype(np.uint8)
        elif obj.sampwidth == 2:
            wav = (wav * 32767).astype(np.int16)
        elif obj.sampwidth == 3 or obj.sampwidth == 4:
            wav = (wav * 2147483647).astype(np.int32)

        return rate, wav

    except Exception as e:
        print(f"Error processing file {path}: {str(e)}")
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


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time
    overlap_percent = args.overlap

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
            else:
                # Create overlapping windows
                for cnt, i in enumerate(range(0, wav.shape[0] - window_size + 1, hop_size)):
                    start = i
                    stop = i + window_size

                    if stop <= wav.shape[0]:
                        sample = wav[start:stop]
                        save_sample(sample, rate, target_dir, fn, cnt)

                # Handle the last window if there's remaining audio
                if stop < wav.shape[0]:
                    last_start = wav.shape[0] - window_size
                    last_sample = wav[last_start:]
                    save_sample(last_sample, rate, target_dir, fn, cnt + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=5.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--overlap', type=float, default=50.0,
                        help='percentage of overlap between windows (0-100)')
    args, _ = parser.parse_known_args()

    split_wavs(args)