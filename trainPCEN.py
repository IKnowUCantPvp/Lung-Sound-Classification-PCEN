import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv2DPCEN
from tqdm import tqdm
from glob import glob
import argparse
import warnings
import librosa
from monitorPCEN import ImprovedPCENMonitor
from models import Conv2DOldPCEN


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_length = int(self.sr * self.dt)  # Samples per audio clip
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # Batch arrays adapted for raw waveforms
        X = np.zeros((self.batch_size, self.input_length, 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            # Load and pad/truncate raw audio waveform
            wav, _ = librosa.load(path, sr=self.sr)
            if len(wav) < self.input_length:
                wav = np.pad(wav, (0, self.input_length - len(wav)))
            else:
                wav = wav[:self.input_length]

            # Add channel dimension (as required by Kapre STFT)
            X[i] = np.expand_dims(wav, axis=-1)  # Shape: (input_length, 1) for STFT
            Y[i] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES': len(os.listdir(args.src_root)),
              'SR': sr,
              'DT': dt}

    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.2,
                                                                  random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(
            len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(
            len(set(label_val)), params['N_CLASSES']))

    # Create data generators
    tg = DataGenerator(wav_train, label_train, sr, dt,
                           params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                           params['N_CLASSES'], batch_size=batch_size)

    model = Conv2DPCEN(**params)

    cp = ModelCheckpoint(
        f'models/{model_type}',  # Remove .h5 extension
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        verbose=1
    )
    csv_logger = CSVLogger(csv_path, append=False)
    pcen_monitor = ImprovedPCENMonitor(log_file=f'logs/{model_type}_pcen_params.csv')

    model.fit(tg, validation_data=vg,
              epochs=30, verbose=1,
              callbacks=[csv_logger, cp, pcen_monitor])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='conv2dpcen',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='wheezecrackle',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=6.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=8000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)

