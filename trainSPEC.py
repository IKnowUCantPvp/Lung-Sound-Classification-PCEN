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
from models import Conv2DSpec
from tqdm import tqdm
from glob import glob
import argparse
import warnings


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size, shuffle=True, augmentation=False, ):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size)) #total number of batches


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #indexes is equal to where you are in the batch.

        wav_paths = [self.wav_paths[k] for k in indexes] #gets the associated wav paths and labels from the indexes that are being used
        labels = [self.labels[k] for k in indexes]

        # make an empty dataset for both X and Y
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)


        for i, (path, label) in enumerate(zip(wav_paths, labels)): #for loop that gets the path and label from the wav_paths and labels
            rate, wav = wavfile.read(path)                         #extracts the rate and wav as in sr*d (8000*6)
            X[i,] = wav.reshape(-1, 1)                             #puts the wav (sr*d) into the batch index and makes the channel 1
            Y[i,] = to_categorical(label, num_classes=self.n_classes) #puts the label into onehotencoded format)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))  #this shuffles the original indexes if you have shuffle on.
        if self.shuffle:                               #it basically resets the entire indexes then shuffles them
            np.random.shuffle(self.indexes)



def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES':len(os.listdir(args.src_root)),
              'SR':sr,
              'DT':dt}
    models = {
              'Conv2DSpec':Conv2DSpec(**params),
              }
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels) #puts the labels into number format)

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.2,
                                                                  random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    model = models[model_type]
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
    model.fit(tg, validation_data=vg,
              epochs=50, verbose=2,
              callbacks=[csv_logger, cp])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='Conv2DSpec',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default=r'C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\CurrentDatasets\UnaugmentedDatasets (10 classes)\UnaugmentedTrainDataset',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16    ,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=6.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=8000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)


