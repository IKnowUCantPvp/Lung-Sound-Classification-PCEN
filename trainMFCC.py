import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv2DMFCC
import argparse
import warnings


def train(args):
    """
    Train the model using pre-computed MFCC features
    """
    # Load features first to get input shape
    features = np.load(os.path.join(args.processed_dir, 'features.npy'))
    label_array = np.load(os.path.join(args.processed_dir, 'labels.npy'))

    # Add channel dimension for Conv2D
    features = features[..., np.newaxis]

    # Create model type dict with correct parameter names
    params = {
        'input_shape': (features.shape[1], features.shape[2]),
        'n_classes': len(os.listdir(args.src_root))
    }

    models = {
        'mfcc_cnn': Conv2DMFCC(**params)
    }

    model_type = args.model_type
    assert model_type in models.keys(), f'{model_type} not an available model'

    # Rest of your existing training code remains the same...
    csv_path = os.path.join('logs', f'{model_type}_history.csv')

    # Prepare labels
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = le.transform(label_array)

    # Train/validation split
    features_train, features_val, label_train, label_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=0
    )

    # Convert labels to categorical
    label_train_cat = to_categorical(label_train, num_classes=params['n_classes'])
    label_val_cat = to_categorical(label_val, num_classes=params['n_classes'])

    # Validation checks
    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['n_classes']:
        warnings.warn(f'Found {len(set(label_train))}/{params["n_classes"]} classes in training data.')
    if len(set(label_val)) != params['n_classes']:
        warnings.warn(f'Found {len(set(label_val))}/{params["n_classes"]} classes in validation data.')

    # Get model
    model = models[model_type]

    # Set up callbacks
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

    # Train model
    model.fit(
        features_train,
        label_train_cat,
        validation_data=(features_val, label_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[csv_logger, cp],
        verbose=1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFCC Classification Training')
    parser.add_argument('--model_type', type=str, default='mfcc_cnn',
                        help='model to run: mfcc_cnn (larger dataset)')
    parser.add_argument('--src_root', type=str, default='unaugmented8khz',
                        help='directory of original audio files')
    parser.add_argument('--processed_dir', type=str, default='2D-MFCC',
                        help='directory containing processed MFCC features')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train')
    args, _ = parser.parse_known_args()

    train(args)