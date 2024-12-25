#this doesn't work for some reason pls fix samer

import tensorflow as tf

# Load your model from the .h5 file
model = tf.keras.models.load_model('conv2dspec.h5')

from trainSPEC import DataGenerator

# Suppose you already have lists of files and labels for validation/test
# For example:
val_generator = DataGenerator(
    trainSPEC.wav_paths_val,
    trainSPEC.labels_val,
    sr=8000,   # or whatever sample rate you're using
    dt=6.0,    # etc.
    n_classes=6,  # or however many classes you have
    batch_size=16
)
