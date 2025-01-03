from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os
from PCENLayer import PCENLayer


def Conv1D(N_CLASSES=10, SR=16000, DT=1.0):
    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
    x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
    x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
    x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
    x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
    x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
    x = layers.Dropout(rate=0.1, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='1d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def Conv2DSpec(N_CLASSES=6, SR=8000, DT=6.0):
    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def Conv2DOldPCEN(N_CLASSES=6, SR=8000, DT=6.0):
    input_shape = (int(SR * DT), 1)

    # Get the base mel spectrogram layer without decibel conversion
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=False,  # No dB conversion before PCEN
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')

    # Apply PCEN
    x = PCENLayer(
        alpha=0.98,  # Higher initial smoothing for lung sounds
        delta=2.0,  # Lower initial bias for subtle variations
        root=0.4,  # Lower root for better noise handling
        smooth_coef=0.05,  # Higher smoothing for temporal features
        floor=1e-7,  # Small constant for numerical stability
        trainable=False,
        name='pcen'
    )(i.output)

    # Keep LayerNorm after PCEN
    x = LayerNormalization(axis=2, name='batch_norm')(x)

    # Rest of the model remains exactly the same
    x = layers.Conv2D(8, kernel_size=(7, 7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

    model = Model(inputs=i.input, outputs=o, name='2d_convolution_pcen')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model





def Conv2DMFCC(input_shape, n_classes=6):
    """
    Create CNN model for MFCC features with exactly the same architecture as Conv2DSpec
    """
    inputs = layers.Input(shape=(*input_shape, 1))

    # Exactly matching Conv2DSpec layers
    x = LayerNormalization(axis=2, name='batch_norm')(inputs)
    x = layers.Conv2D(8, kernel_size=(7, 7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)

    x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)

    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    # Create and compile model with same settings
    model = Model(inputs=inputs, outputs=outputs, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
import librosa
import numpy as np


def Conv2DPCEN(N_CLASSES=6, SR=8000, DT=6.0):
    """
    Modified Conv2DPCEN model to match the Conv2DSpec input format
    Using get_melspectrogram_layer like Conv2DSpec
    """
    input_shape = (int(SR * DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')

    x = ImprovedPCENLayer()(i.output)
    x = LayerNormalization(axis=-1, name='batch_norm')(x)
    x = layers.Conv2D(8, kernel_size=(7, 7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

    model = Model(inputs=i.input, outputs=outputs, name='2d_convolution_pcen')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


class ImprovedPCENLayer(tf.keras.layers.Layer):
    def __init__(self,
                 alpha=0.95,  # Higher initial smoothing for lung sounds
                 delta=1.5,  # Lower initial bias for subtle variations
                 root=0.3,  # Lower root for better noise handling
                 smooth_coef=0.04,  # Higher smoothing for temporal features
                 floor=1e-6,  # Small constant for numerical stability
                 trainable=True,
                 **kwargs):
        super(ImprovedPCENLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.delta = delta
        self.root = root
        self.smooth_coef = smooth_coef
        self.floor = floor
        self.trainable = trainable

    def build(self, input_shape):
        # Initialize trainable parameters with lung-sound-specific constraints
        self.alpha_var = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.alpha),
            # Higher range for alpha to capture longer temporal dependencies
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.9, max_value=0.99),
            trainable=self.trainable
        )

        self.delta_var = self.add_weight(
            name='delta',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.delta),
            # Wider range for delta to handle varying breath intensities
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.8, max_value=3.0),
            trainable=self.trainable
        )

        self.root_var = self.add_weight(
            name='root',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.root),
            # Lower range for root to preserve subtle features
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.2, max_value=0.7),
            trainable=self.trainable
        )

    def call(self, inputs):
        # Ensure non-negative input
        inputs = tf.maximum(inputs, self.floor)

        # Enhanced temporal integration for lung sounds
        frames = tf.shape(inputs)[1]
        # Longer decay kernel for respiratory cycles
        decay_kernel = tf.pow(self.alpha_var, tf.range(frames, dtype=tf.float32))
        decay_kernel = decay_kernel / tf.reduce_sum(decay_kernel)
        decay_kernel = tf.reshape(decay_kernel, [1, -1, 1, 1])

        # Apply smoothing using conv2d
        smoothed = tf.nn.conv2d(
            inputs,
            decay_kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        smoothed = tf.maximum(smoothed, self.floor)

        # PCEN computation with adjusted parameters for lung sounds
        smooth_factor = tf.pow(smoothed, -self.alpha_var)
        inner_term = inputs * smooth_factor + self.delta_var
        pcen = tf.pow(inner_term, self.root_var) - tf.pow(self.delta_var, self.root_var)

        return pcen

    def get_config(self):
        config = super(ImprovedPCENLayer, self).get_config()
        config.update({
            'alpha': float(self.alpha),
            'delta': float(self.delta),
            'root': float(self.root),
            'smooth_coef': float(self.smooth_coef),
            'floor': float(self.floor),
            'trainable': self.trainable
        })
        return config

def LSTM(N_CLASSES=10, SR=16000, DT=1.0):
    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                     n_mels=128,
                                     pad_end=True,
                                     n_fft=512,
                                     win_length=400,
                                     hop_length=160,
                                     sample_rate=SR,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_last',
                                     name='2d_convolution')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
    s = TimeDistributed(layers.Dense(64, activation='tanh'),
                        name='td_dense_tanh')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                             name='bidirectional_lstm')(s)
    x = layers.concatenate([s, x], axis=2, name='skip_connection')
    x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
    x = layers.MaxPooling1D(name='max_pool_1d')(x)
    x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(32, activation='relu',
                         activity_regularizer=l2(0.001),
                         name='dense_3_relu')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model