from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os


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

    x = LibrosaPCENLayer()(i.output)
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


class LibrosaPCENLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LibrosaPCENLayer, self).__init__(**kwargs)
        self.init_gain = 0.98
        self.init_bias = 2.0
        self.init_power = 0.5
        self.eps = 1e-6

    def build(self, input_shape):
        self.gain = self.add_weight(
            name='gain',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_gain),
            trainable=True
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_bias),
            trainable=True
        )

        self.power = self.add_weight(
            name='power',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_power),
            trainable=True
        )

    def call(self, inputs):
        # Normalize inputs to prevent extreme values
        inputs = tf.nn.relu(inputs)  # Ensure non-negative
        inputs = inputs + self.eps

        # Fixed kernel for smoothing
        window_size = 3
        kernel = tf.ones([1, window_size, 1, 1], dtype=inputs.dtype) / window_size

        # Smoothing
        smoothed = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        smoothed = tf.maximum(smoothed, self.eps)  # Ensure strictly positive

        # Safe power operation for denominator
        denominator = tf.where(
            smoothed > 0,
            tf.pow(smoothed, tf.clip_by_value(self.gain, 0.1, 0.999)),
            tf.ones_like(smoothed)
        )

        # PCEN computation
        ratio = inputs / denominator
        inner = ratio + self.bias

        # Safe power operation for output
        pcen = tf.pow(inner, tf.clip_by_value(self.power, 0.1, 0.999)) - tf.pow(self.bias, self.power)

        return pcen

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

