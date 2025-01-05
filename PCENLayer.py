import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PCENLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, pcen_lr_multiplier=10.0):
        super(PCENLearningRateScheduler, self).__init__()
        self.pcen_lr_multiplier = pcen_lr_multiplier

    def on_train_batch_begin(self, batch, logs=None):
        # Find PCEN layer
        pcen_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PCENLayer):
                pcen_layer = layer
                break

        if pcen_layer is not None:
            # Get current learning rate
            current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

            # Set higher learning rate for PCEN layer weights
            pcen_weights = pcen_layer.trainable_weights
            for w in pcen_weights:
                w._learning_rate_multiplier = self.pcen_lr_multiplier


class PCENLayer(layers.Layer):
    def __init__(self,
                 alpha=0.98,
                 delta=2.0,
                 root=0.5,
                 trainable=True,
                 trainable_params=None,
                 name=None):
        super(PCENLayer, self).__init__(name=name)
        self.init_alpha = alpha
        self.init_delta = delta
        self.init_root = root
        self.trainable = trainable
        self.trainable_params = trainable_params or []
        self.last_print_time = 0  # Track last print time

    def build(self, input_shape):
        # Initialize alpha
        self.alpha_kernel = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.random_uniform_initializer(
                self.init_alpha * 0.9,
                self.init_alpha * 1.1
            ),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.5, max_value=1.0),
            trainable=self.trainable and ('alpha' in self.trainable_params if self.trainable_params else self.trainable)
        )

        # Fixed smooth_coef as a non-trainable weight
        self.smooth_coef = self.add_weight(
            name='smooth_coef',
            shape=(1,),
            initializer=tf.constant_initializer(0.04),  # Fixed value from Conv2DOldPCEN
            trainable=False
        )

        # Initialize delta
        self.delta_kernel = self.add_weight(
            name='delta',
            shape=(1,),
            initializer=tf.random_uniform_initializer(
                self.init_delta * 0.9,
                self.init_delta * 1.1
            ),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=1.0, max_value=3.0),
            trainable=self.trainable and ('delta' in self.trainable_params if self.trainable_params else self.trainable)
        )

        # Initialize root
        self.root_kernel = self.add_weight(
            name='root',
            shape=(1,),
            initializer=tf.random_uniform_initializer(
                self.init_root * 0.9,
                self.init_root * 1.1
            ),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.3, max_value=0.7),
            trainable=self.trainable and ('root' in self.trainable_params if self.trainable_params else self.trainable)
        )

    def call(self, inputs):
        eps = 1e-6

        def step(previous, current):
            return (1 - self.smooth_coef) * current + self.smooth_coef * previous

        x_transposed = tf.transpose(inputs, [1, 0, 2, 3])
        init_state = tf.zeros_like(x_transposed[0])
        smoothed = tf.scan(step, x_transposed, initializer=init_state)
        smoothed = tf.transpose(smoothed, [1, 0, 2, 3])

        # PCEN formula
        pcen = (inputs / (eps + smoothed) ** self.alpha_kernel + self.delta_kernel) ** self.root_kernel - \
               self.delta_kernel ** self.root_kernel

        return pcen



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