import tensorflow as tf
from tensorflow.keras import layers


class PCENLayer(layers.Layer):
    def __init__(self,
                 alpha=0.98,
                 smooth_coef=0.03,
                 delta=2.0,
                 root=0.5,
                 floor=1e-6,  # Small numerical stability constant
                 trainable=False,
                 **kwargs):
        super(PCENLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.smooth_coef = smooth_coef
        self.delta = delta
        self.root = root
        self.floor = floor  # Store the floor constant
        self.trainable = trainable

    def build(self, input_shape):
        # input_shape will be (batch, time, mel, channel)
        self.alpha_kernel = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.constant_initializer(self.alpha),
            trainable=self.trainable
        )

        self.smooth_coef_kernel = self.add_weight(
            name='smooth_coef',
            shape=(1,),
            initializer=tf.constant_initializer(self.smooth_coef),
            trainable=self.trainable
        )

        self.delta_kernel = self.add_weight(
            name='delta',
            shape=(1,),
            initializer=tf.constant_initializer(self.delta),
            trainable=self.trainable
        )

        self.root_kernel = self.add_weight(
            name='root',
            shape=(1,),
            initializer=tf.constant_initializer(self.root),
            trainable=self.trainable
        )

    def call(self, inputs):
        # inputs shape: (batch, time, mel, channel)
        eps = self.floor  # Use floor as epsilon

        # Using tf.scan for the IIR filter
        def step(previous, current):
            # previous and current have shape (batch, mel, channel)
            return (1 - self.smooth_coef_kernel) * current + \
                self.smooth_coef_kernel * previous

        # Transpose to move time axis first for scanning
        x_transposed = tf.transpose(inputs, [1, 0, 2, 3])  # (time, batch, mel, channel)

        # Get initial state of zeros
        init_state = tf.zeros_like(x_transposed[0])  # (batch, mel, channel)

        # Scan through time
        smoothed = tf.scan(
            step,
            x_transposed,
            initializer=init_state
        )

        # Transpose back to (batch, time, mel, channel)
        smoothed = tf.transpose(smoothed, [1, 0, 2, 3])

        # PCEN transformation
        pcen = (inputs / (eps + smoothed) ** self.alpha_kernel + self.delta_kernel) ** self.root_kernel - \
               self.delta_kernel ** self.root_kernel

        return pcen  # Shape: (batch, time, mel, channel)

    def get_config(self):
        config = super(PCENLayer, self).get_config()
        config.update({
            'alpha': float(self.alpha_kernel.numpy()),
            'smooth_coef': float(self.smooth_coef_kernel.numpy()),
            'delta': float(self.delta_kernel.numpy()),
            'root': float(self.root_kernel.numpy()),
            'floor': self.floor,  # Add floor to the configuration
            'trainable': self.trainable
        })
        return config
