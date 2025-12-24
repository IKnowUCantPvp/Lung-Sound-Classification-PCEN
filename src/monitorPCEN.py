import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PCENLayer import PCENLayer


class ImprovedPCENMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_file='pcen_parameters.csv'):
        super().__init__()
        self.log_file = log_file
        self.parameter_history = {
            'epoch': [],
            'alpha': [],
            'delta': [],
            'root': []
        }

    def on_epoch_end(self, epoch, logs=None):
        pcen_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PCENLayer):
                pcen_layer = layer
                break

        if pcen_layer is not None:
            self.parameter_history['epoch'].append(epoch)
            self.parameter_history['alpha'].append(float(pcen_layer.alpha_kernel.numpy()))
            self.parameter_history['delta'].append(float(pcen_layer.delta_kernel.numpy()))
            self.parameter_history['root'].append(float(pcen_layer.root_kernel.numpy()))

            print(f"\nPCEN Parameters at epoch {epoch}:")
            print(f"Alpha: {self.parameter_history['alpha'][-1]:.4f}")
            print(f"Delta: {self.parameter_history['delta'][-1]:.4f}")
            print(f"Root: {self.parameter_history['root'][-1]:.4f}")

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.parameter_history)
        df.to_csv(self.log_file, index=False)
        print(f"\nPCEN parameter history saved to {self.log_file}")

    def plot_parameters(self, save_path=None):
        """Plot parameter changes during training"""
        df = pd.DataFrame(self.parameter_history)

        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

        params = [
            ('alpha', 'Alpha Parameter (Smoothing)', 'b-'),
            ('smooth_coef', 'Smooth Coefficient', 'm-'),
            ('delta', 'Delta Parameter (Bias)', 'r-'),
            ('root', 'Root Parameter', 'g-')
        ]

        for (param, title, color), ax in zip(params, axes):
            ax.plot(df['epoch'], df[param], color, label=param.title())
            ax.set_title(f'PCEN {title}')
            ax.set_ylabel(f'{param.title()} Value')
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel('Epoch')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()