import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from models import ImprovedPCENLayer


class ImprovedPCENMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_file='improved_pcen_parameters.csv'):
        """
        Custom callback to monitor ImprovedPCENLayer parameters during training

        Args:
            log_file (str): Path to save the CSV log of parameters
        """
        super().__init__()
        self.log_file = log_file
        self.parameter_history = {
            'epoch': [],
            'alpha': [],
            'delta': [],
            'root': []
        }

    def on_epoch_end(self, epoch, logs=None):
        """
        Collect ImprovedPCENLayer parameters at the end of each epoch

        Args:
            epoch (int): Current training epoch
            logs (dict): Training metrics for the epoch
        """
        # Find the ImprovedPCENLayer
        pcen_layer = None
        for layer in self.model.layers:
            if isinstance(layer, ImprovedPCENLayer):
                pcen_layer = layer
                break

        if pcen_layer is not None:
            # Log current parameter values
            self.parameter_history['epoch'].append(epoch)
            self.parameter_history['alpha'].append(float(pcen_layer.alpha_var.numpy()))
            self.parameter_history['delta'].append(float(pcen_layer.delta_var.numpy()))
            self.parameter_history['root'].append(float(pcen_layer.root_var.numpy()))

            # Print current values for monitoring during training
            print(f"\nPCEN Parameters at epoch {epoch}:")
            print(f"Alpha: {self.parameter_history['alpha'][-1]:.4f}")
            print(f"Delta: {self.parameter_history['delta'][-1]:.4f}")
            print(f"Root: {self.parameter_history['root'][-1]:.4f}")

    def on_train_end(self, logs=None):
        """
        Save parameter history to a CSV file when training ends

        Args:
            logs (dict): Training metrics
        """
        # Convert parameter history to DataFrame and save
        df = pd.DataFrame(self.parameter_history)
        df.to_csv(self.log_file, index=False)
        print(f"\nPCEN parameter history saved to {self.log_file}")

    def plot_parameters(self, save_path=None):
        """
        Visualize PCEN parameter changes during training

        Args:
            save_path (str, optional): Path to save the plot. If None, displays the plot
        """
        df = pd.DataFrame(self.parameter_history)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot alpha parameter
        ax1.plot(df['epoch'], df['alpha'], 'b-', label='Alpha')
        ax1.set_title('PCEN Alpha Parameter (Smoothing)')
        ax1.set_ylabel('Alpha Value')
        ax1.grid(True)
        ax1.legend()

        # Plot delta parameter
        ax2.plot(df['epoch'], df['delta'], 'r-', label='Delta')
        ax2.set_title('PCEN Delta Parameter (Bias)')
        ax2.set_ylabel('Delta Value')
        ax2.grid(True)
        ax2.legend()

        # Plot root parameter
        ax3.plot(df['epoch'], df['root'], 'g-', label='Root')
        ax3.set_title('PCEN Root Parameter')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Root Value')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()