import tensorflow as tf
import numpy as np
from models import Conv2DPCEN
from models import LibrosaPCENLayer

class PCENParameterMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_file='pcen_parameters.csv'):
        """
        Custom callback to monitor PCEN layer parameters during training

        Args:
            log_file (str): Path to save the CSV log of parameters
        """
        super().__init__()
        self.log_file = log_file
        self.parameter_history = {
            'epoch': [],
            'gain': [],
            'bias': [],
            'power': []
        }

    def on_epoch_end(self, epoch, logs=None):
        """
        Collect PCEN layer parameters at the end of each epoch

        Args:
            epoch (int): Current training epoch
            logs (dict): Training metrics for the epoch
        """
        # Find the PCEN layer
        pcen_layer = None
        for layer in self.model.layers:
            if isinstance(layer, LibrosaPCENLayer):
                pcen_layer = layer
                break

        if pcen_layer is not None:
            # Log current parameter values
            self.parameter_history['epoch'].append(epoch)
            self.parameter_history['gain'].append(float(pcen_layer.gain.numpy()))
            self.parameter_history['bias'].append(float(pcen_layer.bias.numpy()))
            self.parameter_history['power'].append(float(pcen_layer.power.numpy()))

    def on_train_end(self, logs=None):
        """
        Save parameter history to a CSV file when training ends

        Args:
            logs (dict): Training metrics
        """
        import pandas as pd

        # Convert parameter history to DataFrame and save
        df = pd.DataFrame(self.parameter_history)
        df.to_csv(self.log_file, index=False)
        print(f"PCEN parameter history saved to {self.log_file}")

# Example usage in training
def train_model():
    # Assuming you have your model and training data ready
    model = Conv2DPCEN()

    # Create the parameter monitor callback
    pcen_monitor = PCENParameterMonitor(log_file='pcen_parameters.csv')


# Visualization option
def plot_pcen_parameters(csv_file='pcen_parameters.csv'):
    """
    Visualize PCEN parameter changes during training

    Args:
        csv_file (str): Path to the CSV file with parameter history
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create subplots for each parameter
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot gain
    ax1.plot(df['epoch'], df['gain'], label='Gain')
    ax1.set_title('PCEN Gain Parameter')
    ax1.set_ylabel('Gain Value')
    ax1.legend()

    # Plot bias
    ax2.plot(df['epoch'], df['bias'], label='Bias', color='orange')
    ax2.set_title('PCEN Bias Parameter')
    ax2.set_ylabel('Bias Value')
    ax2.legend()

    # Plot power
    ax3.plot(df['epoch'], df['power'], label='Power', color='green')
    ax3.set_title('PCEN Power Parameter')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Power Value')
    ax3.legend()

    plt.tight_layout()
    plt.show()