import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_spectrograms(audio_path, output_dir, sr=8000, n_mels=128,
                          hop_length=160, n_fft=512):
    """
    Generate and save STFT spectrogram, mel spectrogram, and PCEN spectrogram for a single audio file

    Parameters:
        audio_path (str): Path to audio file
        output_dir (str): Directory to save spectrograms
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        hop_length (int): Number of samples between successive frames
        n_fft (int): Length of FFT window
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load audio file
    y, _ = librosa.load(audio_path, sr=sr)

    # Get filename without extension
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    # 1. Generate regular STFT spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        D_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT Spectrogram - {filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_stft.png'))
    plt.close()

    # 2. Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_melspec.png'))
    plt.close()

    # 3. Generate PCEN spectrogram with optimized parameters
    pcen = librosa.pcen(
        mel_spec,
        sr=sr,
        gain=0.8,  # Higher initial smoothing for lung sounds
        bias=5.5,  # Lower initial bias for subtle variations
        power=0.7,  # Lower root for better noise handling
        time_constant=0.1,  # Higher smoothing for temporal features
        eps=1e-8  # Small constant for numerical stability
    )

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        pcen,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        norm=plt.Normalize(vmin=0, vmax=np.percentile(pcen, 95))
    )
    plt.colorbar(img, format='%.2f')
    plt.title(f'PCEN Spectrogram - {filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_pcen.png'), dpi=300)
    plt.close()

# /Users/samer/PycharmProjects/Lung-sounds-isef/clean_large/Bronchiectasis/111_1b2_Tc_sc_Meditron_original_2.wav
if __name__ == "__main__":
    # Set your paths here
    audio_file = "/Users/samer/PycharmProjects/Lung-sounds-isef/clean_large/COPD/104_1b1_Al_sc_Litt3200_original_2.wav"  # Change this to your WAV file path
    output_directory = "images"  # Where to save the spectrograms

    print(f"Processing file: {audio_file}")
    generate_spectrograms(audio_file, output_directory)
    print("Finished processing!")