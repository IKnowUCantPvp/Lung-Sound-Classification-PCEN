import os
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


def bandstop_filter(data, sr, low_cutoff, high_cutoff):
    """
    Apply a bandstop filter to the audio data.
    """
    nyquist = 0.5 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(2, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


def undo_filter(audio_path, save_path, filter_type):
    """
    Undo the specified filter type and save the processed file.
    """
    # Load audio file
    data, sr = librosa.load(audio_path, sr=None)

    # Undo filtering based on filter type
    if filter_type == 'D':  # DiaphragmWFilter (WITH FILTER STILL APPLIED) Mode: [100-500 Hz emphasis]
        data = bandstop_filter(data, sr, low_cutoff=100, high_cutoff=500)
    elif filter_type == 'B':  # Bell Mode: [20-200 Hz emphasis]
        data = bandstop_filter(data, sr, low_cutoff=20, high_cutoff=200)
    elif filter_type == 'E':  # Extended Mode: [50-500 Hz emphasis]
        data = bandstop_filter(data, sr, low_cutoff=50, high_cutoff=500)
    else:
        print(f"Unknown filter type: {filter_type}")
        return

    # Save the processed audio
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write(save_path, data, sr)
    print(f"Processed: {audio_path} -> {save_path}")


def process_directory(input_dir, output_dir, filter_type):
    """
    Process all .wav files in a directory tree.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                # Construct the output path, preserving the directory structure
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                # Undo the filter and save the file
                undo_filter(input_path, output_path, filter_type)


# Main script
if __name__ == "__main__":
    # Define the source and output directories
    input_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\cleandownsampleMore\Diaphragm"  # Replace with source directory
    output_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\DiaphragmToNormal"  # Replace with destination directory

    # Specify the filter type for the folder ('D', 'B', or 'E')
    filter_type = 'D'  # Change as needed

    # Process the directory
    process_directory(input_directory, output_directory, filter_type)
    print("All files processed successfully.")
