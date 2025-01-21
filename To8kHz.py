import os
import librosa
import soundfile as sf


def resample_wav_files_to_8k(input_dir, output_dir):
    """
    Recursively resample all WAV files in `input_dir` to 8 kHz (mono)
    and save them to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                input_path = os.path.join(root, filename)

                # Recreate subdirectory structure in output_dir
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                output_path = os.path.join(target_dir, filename)

                # Load and resample
                audio, sr = librosa.load(input_path, sr=8000, mono=True)
                sf.write(output_path, audio, 8000)

                print(f"Resampled {input_path} -> {output_path}")


if __name__ == "__main__":
    input_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\unaugmentedfiles"
    output_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\unaugmented8khz"
    resample_wav_files_to_8k(input_directory, output_directory)
