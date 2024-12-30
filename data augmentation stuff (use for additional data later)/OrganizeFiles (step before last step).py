import os
from glob import glob
from tqdm import tqdm
from shutil import copy2


def transfer_files(src_root, dst_root):
    """
    Transfers audio files from a structured folder system into a new structure
    where files are grouped by lung class with augmentation types preserved.

    Parameters:
    - src_root: Path to the source folder (e.g., `cleanaug`).
    - dst_root: Path to the destination folder (e.g., `clean`).
    """
    # Ensure destination directory exists
    os.makedirs(dst_root, exist_ok=True)

    # Find all .wav files in the source directory, preserving folder structure
    wav_paths = glob(os.path.join(src_root, '**', '*.wav'), recursive=True)

    for wav_path in tqdm(wav_paths, desc="Transferring Files"):
        # Parse the augmentation type and class label from the file path
        parts = wav_path.split(os.sep)
        if len(parts) < 3:
            continue  # Skip malformed paths

        # Extract augmentation type and class label
        aug_type = parts[-3]  # Augmentation type is the third-to-last folder
        class_label = parts[-2]  # Class label is the second-to-last folder
        file_name = parts[-1]  # File name

        # Construct new folder structure
        target_dir = os.path.join(dst_root, class_label)
        os.makedirs(target_dir, exist_ok=True)

        # Modify file name to include augmentation type
        new_file_name = f"{os.path.splitext(file_name)[0]}_{aug_type}.wav"
        target_path = os.path.join(target_dir, new_file_name)

        # Copy file to the new location
        copy2(wav_path, target_path)


if __name__ == '__main__':
    # Define source and destination directories
    SRC_ROOT = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\Step 3 (FILTERED FILES)"
    DST_ROOT = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\Step 4 (final FILTERED FILES)"

    # Transfer files
    transfer_files(SRC_ROOT, DST_ROOT)
