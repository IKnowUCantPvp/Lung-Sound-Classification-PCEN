import os
import numpy as np
import librosa
from pathlib import Path

def analyze_audio_file(file_path):
    """
    Analyze the scaling of an audio file and print detailed statistics
    
    Args:
        file_path: Path to the audio file
    """
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    print("-" * 50)
    
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Calculate basic statistics
    abs_max = np.abs(y).max()
    abs_min = np.abs(y).min()
    mean = np.mean(y)
    std = np.std(y)
    rms = np.sqrt(np.mean(y**2))
    
    # Check if it's in common ranges
    is_normalized = abs_max <= 1.1  # Allow some overhead
    is_16bit = abs_max <= 32768
    is_32bit = abs_max <= 2**31
    
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    print("\nAmplitude Statistics:")
    print(f"Maximum absolute value: {abs_max}")
    print(f"Minimum absolute value: {abs_min}")
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")
    print(f"RMS value: {rms}")
    
    print("\nScaling Analysis:")
    print(f"Is in [-1, 1] range? {'Yes' if is_normalized else 'No'}")
    print(f"Is in 16-bit range? {'Yes' if is_16bit else 'No'}")
    print(f"Is in 32-bit range? {'Yes' if is_32bit else 'No'}")
    
    # Histogram analysis
    hist, bins = np.histogram(y, bins=50)
    peak_bin = bins[np.argmax(hist)]
    print(f"\nHistogram peak around: {peak_bin:.3f}")
    
    return {
        'file': os.path.basename(file_path),
        'max': abs_max,
        'normalized': is_normalized,
        'rms': rms
    }

def analyze_directory(directory):
    """
    Analyze all wav files in a directory
    
    Args:
        directory: Path to directory containing wav files
    """
    print(f"Analyzing audio files in: {directory}")
    print("=" * 50)
    
    # Get all wav files
    wav_files = list(Path(directory).glob('*.wav'))
    
    if not wav_files:
        print("No wav files found in directory!")
        return
    
    # Analyze each file
    results = []
    for wav_file in wav_files:
        result = analyze_audio_file(str(wav_file))
        results.append(result)
    
    # Summary
    print("\nSummary:")
    print("=" * 50)
    print(f"Total files analyzed: {len(results)}")
    
    normalized_files = sum(1 for r in results if r['normalized'])
    print(f"\nFiles in [-1, 1] range: {normalized_files}/{len(results)}")
    
    # Print files that might need attention
    print("\nFiles outside [-1, 1] range:")
    for r in results:
        if not r['normalized']:
            print(f"- {r['file']} (max: {r['max']:.2f})")

if __name__ == "__main__":
    # Example usage:
    # analyze_directory('path/to/your/wav/files')
    
    # Or analyze a single file:
    # analyze_audio_file('path/to/your/wav/file.wav')
    
    print("To use this script:")
    print("1. For a single file:")
    print("   analyze_audio_file('path/to/your/wav/file.wav')")
    print("2. For a directory:")
    print("   analyze_directory('path/to/your/wav/files')")

analyze_audio_file()