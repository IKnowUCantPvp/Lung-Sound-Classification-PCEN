import os
import zipfile
import subprocess
import shutil
import sys
import glob

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_file(file_id, output_path):
    print(f"Downloading {output_path}...")
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def organize_data():
    """
    After unzipping, the folders might have long names.
    We want to standardize them to data/train and data/test.
    """
    # Check what folders exist in data/
    # Expected names based on zip filenames usually, but sometimes different.
    # cleanTrainDataset.zip -> cleanTrainDataset (nonoise and COPD cut)
    # cleanEvaluationDataset.zip -> cleanEvaluationDataset (noise and COPD cut)
    
    # We will look for directories containing 'Train' and 'Evaluation'
    
    # Define destination paths
    train_dest = os.path.abspath(os.path.join('data', 'train'))
    test_dest = os.path.abspath(os.path.join('data', 'test'))
    
    # Check for extracted folders
    # Train
    train_candidates = glob.glob('data/cleanTrainDataset*')
    for candidate in train_candidates:
        if os.path.isdir(candidate) and 'train' not in os.path.basename(candidate).lower():
             print(f"Renaming {candidate} to {train_dest}")
             if os.path.exists(train_dest):
                 shutil.rmtree(train_dest)
             os.rename(candidate, train_dest)

    # Test
    test_candidates = glob.glob('data/cleanEvaluationDataset*')
    for candidate in test_candidates:
        if os.path.isdir(candidate) and 'test' not in os.path.basename(candidate).lower():
             print(f"Renaming {candidate} to {test_dest}")
             if os.path.exists(test_dest):
                 shutil.rmtree(test_dest)
             os.rename(candidate, test_dest)

    if not found_train and os.path.exists(train_dest):
        print("Train directory already exists.")
    if not found_test and os.path.exists(test_dest):
        print("Test directory already exists.")

def main():
    # Install gdown if missing
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        install_package("gdown")

    os.makedirs('data', exist_ok=True)

    # Train Dataset
    train_zip = 'cleanTrainDataset.zip'
    if not os.path.exists(train_zip):
        download_file('14n9yAQ2695hqiiVtIgxlLblvVgJu6Ssp', train_zip)
    
    # Eval Dataset
    eval_zip = 'cleanEvaluationDataset.zip'
    if not os.path.exists(eval_zip):
        download_file('1EdXFr8iA2bwlO3kZr-wQwXsT3xhj00oO', eval_zip)

    # Extract
    unzip_file(train_zip, 'data/')
    unzip_file(eval_zip, 'data/')
    
    # Organize
    organize_data()
    
    print("\nData Setup Complete!")
    print(f"Train Data: {os.path.abspath('data/train')}")
    print(f"Test Data:  {os.path.abspath('data/test')}")
    print("\nYou can now run training:")
    print("python3 src/trainPCEN.py --data_dir data/train")
    print("\nAnd evaluation:")
    print("python3 src/evalMetricsPytorch.py --data_dir data/test")

if __name__ == "__main__":
    main()
