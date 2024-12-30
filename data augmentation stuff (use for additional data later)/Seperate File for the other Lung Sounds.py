import os
import shutil
import re

# Function to parse the filename and extract metadata
def parse_filename(filename):
    # Updated regex pattern
    pattern = r"^(B|D|E)P(\d+)_([A-Za-z\s+&]+),([A-Za-z\s]+),([AP]\s?[LR]\s?[LMU]?\s?[LR]?)\s?,\s?(\d+),([MF])\.wav$"

    match = re.match(pattern, filename.strip())

    if match:
        return {
            "Filter": match.group(1),  # Filter type
            "Patient Number": int(match.group(2)),  # Patient number
            "Diagnosis": [d.strip() for d in match.group(3).split("+")],  # Split multiple diagnoses
            "Sound Type": match.group(4).strip(),  # Sound type
            "Chest Zone": match.group(5).replace(" ", ""),  # Remove spaces in chest zone
            "Age": int(match.group(6)),  # Age
            "Gender": match.group(7).upper(),  # Gender
        }
    else:
        return None

# Function to organize files by filter and diagnosis
def organize_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    unmatched_files = []  # Keep track of unmatched files

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                metadata = parse_filename(file)
                if metadata:
                    # Map filter code to folder name
                    filter_map = {"B": "Bell", "D": "Diaphragm", "E": "Extended"}
                    filter_folder = filter_map.get(metadata["Filter"])
                    if not filter_folder:
                        print(f"Unknown filter for file: {file}")
                        continue

                    filter_dir = os.path.join(dst_dir, filter_folder)
                    os.makedirs(filter_dir, exist_ok=True)

                    # Check for multiple diagnoses
                    if len(metadata["Diagnosis"]) > 1:
                        # Place in a dedicated "multiple_diagnoses" folder
                        multi_diag_dir = os.path.join(filter_dir, "multiple_diagnoses")
                        os.makedirs(multi_diag_dir, exist_ok=True)
                        dst_path = os.path.join(multi_diag_dir, file)
                    else:
                        # Single diagnosis folder
                        diagnosis_dir = os.path.join(filter_dir, metadata["Diagnosis"][0].replace(" ", "_"))
                        os.makedirs(diagnosis_dir, exist_ok=True)
                        dst_path = os.path.join(diagnosis_dir, file)

                    # Copy the file
                    src_path = os.path.join(root, file)
                    shutil.copy(src_path, dst_path)
                else:
                    unmatched_files.append(file)

    # Log unmatched files
    if unmatched_files:
        print("The following files did not match the expected pattern:")
        for file in unmatched_files:
            print(f"  - {file}")

# Main execution
if __name__ == "__main__":
    src_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\Additional Lung Sounds\Audio Files"
    dst_directory = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-sounds-isef\data augmentation stuff (use for additional data later)\cleandownsampleMore"

    organize_files(src_directory, dst_directory)
    print("Files organized successfully.")



