import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import yaml


# Define your base, train, and validation directories
base_dir = "./datasets/Converted/"

# Set the destination for train and val data
train_dir = "./datasets/aggregated/03_13_24_Batch4_Batch5_Batch6_exported_datasets_yolo/train/" 
val_dir   = "./datasets/aggregated/03_13_24_Batch4_Batch5_Batch6_exported_datasets_yolo/val/"
yaml_path = "./datasets/aggregated/03_13_24_Batch4_Batch5_Batch6_exported_datasets_yolo/data.yaml"

# Your class configuration
nc = 1
names = ['car']

def ask_permission_and_clear(directory):
    """Asks for permission to clear the directory if it exists, then clears it."""
    # if os.path.exists(directory):
    #     response = input(f"The directory {directory} already exists. Do you want to replace it? (y/n): ").lower()
    #     if response == 'y':
    #         shutil.rmtree(directory)
    #         os.makedirs(directory)
    #     else:
    #         print("Operation cancelled.")
    #         exit()
    # else:
    #     os.makedirs(directory)

def copy_files(files, dest_dir, description="Copying files"):
    """Copies files to the destination directory with a progress bar."""
    os.makedirs(dest_dir, exist_ok=True)
    for file in tqdm(files, desc=description):
        try:
            shutil.copy(file, dest_dir)
        except Exception as e:
            print(f"Failed to copy {file}: {e}")


def get_files_and_directories(base_dir, pattern, filter_text='_converted_to_segments'):
    """Returns files matching pattern within directories containing filter_text, and their unique directories."""
    all_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    filtered_files = [file for file in all_files if filter_text in os.path.dirname(file)]
    directories = set(os.path.dirname(file) for file in filtered_files)
    return filtered_files, directories

def create_data_yaml(train_dir, val_dir, nc, names, yaml_path="data.yaml"):
    data = {
        'train': "../train/images",
        'val': "../val/images",
        # Optionally add a 'test' key if you have a test set
        'nc': nc,
        'names': names
    }
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def match_images_with_labels(images, labels):
    """
    Matches images with their corresponding label files.
    Assumes that image and label files share the same basename but differ in extensions.
    Returns matched image and label files.
    """
    matched_images = []
    matched_labels = []
    # Create a set of base filenames for labels for quick lookup
    label_bases = {os.path.splitext(os.path.basename(label))[0]: label for label in labels}
    
    for image in images:
        image_base = os.path.splitext(os.path.basename(image))[0]
        if image_base in label_bases:
            matched_images.append(image)
            matched_labels.append(label_bases[image_base])
        else:
            print(f"No corresponding label found for image: {image}")
    
    return matched_images, matched_labels

# Ask for permission to replace directories if they exist
ask_permission_and_clear(train_dir)
ask_permission_and_clear(val_dir)

# Collect all images and labels and their directories, filtered by '_converted_to_segments'
all_images, image_dirs = get_files_and_directories(base_dir, '**/images/*', '_converted_to_segments')
all_labels, label_dirs = get_files_and_directories(base_dir, '**/labels/*', '_converted_to_segments')


# Match images with labels to ensure each image has a corresponding label
all_images, all_labels = match_images_with_labels(all_images, all_labels)

# Print detected directories for a sanity check
print("Detected image directories:")
for dir in sorted(image_dirs):
    print(dir)
    
print("\n")
print("\nDetected label directories:")
for dir in sorted(label_dirs):
    print(dir)

# Ensure each image has a corresponding label file (optional validation step)
# This step assumes file names without extensions are matching for images and labels

# Split data into training and validation
images_train, images_val, labels_train, labels_val = train_test_split(
    all_images, all_labels, test_size=0.1, random_state=42
)

# # Copy the files to the respective directories with progress bars
# copy_files(images_train, os.path.join(train_dir, 'images'), "Copying training images")
# copy_files(labels_train, os.path.join(train_dir, 'labels'), "Copying training labels")
# copy_files(images_val, os.path.join(val_dir, 'images'), "Copying validation images")
# copy_files(labels_val, os.path.join(val_dir, 'labels'), "Copying validation labels")

# After organizing the dataset, create the data.yaml file
create_data_yaml(train_dir, val_dir, nc, names, yaml_path)

print(f"Data configuration file created at: {yaml_path}")

print(f"Unique images to copy: {len(set(all_images))}")
print(f"Actual images list length: {len(all_images)}")