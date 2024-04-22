import os
import shutil

# Define your source and destination directories
src_image_dir = '/mnt/roar_shared_disk/YOLO_datasets/jiaming/multicar_datasets/compiled_bbox_datasets/batch_5and6/Batch_5/mako3_to_be_labeled/mako3_to_be_labeled_converted_to_segments/train/images'
src_label_dir = '/mnt/roar_shared_disk/YOLO_datasets/jiaming/multicar_datasets/compiled_bbox_datasets/batch_5and6/Batch_5/mako3_to_be_labeled/mako3_to_be_labeled_converted_to_segments/train/labels'

src_mask_val_dir = '/mnt/roar_shared_disk/YOLO_datasets/jiaming/multicar_datasets/compiled_bbox_datasets/batch_5and6/Batch_5/mako3_to_be_labeled/mako3_to_be_labeled_converted_to_segments/train/mask_validation'
dest_dir = '/mnt/roar_shared_disk/YOLO_datasets/jiaming/multicar_datasets/compiled_bbox_datasets/batch_5and6/Batch_5/mako3_722_to_740/mako3_722_to_740_converted_to_segments/train'

# Create new directories if they do not exist
os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'mask_validation'), exist_ok=True)

# Fetch and sort the files in the source directories
image_files = sorted([f for f in os.listdir(src_image_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(src_label_dir) if f.endswith('.txt')])
mask_val_files = sorted([f for f in os.listdir(src_mask_val_dir) if f.endswith('.jpg')])
# Check that the number of files matches
assert len(image_files) == len(label_files) == len(mask_val_files), "The number of image and label files does not match."

#print(image_files[:30])
#print(label_files[:30])
# Move the first 4700 image files and their labels
for image_file, label_file, mask_val_file in zip(image_files[722:740], label_files[722:740], mask_val_files[722:740]):
    # Construct the source and destination paths for images and labels
    src_image_path = os.path.join(src_image_dir, image_file)
    src_label_path = os.path.join(src_label_dir, label_file)
    src_mask_val_path = os.path.join(src_mask_val_dir, mask_val_file)
    dest_image_path = os.path.join(dest_dir, 'images', image_file)
    dest_label_path = os.path.join(dest_dir, 'labels', label_file)
    dest_mask_val_path = os.path.join(dest_dir, 'mask_validation', mask_val_file)
    #try:
    shutil.copy(src_image_path, dest_image_path)
    shutil.copy(src_label_path, dest_label_path)
    shutil.copy(src_mask_val_path, dest_mask_val_path)
    #except Exception as e:
    #    print(f"Failed to copy {src_image_path} or {src_label_path} to destination due to: {e}")
    
    

print("Files have been moved successfully.")