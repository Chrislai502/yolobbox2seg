 
from tqdm import tqdm
import cv2
import numpy as np
import os
from PIL import Image

# if you would like to plot and view the segmentation masks then set Objects list from the yaml file
Objects = []
# color rgb values for each class
color = []

SAVE_MASK_IMAGE = True

# get all files with .jpg in all directories in ./Yolo_Dataset_2
import glob
bbox_dataset = 'datasets/Batch_6/LABELLED'
seg_dataset = 'datasets/Batch_6/CONVERTED'
image_files = glob.glob(f"./{bbox_dataset}/*.jpg", recursive=True)
label_files = glob.glob(f"./{bbox_dataset}/*.txt", recursive=True)

print("Number of images:", len(image_files))
print("Number of labels:", len(label_files))
# iterate through each image file and add it to a tuple 
image_lables = []
for imgPath in image_files:
    # get the label file path
    labelPath = imgPath.replace(".jpg", ".txt")
    # rplace images with labels
    labelPath = labelPath.replace("images", "labels")
    # add the image and label path to a tuple
    image_lables.append((imgPath, labelPath))


def save_all_masks_on_one_image(raw_image, masks, save_dir, save_filename="all_masks_overlay"):
    raw_image_array = np.array(raw_image, dtype=np.float32)  # Convert to float for blending
    overlay_image = raw_image_array.copy()

    # Ensure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use a fixed color for visibility and debugging
    color = np.array([255.0, 0.0, 0.0, 153.0])  # Red with alpha (60% opacity when considering 255 scale)
    alpha = 0.6

    for i, mask in enumerate(masks):
        mask = mask.astype(np.float32)  # Ensure mask is float
        h, w = mask.shape
        
        # Create a colored mask
        colored_mask = np.zeros((h, w, 4), dtype=np.float32)  # Include alpha channel for the mask
        colored_mask[..., :3] = color[:3]  # Apply color
        colored_mask[..., 3] = mask * color[3]  # Apply mask's alpha channel
        
        # Alpha blending
        alpha = colored_mask[..., 3:] / 255.0  # Normalize alpha to [0, 1]
        
        overlay_image = (1 - alpha) * overlay_image + alpha * colored_mask[..., :3]
    
    # Ensure the resulting image is in the correct data type and range
    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

    # Convert array to image after all masks have been applied
    overlay_pil = Image.fromarray(overlay_image)
    save_path = os.path.join(save_dir, save_filename)  # Use PNG to preserve quality
    overlay_pil.save(save_path)
    print(f"Saved: {save_path}")        

def parse_seg_label_file(seg_label_path, image_shape):
    """
    Parses a segmentation label file.

    Parameters:
    - seg_label_path: Path to the segmentation label file.
    - image_shape: Tuple of (height, width) of the corresponding image.

    Returns:
    - masks: A list of mask arrays.
    - class_ids: A list of class IDs associated with each mask.
    """
    masks = []
    class_ids = []

    with open(seg_label_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])
            # Assuming the rest of the line is normalized mask coordinates
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            # Un-normalize coordinates
            coords[:, 0] *= image_shape[1]  # Width
            coords[:, 1] *= image_shape[0]  # Height
            # Create a blank mask
            mask = np.zeros(image_shape, dtype=np.uint8)
            # Draw the polygon on the mask
            coords = np.array([coords], dtype=np.int32)  # cv2.fillPoly expects a 3D array
            cv2.fillPoly(mask, coords, 1)
            masks.append(mask)
            class_ids.append(class_id)

    return masks, class_ids


# Generate a random color for each object in the Objects list
for objects in Objects:
    color.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))

loopCount = 0  # Initialize loop count for tracking number of processed labels

# get image size from one image
image = cv2.imread(imgPath, cv2.IMREAD_COLOR)  # Read the image file

destination = f'{seg_dataset}/train'  # Default destination folder for training data
# if 'valid' in imgPath:  # Change destination if the image is for validation
#     destination = f'{seg_dataset}/valid'

if SAVE_MASK_IMAGE:
    mask_destination = f'{destination}/mask_validation'

counter = 0
has_seg = 0
print("Counter Num Images", counter)

# Assuming image_labels is a list of tuples containing image paths and corresponding label paths
for imgPath, labelPath in tqdm(image_lables):
    
    # Extract the file name without extension to use for the label file
    label_file = imgPath.split('/')[-1].split('.')[0]
    seg_label_path = os.path.join(destination, f'labels/{label_file}.txt')
    
    # Skip processing if label file already exists in the destination
    if os.path.exists(seg_label_path):
        has_seg +=1 
        print(f'{label_file} already exists in {destination}')
        
        mask_file = os.path.join(mask_destination, os.path.basename(imgPath))
        print(mask_file)
        if (not os.path.exists(mask_file)) and SAVE_MASK_IMAGE:
            print("Saving Validation Mask...")
            # Convert the PIL image for compatibility
            raw_image = Image.open(imgPath).convert("RGB")
            raw_image_array = np.array(raw_image)
            h, w = raw_image_array.shape[:2]
            
            # Parse the segmentation label file
            masks, class_ids = parse_seg_label_file(seg_label_path, (h, w))

            # Save the masks on the image

            save_all_masks_on_one_image(raw_image, masks, mask_destination, save_filename=os.path.basename(imgPath))
        continue

    counter += 1
    print(counter, "Has Seg:", has_seg)