import os
import cv2
import numpy as np
from PIL import Image

def getLabels(labelPath):
    # Check if the file exists
    if not os.path.exists(labelPath):
        # Return an empty list if the file doesn't exist
        return None
    
    with open(labelPath) as f:
        # Preparing list for annotation of BB (bounding boxes)
        labels = []
        for line in f:
            labels.append(line.rstrip())

    return labels

def getConvertedBoxes(labels, image_width, image_height):
    converted_boxes = []
    class_ids = []
    for i in range(len(labels)):
        bb_current = labels[i].split()
        class_id = int(bb_current[0])
        x_center, y_center = float(bb_current[1]), float(bb_current[2])
        box_width, box_height = float(bb_current[3]), float(bb_current[4])
        
        # Convert to top left and bottom right coordinates
        x0 = int((x_center - box_width / 2) * image_width)
        y0 = int((y_center - box_height / 2) * image_height)
        x1 = int((x_center + box_width / 2) * image_width)
        y1 = int((y_center + box_height / 2) * image_height)
        class_ids.append(class_id)
        converted_boxes.append([x0, y0, x1, y1])
    return  class_ids, converted_boxes

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
    # print(f"Saved: {save_path}")        

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