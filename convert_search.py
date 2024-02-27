import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import cv2
import numpy as np
import os
from PIL import Image
import glob
import torch
import shutil
from utils import getLabels, getConvertedBoxes, save_all_masks_on_one_image, parse_seg_label_file
import subprocess

# Initialize the SAM model
print("Initializing SAM Model...")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM Model Initialized!")

SAVE_MASK_IMAGE = True
ROOT_FILEPATH = "./datasets/temp/"

# Function to recursively find directories containing .jpg files, excluding certain directories
def find_image_directories(root_path, exclude_suffix="_converted_to_segments"):
    directories = set()
    for root, dirs, files in os.walk(root_path):
        # Skip directories that end with the exclude_suffix
        if root.endswith(exclude_suffix):
            dirs[:] = []  # Prevents walk from looking into any subdirectories
            continue

        for file in files:
            if file.endswith(".jpg"):
                directories.add(root)
                break  # Stop searching this directory once a .jpg is found
    return list(directories)

def process_directory(bbox_dataset):
    
    seg_dataset = f"{bbox_dataset}_converted_to_segments"
    image_files = glob.glob(f"{bbox_dataset}/*.jpg")
    label_files = glob.glob(f"{bbox_dataset}/*.txt")

    print("\n Processing directory:", bbox_dataset)
    print("Number of images:", len(image_files))
    print("Number of labels:", len(label_files))
        
    image_lables = []
    
    # if you would like to plot and view the segmentation masks then set Objects list from the yaml file
    Objects = []
    # color rgb values for each class
    color = []
    
    for imgPath in image_files:
        # get the label file path
        labelPath = imgPath.replace(".jpg", ".txt")
        # rplace images with labels
        labelPath = labelPath.replace("images", "labels")
        # add the image and label path to a tuple
        image_lables.append((imgPath, labelPath))
        
    # Generate a random color for each object in the Objects list
    for objects in Objects:
        color.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))

    loopCount = 0  # Initialize loop count for tracking number of processed labels

    # get image size from one image
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)  # Read the image file

    # Assuming image_labels is a list of tuples containing image paths and corresponding label paths
    for imgPath, labelPath in tqdm(image_lables):
        destination = f'{seg_dataset}/train'  # Default destination folder for training data
        if 'valid' in imgPath:  # Change destination if the image is for validation
            destination = f'{seg_dataset}/valid'
        
        if SAVE_MASK_IMAGE:
            mask_destination = f'{destination}/mask_validation'
        
        # Extract the file name without extension to use for the label file
        label_file = imgPath.split('/')[-1].split('.')[0]
        seg_label_path = os.path.join(destination, f'labels/{label_file}.txt')
        
        # Skip processing if label file already exists in the destination
        if os.path.exists(seg_label_path):
            # print(f'{label_file} already exists in {destination}')
            mask_file = os.path.join(mask_destination, os.path.basename(imgPath))
            if (not os.path.exists(mask_file)) and SAVE_MASK_IMAGE:
                # Convert the PIL image for compatibility
                raw_image = Image.open(imgPath).convert("RGB")
                raw_image_array = np.array(raw_image)
                h, w = raw_image_array.shape[:2]
                
                # Parse the segmentation label file
                masks, class_ids = parse_seg_label_file(seg_label_path, (h, w))

                # Save the masks on the image
                # print("Saving Validation Mask...")
                save_all_masks_on_one_image(raw_image, masks, mask_destination, save_filename=os.path.basename(imgPath))
            continue
        
        labels = getLabels(labelPath)  # Assuming getLabels is a function to parse label files
        if labels == None:
            continue # yolo format skip if there's no labels.
        
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)  # Read the image file
        predictor.set_image(image)  # Assuming predictor is a pre-defined object for predictions
        raw_image = Image.open(imgPath).convert("RGB")  # Open the image with PIL for additional operations if needed
        h, w = image.shape[:2]  # Get image dimensions
        
        # Convert bounding boxes according to the image dimensions
        class_ids, bounding_boxes = getConvertedBoxes(labels, w, h)
        
        # Convert bounding boxes to tensor and apply any necessary transformations
        input_boxes = torch.tensor(bounding_boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        # Predict masks based on the transformed bounding boxes
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Process each mask generated by the predictor
        for i, mask in enumerate(masks):
            binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)  # Convert mask to binary (0 or 1) format
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
            
            try:
                largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour
                segmentation = largest_contour.flatten().tolist()  # Flatten the largest contour to a list
                mask = np.array(segmentation).reshape(-1, 2)  # Reshape for normalization
                mask_norm = mask / np.array([w, h])  # Normalize the pixel coordinates
                class_id = class_ids[i]  # Get the class ID for the current mask
                yolo = mask_norm.reshape(-1)  # Flatten the normalized mask coordinates
                
                if not os.path.exists(destination):  # Ensure the destination directory exists
                    os.makedirs(destination)
                
            except Exception as e:
                print(e)
                continue  # Skip to the next mask if any errors occur
            
            loopCount += 1  # Increment the processed label count

            # Ensure the labels directory exists
            if not os.path.exists(os.path.join(destination, 'labels')):
                os.makedirs(os.path.join(destination, 'labels'))
            
            # Write the normalized mask coordinates to the label file
            with open(seg_label_path, "a") as f:
                for val in yolo:
                    f.write(f"{class_id} {val:.6f}")
                f.write("\n")
        
        if SAVE_MASK_IMAGE:
            # Ensure the mask_destination directory exists
            if not os.path.exists(mask_destination):
                os.makedirs(mask_destination)
            
            # Call the function with the correct parameters
            the_size = masks.size()
            h, w = the_size[-2], the_size[-1]
            masks = masks.cpu().numpy().astype(np.uint8).reshape(-1, h, w)
            save_all_masks_on_one_image(raw_image, masks, mask_destination, save_filename=os.path.basename(imgPath))

        # Ensure the images directory exists and copy the current image to it
        if not os.path.exists(os.path.join(destination, 'images')):
            os.makedirs(os.path.join(destination, 'images'))
        shutil.copy(imgPath, f'{destination}/images')

    # After processing the images and saving masks, call ffmpeg to create a video from the masks
    mask_validation_dir = f"{destination}/mask_validation"
    output_video_path = f"{destination}/validation.mp4"
    
    # Ensure the mask_validation directory exists and contains jpg files before calling ffmpeg
    if os.path.exists(mask_validation_dir) and len(glob.glob(f"{mask_validation_dir}/*.jpg")) > 0:
        ffmpeg_command = [
            "ffmpeg",
            "-framerate", "50",
            "-pattern_type", "glob",
            "-i", f"{mask_validation_dir}/*.jpg",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        
        try:
            print("Creating video from masks...")
            subprocess.run(ffmpeg_command, check=True)
            print("Video created successfully:", output_video_path)
        except subprocess.CalledProcessError as e:
            print("Failed to create video from masks:", e)
    else:
        print("No masks found to create a video in", mask_validation_dir)
    
def main(root_path):
    image_directories = find_image_directories(root_path)
    print("Processing These Image Directories:\n")
    for i, image_dir in enumerate(image_directories):
        print(f"{i}. ", image_dir)
        
    for image_dir in image_directories:
        process_directory(image_dir)

if __name__ == "__main__":
    main(ROOT_FILEPATH)