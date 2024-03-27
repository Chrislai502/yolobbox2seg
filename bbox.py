import cv2
import os

def draw_bounding_boxes(image_path, label_path, output_path):
    """
    Draws bounding boxes on the image based on the label file and saves the image.

    Args:
    - image_path (str): The path to the input image.
    - label_path (str): The path to the label file.
    - output_path (str): The path where the output image will be saved.
    """
    # Check if the image and label files exist
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print("Image or label file does not exist.")
        return
    
    # Read the image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Read the labels
    with open(label_path, 'r') as file:
        labels = file.readlines()
    
    # Draw each bounding box on the image
    for label in labels:
        class_id, x_center, y_center, width, height = [float(x) for x in label.split()]

        # Convert normalized positions to absolute pixel values
        x_center, y_center, width, height = (x_center * image_width, y_center * image_height,
                                             width * image_width, height * image_height)
        
        # Calculate the top left and bottom right corners
        x0, y0 = int(x_center - width / 2), int(y_center - height / 2)
        x1, y1 = int(x_center + width / 2), int(y_center + height / 2)
        
        # Draw the bounding box
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)
    print(f"Image with bounding boxes saved at {output_path}")

# Example usage
image_path = "path/to/your/image.jpg"
label_path = "path/to/your/label.txt"
output_path = "path/to/save/image_with_boxes.jpg"
draw_bounding_boxes(image_path, label_path, output_path)
