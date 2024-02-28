# BBox to Segmentation Converter

## Overview
This script, `convert_search.py`, is designed to automate the conversion of bounding box annotations to segmentation masks for image datasets. It leverages the SAM (Segment Anything Model) for generating high-quality segmentation masks from images and their corresponding bounding box annotations.

## Requirements
Create a conda environment from the `conda_env.yml` or:
- Python 3.8+
- OpenCV
- NumPy
- PIL
- PyTorch
- tqdm
- SAM Model Checkpoint (e.g., `sam_vit_h_4b8939.pth`)

## Installation
1. Ensure you have the required Python version and all necessary libraries installed. You can install the dependencies via pip:
    ```bash
    pip install opencv-python numpy Pillow torch tqdm
    ```

2. Download the SAM Model checkpoint file (`sam_vit_h_4b8939.pth`) and place it in the root directory of the project.

## Usage
1. Prepare your dataset with images and their corresponding YOLO format label files in the same directory. The script expects the following directory structure:
    ```
    datasets
    ├── Batch_4
    ├── Batch_6
    └── temp
    ```

2. Modify the `ROOT_FILEPATH` in the script to point to the root directory of your dataset. By default, it is set to `"./datasets/temp/"`.

3. Run the script with Python:
    ```bash
    python convert_search.py
    ```

4. The script will process each directory containing `.jpg` files, generate segmentation masks, and save them along with the original images in a new directory suffixed with `_converted_to_segments`.

## Features
- **Directory Traversal**: Automatically finds directories containing `.jpg` images and processes them.
- **Segmentation Mask Generation**: Uses the SAM model to generate segmentation masks from bounding box annotations.
- **Validation Masks and Videos**: Optionally, saves validation masks as images and compiles them into a video for easy review.

## Configuration
- **SAM Model and Device**: Configure the SAM model checkpoint and the device (`cuda` or `cpu`) at the beginning of the script.
- **Exclusion of Directories**: Directories ending with `_converted_to_segments` are automatically excluded from processing to avoid duplication.

## Notes
- Ensure the SAM Model checkpoint matches the model architecture specified in the script.
- The script includes utilities for parsing label files, converting bounding boxes, and saving masks, which can be customized as needed.

## Contribution
Feel free to fork the repository, make improvements, and submit pull requests. We appreciate your contributions to enhancing this tool.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
