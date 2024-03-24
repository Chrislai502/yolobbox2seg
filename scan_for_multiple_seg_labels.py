import os
import shutil

import os
import shutil

def find_directories_with_multiple_segments(root_path):
    """
    Identifies directories within 'root_path' containing files in their 'labels/' subdirectories
    where each file has more than two lines of text, excluding empty lines. Assumes directories
    to have a specific structure, ending in '_converted_to_segments'.

    Parameters:
    - root_path: The root directory path to start searching from.

    Returns:
    - A list of paths to the 'labels/' directories meeting the criteria.
    """
    directories_with_multiple_segments = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if dirpath.endswith('labels') and '_converted_to_segments' in dirpath:
            # print(f"Checking: {dirpath}")
            for filename in filenames:
                if filename.endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        non_empty_lines = [line for line in lines if line.strip()]
                        if len(non_empty_lines) > 2:
                            directories_with_multiple_segments.append(dirpath)
                            print(f"Found eligible directory: {dirpath}")
                            break

    return directories_with_multiple_segments

def move_directories_with_multiple_segments(root_path):
    """
    Moves directories with multiple segment files and their corresponding directories
    into a 'multiple_segments' directory within the root_path, preserving the original
    file structure relative to the root directory.

    Parameters:
    - root_path: The root directory path to start searching from.
    """
    
    directories = find_directories_with_multiple_segments(root_path)
    target_root = os.path.join(root_path, 'multiple_segments')

    print("\n Directories found!! Moving Directories ~\n\n")

    for labels_path in directories:
        relative_path = os.path.relpath(labels_path, root_path)
        base_path = relative_path.replace('/train/labels', '')
        base_segment_path = '/'.join(base_path.split('/')[:-1])

        new_base_path = os.path.join(target_root, base_path)
        new_base_segment_path = os.path.join(target_root, base_segment_path)

        if not os.path.exists(os.path.dirname(new_base_path)):
            os.makedirs(os.path.dirname(new_base_path))
            # print(f"Created directory structure: {os.path.dirname(new_base_path)}")

        if not os.path.exists(os.path.dirname(new_base_segment_path)):
            os.makedirs(os.path.dirname(new_base_segment_path))
            # print(f"Created directory structure: {os.path.dirname(new_base_segment_path)}")

        # Move the directories
        seg_path = os.path.join(root_path, base_path)
        non_seg_path = seg_path.replace("_converted_to_segments", "")
        move_to = os.path.dirname(new_base_path)
        if os.path.exists(seg_path) and os.path.exists(non_seg_path):
            shutil.move(seg_path, move_to)
            shutil.move(non_seg_path, move_to)
            print(f"Moved {seg_path} to {move_to}")
            print(f"Moved {non_seg_path} to {move_to}")
        else:
            print(f"ERROR: {seg_path} or {non_seg_path} doesn't exist, Exiting...")
            exit(1)
            


# Example usage
root_directory = '/home/chris-lai/yolov8/bbox_2_segmentation/datasets/23-03-2024'  # Replace with your actual root directory path
move_directories_with_multiple_segments(root_directory)
