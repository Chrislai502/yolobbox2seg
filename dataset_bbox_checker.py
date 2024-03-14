import os
import re

def count_files_with_pattern(directory, pattern):
    # Compile the regular expression pattern for efficiency
    compiled_pattern = re.compile(pattern)
    
    # Initialize a count of files that match the pattern
    matching_files_count = 0

    # # Walk through all files in the specified directory
    # for root, dirs, files in os.walk(directory):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         with open(file_path, 'r') as f:
    #             # Read the file's contents
    #             contents = f.read()
    #             # Search for the pattern in the contents
    #             if compiled_pattern.search(contents):
    #                 matching_files_count += 1
    #                 print(file_path)
    #                 # break  # Found a match, no need to check further in this file
                    
    # Walk through all files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                # Check each line in the file
                for line in f:
                    # Find all matches of the pattern in the line
                    matches = re.findall(pattern, line)
                    if matches:
                        # If there's at least one match in the line, count the file
                        matching_files_count += 1
                        print(file_path)
                        break  # Stop checking more lines in this file

    return matching_files_count

# Directory to search
directory = './datasets/aggregated/03_13_24_Batch4_Batch5_Batch6_exported_datasets_yolo/train/labels/'

# Pattern to search for - adjust as needed
# This pattern looks for lines starting with a digit, followed by any number of 5-value groups.
pattern = r'^\d(?:\s+\d+\.\d+){0,4}\s*$'

# Count files that match the pattern
count = count_files_with_pattern(directory, pattern)
print(f'Number of files with the specified pattern: {count}')