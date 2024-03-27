def process_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines or len(lines) == 1:
        return  # Skip empty files

    # Extract class from the first line (assuming all lines have the same class)
    first_line_parts = lines[0].split()
    instance_class = first_line_parts[0]
    
    # Use a set to store unique coordinates (as tuples to ensure hashability)
    unique_coords = set()
    for line in lines:
        coords = line.split()[1:]  # Skip the class part
        # Process coordinates in pairs and add them to the set
        for i in range(0, len(coords), 2):
            coord_pair = (coords[i], coords[i+1])  # Create a tuple for each coordinate pair
            unique_coords.add(coord_pair)
    
    # Convert unique coordinate tuples back to a list of strings for writing to file
    all_coords = [coord for pair in unique_coords for coord in pair]  # Flatten the set of tuples back into a list

    # Check if the unique merged instance has less than 5 points
    if len(all_coords) < 5 * 2:  # Each point has two coordinates (x, y)
        with open(file_path, 'w') as file:  # Empty the file
            file.write("")
    else:
        # Write the unique merged instance back to the file
        with open(file_path, 'w') as file:
            merged_line = instance_class + ' ' + ' '.join(all_coords) + '\n'
            file.write(merged_line)

# Example usage:
# Adjust 'label_file_path' to your specific label file path
label_file_path = './datasets/aggregated/03_13_24_Batch4_Batch5_Batch6_exported_datasets_yolo/train/sanity.txt'
process_label_file(label_file_path)