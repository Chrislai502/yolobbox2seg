MIN_SEG_POINTS = 4
print('hello')
def process_label_file(file_path):
    print(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    print(f"Processing: {file_path}")
    
    out =[]
    for line in lines:
        # if a line (segement) has less than MIN_SEG_POINTS points, ignore that segement
        coords = line.split()[1:]  # Skip the class part
        if len(coords) >= MIN_SEG_POINTS * 2:
            out.append(line)

    print('out', out)
    
    # Write the good, significant segements back to the file
    with open(file_path, 'w') as file:
        for l in out:
            file.write(l)

process_label_file('./test_label_segement.txt')