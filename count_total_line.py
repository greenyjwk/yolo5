import os

def count_lines_in_directory(directory):
    total_lines = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Count lines in each text file
            with open(file_path, 'r') as file:
                lines = file.readlines()
                total_lines += len(lines)
                print(f"File: {filename}, Lines: {len(lines)}")
    
    return total_lines

# Specify the directory path
directory_path = "/mnt/storage/ji/yolov5_2/runs/val/2class.10Rand_ContrLearn/labels"

# Count total lines
total_lines = count_lines_in_directory(directory_path)

print(f"\nTotal number of lines in all text files: {total_lines}")