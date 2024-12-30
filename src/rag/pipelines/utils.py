import os

def get_all_file_paths(directory):
    file_paths = []
    
    # Walk the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            # Construct the full file path and add it to the list
            file_paths.append(os.path.join(root, file))
    
    return file_paths