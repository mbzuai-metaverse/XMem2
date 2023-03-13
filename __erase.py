import os
import shutil

folder_path = '/home/ariana/RESEARCH/Datasets/metaverse/to_process/pimples/JPEGImages' # Replace with the path to your folder
new_prefix = 'frame_' # Replace with your new prefix

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Sort the files by name to ensure they're in the correct order
files.sort()

# Loop through each file and rename it
for i, file_name in enumerate(files):
    # Generate the new file name
    new_file_name = '{}{:06d}.jpg'.format(new_prefix, i)
    
    # Get the full path to the original file
    old_path = os.path.join(folder_path, file_name)
    
    # Get the full path to the new file
    new_path = os.path.join(folder_path, new_file_name)
    
    # Rename the file
    shutil.move(old_path, new_path)
