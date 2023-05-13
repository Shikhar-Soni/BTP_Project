import os
import random

# Set the path to your folder containing the images
path_to_folder = r"C:\Users\soggy\Videos\Diffusion\MRI_Augmented_512"

# Get a list of all the files in the folder
file_list = os.listdir(path_to_folder)

# Shuffle the file list to randomly drop images
random.shuffle(file_list)

# Determine the number of files to keep and drop
num_files_to_keep = 2000
num_files_to_drop = len(file_list) - num_files_to_keep

# Loop through the file list and delete the extra files
for i in range(num_files_to_drop):
    os.remove(os.path.join(path_to_folder, file_list[i]))
