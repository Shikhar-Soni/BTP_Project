import os
import fastmri
from fastmri.data import transforms as T
import h5py
import numpy as np
from matplotlib import pyplot as plt
import PIL
import cv2
import sys
from fastmri.data.subsample import RandomMaskFunc

total_images = 0

BASE_FOLDER = r"C:\Users\Soumya\Downloads\brain_multicoil_challenge_transfer\brain_multicoil_challenge_transfer~\multicoil_challenge_transfer"

SAVE_FOLDER = r"C:\Users\Soumya\Desktop\images"

for file_name in os.listdir(BASE_FOLDER):
    
    # the file path for individual MRIs
    if(file_name.split(".")[-1] != "h5"):
        print("Unknown file type")
        continue

    full_file_path = BASE_FOLDER + "\\" + file_name
    print("File path:", full_file_path)
    hf = h5py.File(full_file_path)
    
    volume_kspace = hf['kspace']
    slice_count = volume_kspace.shape[0]
    
    total_images += slice_count
    
    for i in range(slice_count):
        ith_slice = volume_kspace[i]
        tensor_ith_slice = T.to_tensor(ith_slice)      # Convert from numpy array to pytorch tensor
        image_space_ith_slice = fastmri.ifft2c(tensor_ith_slice)           # Apply Inverse Fourier Transform to get the complex image
        image_abs_ith_slice = fastmri.complex_abs(image_space_ith_slice)   # Compute absolute value to get a real image
        
        image_abs_rss_ith_slice = fastmri.rss(image_abs_ith_slice, dim=0)
        
        image_file_name = os.path.join(SAVE_FOLDER + "\\fully_sampled", file_name.split('.')[0] + "_" + str(i) + "_" + ".png")

        print(image_file_name)
        plt.imsave(image_file_name, np.abs(image_abs_rss_ith_slice.numpy()), cmap='gray')

        mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object
        masked_kspace, mask, _ = T.apply_mask(tensor_ith_slice, mask_func)

        sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
        sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
        sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

        image_file_name = os.path.join(SAVE_FOLDER + "\\under_sampled", file_name.split('.')[0] + "_" + str(i) + "_" + ".png")

        plt.imsave(image_file_name, np.abs(sampled_image_rss.numpy()), cmap='gray')

        image_file_name = os.path.join(SAVE_FOLDER + "\\masks", file_name.split('.')[0] + "_" + str(i) + "_" + ".png")

        plt.imsave(image_file_name, mask.numpy(), cmap='gray')

print("Total Images extracted = ", total_images)