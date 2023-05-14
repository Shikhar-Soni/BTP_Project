import os
import fastmri
from fastmri.data import transforms as T
import h5py
import numpy as np
from matplotlib import pyplot as plt
import PIL
import cv2
import sys

total_images = 0

BASE_FOLDER = r"C:\Users\Soumya\Downloads\brain_multicoil_challenge_transfer\brain_multicoil_challenge_transfer~\multicoil_challenge_transfer"

SAVE_FOLDER = r"C:\Users\Soumya\Desktop\images"

def cropTheImageIntoABox(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_TOZERO)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    print(len(contours))

    x, y, w, h = 0, 0, 0, 0
    for i in range(len(contours)):
        cnt = cv2.boundingRect(contours[i])
        if w * h < cnt[3] * cnt[2]:
            x, y, w, h = cnt

    crop = img
    if(img.shape[1] < h):
        crop = img[y:y+h,:]
    else:
        ww = img.shape[1]
        hh = img.shape[0]
        crop = img[max(y+h//2-ww//2,0):min(hh, y+h//2+ww//2),:]

    crop = cv2.resize(crop, (248, 248))
    cv2.imwrite(img_path, crop)

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
        
        image_file_name = os.path.join(SAVE_FOLDER, file_name.split('.')[0] + "_" + str(i) + "_" + ".png")

        print(image_file_name)

        plt.imsave(image_file_name, np.abs(image_abs_rss_ith_slice.numpy()), cmap='gray')

        # im = np.abs(image_abs_rss_ith_slice.numpy())
        cropTheImageIntoABox(image_file_name)

    # sys.exit(-1)

print("Total Images extracted = ", total_images)