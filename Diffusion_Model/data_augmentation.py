import imgaug.augmenters as iaa
import os
from PIL import Image
import numpy as np

# Define the augmentations to be performed
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Flipud(0.5),
    # iaa.Crop(percent=(0, 0.1)), # crop up to 10% of the image
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur the image with a sigma between 0 and 3.0
    # iaa.Affine(scale=(0.5, 1.5)) # scale the image between 50% and 150%
])

checkseq = iaa.Sequential([
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.Flipud(0.5),
    # iaa.Crop(percent=(0, 0.1)), # crop up to 10% of the image
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur the image with a sigma between 0 and 3.0
    # iaa.Affine(scale=(0.5, 1.5)) # scale the image between 50% and 150%
])

# Define the path to the folder of images to be augmented
input_folder = r'C:\Users\soggy\Videos\Diffusion\MRI_Original_512'

# Define the output folder for the augmented images
output_folder = r'C:\Users\soggy\Videos\Diffusion\MRI_Augmented_512'

# Loop through the images in the input folder and perform augmentation
for filename in os.listdir(input_folder):
    img = Image.open(os.path.join(input_folder, filename))
    img = np.asarray(img) # convert the image to a numpy array
    if img.ndim == 3 and img.shape[-1] >= 32 and img.shape[0:2] != (1, 1) or 1:
        
        print("gae")
        img_aug = seq(image=img)
        img_check = checkseq(image=img)
        img_aug = Image.fromarray(img_aug.astype('uint8')) # convert the augmented image back to a PIL Image
        if img_aug != img_check:
            img_aug.save(os.path.join(output_folder, f'{filename[:-4]}.png'))
        else:
            print("nope")
