import copy
import cv2
import os

from matplotlib import pyplot as plt

directory = r"C:\Users\soggy\Videos\MRIImagesDataset\allimgs"

out_dir = r"C:\Users\soggy\Videos\MRIImagesDataset\two_dataset\lowres"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

        IMG_IN = f

        save_name = os.path.join(out_dir, "lowres_"+filename)
        print(save_name)
        # keep a copy of original image
        original = cv2.imread(IMG_IN)

        # Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
        img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

        height, width = img.shape

        if(height==width):
            cv2.imwrite(save_name, cv2.resize(img, (62, 62)))
            continue

        # use binary threshold, all pixel that are beyond 3 are made white
        _, thresh_original = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

        # Now find contours in it.
        thresh = copy.copy(thresh_original)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # get contours with highest height
        lst_contours = []
        for cnt in contours:
            ctr = cv2.boundingRect(cnt)
            lst_contours.append(ctr)
        x,y,w,h = sorted(lst_contours, key=lambda coef: coef[3])[-1]


        # draw contours
        ctr = copy.copy(original)
        cv2.rectangle(ctr, (x,y),(x+w,y+h),(0,255,0),2)

        midpoint = (y + y + h)//2

        crop = original[midpoint - (width//2):midpoint + (width//2),:]
        cv2.imwrite(save_name, cv2.resize(crop, (62, 62)))
#         plt.imshow(crop)
        # cv2.imwrite('sofwinres.png',crop)