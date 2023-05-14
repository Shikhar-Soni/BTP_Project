import cv2
import os

# base directory
train_dir = r"C:\Users\Soumya\Desktop"

for img in os.listdir(train_dir + "\\images"):

    if(img.split(".")[-1] != "png"):
        print("Unknown file type")
        continue

    print(img)

    img_array = cv2.imread(train_dir + "\\images\\" + img)
    
    # bicubic interpolation
    img_array = cv2.resize(img_array, (100,100), interpolation=cv2.INTER_CUBIC)
    lr_img_array = cv2.resize(img_array,(50,50), interpolation=cv2.INTER_CUBIC)

    # write images

    cv2.imwrite(train_dir + "\\hr_images\\" + img, img_array)
    cv2.imwrite(train_dir + "\\lr_images\\" + img, lr_img_array)