import cv2
import os
from PIL import Image


def load_images(folder):
        image_list = []
        for filename in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, filename))
            image_list.append(img)
        return image_list


def RemoveBackground(Image):
    IGray = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    ret,BW = cv2.threshold(IGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,thresh2 = cv2.threshold(BW,127,255,cv2.THRESH_BINARY_INV)
    maskedImage = cv2.bitwise_and(Image,Image,mask=thresh2)
    return maskedImage


def Resize_Images(path,path2):
    list_dir = os.listdir(path)
    for x in list_dir:
        base = x
        x = path + '/' + x
        print(os.path.isdir(x))
    im1 = Image.open(x)
    im2 = im1.resize((64, 64))    
    im2.save(path2+"/"+base)
    print("Images are stored at {}".format(path2)) 
