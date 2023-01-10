import numpy as np
import cv2
import os 

VID = 0
IMG = 1

def obtain_images(name_images, image_folder, mode, output_width = 500):
    """
    input : name and foler of the images location
    returns : list containing all the images in the folder in RGB format
    """
    images = []
    if mode == VID : 
        for name in name_images:
            image = cv2.imread(os.path.join(image_folder, name))
            if image is not None : 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image_resize(image, output_width = output_width)
                images.append(image)

    if mode == IMG :
        image = cv2.imread(os.path.join(image_folder, name_images))
        if image is not None : 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image_resize(image, output_width=output_width)
            images.append(image)

    if (len(images) != 0) : 
        return images
    
    return None

def image_resize(image, output_width = 500):
    """
    input : image
    output : resized image
    """
    wpercent = (output_width/float(image.shape[1]))
    hsize = int((float(image.shape[0])*float(wpercent)))
    image = cv2.resize(image, (output_width,hsize), interpolation = cv2.INTER_AREA) 

    return image

def obtain_name_images(image_folder):
    """
    input : name of the folder with the images
    output : list of name of the files to be open
    """
    lst = os.listdir(image_folder)
    lst.sort()
    name_images = [img for img in lst 
                if img.endswith(".jpg") or
                    img.endswith(".jpeg") or
                    img.endswith(".JPG") or
                    img.endswith("png")]

    return name_images 