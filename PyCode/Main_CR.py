import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
import os
import ImageAnnotation


VID = 0
SING_IMG = 1


def obtain_name_images(image_folder):

    lst = os.listdir(image_folder)
    lst.sort()

    name_images = [img for img in lst 
                if img.endswith(".jpg") or
                    img.endswith(".jpeg") or
                    img.endswith("png")]

    return name_images 

def obtain_images(name_images, image_folder, mode):
    imgs = []

    if mode == VID : 
        for name in name_images:
            img = cv2.imread(os.path.join(image_folder, name))
            if img is not None : 
                img_small = img_resize(img)
                imgs.append(img_small)

    if mode == SING_IMG :
        print('image read : ', os.path.join(image_folder, name_images))
        img = cv2.imread(os.path.join(image_folder, name_images))
        if img is not None : 
            img_small = img_resize(img)
            imgs.append(img_small)
        else : 
            print('no image read')

    if (len(imgs) != 0) : 
        return imgs
    
    return None


def img_resize(img):
    #resize
    output_width = 900  #set the output size
    wpercent = (output_width/float(img.shape[1]))
    hsize = int((float(img.shape[0])*float(wpercent)))
    img = cv2.resize(img, (output_width,hsize), interpolation = cv2.INTER_AREA) #this resize makes the video not work??
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


if __name__ == "__main__":

    mode = VID

    if (mode == VID): 
        #distinguer entre mode video et mode single image?
        vid_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        video_name = 'CropDetectionVID.avi'
        os.chdir("/home/roxane/Desktop/M3_2022/Caterra/vid_assembled")

        name_images = obtain_name_images(vid_folder)
        images = obtain_images(name_images,vid_folder, mode)

        if images is not None:
            images_post_process = ImageAnnotation.annotation(images, 1) 
            height, width, layers = images_post_process[0].shape  
            #print(images_post_process[0].shape)

            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width, height)) 

            #cv2.imshow('img post process', images_post_process[10])
            #cv2.waitKey(10000)

            # Appending the images to the video one by one
            for img in images: 
                video.write(img)

            # Deallocating memories taken for window creation
            cv2.destroyAllWindows() 
            video.release()  # releasing the video generated
        else : 
            print('error : Images not found')

    if (mode == SING_IMG): 
        print('sing img')

        image_folder = '/home/roxane/Desktop/M3_2022/Caterra/crop-detection/Images_Preprocess/'
        name_images = 'crop_row_256.jpg'

        sing_image = obtain_images(name_images, image_folder, mode)

        if sing_image is not None : 
            #print(sing_image)
            #cv2.imshow('single analyzed image', sing_image[0])
            #cv2.waitKey(1000)
            image_post_process = ImageAnnotation.annotation(sing_image, 0) 

        else : 
            print('None')
        #image_post_process = ImageAnnotation.annotation(sing_image) 

        cv2.imshow('single analyzed image', image_post_process[0])
        cv2.waitKey(1000)

        