import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
import os
import ImageAnnotation
import MaskingProcess

VID = 0
SING_IMG = 1
nb_row = 4

if __name__ == "__main__":

    mode = VID

    if (mode == VID): 
        #distinguer entre mode video et mode single image?
        vid_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        video_name = 'CropDetectionVID.avi'
        os.chdir("/home/roxane/Desktop/M3_2022/Caterra/vid_assembled")

        name_images = MaskingProcess.obtain_name_images(vid_folder)
        images = MaskingProcess.obtain_images(name_images,vid_folder, mode)

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

        sing_image = MaskingProcess.obtain_images(name_images, image_folder, mode)

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

        