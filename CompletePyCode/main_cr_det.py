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
mode = SING_IMG


if __name__ == "__main__":

    if (mode == VID): 
        #distinguer entre mode video et mode single image?
        vid_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        video_name = 'CropDetectionVID.avi'
        os.chdir("/home/roxane/Desktop/M3_2022/Caterra/vid_assembled")

        name_images = MaskingProcess.obtain_name_images(vid_folder)
        images = MaskingProcess.obtain_images(name_images,vid_folder, mode)

        if images is not None:
            for img in images:
                img_annotated = ImageAnnotation.analyze_img(img)
                cv2.imshow('vid : ', img_annotated)
                if cv2.waitKey(1) == ord('q'):
                    break           

        else : 
            print('error : Images not found')

    if (mode == SING_IMG): 
        print('sing img')
        image_folder = '/home/roxane/Desktop/M3_2022/Caterra/crop-detection/Images_Preprocess/'
        name_images = 'crop_row_256.jpg'

        sing_image = MaskingProcess.obtain_images(name_images, image_folder, mode)

        if sing_image is not None : 
            img_annotated = ImageAnnotation.analyze_img(sing_image[0])
            cv2.imshow('single analyzed image', img_annotated)
            cv2.waitKey(0)
            if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows() 

        else : 
            print('No image')



        