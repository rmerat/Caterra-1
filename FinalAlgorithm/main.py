import cv2
import numpy as np
import Evaluation
import Preprocessing
import RansacProcess
import HoughProcess
import ExtractfromRosPackage
import SettingUp
from skimage.color import rgb2lab


VID = 0
IMG = 1
EVAL = 0

if __name__ == "__main__":
    """
    hyperparameters : 
        - nb_crops : by def = 4
        - mode : single img or video?
        - if VID : folder containing the images
        - if IMG : foler + name of the image
    """

    mode = VID
    nb_row = 4
    folder = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines' 
    folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58'

    #initialisation of param : 
    hough_flag = 1
    list_rows = []
    list_validity = []
    idx_since_hough = 0
    av_info = 0

    if (mode == IMG):
        name_images = 'crop_row_194.JPG' #'rgb397.jpg' #'crop_row_001.JPG' #'rgb397.jpg'
    
    if (mode == VID):
        name_images = SettingUp.obtain_name_images(folder)

    images = SettingUp.obtain_images(name_images, folder, mode)
    height_sky, col_veg, av_info = Preprocessing.init(images[0], mode)

    for idx, image in enumerate(images) : 
        
        if(idx%10==0):
            print(idx)

        image = image[height_sky:,:,:]
        vegetation_mask = Preprocessing.get_vegetation_mask(image, height_sky, col_veg, mode, av_info)
        valid = 0
        
        while(valid==0):
            if(hough_flag==1):
                masks, vp = HoughProcess.find_approx(image)
                hough_flag = 0
                idx_since_hough = 0


            if(hough_flag==0):
                rows, masks, img_annotated = RansacProcess.find_rows(image, masks, vp)
                valid = RansacProcess.return_validity(rows, idx_since_hough)
                
                if (idx_since_hough==0 and valid == 1):
                    list_rows.append(rows)
                    list_validity.append(1)

                if (idx_since_hough == 0 and valid == 0): #we just did the HT but still no good result 
                    list_rows.append(rows)
                    list_validity.append(0)
                    valid = 1 # we did our best, go to next frame anyway 

                if (idx_since_hough>0 and valid == 1): 
                    list_rows.append(rows)
                    list_validity.append(1)

                if (idx_since_hough>0 and valid == 0): #do the hough process 
                    hough_flag = 1
        
        idx_since_hough = idx_since_hough + 1 

        
        cv2.imshow('img_annotated', vegetation_mask)
        filename = '/home/roxane/Desktop/img_annotated_clean/img_' + str(idx).zfill(3) + '.jpg'
        cv2.imwrite(filename,  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

