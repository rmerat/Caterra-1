import cv2
import numpy as np
import Evaluation
import Preprocessing
import RansacProcess
import HoughProcess
import ExtractfromRosPackage
import SettingUp


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

    mode = IMG
    nb_row = 5
    folder = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines' 
    folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58'
    folder = '/home/roxane/Desktop/M3_2022/Caterra/crop-detection/Images_Preprocess'

    #initialisation of param : 
    hough_flag = 1
    list_rows = []
    list_validity = []
    idx_since_hough = 0
    av_info = 0

    if (mode == IMG):
        name_images = 'crops010.png' #crop_row_001.JPG' # 'rgb000.jpg' #'rgb397.jpg' #'crop_row_001.JPG' #'rgb397.jpg'
    
    if (mode == VID):
        name_images = SettingUp.obtain_name_images(folder)

    images = SettingUp.obtain_images(name_images, folder, mode)
    height_sky, col_veg, av_info = Preprocessing.init(images[0], mode)

    for idx, image in enumerate(images) : 
        print('image ', idx, 'of ', len(images))

        image = image[height_sky:,:,:]
        vegetation_mask = Preprocessing.get_vegetation_mask(image, height_sky, col_veg, mode, av_info)
        valid = 0
        cv2.imshow('vegetation_mask', vegetation_mask)
        
        while(valid==0):
            if(hough_flag==1):
                masks, vp, hough_image = HoughProcess.find_approx(image, vegetation_mask, nb_row)
                hough_flag = 0
                idx_since_hough = 0


            if(hough_flag==0):
                valid, masks, pts1, pts2, img_annotated = RansacProcess.find_rows(image, masks, vp, vegetation_mask)
                rows = [pts1, pts2]

                if ((idx_since_hough%5==0) and (idx_since_hough>0)):
                    valid = 0
                
                if (idx_since_hough==0 and valid == 1):
                    list_rows.append(rows)
                    list_validity.append(1)

                if (idx_since_hough == 0 and valid == 0): 
                    list_rows.append(rows)
                    list_validity.append(0)
                    valid = 1 # we did our best, go to next frame anyway 

                if (idx_since_hough>0 and valid == 1): 
                    list_rows.append(rows)
                    list_validity.append(1)

                if (idx_since_hough>0 and valid == 0): #do the hough process 
                    hough_flag = 1
        
        idx_since_hough = idx_since_hough + 1 

        cv2.imshow('img_annotated',  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
        filename = '/home/roxane/Desktop/img_annotated_clean/img_' + str(idx).zfill(3) + '.jpg'
        cv2.imwrite(filename,  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
        if(mode==VID):
            if cv2.waitKey(1) == ord('q'):
                break
        
        if(EVAL == True):
            #if(mode == IMG):
            crop_only = Evaluation.SaveData(img_annotated, list_rows[0])
            GTImage, cv, dv, v0, array_GT = Evaluation.LoadGroundTruth(name_images)
            score, precision = Evaluation.evaluate_results(cv, dv, v0, array_GT, nb_crop_row=nb_row)
            cv2.imshow('Ground Truth Image : ', cv2.cvtColor(GTImage, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(0)

        if(mode==IMG):
            cv2.waitKey(0)
    cv2.destroyAllWindows()

