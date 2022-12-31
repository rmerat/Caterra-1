import numpy as np
import cv2
import os
import PerImageProcessing
import MaskingProcess
import Evaluation


VID = 0
SING_IMG = 1
INI_PROCESS = 0
SPEED_PROCESS = 1
FINAL_PROCESS = 2

mode = VID
nb_row = 4

def detection_process(images, mode):
    """
    input : list of img to be analyzed in rgb format
    output : not yet defined, prob data + flag set to one if analyzing went smoothly 
    """

    imgs_annotated = []
    stage = INI_PROCESS

    if stage == INI_PROCESS: #longer but needed to create initial mask of images 
        print('initial process...')
        hough_img, arr_mask, col_best_mask, vp_pt = PerImageProcessing.Initial_Process(images[0])
        stage = SPEED_PROCESS


    if stage == SPEED_PROCESS: #quick, use of ransac 
        print('...speed process...')
        for img in images:
            img_annotated, arr_mask = PerImageProcessing.Speed_Process(img, arr_mask, col_best_mask, vp_pt)
            imgs_annotated.append(img_annotated)
            if(mode ==VID): 
                cv2.imshow('vid : ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows() 
                    break   
            stage = FINAL_PROCESS



    if stage == FINAL_PROCESS : #save data and evaluate it 
        print('...processing done!')
        data = Evaluation.SaveData(imgs_annotated)
        #TODO:implement evaluation 
        if(mode ==SING_IMG):
            cv2.imshow('img : ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows() 

    return data #return something en rapport avec le bon acheminement du process

if __name__ == "__main__":

    #first, get the name of the files we are going to analyze
    if (mode == VID):
        #distinguer entre mode video et mode single image?
        imgs_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        name_images = MaskingProcess.obtain_name_images(imgs_folder)

    if (mode == SING_IMG): 
        print('sing img')
        imgs_folder = '/home/roxane/Desktop/M3_2022/Caterra/crop-detection/Images_Preprocess/'
        name_images = 'crop_row_256.jpg'

    #open and resize images for consistency --> returns img in rgb format
    images = MaskingProcess.obtain_images(name_images,imgs_folder, mode)

    # Main Detection function 
    if images is not None : 
        detection_process(images, mode) 

    else : 
        print('No image')



        