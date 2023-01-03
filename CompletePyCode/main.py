import numpy as np
import cv2
import PerImageProcessing
import MaskingProcess
import Evaluation


VID = 0
SING_IMG = 1
INI_PROCESS = 0
SPEED_PROCESS = 1
FINAL_PROCESS = 2

nb_row = 3

def detection_process(images, mode, nb_row = 6, sky = 1):
    """
    input : list of img to be analyzed in rgb format
    output : not yet defined, prob data + flag set to one if analyzing went smoothly 
    """

    imgs_annotated = []
    imgs_crops_only = []
    stage = INI_PROCESS

    for idx, img in enumerate(images):
        #print(idx, 'stage : ', stage)
        if stage == INI_PROCESS: #longer but needed to create initial mask of images 
            print('initial process...')
            hough_img, arr_mask, col_best_mask, vp_pt = PerImageProcessing.Initial_Process(img, nb_row = nb_row, sky = sky)
            height_original = images[0].shape[0]
            height, _, = arr_mask[0].shape
            stage = SPEED_PROCESS
            print('...speed process...')
        
        if stage == SPEED_PROCESS: #quick, use of ransac 
            #print('...speed process...')

            #print('...speed process...')
            #print(idx, 'vp : ', vp_pt)
            img = img[height_original-height:,:,:]
            arr_mask, img_annotated, vp_pt, cond_speed, img_crops_only = PerImageProcessing.speed_process_lines(img, col_best_mask, arr_mask, vp_pt)
            
            if(cond_speed==0):
                print('back to ini!')
                stage = INI_PROCESS
            else : 
            #img_annotated = img
            #vp point to recaulculate 
            # img_annotated, arr_mask = PerImageProcessing.Speed_Process(img, arr_mask, col_best_mask, vp_pt)
                imgs_annotated.append(img_annotated)
                imgs_crops_only.append(img_crops_only)
            
                if(mode ==VID): 
                    cv2.imshow('vid :q ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
                    #cv2.imshow('no annotation', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    if cv2.waitKey(1) == ord('q'):
                        cv2.destroyAllWindows() 
                        break   
                if(mode == SING_IMG):
                    cv2.imshow('crop only : ', img_crops_only)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
    
    stage = FINAL_PROCESS

    if stage == FINAL_PROCESS : #save data and evaluate it 
        print('...processing done!')
        Evaluation.SaveData(img_crops_only)
        """data = Evaluation.SaveData(crops_only)
        #TODO:implement evaluation """
        if(mode ==SING_IMG):
            cv2.imshow('img : ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()

    return 0 #return something en rapport avec le bon acheminement du process


if __name__ == "__main__":

    sky_on = 1
    mode = SING_IMG


    #first, get the name of the files we are going to analyze
    if (mode == VID):
        #distinguer entre mode video et mode single image?
        imgs_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        name_images = MaskingProcess.obtain_name_images(imgs_folder)
        sky_on = 1
        nb_row = 6


    if (mode == SING_IMG): 
        print('sing img')
        imgs_folder = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines'
        name_images = 'crop_row_001.JPG' #crop_row_001, crop_row_020, crop_row_053
        #imgs_folder = '/home/roxane/Desktop/M3_2022/crop_dataset/'
        #name_images = 'crop_row_001.JPG'
        sky_on = 0
        nb_row = 3


    #open and resize images for consistency --> returns img in rgb format
    images = MaskingProcess.obtain_images(name_images,imgs_folder, mode)

    # Main Detection function 
    if images is not None : 
        detection_process(images, mode, nb_row = nb_row, sky = sky_on)

    else : 
        print('No image')


        