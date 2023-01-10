import numpy as np
import cv2
import PerImageProcessing
import MaskingProcess
import Evaluation
import random


VID = 0
SING_IMG = 1
INI_PROCESS = 0
SPEED_PROCESS = 1
FINAL_PROCESS = 2

def detection_process(images, mode, name_images, eval_on = 0, nb_row = 3, sky = 1, vp_on = 1): 
    """
    input : list of img to be analyzed in rgb format
    output : not yet defined, prob data + flag set to one if analyzing went smoothly 
    """
    imgs_annotated = []
    imgs_crops_only = []
    stage = INI_PROCESS
    vp_pt = 0
    col_best_mask = None 
    sky_height = 0

    for idx, img in enumerate(images):
            #cv2.waitKey(0)

            if stage == INI_PROCESS: #longer but needed to create initial mask of images 
                hough_img, arr_mask, col_best_mask, vp_pt, sky_height = PerImageProcessing.Initial_Process(img, idx, col_best_mask, sky_height, nb_row = nb_row, sky = sky)
                stage = SPEED_PROCESS
            
            if stage == SPEED_PROCESS: #quick, use of ransac 
                print('img ', idx+1, 'out of', len(images))
                img_no_sky = np.copy(img[sky_height:, :,:])

                arr_mask_new, img_annotated, vp_pt, cond_speed, img_crops_only, pts1, pts2 = PerImageProcessing.speed_process_lines(img_no_sky, col_best_mask, arr_mask, vp_pt, vp_on=vp_on)
                if (((idx+1)%10)==0):
                    print('%10')
                    cond_speed = 0
                
                if(cond_speed==0):
                    cond_speed = 1
                    hough_img, arr_mask, col_best_mask, vp_pt, sky_height = PerImageProcessing.Initial_Process(img, idx, col_best_mask, sky_height, nb_row = nb_row, sky = sky)
                    img_no_sky = np.copy(img[sky_height:, :,:])
                    arr_mask_new, img_annotated, vp_pt, cond_speed, img_crops_only, pts1, pts2 = PerImageProcessing.speed_process_lines(img_no_sky, col_best_mask, arr_mask, vp_pt, vp_on=vp_on, nb_crops=nb_row)
                    print('new speed cond : ', cond_speed)
                
                imgs_annotated.append(img_annotated)
                r_acc, th_acc,threshold_acc = MaskingProcess.get_r_theta(pts1, pts2)
                if(cond_speed==1):
                    arr_mask = arr_mask_new
                    vp_pt, _ = MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, stage)
                    vp_pt = np.asarray(vp_pt)
               

                if(mode ==VID): 
                    cv2.imshow('vid : ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
                    #cv2.waitKey(0)
                    if cv2.waitKey(1) == ord('q'):
                        cv2.destroyAllWindows() 
                        break 

                if(mode == SING_IMG): 
                    cv2.imshow('single image : ', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
                    #cv2.waitKey(0)
                    if cv2.waitKey(1) == ord('q'):
                        cv2.destroyAllWindows() 
                        break   
            
            filename = '/home/roxane/Desktop/img_annotated_clean/img_' + str(idx).zfill(3) + '.jpg'
            cv2.imwrite(filename,  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))

                
        #stage = FINAL_PROCESS


    """
        if stage == FINAL_PROCESS : #save data and evaluate it 
            print('...processing done!')
            if(mode ==SING_IMG):
                crop_only = Evaluation.SaveData(img_crops_only, pts1, pts2)
                cop, cv, dv, v0, array_GT = Evaluation.LoadGroundTruth(name_images, img_annotated)
                score, precision = Evaluation.evaluate_results(cv, dv, v0, array_GT, nb_crop_row=nb_row)
                total_score = score
                cv2.imshow('My Results : ', cv2.cvtColor(cop, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
    """

    return imgs_annotated #, score #return something en rapport avec le bon acheminement du process

if __name__ == "__main__":
    """
    hyperparameters : 
        - sky_on
        - nb_crops
        - vp_on
        - mode
    """
    vp_on = 1
    sky_on = 1
    mode = SING_IMG
    nb_row = 4

    if (mode == VID):
        #distinguer entre mode video et mode single image?
        imgs_folder = '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/'
        name_images = MaskingProcess.obtain_name_images(imgs_folder)
        nb_row = 5

    if (mode == SING_IMG): 
        imgs_folder = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines' #/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/' # '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines' # '/home/roxane/Desktop/M3_2022/USB/Realsense_18-08-2022_10-46-58/' 
        name_images = 'crop_row_001.JPG' # 'crop_row_256.JPG' #'rgb000.jpg' # crop_row_001, crop_row_020, crop_row_053
        sky_on = 1
        nb_row = 3
        vp_on = 1

    # open and resize images for consistency --> returns img in rgb format
    images = MaskingProcess.obtain_images(name_images,imgs_folder, mode)

    # Main Detection function 
    if images is not None : 
        imgs_annotated = detection_process(images, mode, name_images, nb_row = nb_row, sky = sky_on, vp_on = vp_on)

    else : 
        print('No image')


        