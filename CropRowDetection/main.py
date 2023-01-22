import cv2
import numpy as np
import Evaluation
import Preprocessing
import RansacProcess
import HoughProcess
#import ExtractfromRosPackage
import SettingUp
import os 
from argparse import ArgumentParser

VID = 0
IMG = 1

if __name__ == "__main__":
    """
    inputs : - mode = picture or video
            - k : for video mode = Number of frames before doing Hough Process 
            - n : number of crop rows to be detected (default = 5, can decrease if not enough crops are found)
            - name : for picture mode = name of the files to be analyzed
            - e : for picture mode = evaluation mode 
    """

    p = ArgumentParser(description="Detect crop rows!")
    p.add_argument("mode", choices=["video", "picture"], default="video", help= 'specify mode ')
    p.add_argument("-k", type=int, default=5, help="number of frames before complete process (video only)", required=False)
    p.add_argument("-n", type=int, default=5, help="number of crop rows", required=False)
    p.add_argument("--name", "-p", type=str, default = 'crop_row_001.JPG', help="name of the picture", required=False)
    p.add_argument("-e", type=bool, default = False, help="Evaluation on/off", required=False)

    args = p.parse_args()

    if args.mode == "video":
        mode = VID
        folder = os.getcwd() + '/VideoDataset'
        name_images = SettingUp.obtain_name_images(folder)

    elif args.mode == "picture":
        mode = IMG
        folder = os.getcwd()
        name_images = '/ImagesCropRow/' + args.name 

    nb_row = args.n
    K = args.k
    hough_flag = 1
    list_rows = []
    list_validity = []
    idx_since_hough = 0
    av_info = 0

    images = SettingUp.obtain_images(name_images, folder, mode)
    height_sky, col_veg, av_info = Preprocessing.init(images[0], mode)

    for idx, image in enumerate(images) : 
        if(mode==VID):
            print('image ', idx, 'of ', len(images))

        image = image[height_sky:,:,:]
        vegetation_mask = Preprocessing.get_vegetation_mask(image, col_veg, mode, av_info)
        valid = 0
        
        while(valid==0):
            if(hough_flag==1):
                masks, vp, hough_image, pts1, pts2, nb_row = HoughProcess.find_approx(image, vegetation_mask, nb_row)
                hough_flag = 0
                idx_since_hough = 0

            if(hough_flag==0):
                valid, masks, pts1, pts2, img_annotated = RansacProcess.find_rows(image, masks, vp, vegetation_mask)
                rows = [pts1, pts2]

                if ((idx_since_hough%K==0) and (idx_since_hough>0)):
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

        if(mode==VID):
            cv2.imshow('img_annotated',  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            filename = os.getcwd() + '/VideoDatasetResults/' + 'img_' + str(idx).zfill(3) + '.jpg'
            cv2.imwrite(filename,  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break
        
        if(mode==IMG):
            if(args.e == True):
                name = Evaluation.SaveData(img_annotated.shape[0], list_rows[0]) 
                GTImage, ComparaisonImage, dv, v0, array_GT = Evaluation.LoadGroundTruth(args.name, img_annotated, height_sky)
                score = Evaluation.evaluate_results(dv, v0, height_sky, array_GT, name, nb_crop_row=nb_row)
                print('SCORE : ', score)
                cv2.imshow('comparaison image ', ComparaisonImage)

            filename = os.getcwd() + '/ImagesResults/' + args.name 
            cv2.imwrite(filename,  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            cv2.imshow('img_annotated',  cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    cv2.destroyAllWindows()
