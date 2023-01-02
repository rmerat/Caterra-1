#Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import extcolors
import skimage
import scipy
from PIL import Image
import math
import skimage
from sklearn.linear_model import LinearRegression
import pandas as pd
from colormap import rgb2hex
import numpy as np
import cv2
import os
import MaskingProcess


def Initial_Process(img, nb_row = 4, sky = 0):

    #Cut off sky
    if(sky==1) : 
        grad_sky = MaskingProcess.get_sky_region_gradient(img)
        img_no_sky = MaskingProcess.cut_image_from_mask(grad_sky, img)
    else : 
        img_no_sky = img

    best_mask_median, col_best_mask = MaskingProcess.veg_segmentation(img, img_no_sky)
    best_mask_median_edge = cv2.Canny(best_mask_median,100,200)

    arr_mask, th_acc, r_acc, threshold_acc, best_mask_evaluate = MaskingProcess.keep_mask_max_acc_lines(best_mask_median_edge, img_no_sky, nb_row)

    vp_pt = np.asarray(MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, best_mask_median_edge))

    cv2.imshow('after initial process', best_mask_evaluate)
    cv2.waitKey(0)

    return best_mask_evaluate, arr_mask, col_best_mask, vp_pt

def Speed_Process(img, arr_mask, col_best_mask, vp_pt, nb_row = 4,  sky_on = 0):

    annotated_img = np.copy(img)
    img_lab = skimage.color.rgb2lab(img/255)

    col_best_mask_lab = skimage.color.rgb2lab((col_best_mask[0]/255, col_best_mask[1]/255, col_best_mask[2]/255))
    
    # vegetation segmentation using mask of the detected vegetal color
    best_mask = MaskingProcess.mask_vegetation(img_lab, col_best_mask_lab)
    best_mask_median = cv2.medianBlur(best_mask,3)

    #cv2.imshow('best_mask_median', best_mask_median)
    #cv2.waitKey(1000)

    #print(best_mask_median.shape)
    model = MaskingProcess.pattern_ransac(arr_mask, vp_pt, best_mask_median) 


    for crop in model:
        crop = np.asarray(crop)
        diff = (crop[1]-crop[0])
        cv2.line(annotated_img, crop[0], crop[1]+5*diff, (50,200,50), 3)


    cv2.line(annotated_img, (10,100), (100,10), (255,0,0), 10)
    cv2.line(annotated_img, (100,10), (10,100), (0,255,0), 5)
    cv2.line(annotated_img, (10,10), (100,100), (0,0,255), 2)

    return annotated_img, arr_mask

