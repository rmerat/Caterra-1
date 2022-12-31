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

def veg_segmentation(img):

    # extract the main colors from the image 
    colors_rgb = MaskingProcess.extract_rgb_colors(img)

    # extract greenest color 
    col_best_mask = MaskingProcess.greenest_color(colors_rgb)

    # convert color and image to lab space
    img_lab = skimage.color.rgb2lab(img/255)
    col_best_mask_lab = skimage.color.rgb2lab((col_best_mask[0]/255, col_best_mask[1]/255, col_best_mask[2]/255))

    # vegetation segmentation using mask of the detected vegetal color
    best_mask = MaskingProcess.mask_vegetation(img_lab, col_best_mask_lab)
    best_mask_median = cv2.medianBlur(best_mask,3)
    
    return best_mask_median, col_best_mask


def Initial_Process(img, sky_on = 0, nb_row = 4):

    best_mask_median, col_best_mask = veg_segmentation(img)
    best_mask_median_edge = cv2.Canny(best_mask_median,100,200)
    cv2.imshow('best mask with median filter', best_mask_median)
    cv2.waitKey(3000)

    arr_mask, th_acc, r_acc, threshold_acc, best_mask_evaluate = MaskingProcess.keep_mask_max_acc_lines(best_mask_median_edge, img, 4)

    cv2.imshow('crop detected with only HT', best_mask_evaluate)
    cv2.waitKey(3000)

    arr_mask = []

    return img, arr_mask, col_best_mask

def Speed_Process(img, arr_mask, sky_on = 0, nb_row = 4):

    annotated_img = np.copy(img)

    cv2.line(annotated_img, (10,100), (100,10), (255,0,0), 10)
    cv2.line(annotated_img, (100,10), (10,100), (0,255,0), 5)
    cv2.line(annotated_img, (10,10), (100,100), (0,0,255), 2)

    return annotated_img, arr_mask
