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


def Initial_Process(img, sky_on = 0, nb_row = 4):

    best_mask_median, col_best_mask = MaskingProcess.veg_segmentation(img)
    best_mask_median_edge = cv2.Canny(best_mask_median,100,200)
    cv2.imshow('best mask with median filter', best_mask_median)
    cv2.waitKey(3000)

    arr_mask, th_acc, r_acc, threshold_acc, best_mask_evaluate = MaskingProcess.keep_mask_max_acc_lines(best_mask_median_edge, img, 4)

    vp_pt = np.asarray(MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, best_mask_median_edge))

    cv2.imshow('crop detected with only HT', best_mask_evaluate)
    cv2.waitKey(3000)

    return best_mask_evaluate, arr_mask, col_best_mask, vp_pt

def Speed_Process(img, arr_mask, col_best_mask, vp_pt, sky_on = 0, nb_row = 4):

    annotated_img = np.copy(img)

    cv2.line(annotated_img, (10,100), (100,10), (255,0,0), 10)
    cv2.line(annotated_img, (100,10), (10,100), (0,255,0), 5)
    cv2.line(annotated_img, (10,10), (100,100), (0,0,255), 2)

    return annotated_img, arr_mask
