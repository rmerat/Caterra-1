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


def find_hough_lines() : 

    return 0

def Initial_Process(img, idx, col_best_mask, sky_height = 0, nb_row = 4, sky = 0):

    #Cut off sky
    if(sky==1) : 
        if(idx == 0):
            grad_sky = MaskingProcess.get_sky_region_gradient(img)
            img_no_sky = MaskingProcess.cut_image_from_mask(grad_sky, img)
            sky_height = img.shape[0] - img_no_sky.shape[0]
        if(idx!=0):
            img_no_sky = img[sky_height:, :, :]
        
    else : 
        img_no_sky = img

    #cv2.imshow('ini image : ', img_no_sky)
    best_mask_median, best_mask_brut, col_best_mask = MaskingProcess.veg_segmentation(img, img_no_sky, idx, col_best_mask)

    #remove bushy regions 
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(best_mask_brut, kernel, iterations=1)
    best_mask_nobushes = best_mask_brut-eroded

    best_mask_median_edge = cv2.Canny(best_mask_median,100,200)
    best_mask_brut_edge = cv2.Canny(best_mask_brut,100,200) 
    best_mask_nobushes_edge = cv2.Canny(best_mask_nobushes,100,200)

    #cv2.imshow('vegetation best_mask_median', best_mask_median)

    #cv2.imshow('vegetation best_mask_median_edge', best_mask_median_edge)

    kernel = np.ones((2, 2), np.uint8)
    best_mask_median_edge_dil22 = cv2.dilate(best_mask_median_edge, kernel, iterations = 1)
    #cv2.imshow('best_mask_median_edge dilated 22', best_mask_median_edge_dil22)

    kernel = np.ones((2, 1), np.uint8)
    best_mask_median_edge_dil21 = cv2.dilate(best_mask_median_edge, kernel, iterations = 1)
    #cv2.imshow('best_mask_median_edge dilated 21 ', best_mask_median_edge_dil21)

    #kernel = np.ones((1, 2), np.uint8)
    #best_mask_median_edge_dil12 = cv2.dilate(best_mask_median_edge, kernel, iterations = 1)
    #cv2.imshow('best_mask_median_edge dilated 12 ', best_mask_median_edge_dil12)

    #cv2.imshow('vegetation best_mask_brut',  best_mask_brut)

    #cv2.imshow('vegetation best_mask_brut_edge',  best_mask_brut_edge)


    #cv2.imshow('vegetation best_mask_nobushes',  best_mask_nobushes)

    #cv2.imshow('vegetation best_mask_nobushes_edge',  best_mask_nobushes_edge)

    # Noise removal using Morphological
    # closing operation
    #kernel = np.ones((2, 2), np.uint8)
    best_mask_nobushes_dil = cv2.dilate(best_mask_nobushes, kernel, iterations = 1)
    #cv2.imshow('best mask no bushes dilated ', best_mask_nobushes_dil)
    # Background area using Dilation

    #cv2.waitKey(0)


    arr_mask, th_acc, r_acc, threshold_acc, best_mask_evaluate, pts1, pts2 = MaskingProcess.keep_mask_max_acc_lines(best_mask_median_edge, img_no_sky, nb_row+1)

    vp_pt, outlier = MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, stage=0)
    vp_pt = np.asarray(vp_pt)
    if outlier is not None:
        print('outlier : ', outlier)
        arr_mask.pop(outlier[0])
        th_acc.pop(outlier[0])
        r_acc.pop(outlier[0])
        threshold_acc.pop(outlier[0])
        pts1.pop(outlier[0])
        pts2.pop(outlier[0])
        vp_pt, outlier = MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, stage=0)
        vp_pt = np.asarray(vp_pt)

    else : #remove last line 
        print('no outliers lines')
        arr_mask.pop()
        th_acc.pop()
        r_acc.pop()
        threshold_acc.pop()
        pts1.pop()
        pts2.pop()



    #cv2.imshow('after initial process', best_mask_evaluate)
    #cv2.waitKey(0)

    #arr_mask = check_equidistance(arr_mask)

    return best_mask_evaluate, arr_mask, col_best_mask, vp_pt, sky_height

def eliminate_outlier_line(pts1,pts2, vp_point, th_acc, r_acc):

    new_line = []
    """
    # NumPy array of lines, where each row represents a line defined by two points (x1, y1, x2, y2)
    lines = tot

    # Given point
    point = (193, -35)

    # Initialize a variable to store the farthest distance
    farthest_distance = 0

    # Iterate through the lines and find the farthest one from the given point
    for line in lines:
        x1, y1, x2, y2 = line
        # Calculate the distance from the point to the line
        distance = abs((y2 - y1) * point[0] - (x2 - x1) * point[1] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if distance > farthest_distance:
            farthest_distance = distance
            farthest_line = line
    """

    #recalculate VP -->maybe put threshold on mean distance to VP?


    return pts1,pts2, th_acc, r_acc#, best_mask_evaluate, arr_mask

def Speed_Process(img, arr_mask, col_best_mask, vp_pt, nb_row = 4,  sky_on = 0):

    annotated_img = np.copy(img)
    img_lab = skimage.color.rgb2lab(img/255)

    col_best_mask_lab = skimage.color.rgb2lab((col_best_mask[0]/255, col_best_mask[1]/255, col_best_mask[2]/255))
    
    # vegetation segmentation using mask of the detected vegetal color
    best_mask = MaskingProcess.mask_vegetation(img_lab, col_best_mask_lab)
    best_mask_median = cv2.medianBlur(best_mask,3)

    model = MaskingProcess.pattern_ransac(arr_mask, vp_pt, best_mask_median) 


    for crop in model:
        crop = np.asarray(crop)
        diff = (crop[1]-crop[0])
        cv2.line(annotated_img, crop[0], crop[1]+5*diff, (50,200,50), 3)


    cv2.line(annotated_img, (10,100), (100,10), (255,0,0), 10)
    cv2.line(annotated_img, (100,10), (10,100), (0,255,0), 5)
    cv2.line(annotated_img, (10,10), (100,100), (0,0,255), 2)

    return annotated_img, arr_mask

def check_equidistance(arr_mask):

    return arr_mask

def speed_process_lines(image, col_best_mask, arr_mask, vp_pt, vp_on, nb_crops=5):

    img_lab = skimage.color.rgb2lab(image/255) #calculate best color mask based on previously calculated color 
    col_best_mask_lab = skimage.color.rgb2lab((col_best_mask[0]/255, col_best_mask[1]/255, col_best_mask[2]/255))

    best_mask = MaskingProcess.mask_vegetation(img_lab, col_best_mask_lab)
    #best_mask = cv2.medianBlur(best_mask,3)
    band_width = int(image.shape[1]/25)
    cv2.imshow('best_mask before speed process line : ', best_mask)

    pts1 = []
    pts2 = []
    acc_m = []
    masked_images = []
    img_ransac_lines = np.copy(image)
    crops_only = np.zeros_like(image)
    arr_mask_new = []
    
    for i in range(nb_crops):
        #print('size best_mask and arr_mask : ', best_mask.shape, arr_mask[0].shape)
        masked_images.append(cv2.bitwise_and(best_mask, arr_mask[i]))
        #cv2.imshow('masked image of i :'+ str(i), masked_images[i])
        #cv2.imshow('mask i :'+ str(i), arr_mask[i])

        #cv2.waitKey(0)
    print('len array mask des diff crops : ', len(arr_mask))


    
    for i in range(len(arr_mask)): #for each row
        mask_single_crop = np.zeros_like(best_mask)
        cond = m = cond_horizon = cond_double = 0
        #cv2.imshow('masked images : ', masked_images[i])
        #cv2.imshow('best_mask', best_mask)
        #cv2.waitKey(0)

        p1, p2, m, cond_speed = MaskingProcess.apply_ransac(image, masked_images[i], vp_pt, vp_on, best_mask, arr_mask[i], i) #HERE JUST CHANGED 
        
        if (cond_speed==0): 
            return arr_mask, img_ransac_lines, vp_pt, cond_speed, crops_only, pts1, pts2

        masked_images[i], cond_horizon = MaskingProcess.remove_horizon(p1, p2, m, masked_images[i], band_width)
        masked_images[i], cond_double = MaskingProcess.remove_double(p1, p2, m, acc_m, masked_images[i], band_width)

        if ((cond_horizon==0) or (cond_double==0)):
            cond_speed = 0
            return arr_mask, img_ransac_lines, vp_pt, cond_speed, crops_only, pts1, pts2

        
        pts1.append(p1)
        pts2.append(p2)
        acc_m.append(m)
        cv2.line(img_ransac_lines, p1, p2, (255,0,0), 2)
        cv2.line(crops_only, p1, p2, (255,0,0), 1)
        cv2.line(mask_single_crop, p1, p2, (255,0,0), band_width)
       # cv2.imshow('mask_single_crop', mask_single_crop)
        #cv2.imshow('img_ransac_lines', img_ransac_lines)
        #cv2.waitKey(0)
        arr_mask[i] = mask_single_crop

    return arr_mask, img_ransac_lines, vp_pt, cond_speed, crops_only, pts1, pts2
