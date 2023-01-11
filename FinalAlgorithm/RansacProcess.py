import numpy as np
import cv2
from skimage import measure


def find_rows(image, masks, vp, vegetation_mask):
    """ 
    input : image to analyse, vanishing point, 
            list of the mask to separate points belonging to different crops, 
            image of the segmented vegetation
    output :  validity of the crops detected, new masks to seperate points belonging to different
            crops, pts1 and pts 2 = list containing one point per crops, 
            img_ransac_lines = img to visualize the crops   
    """
    
    pts1 = []
    pts2 = []
    acc_m = []
    masked_images = []
    img_ransac_lines = np.copy(image)
    masks_new = []
    validity = m = 0
    band_width = int(image.shape[1]/35) #before : 10

    for i in range(len(masks)):
        masked_images.append(cv2.bitwise_and(vegetation_mask, masks[i]))
        #inclure dès ici la condition sur la size des donnöes 

    for i in range(len(masks)): #for each row
        mask_single_crop = np.zeros_like(masks[0])
        p1, p2, m, validity = ransac_find_single_crop(masked_images[i], vp, acc_m) #HERE JUST CHANGED 
        
        if(validity == 1):
            pts1.append(p1)
            pts2.append(p2)
            acc_m.append(m)
            cv2.line(img_ransac_lines, p1, p2, (255,0,0), 2)
            cv2.line(mask_single_crop, p1, p2, (255,0,0), 10)
            masks_new.append(mask_single_crop)

        else :
            return validity, masks, pts1, pts2, img_ransac_lines

    return validity, masks_new, pts1, pts2, img_ransac_lines


def ransac_find_single_crop(masked_images_i, vp, acc_m):
    """
    input : image containing the points belonging to a single slope, vanishing point, slopes of other crops
    output : two points of a single crop, its slope and its validity
    """
    
    validity = 1
    x,y = np.where(masked_images_i>0)
    data = np.column_stack([x, y])
    vp = np.asarray(vp)
    n = int(data.shape[0])
    vp_data_x = np.full((n,1), vp[1])
    vp_data_y = np.full((n,1), vp[0])
    data_vp = np.column_stack([vp_data_x, vp_data_y])
    data = np.row_stack([data, data_vp])

    if(data.shape[0]>700):
        data = data[np.random.choice(data.shape[0], 500, replace=False), :]

    if(data.shape[0]<20):
        validity = 0
        return None, None, None, validity
    else : 
        p1, p2, m = fit_line(data, masked_images_i)

    validity = valid_crop_slope(m, acc_m)

    return p1, p2, m, validity 


def fit_line(data, masked_images_i):
    """
    input : data = np array, on which we want to fit a line 
    output : two points of the line fitted and its slope
    """
    model, _ = measure.ransac(data, measure.LineModelND, min_samples=2, residual_threshold=1, max_trials=500) #before 1000
    y0, x0 = model.params[0]#.astype(int)
    t1, t0 = model.params[1]
    m = -t1/t0
    k1 = (masked_images_i.shape[0]-y0)/m
    k2 = -(y0)/m
    p2 = [int(x0 - k2), int(y0 + k2*m)]
    p1 = [int(x0 - k1), int(y0 + k1*m)]

    return p1, p2, m


def valid_crop_slope(m, acc_m):
    """
    input : slope of a single line
    output : validity of this slope
    """

    #first : is it horizon?
    thr = 0.2 #before : 0.1
    validity = 1 

    if (abs(m)<thr):
        validity = 0
    
    #second : is it the same slopes as other crops? 
    if (len(acc_m)>=1):
        for m_others in acc_m:
            if (abs(m-m_others)<0.1): # angle already detected 
                validity = 0
                break

    return validity

