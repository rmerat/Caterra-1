import math
import numpy as np
import cv2 
from sys import exit

def find_approx(image, vegetation_mask, nb_row):
    """
    input : image, image with veg segmented, nb of crops to be detected
    output : array of the mask per crops, vanishing point 
    """
    bushy = 0
    vegetation_mask_new = np.copy(vegetation_mask)

    if((len(np.nonzero(vegetation_mask)[0])/(np.size(vegetation_mask)))>0.1): #if bushy 
        bushy = 1 
        vegetation_mask_new = cv2.Canny(vegetation_mask,100,200)

    outlier = [0, 0, 0]
    pts1 = []
    pts2 = []
    acc_theta = []
    acc_rho = []
    acc_weight = []
    row_image = np.copy(vegetation_mask_new)
    hough_image = np.copy(cv2.cvtColor(vegetation_mask_new, cv2.COLOR_GRAY2BGR))


    it = 0

    while(outlier is not None): 
        it+=1
        pts1_new, pts2_new, acc_theta_new, acc_rho_new, acc_weight_new, hough_image_new = calculate_main_rows(row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier) 
        vp = vp_detection(acc_theta_new, acc_rho_new, acc_weight_new) 
        outlier, acc_theta, acc_rho,  acc_weight, pts1, pts2, row_image = check_outliers(acc_theta_new, acc_rho_new, acc_weight_new, pts1_new, pts2_new, vp, row_image) 
        
        if(it>5): #too many crop are trying to be detected -> restarts process trying to found less crop rows 
            row_image = np.copy(vegetation_mask)
            hough_image = np.copy(image)
            pts1 = []
            pts2 = []
            acc_theta = []
            acc_rho = []
            acc_weight = []
            outlier = [0, 0, 0]
            nb_row = nb_row -1 
            it = 0
        
    masks = make_masks(pts1, pts2, vegetation_mask, bushy) 

    return masks, vp, hough_image, pts1, pts2, nb_row


def check_outliers(acc_theta_new, acc_rho_new, acc_weight_new,  pts1_new, pts2_new, vp, row_image):
    """
    input : info about the detcted lines (parameters in hough space and their points), vanishing point 
    output : outliers line informations, list of the line's parameters updated (outliers removed)
    """

    var_residual = 0
    eps_acc = []
    x0,y0 = vp
    V = sum(acc_weight_new)
    outlier = None 

    for t,r,w in zip(acc_theta_new, acc_rho_new, acc_weight_new):
        #calculate the distance between the VP and each crops
        eps = r - (x0*np.cos(t) + y0 * np.sin(t))  
        eps_acc.append(abs(eps))
        var_residual = var_residual + (w/V)*pow(eps,2) 

    var_residual = np.sqrt(var_residual)
    eps_max = max(eps_acc)
    idx_max = eps_acc.index(eps_max)

    if ((eps_max>1.25*var_residual) and (eps_max>20)):
        idx_outlier = idx_max
        outlier = [idx_outlier, acc_theta_new[idx_outlier], acc_rho_new[idx_outlier]]

        # remove the outlier from the lists describing the faulty crops
        acc_theta_new.pop(idx_outlier)
        acc_rho_new.pop(idx_outlier)
        acc_weight_new.pop(idx_outlier)
        pts1_new.pop(idx_outlier)
        pts2_new.pop(idx_outlier)

    else : 
        outlier = None 
        idx_outlier = None

    return outlier, acc_theta_new, acc_rho_new,  acc_weight_new, pts1_new, pts2_new, row_image


def make_masks(pts1, pts2, vegetation_mask, bushy):
    """
    inputs : points of different detected lines, vegetation mask, bushyness of the image
    returns : mask = image with 1 around line detected and 0 everywhere else 
    """

    masks = []
    if(bushy==0):
        band_width = int(vegetation_mask.shape[1]/20)
    if(bushy>0):
        band_width = int(vegetation_mask.shape[1]/10)

    for p1, p2 in zip(pts1, pts2):
        mask = np.zeros_like(vegetation_mask)
        cv2.line(mask, p1, p2, (255,255,255), int(band_width))
        masks.append(mask)

    return masks


def vp_detection(acc_theta, acc_rho, acc_weight):
    """
    input : all theta, rhos, and value in the accumulator of the main lines detected 
    output : the vanishing point = tuple, closest point to an intersection of the lines 
    """

    #Initialisation of variables 
    A = B = C = D = E = 0
    idx_outlier = None 
    outlier = None

    for t,r,w in zip(acc_theta, acc_rho, acc_weight):
        a = np.cos(t)
        b = np.sin(t)
        A = A + w*pow(a,2)
        B = B + w*pow(b,2)
        C = C + w*a*b
        D = D + w*a*r
        E = E + w*b*r

    M = np.array([[A,C],[C,B]])
    b = np.array([D,E]) 

    if(np.linalg.det(M)!=0): 
        x0,y0 = np.linalg.solve(M,b)
        x0 = int(x0)
    else : 
        x0 = y0 = 0
        print('ERROR : Vanishing point cant be calculated')
        exit(0)
    return ((x0+1),y0)


def find_acc_hough(mask, angle_acc, outlier):
    """
    input : 2D mask + list containing the angle previously found
    output : accumulator + array to convert theta and rhos to accumulator coordinates
    """

    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(0, 180)) #maybe dont make it so precise !
    width, height = mask.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, num = diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64)
    accumulator_copy = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(mask)  

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(len(thetas)):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
            accumulator_copy[rho, t_idx] += 1
            if (abs(np.rad2deg(thetas[t_idx])-90)<25): #if horizontale lignes 
                accumulator[:, t_idx] = 0
            
            for angle in angle_acc:
                if (abs(np.rad2deg(thetas[t_idx])-np.rad2deg(angle))<10): #if angle already detected 
                    accumulator[:, t_idx] = 0
    
    return accumulator, thetas, rhos


def calculate_main_rows(row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier):
    """
    Take the values from most nb_row most prominent line in the Hough Space
    input : image segmented, nb rows to be detected, informations about previously found lines (non - 0 in case of outliers)
    output : pts1, pts2 = list of points describing the detected lines + their hough space parameters rho, theta and their value and the accumulator
    """

    band_width = int(hough_image.shape[1]/30)

    while(len(pts1)<nb_row):
        accumulator, thetas, rhos = find_acc_hough(row_image, acc_theta, outlier)
        th_max = accumulator.max()
        r_idx, th_idx = np.where(accumulator>=th_max)
        rho = rhos[r_idx[0]]#in case multiple same max 
        theta = thetas[th_idx[0]]

        acc_theta.append(theta)
        acc_rho.append(rho)
        acc_weight.append(th_max)

        p1, p2 = r_th_to_pts(rho, theta)
        pts1.append(p1)
        pts2.append(p2)

        cv2.line(row_image, p1, p2, (0,0,0), int(band_width))
        cv2.line(hough_image, p1, p2, (0,255,0), 2)

    return pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image


def r_th_to_pts(rho, theta):
    """
    input : rho, theta = parameter of the hough space
    output : two points of the line in the image space
    """

    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    p1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    p2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    return p1, p2 

