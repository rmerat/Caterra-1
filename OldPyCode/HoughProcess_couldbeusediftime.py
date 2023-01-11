import math
import numpy as np
import cv2 
import matplotlib.pyplot as plt 


def find_approx(image, vegetation_mask, nb_row):
    """
    input : image with veg segmented
    output : array of the mask per crops, vanishing point 
    """
    #put cond on bushes ?
    #here put a loop with while(outliers)

    outlier = [0, 0, 0]
    pts1 = []
    pts2 = []
    acc_theta = []
    acc_rho = []
    acc_weight = []
    row_image = np.copy(vegetation_mask) #here we delete little by little the row detected
    hough_image = np.copy(image)

    #il faut faire 2 acc theta et tout : l'un ou l'on remove les outliers et l'autre pas (pour ne pas les redecouvrire)
    while(outlier is not None):
        print('OUTLIER : ', outlier)
        pts1_new, pts2_new, acc_theta_new, acc_rho_new, acc_weight_new, hough_image_new, eliminated = calculate_main_rows(vegetation_mask, row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier) #il faut envoyer une image avec les rows des outliers supprimös 
        #prob inckuse in the argument acc_ stuff because some stuff will be deletd in eliminate outliers
        vp = vp_detection(acc_theta_new, acc_rho_new, acc_weight_new) 
        #outliers = None 
        outlier, idx_outliers, acc_theta, acc_rho,  acc_weight, pts1, pts2, row_image = check_outliers(acc_theta_new, acc_rho_new, acc_weight_new, pts1_new, pts2_new, vp, row_image, eliminated) #peut etre donner pts ET pts1_tot et dans cette fonction els differencier ?
        #pts1, pts2 = remove_outliers(pts1, pts2, acc_theta, acc_rho, acc_weight, outliers, idx_outliers)

    cv2.imshow('hough image', hough_image)
    cv2.waitKey(0)

    masks = make_masks(pts1, pts2, vp, vegetation_mask) #question : est ce que le mask doit passer par le vp?


    return masks, vp, hough_image

def remove_outliers(pts1, pts2, acc_theta, acc_rho, acc_weight, outliers, idx_outliers):
    if outliers is not None: 
        pts1.pop(idx_outliers)
        pts2.pop(idx_outliers)

    return pts1, pts2

def check_outliers(acc_theta_new, acc_rho_new, acc_weight_new,  pts1_new, pts2_new, vp, row_image, eliminated ):

    var_residual = 0
    eps_acc = []
    x0,y0 = vp
    V = sum(acc_weight_new)
    outlier = None 

    print('acc_weight_old', acc_weight_new)


    for t,r,w in zip(acc_theta_new, acc_rho_new, acc_weight_new):
        eps = r - (x0*np.cos(t) + y0 * np.sin(t)) #fct of 
        eps_acc.append(abs(eps))
        var_residual = var_residual + (w/V)*pow(eps,2) 
        # peut eetre que toutes les lignes deraient etre considere de la meme importance
        #print('\n eps : ', eps, 'estimates rho : ', (x0*np.cos(t) + y0 * np.sin(t)), 'rho : ', r, 'var residual : ', var_residual)

    var_residual = np.sqrt(var_residual)
    eps_max = max(eps_acc)
    idx_max = eps_acc.index(eps_max)

    if ((eps_max>1.25*var_residual) and (eps_max>20)):
        idx_outlier = idx_max
        outlier = [idx_outlier, acc_theta_new[idx_outlier], acc_rho_new[idx_outlier]]
        print('outlier find : ', outlier) # , idx_outlier, eps_max, var_residual)
        #remove it from the lists of lines 
        acc_theta_new.pop(idx_outlier)
        acc_rho_new.pop(idx_outlier)
        acc_weight_new.pop(idx_outlier)
        pts1_new.pop(idx_outlier)
        pts2_new.pop(idx_outlier)
        #add back the outliers to row_image 
        cv2.imshow('before reputting outliers : ', row_image)

        cv2.imshow('what we are gonna reput : ', eliminated[idx_outlier])

        row_image = row_image + eliminated[idx_outlier]
        cv2.imshow('after reputting outliers : ', row_image)
        #now the
    else : 
        outlier = None 
        idx_outlier = None
        print('no outliers')

    print('en of fct : ', outlier)
    print('acc_weight_new', acc_weight_new)
    #maybe remove outlier from pts1 and pts2
    return outlier, idx_outlier, acc_theta_new, acc_rho_new,  acc_weight_new, pts1_new, pts2_new, row_image

def make_masks(pts1, pts2, vp, vegetation_mask):

    masks = []
    band_width = int(vegetation_mask.shape[1]/30)

    for p1, p2 in zip(pts1, pts2):
        #print('p1 p2 : ', p1, p2)
        mask = np.zeros_like(vegetation_mask)
        cv2.line(mask, p1, p2, (255,255,255), int(band_width))
        masks.append(mask)


    return masks

def vp_detection(acc_theta, acc_rho, acc_weight):

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
    b = np.array([D,E]) #TODO : find another name 

    if(np.linalg.det(M)!=0): 
        x0,y0 = np.linalg.solve(M,b)
        x0 = int(x0)
    else : 
        x0 = y0 = 0
        print('sing matrix ') #TODO : reflechir à ce que ca veut dire 

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
    y_idxs, x_idxs = np.nonzero(mask)  

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(len(thetas)):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
            if (abs(np.rad2deg(thetas[t_idx])-90)<25): #if horizontale lignes 
                accumulator[:, t_idx] = 0
            
            for angle in angle_acc:
                if (abs(np.rad2deg(thetas[t_idx])-np.rad2deg(angle))<10): #if angle already detected 
                    accumulator[:, t_idx] = 0

            #if (outlier[0]!=0):
                #print('rho : ', rho)
                #print('rho outliers : ', outlier[2])

            #here remove square in acc around outlier detected 
            if (((abs(np.rad2deg(thetas[t_idx]))-outlier[1])<5) and (rho-outlier[2]<3)):
                print('the weight would have been : ', accumulator[:, t_idx])
                print('rho and rho outliers : ', rho, outlier[2])
                print('angle and engle outliers : ', abs(np.rad2deg(thetas[t_idx])), outlier[1])
                accumulator[:, t_idx] = 0
        

            #add here something about outliers 
    
    return accumulator, thetas, rhos

def calculate_main_rows(vege_image, row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier):

    band_width = int(hough_image.shape[1]/30)
    i = 0
    eliminated = []
    print('begining fct : len pts', len(pts1))

    while(len(pts1)<nb_row):
        eliminate_single = np.zeros_like(row_image)

        i=i+1

        print('step ', i, 'of ', nb_row)

        accumulator, thetas, rhos = find_acc_hough(row_image, acc_theta, outlier)
        plt.imshow(accumulator)
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
        cv2.line(eliminate_single, p1, p2, (255,0,0), int(band_width))
        eliminated.append(cv2.bitwise_and(vege_image, vege_image, mask = eliminate_single))
        #cv2.imshow('on_going_mask', on_going_mask)
        #cv2.waitKey(0)
        cv2.line(hough_image, p1, p2, (255,0,0), 2)
        #cv2.imshow('hough_image', hough_image)
        #cv2.waitKey(0)
        cv2.imshow('row image : ', row_image)
        cv2.waitKey(0)

    print('end fct : len pts', len(pts1))

    return pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, eliminated

def r_th_to_pts(rho, theta):
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        p1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        p2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        return p1, p2 

