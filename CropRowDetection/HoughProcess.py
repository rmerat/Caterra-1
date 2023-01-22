import math
import numpy as np
import cv2 


def find_approx(image, vegetation_mask, nb_row):
    """
    input : image with veg segmented
    output : array of the mask per crops, vanishing point 
    """
    bushy = 0
    vegetation_mask_new = np.copy(vegetation_mask)

    if((len(np.nonzero(vegetation_mask)[0])/(np.size(vegetation_mask)))>0.1): #if bushy 
        print('bushy')
        bushy = 1 
        vegetation_mask_new = cv2.Canny(vegetation_mask,100,200)

    outlier = [0, 0, 0]
    pts1 = []
    pts2 = []
    acc_theta = []
    acc_rho = []
    acc_weight = []
    row_image = np.copy(vegetation_mask_new) #here we delete little by little the row detected
    hough_image = np.copy(cv2.cvtColor(vegetation_mask_new, cv2.COLOR_GRAY2BGR))


    it=0
    #il faut faire 2 acc theta et tout : l'un ou l'on remove les outliers et l'autre pas (pour ne pas les redecouvrire)
    while(outlier is not None):
        it+=1
        pts1_new, pts2_new, acc_theta_new, acc_rho_new, acc_weight_new, hough_image_new = calculate_main_rows(row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier) 
        vp = vp_detection(acc_theta_new, acc_rho_new, acc_weight_new) 
        outlier, idx_outliers, acc_theta, acc_rho,  acc_weight, pts1, pts2, row_image = check_outliers(acc_theta_new, acc_rho_new, acc_weight_new, pts1_new, pts2_new, vp, row_image) 
        
        if(it>5):
            row_image = np.copy(vegetation_mask) #here we delete little by little the row detected
            hough_image = np.copy(image)
            pts1 = []
            pts2 = []
            acc_theta = []
            acc_rho = []
            acc_weight = []
            outlier = [0, 0, 0]
            nb_row = nb_row -1 
            print('nb_crop -=1')
            it = 0
        
        """if outlier is not None:
            print(pts1_new,pts2_new, 'idx outliers : ', idx_outliers)
            print('OUTLIERS')
            #for p1, p2 in zip(pts1_new, pts2_new):
                #cv2.line(hough_image, p1,p2, (255,0,0), 1)
            
            th = outlier[1]
            rho = outlier[2]
            p1,p2 = r_th_to_pts(rho, th)
            cv2.line(hough_image, p1, p2, (0,0,255), int(2))


            cv2.imshow('hough image with outliers : ', hough_image)
            cv2.waitKey(0)"""

    masks = make_masks(pts1, pts2, vp, vegetation_mask, bushy) 

    return masks, vp, hough_image, pts1, pts2, nb_row


def check_outliers(acc_theta_new, acc_rho_new, acc_weight_new,  pts1_new, pts2_new, vp, row_image):

    var_residual = 0
    eps_acc = []
    x0,y0 = vp
    V = sum(acc_weight_new)
    outlier = None 

    for t,r,w in zip(acc_theta_new, acc_rho_new, acc_weight_new):
        #calculate the distance between the VP and each crops
        eps = r - (x0*np.cos(t) + y0 * np.sin(t)) #fct of 
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

    return outlier, idx_outlier, acc_theta_new, acc_rho_new,  acc_weight_new, pts1_new, pts2_new, row_image


def make_masks(pts1, pts2, vp, vegetation_mask, bushy):

    masks = []
    #cv2.imshow('vegetation_mask', vegetation_mask)
    #cv2.waitKey(0)
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

            #add here something about outliers 
    #plt.imshow(accumulator_copy)
    #plt.xlabel("\u03C1")
    #plt.xlim(-diag_len, diag_len)
    #plt.ylabel("\u03F4")
    #plt.ylim(0, np.pi)

    #plt.show()
    
    return accumulator, thetas, rhos


def calculate_main_rows(row_image, nb_row, pts1, pts2, acc_theta, acc_rho, acc_weight, hough_image, outlier):
    """
    Take the values from most nb_row most prominent line in the Hough Space
    input : 
    output :
    """

    band_width = int(hough_image.shape[1]/30)
    i = 0

    while(len(pts1)<nb_row):

        i=i+1

        #print('step ', i, 'of ', nb_row)

        accumulator, thetas, rhos = find_acc_hough(row_image, acc_theta, outlier)
        #plt.imshow(accumulator)
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

    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    p1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    p2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    return p1, p2 

