import numpy as np
import matplotlib.pyplot as plt
import cv2
import MaskingProcess
#import Final_PyCode.MaskingProcess as MaskingProcess
from PIL import Image 
import extcolors
import skimage


INI_PROCESS = 0
SPEED_PROCESS = 1
ENDED_PROCESS = 2

def initial_process(img, sky_on):
    # keep also the color of the mask examined for more speeeeed
    # and the size of the image with no sky

    #Cut off sky
    if(sky_on==1) : 
        grad_sky = MaskingProcess.get_sky_region_gradient(img)
        img_no_sky = MaskingProcess.cut_image_from_mask(grad_sky, img)
    else : 
        img_no_sky = img

    #cv2.imshow('img_no_sky', img_no_sky)
    #cv2.waitKey(1000)

    #extract the main colors from the image 
    im_pil = Image.fromarray(img)
    colors_x = extcolors.extract_from_image(im_pil, tolerance = 12, limit = 12) 
    colors_rgb, colors_lab = MaskingProcess.colors_to_array(colors_x)

    #cv2.imshow('image ', img_no_sky)
    #cv2.waitKey(1000)

    #array with the masking of all main colors : 
    img_no_sky_lab = skimage.color.rgb2lab(img_no_sky/255)
    arr_mask = []
    for ref_color in colors_lab : 
        mask = MaskingProcess.mask_vegetation(img_no_sky_lab, ref_color)
        mask = cv2.medianBlur(mask,3)
        arr_mask.append(mask) 

    #HT to find keep the best mask : 
    best_mask, col_best_mask = MaskingProcess.keep_best_mask(arr_mask, img_no_sky, colors_lab)
    best_mask_edge = cv2.Canny(best_mask,100,200)

    #cv2.imshow('best mask ', best_mask)
    #cv2.imshow('best mask edge ', best_mask_edge)
    #cv2.waitKey(1000)
    
    #plt.imshow(best_mask_edge)
    crop_nb = 6
    arr_mask, th_acc, r_acc, threshold_acc = MaskingProcess.keep_mask_max_acc_lines(best_mask_edge, img_no_sky, crop_nb)
    x0,y0 = MaskingProcess.VP_detection(th_acc, r_acc, threshold_acc, best_mask_edge)
    vp_point = [x0,y0]
    img_no_sky_copy = np.copy(img_no_sky)
    #cv2.circle(img_no_sky_copy, (x0, y0), 10, (255,255,255), 5)
    #cv2.imshow('VP point', img_no_sky_copy)
    #cv2.waitKey(1000)
    #vp_point = 0
    return arr_mask, col_best_mask, vp_point

def quick_process(image, arr_mask, col_best_mask):

    #calculate best color mask based on previously calculated color 
    img_lab = skimage.color.rgb2lab(image/255)
    mask_col = MaskingProcess.mask_vegetation(img_lab, col_best_mask)
    mask_col_edge = cv2.Canny(mask_col,100,200)

    #offset, slope = MaskingProcess.ransac_on_existing_mask(arr_mask, mask_col_edge)
    #masked_img = MaskingProcess.draw_ransac_lines(image, offset, slope,mask_col_edge )
    masked_img = MaskingProcess.seq_line_det(arr_mask, mask_col_edge, image)


    rows = [0,0]
    return image, masked_img, rows

def annotation(images, sky_on):

    stage = INI_PROCESS

    if stage == INI_PROCESS: 
        print('initial process...')
        height_original = images[0].shape[0]
        arr_mask, col_best_mask, _ = initial_process(images[0], sky_on)
        height, _, _ = arr_mask[0].shape
        stage = SPEED_PROCESS
        #print(images[0].shape)


    if stage == SPEED_PROCESS:
        print('...speed process...')
        img_save = []
        arr_rows = []
        for idx, sing_img in enumerate(images) : 
            sing_img = sing_img[height_original-height:,:,:]
            #if ((idx%100)==0):
                #arr_mask, col_best_mask, _ = initial_process(sing_img,0)

            image_quick_process, arr_mask, rows_sing_frame = quick_process(sing_img, arr_mask, col_best_mask)
            img_save.append(image_quick_process)
            arr_rows.append(rows_sing_frame)
            
            #print('image nb ', idx)
            #cv2.imshow('results : ', image_quick_process)
            if cv2.waitKey(1) == ord('q'):
                break
        #print(sing_img.shape)
        images_annotated = img_save
        stage = ENDED_PROCESS


    if stage == ENDED_PROCESS :
        print('...processing done!')
        height, width, _ = images_annotated[0].shape
        """
        for i in range(len(images_annotated)):
            images_annotated[i] = cv2.resize(images_annotated[i], images_annotated[0], interpolation = cv2.INTER_AREA) #this resize makes the video not work?
            s = images_annotated[0].shape
            if ((images_annotated[i].shape) is not s):
                print('WDF')
        """
        return images_annotated

    print('error')

    return 0