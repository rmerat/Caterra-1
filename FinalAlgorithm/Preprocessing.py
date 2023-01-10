import numpy as np 
import cv2 
from scipy.signal import medfilt
from PIL import Image
from colormap import rgb2hex
from skimage.color import rgb2lab
import extcolors

def init(image):

    height_sky = sky_process(image)
    col_veg = extract_veg_colour(image) #maybe to do it image with no sky?

    return height_sky, col_veg


def get_sky_region_gradient(image):
    """
    input : rgb image 
    output : image with ground px put to 0
    """

    w = image.shape[1]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (9, 3))
    img_gray = cv2.medianBlur(img_gray, 5)
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)

    gradient_mask = (lap < 5).astype(np.uint8) # keep region with small laplacian

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel) #erosion, minimum of neighbouring px

    for i in range(w): #for each column
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20: # if the sky is bigger then 20 px starting from the top
                mask[first_one_index:first_zero_index, i] = 1 # put 1 between sky and land
                mask[first_zero_index:, i] = 0 # put 0 in land (appears black)
                mask[:first_one_index, i] = 0 # put 0 before the sky starts 
        except:
            continue

    grad_sky = cv2.bitwise_and(image, image, mask = mask)

    return grad_sky


def cut_image_from_mask(grad_sky,img):
    """
    input : gradient of the image, image 
    output : height of the sky 
    """

    low = np.array([1,1,1])
    high = np.array([256,256,256])
    masked_sky = cv2.inRange(grad_sky, low, high)


    h,w = masked_sky.shape
    i = h #start from the buttom of the image
    count = flag = j = 0

    while((i>0) & (flag==0)): # slowly go up, stopping if 50% of a row is considered sky 
        i = i-1 
        j = count = 0
        while(j<w):
            if (masked_sky[i,j] == 255) : # if px is sky 
                count = count + 1
            j = j+1
            if (count > (w*0.5)): 
                sky_height = i; 
                flag = 1
    
    return sky_height


def sky_process(image) : 
    """
    input : image rgb with sky on 
    output : height of the sky
    """

    grad_sky = get_sky_region_gradient(image)
    height_sky = cut_image_from_mask(grad_sky, image)

    if (height_sky>(image.shape[0])/2): #if too much sky 
        height_sky = 0

    return height_sky


def colors_to_array(colors_x) : 
    """
    input : tuple containing list of the main colors cluster of the images
    output : np-array containg the cluster colours in rgb space
    """
    colors_rgb = np.zeros((len(colors_x[0]),3))

    for i in range(len(colors_x[0])):
        col = colors_x[0][i][0]
        colors_rgb[i] = col

    return colors_rgb


def extract_rgb_colours(image):
    """
    input : rgb image as np array
    output : list of main rgb colors 
    """
    image_pil = Image.fromarray(image)
    colors_x = extcolors.extract_from_image(image_pil, tolerance = 12, limit = 8)
    rgb_colours = colors_to_array(colors_x)
    #donuts(colors_x)# for debugging 
    return rgb_colours


def extract_greenest_colour(colors_rgb, image_rgb=0):
    """
    input : list of rgb colors
    output : greenest color in rgb format
    """
    diff = smallest_diff = float('inf')
    col_best_mask = [0, 255, 0]

    for col in colors_rgb:
        name = 'diferent maske for col ' + str(col[0]) + str(col[1])+ str(col[2])
        col_lab = rgb2lab((col[0]/255, col[1]/255, col[2]/255))
        masked_img = cv2.inRange(rgb2lab(image_rgb/255), col_lab - [8,8,8], col_lab + [8,8,8])
        cv2.imshow(name, masked_img)


        diff = np.linalg.norm(np.asarray(col - [0,255,0])) # maybe better to use lab format?
        if diff < smallest_diff: # if closest to green
            smallest_diff = diff
            col_best_mask = col

    col_best_mask = col_best_mask.astype(int)
    r,g,b = (col_best_mask.data)
    
    return col_best_mask


def extract_veg_colour(image):
    """
    input : image, rgb space
    output : colour of the vegetation to be used in later masking 
    """
    
    rgb_colours = extract_rgb_colours(image)
    colour = extract_greenest_colour(rgb_colours, image)

    return colour 

def mask_vegetation(image_lab, colour_veg_lab):
    """
    input : image in lab space
    output : vegetation mask
    """
    # Using inRange method, to create a mask
    thr = [8,8,8] # 4,4,4 for vid? before : 8 20 8 , TODO : maybe change  thr according to histogram ??
    lower_col = colour_veg_lab - thr
    upper_col = colour_veg_lab + thr
    mask = cv2.inRange(image_lab, lower_col, upper_col)

    return mask

def get_vegetation_mask(image, height_sky, colour_veg_rgb):

    # convert color and image to lab space
    image_lab = rgb2lab(image/255)
    colour_veg_lab = rgb2lab((colour_veg_rgb[0]/255, colour_veg_rgb[1]/255, colour_veg_rgb[2]/255))

    # vegetation segmentation using mask of the detected vegetal color
    vegetation_mask = mask_vegetation(image_lab, colour_veg_lab)
    #best_mask_median = cv2.medianBlur(best_mask,3) #may be useful?

    return vegetation_mask