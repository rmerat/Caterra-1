import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
import os
import MaskingProcess

def analyze_img(img):
    analyzed_img = np.copy(img)
    cv2.line(analyzed_img, (10,100), (100, 10), (0,0,0), 30)
    return analyzed_img