import numpy as np
import cv2
import csv
import os

def SaveData(cr, row, type):

    acc_m = []
    acc_b = []
    pts1 = row[0]
    pts2 = row[1]

    test = np.zeros_like(cr)
    for p1, p2 in zip(pts1, pts2):
        x1,y1 = p1
        x2,y2 = p2
        m = (x2-x1) / (y2-y1)
        b = x1 - m*y1

        acc_m.append(m)
        acc_b.append(b)
    

    if(type ==0): 
        name = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/FinalAlgorithm/CropsPersonnalResultsRANSAC.txt'

    if(type== 1):
        name = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/FinalAlgorithm/CropsPersonnalResultsHough.txt'


    with open(name, 'w') as f:
        for line in range(cr.shape[0]):
            pts = []
            for m,b in zip(acc_m, acc_b):
                pt = (line, int(m * line + b))
                pts.append(pt)
            
            pts = str(pts)
            f.write(pts)
            f.write('\n')

        #cr = cv2.resize(cr, (320,240), interpolation = cv2.INTER_AREA) 

    return cr


def LoadGroundTruth(name_images, img_annotated, height_sky, imageHeight = 240, imageWidth = 320, nb_crop = 5):

    GroundTruthPath = '/home/roxane/Desktop/M3_2022/crop_dataset_annoted/GT data/'
    GroundTruthName = name_images.replace(".JPG", ".crp").replace(".jpg", ".crp") #crop_row_095.crp' #replace by name_images[0].crp and not .jpg
    GroundTruthLink = os.path.join(GroundTruthPath, GroundTruthName)
    imagePath = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines/'
    halfWidth = imageWidth/2;
    ComparaisonImage =  np.copy(img_annotated) 

    GTImage = cv2.imread(os.path.join(imagePath, name_images)) #+ .jpg??
    #imageHeight = np.shape(img_annotated)[0]

    with open (GroundTruthLink, 'r') as f:
        cv = [row[0] for row in csv.reader(f,delimiter='\t')]
    with open (GroundTruthLink, 'r') as f:
        dv = [row[1] for row in csv.reader(f,delimiter='\t')]

    v0 = imageHeight - np.shape(dv)[0]# + 1 #first image row where crop rows are present --> why +1 before?

    if(v0<=height_sky):
        GTImage = GTImage[height_sky:,:]
        cv = cv[abs(v0-height_sky):]
        dv = dv[abs(v0-height_sky):]
        array_GT = np.zeros(((imageHeight-height_sky),11))
        start = height_sky
    
    if(v0>height_sky):
        array_GT = np.zeros(((imageHeight-v0),11))
        start = v0
        GTImage = GTImage[v0:,:]
        ComparaisonImage = ComparaisonImage[abs(v0-height_sky):,:]

    Image =  np.copy(GTImage) 

    for v in range(0, imageHeight-start):
        center_dist = int(float(cv[v]))
        spacing = int(float(dv[v]))
        dist_x = halfWidth + center_dist
        cv2.circle(GTImage, (int(dist_x), v), 1, (0,255,0), 1)
        cv2.circle(ComparaisonImage, (int(dist_x), v), 1, (0,255,0), 1)

        while(dist_x>0):
            dist_x = dist_x - spacing
            pt = (int(dist_x), v)
            cv2.circle(GTImage, pt, 1, (0,255,0), 1)
            cv2.circle(ComparaisonImage, (int(dist_x), v), 1, (0,255,0), 1)

        
        dist_x = halfWidth + center_dist
        while(dist_x<imageWidth):
            dist_x = dist_x + spacing
            pt = (int(dist_x),v)
            cv2.circle(GTImage, pt, 1, (0,255,0), 1)
            cv2.circle(ComparaisonImage, (int(dist_x), v), 1, (0,255,0), 1)


        for i in range(6):
            array_GT[v,5 + i] = halfWidth + center_dist + i*spacing
            array_GT[v,5 - i] = halfWidth + center_dist - i*spacing #before v0 instead of height sky

    #cv2.imshow('ground truth image : ', GTImage)
    #cv2.imshow('ground truth + personal result image : ', imagePerso)

    return GTImage, ComparaisonImage, Image, cv, dv, v0, array_GT


def evaluate_results(cv, dv, v0, height_sky, array_GT, type=0, imageHeight = 240, imageWidth = 320, nb_crop_row = 5):
    halfWidth = imageWidth/2

    path_my_results = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/'

    if(type ==0): 
        #name = 1
        #name = 'FinalAlgorithm/CropsPersonnalResultsRANSAC.txt'
        path = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/FinalAlgorithm/CropsPersonnalResultsRANSAC.txt'
        #path = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/CropsPersonnalResultsRANSAC.txt'

    if(type== 1):
        #name = 'FinalAlgorithm/CropsPersonnalResultsHough.txt'
        #path_my_results = os.path.join(path_my_results, "FinalAlgorithm", "CropsPersonnalResultsHough.txt")
        path = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/FinalAlgorithm/CropsPersonnalResultsHough.txt'
        #path = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/CropsPersonnalResultsHough.txt'


    with open (path, 'r') as f:
        cv_myresult = [row for row in csv.reader(f)]

    if(v0<=height_sky):
        array_myresults = np.zeros((imageHeight-height_sky,nb_crop_row*2))

    if(v0>height_sky):
        array_myresults = np.zeros((imageHeight-v0,nb_crop_row*2))
        cv_myresult = cv_myresult[abs(v0-height_sky):]

    for idx_line, line in enumerate(cv_myresult):
        for idx_row, row in enumerate(line) :
            if (idx_row<nb_crop_row*2) : 
                b = int(row.strip(']').strip('[').strip(' (').strip(')'))
                array_myresults[idx_line, idx_row] = b

    #m, sigma, score, test = [3, 0.3, 0, 0]
    #array_myresults = array_myresults[v0:,:]
    array_myresults = array_myresults[:,1::2]
    score_crda = 0
    sigma = 0.3

    for v in range(array_GT.shape[0]):
        d_v = int(float(dv[v]))
        for i in range(array_GT.shape[1]):
            for j in range(nb_crop_row):
                s = max(1-(pow(((array_GT[v,i] - array_myresults[v,j])/(sigma*d_v)),2)),0)
                score_crda = score_crda + s
    score_crda = score_crda/(array_GT.shape[0]*nb_crop_row)
    
    """if(type ==0): 
        print('score RANSAC: ', score_crda)

    if(type== 1):
        print('score Hough: ', score_crda)"""

    return score_crda, cv_myresult


