import numpy as np
import cv2
import csv
import os


VID = 0
IMG = 1

def SaveData(height, row, img_name, mode, validity = 1):
    """
    inputs : height of the image, row = concatenation of 2 lists containing the points of the detected crop row
    outputs : name of where the results are saved 
    """

    acc_m = []
    acc_b = []
    pts1 = row[0]
    pts2 = row[1]

    for p1, p2 in zip(pts1, pts2):
        x1,y1 = p1
        x2,y2 = p2
        m = (x2-x1) / (y2-y1)
        b = x1 - m*y1

        acc_m.append(m)
        acc_b.append(b)


    folder = os.getcwd()
    if(mode == IMG):
        name = img_name.replace(".JPG", "_Results.txt").replace(".jpg", "_Results.txt") 
        name = os.path.join(folder, "ImagesResults", name)

    else : 
        name = os.path.join(folder, "VideoDatasetResults", img_name + "Results.txt")

    with open(name, 'w') as f:
        for line in range(height):
            pts = []
            for m,b in zip(acc_m, acc_b):
                if(validity==1):
                    pt = (line, int(m * line + b))
                else :
                    pt = 0
                    print('non valid frame ')
                pts.append(pt)
            
            
            pts = str(pts)
            f.write(pts)
            f.write('\n')
    
    return name


def LoadGroundTruth(name_images, img_annotated, height_sky, imageHeight = 240, imageWidth = 320):
    """
    Creates array_GT = array containing for each line of the picture, the point of the ground truth's crops
    input : name of the image, image, height of the sky
    output : Image with the Ground Truth crops drawned, Comparaison Image, dv = distance between crops for each line, v0 = line where the GT information starts, array_GT
    
    """

    GroundTruthPath = os.getcwd() +'/GroundTruth/'
    GroundTruthName = name_images.replace(".JPG", ".crp").replace(".jpg", ".crp") 
    GroundTruthLink = os.path.join(GroundTruthPath, GroundTruthName)
    imagePath = os.getcwd() + '/Images/'
    halfWidth = int(imageWidth/2)
    ComparaisonImage =  np.copy(img_annotated) 

    GTImage = cv2.imread(os.path.join(imagePath, name_images)) 

    with open (GroundTruthLink, 'r') as f:
        cv = [row[0] for row in csv.reader(f,delimiter='\t')]
    with open (GroundTruthLink, 'r') as f:
        dv = [row[1] for row in csv.reader(f,delimiter='\t')]

    v0 = imageHeight - np.shape(dv)[0] 

    if(v0<=height_sky) :#first row analyze = height sky 
        GTImage = GTImage[height_sky:,:]
        cv = cv[abs(v0-height_sky):]
        dv = dv[abs(v0-height_sky):]
        array_GT = np.zeros(((imageHeight-height_sky),11))
        start = height_sky
    
    if(v0>height_sky): # first row analyze = v0
        array_GT = np.zeros(((imageHeight-v0),11))
        start = v0
        GTImage = GTImage[v0:,:]
        ComparaisonImage = ComparaisonImage[abs(v0-height_sky):,:]

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
            array_GT[v,5 - i] = halfWidth + center_dist - i*spacing 


    return GTImage, ComparaisonImage, dv, v0, array_GT


def evaluate_results(dv, v0, height_sky, array_GT, name, imageHeight = 240, imageWidth = 320, nb_crop_row = 5):
    """
    Calculates the CRDA score per Image 
    """

    with open (name, 'r') as f:
        cv_myresult = [row for row in csv.reader(f)] #contains for each row of the image the position of the detected crop

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

    array_myresults = array_myresults[:,1::2] #keep only vertical position
    score_crda = 0
    sigma = 0.3 #if a point is more then 30% of the distance between two adjacant crop wrong, put to 0

    for v in range(array_GT.shape[0]): #CRDA formula 
        d_v = int(float(dv[v]))
        for i in range(array_GT.shape[1]):
            for j in range(nb_crop_row):
                s = max(1-(pow(((array_GT[v,i] - array_myresults[v,j])/(sigma*d_v)),2)),0)
                score_crda = score_crda + s

    score_crda = score_crda/(array_GT.shape[0]*nb_crop_row)
    
    return score_crda


