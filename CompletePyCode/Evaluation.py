import os 
import cv2
import matplotlib.pyplot as plt
import csv
import numpy as np

def SaveData(cr, pts1, pts2):
    acc_m = []
    acc_b = []

    print('point ligne 0 : ', pts1[0], pts2[0])
    test = np.zeros_like(cr)
    for p1, p2 in zip(pts1, pts2):
        x1,y1 = p1
        x2,y2 = p2
        m = (x2-x1) / (y2-y1)
        b = x1 - m*y1

        acc_m.append(m)
        acc_b.append(b)
    
    with open('CropsPersonnalResults.txt', 'w') as f:
        for line in range(cr.shape[0]):
            pts = []
            for m,b in zip(acc_m, acc_b):
                pt = (line, int(m * line + b))
                pts.append(pt)
            
            pts = str(pts)
            f.write(pts)
            f.write('\n')

        """
        print(cr.shape[0])
        for line in range(cr.shape[0]):
            print(line)
            for m,b in zip(acc_m, acc_b):
                y = m * line + b
                cv2.circle(test, (int(line), int(y)), 1, (255,0,0),1 )

        cv2.imshow('cr : ', test )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cr = cv2.resize(cr, (320,240), interpolation = cv2.INTER_AREA) 
        """

    return cr

def LoadGroundTruth(name_images, imageHeight = 240, imageWidth = 320, nb_crop = 4):

    GroundTruthLink = '/home/roxane/Desktop/M3_2022/crop_dataset_annoted/GT data/crop_row_095.crp' #replace by name_images[0].crp and not .jpg
    imagePath = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines/'
    halfWidth = imageWidth/2;
    GTImage = cv2.imread(os.path.join(imagePath, name_images)) #+ .jpg??

    with open (GroundTruthLink, 'r') as f:
        cv = [row[0] for row in csv.reader(f,delimiter='\t')]
    with open (GroundTruthLink, 'r') as f:
        dv = [row[1] for row in csv.reader(f,delimiter='\t')]
    v0 = imageHeight - np.shape(dv)[0] + 1 #first image row where crop rows are present

    array_GT = np.zeros((imageHeight-v0, 11))

    for v in range(v0, imageHeight):
        center_dist = int(float(cv[v-v0]))
        spacing = int(float(dv[v-v0]))
        dist_x = halfWidth + center_dist
        cv2.circle(GTImage, (int(dist_x), v), 1, (255,0,0), 1)
        while(dist_x>0):
            dist_x = dist_x - spacing
            pt = (int(dist_x), v)
            cv2.circle(GTImage, pt, 1, (255,0,0), 1)
        
        dist_x = halfWidth + center_dist
        while(dist_x<imageWidth):
            dist_x = dist_x + spacing
            pt = (int(dist_x),v)
            cv2.circle(GTImage, pt, 1, (255,0,0), 1)

        for i in range(6):
            array_GT[(v-v0),5 + i] = halfWidth + center_dist + i*spacing
            array_GT[(v-v0),5 - i] = halfWidth + center_dist - i*spacing

        """array_GT[(v-v0), 0] = halfWidth + center_dist
        array_GT[(v-v0), 1] = halfWidth + center_dist + 2*spacing
        array_GT[(v-v0), 2] = halfWidth + center_dist + 3* spacing
        array_GT[(v-v0), 3] = halfWidth + center_dist + 4 *spacing"""
    
        #TODO : how to choose that automaticlly and not manually ??
        #  -->prob take all of the row of the ground truth and then after it will not matter and be replaced with 0
    print(array_GT[24, :])


    cv2.imshow('ground truth image : ', GTImage)
    cv2.waitKey(0)

    return GTImage, cv, dv, v0, array_GT


def evaluate_results(cv, dv, v0, array_GT, imageHeight = 240, imageWidth = 320, nb_crop_row = 4):
    halfWidth = imageWidth/2;

    path_my_results = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/CropsPersonnalResults.txt'
    with open (path_my_results, 'r') as f:
        cv_myresult = [row for row in csv.reader(f)]
    
    array_myresults = np.zeros((imageHeight,nb_crop_row*2))

    for idx_line, line in enumerate(cv_myresult):
        for idx_row, row in enumerate(line) :
            if (idx_row<nb_crop_row*2) : 
                b = int(row.strip(']').strip('[').strip(' (').strip(')'))
                array_myresults[idx_line, idx_row] = b

    m, sigma, score, test = [3, 0.2, 0, 0]
    array_myresults = array_myresults[v0:,:]
    array_myresults = array_myresults[:,1::2]

    #print(array_myresults)
    #print(array_GT)


    score_crda = 0

    for v in range(array_GT.shape[0]):
        d_v = int(float(dv[v]))
        for i in range(array_GT.shape[1]):
            for j in range(nb_crop_row):
                s = max(1-(pow(((array_GT[v,i] - array_myresults[v,j])/(sigma*d_v)),2)),0)
                score_crda = score_crda + s
    score_crda = score_crda/(array_GT.shape[0]*nb_crop_row)
    print('score : ', score_crda)
    return score_crda, 0


