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
        #m = (y2-y1) / (x2-x1)
        #b = y1 - m* x1

        m = (x2-x1) / (y2-y1)
        b = x1 - m*y1

        #print('equation of line : y = ', m, '*x + ', b)
        acc_m.append(m)
        acc_b.append(b)
    
    #print('acc  :', acc_m, 'acc b : ', acc_b)

    with open('CropsPersonnalResults.txt', 'w') as f:
        for line in range(cr.shape[0]):
            pts = []
            for m,b in zip(acc_m, acc_b):
                pt = (line, int(m * line + b))
                pts.append(pt)
            
            pts = str(pts)
            #print('what will be written : ', pts)
            #print(pt)
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

        """cv2.circle(test, (int(line), int(y)), 1, (255,0,0),1 )

    
        for nb_line, line in enumerate(cr) : #= cr[0] 
            x = np.where(line>0)
            pt = []
            for x_sing in x[0] : 
                #pt.append((x_sing, nb_line))
                pt.append((x_sing, nb_line))
            pt = str(pt)
            #print(pt)
            f.write(pt)
            f.write('\n')"""



    return cr

def LoadGroundTruth(name_images, imageHeight = 240, imageWidth = 320):

    GroundTruthLink = '/home/roxane/Desktop/M3_2022/crop_dataset_annoted/GT data/crop_row_023.crp' #replace by name_images[0]
    imagePath = '/home/roxane/Desktop/M3_2022/Caterra/dataset_straigt_lines/'
    halfWidth = imageWidth/2;
    name_temp = 'crop_row_023.JPG'
    print(os.path.join(imagePath, name_temp)) # name_images[0]))
    GTImage = cv2.imread(os.path.join(imagePath, name_temp)) #+ .jpg??
    
    with open (GroundTruthLink, 'r') as f:
        cv = [row[0] for row in csv.reader(f,delimiter='\t')]
    with open (GroundTruthLink, 'r') as f:
        dv = [row[1] for row in csv.reader(f,delimiter='\t')]
        #first image row where crop rows are present
    v0 = imageHeight - np.shape(dv)[0] + 1

    cop = np.copy(GTImage)

    for v in range(v0, imageHeight):
        center_dist = int(float(cv[v-v0]))
        spacing = int(float(dv[v-v0]))


        #print(v,halfWidth + c_int)
        dist_to_center = int(halfWidth + center_dist)
        #c_tot_2 = int(halfWidth + c_int - d_int)
        p1 = (center_dist, v)
        dist_x = center_dist

        #TODO : stop doing it manually, implement it in the argument with nb rows 
        while(dist_x>0):
            dist_x = dist_x - spacing
            cv2.circle(cop, (dist_x,v), 1, (255,0,0), 1)
            #remove spacing from p1
        while(dist_x<imageWidth):
            dist_x = dist_x + spacing
            cv2.circle(cop, (dist_x,v), 1, (255,0,0), 1)

            #add spacing to p1


        """p2 = (center_dist + spacing, v)
        p3 = (center_dist - spacing, v)"""
        #print(p1,p2,p3)

        #cv2.imshow('basic image : ', cop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #p4 = (c_tot_1 + 2*d_int, v)

        """cv2.circle(cop, p1, 1, (255,0,0), 1)
        cv2.circle(cop, p2, 1, (0,255,0), 1)
        cv2.circle(cop, p3, 1, (0,0, 255), 1)"""
        #cv2.circle(cop, p4, 1, (0,255, 255), 1)

    cv2.imshow('ground truth image : ', cop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return cop, cv, dv, v0


def evaluate_results(cv,dv, v0, imageHeight = 240, imageWidth = 320, nb_crop_row=3):
    halfWidth = imageWidth/2;

    path_my_results = '/home/roxane/Desktop/M3_2022/Caterra_2912/Caterra/CropsPersonnalResults.txt'
    with open (path_my_results, 'r') as f:
        cv_myresult = [row for row in csv.reader(f)]
        #print(len(cv_myresult))

    array = np.zeros((imageHeight,nb_crop_row*2))
    for idx_line, line in enumerate(cv_myresult):
        #print('line : ', len(line), line)
        for idx_row, row in enumerate(line) :
            if (idx_row<nb_crop_row*2) : 
                b = int(row.strip(']').strip('[').strip(' (').strip(')'))
                #print(b)
                array[idx_line, idx_row] = b

    m, sigma, score, test = [3, 0.2, 0, 0]

    for v in range(v0, imageHeight):
        c_GT = int(float(cv[v-v0]))
        d_GT = int(float(dv[v-v0]))

        #print(v,halfWidth + c_GT)
        for i in range(-1,2) : 
            u = array[v][2*(i+1)] # int(halfWidth + c_GT) + i*d_GT  # array[v][2*(i+1)] #int(halfWidth + c_GT) + i*d_GT 
            u_GT = int(halfWidth + c_GT) + i*d_GT
            diff = u_GT-u
            #print(diff)
            test = test + diff
            #print(u_GT, array[v][2*(i+1)])
            t = pow((diff/(sigma*d_GT)),2)
            #print(t)
            s = max(1-t,0)
            score = score + s
    
    score = score / (m*(imageHeight-v0))
            
    #mean = test/(3*(imageHeight-v0))
    #precision = 100 - 100*(mean/imageWidth)

    print('score : ', score, '\naverage precision : ')

    return score, 0
"""

m, sigma, result, test = [3, 0.2, 0, 0]

for v in range(v0, imageHeight):
    c_GT = int(float(cv[v-v0]))
    d_GT = int(float(dv[v-v0]))

    #print(v,halfWidth + c_GT)
    for i in range(-1,3) : 
        u = array[v][2*(i+1)] # int(halfWidth + c_GT) + i*d_GT  # array[v][2*(i+1)] #int(halfWidth + c_GT) + i*d_GT 
        u_GT = int(halfWidth + c_GT) + i*d_GT
        diff = u_GT-u
        test = test + diff
        #print(u_GT, array[v][2*(i+1)])
        t = pow((diff/(sigma*d_GT)),2)
        s = max(1-t,0)
        result = result + s
        
mean = test/(3*(imageHeight-v0))
prec = 100*(mean/imageWidth)
print('score : ', result, '\naverage precision : ', 100 - prec, '%')"""



