# Autonomous Crop Row Detection of a Laser-based Weeding Robot

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Output](#output)

## General info
This project aims to develop a computer vision-based system for detecting crop rows
in agricultural fields, which is a crucial step in the development of an autonomous
agricultural rover robot, enabling the autonomous navigation by providing propri-
oceptive information. The system utilizes a combination of image processing tech-
niques, including color clustering for vegetation segmentation, Hough transform for
crop row detection, as well as vanishing point detection and RANSAC for improved
accuracy. 
Results show that the system is able to detect crop rows with a median
accuracy of 82% following the formula of the crop row detection accuracy.

	
## Setup
To run this project, clone it in a chosen repository and use the following command:

```
$ cd ../*your repo*
$ pip install -r requirements.txt
$ python main.py [picture/video]
```

The arguments are as follow : 
- mandatory : mode = 'picture' or 'video'
- optional : -n = maximum number of crop row to be detected, default = 5 
- if mode = video : 
    * optional : -k = number of frame before recalculation of the Hough transform, default = 5
- if mode = picture : 
    * optional : -p = name of the file to be analyzed 
    * optional : -e = evaluation on/off 


## Output


For Video Mode : 
- Annotated frames will be saved in *CropRowDetection/VideoDatasetResults* under the name *img_[idx].jpg*
- Numerical results per frame will be saved in the same folder under the name *img_[idx]_Results.txt"*

For Picture Mode : 
- Numerical results will be saved in *CropRowDetection/ImagesResults* under *[nameoftheoriginalimage]_Results.txt*
- Annotated images will be saved under their original non-annotated name in the same folder
