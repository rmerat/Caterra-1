#Autonomous Crop Row Detection of a Laser-based Weeding Robot

## Table of contents
* [General info](#general-info)
* [Setup](#setup)

## General info
This project aims to develop a computer vision-based system for detecting crop rows
in agricultural fields, which is a crucial step in the development of an autonomous
agricultural rover robot, enabling the autonomous navigation by providing propri-
oceptive information. The system utilizes a combination of image processing tech-
niques, including color clustering for vegetation segmentation, Hough transform for
crop row detection, as well as vanishing point detection and RANSAC for improved
accuracy. 
Results show that the system is able to detect crop rows with a median
accuracy of 82% following the formula of the crop row detection accuracy developed
by Vidovi ́c et al	

	
## Setup
To run this project, clone it in a chosen repository and use the following command:

```
$ cd ../*your repo*
$ python main.py [picture/video]
```

The argument are as follow : 
- madatory argument : mode = 'picture' or 'video'
- optionnal argument : -n = number of crop row to be detected, default = 5. If less crop row then asked are visible on the image, the algorithm will automaticly detect less
- if mode = video : 
*optionnal argument : -k = number of frame before recalculation of the Hough transform, default = 5
- if mode = picture : 
*optionnal argument : -p = name of the file to be analyzed 
*optionnal argument : -e = evaluation on/off 