import numpy as np
import cv2 as cv
import glob
import yaml
import os
from src.stereo_rectify import rectify_images
from src.utils import load_and_preprocess_cube_front_images
from src.calibrate_Cube import compute_and_save_calibration, get_cube_front, get_cube_subs, read_calibration, undistort_and_crop, undistort_image

# Chessboard' size = nb of inner corners
chessboard_size = (9,6) 
square_size = 0.025

#Path to calibration matrix
calibration_path="/Users/michaelargi/projects/panneaux/stereo_cam/calibration_matrix.yaml"

#get or compute calibration matrix
if os.path.exists(calibration_path):
    mtx, dist,rmse = read_calibration(calibration_path)
else:
    images = glob.glob('/Users/michaelargi/projects/panneaux/stereo_cam/Calibrate_CUBE/*.png')
    mtx, dist = compute_and_save_calibration(images,chessboard_size,square_size)

print(mtx)
print(dist)

folder_name = "undistorted_CUBE"
folder_path = os.path.join(os.getcwd(),folder_name)


images =load_and_preprocess_cube_front_images()
nb_pairs = int(len(images)/2)

for i in range(nb_pairs):
    leftImg = images[2*i]
    rightImg = images[2*i+1]
    undistorted_left,newcameramtx = undistort_and_crop(leftImg, mtx, dist)
    undistorted_right,newcameramtx = undistort_and_crop(rightImg, mtx, dist)
    output_path = os.path.join(folder_path, f'{i}_left.png')
    cv.imwrite(output_path, undistorted_left)
    output_path = os.path.join(folder_path, f'{i}_right.png')
    cv.imwrite(output_path, undistorted_right)
    h,w=undistorted_left.shape[:2]
    new_width=664
    new_height=int((h/w)*new_width)
    undistorted_left = cv.resize(undistorted_left, (new_width, new_height))
    undistorted_right = cv.resize(undistorted_right, (new_width, new_height))
    img1_rect,img2_rect = rectify_images(undistorted_left,undistorted_right)
    cv.imwrite(f'/Users/michaelargi/projects/panneaux/stereo_cam/undistorted_CUBE/{i}_rectified_left.jpg', img1_rect)
    cv.imwrite(f'/Users/michaelargi/projects/panneaux/stereo_cam/undistorted_CUBE/{i}_rectified_right.jpg', img2_rect)

print(newcameramtx)
#now get calibration with und

""" for i in range(0,len(images),2):
    focal_length,cost,_,_ =compute_auto_calibration_for_images(images[i:i+2])
    costs.append(cost)
    focal_lengths.append(focal_length) """


    
    
