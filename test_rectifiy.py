import cv2
from src.stereo_rectify import rectify_images
from src.utils import load_and_preprocess_cube_front_images

images =load_and_preprocess_cube_front_images()

for i in range(0,14):
    leftImg = images[2*i]
    rightImg = images[2*i+1]
    cv2.imshow('left', leftImg)
    cv2.imshow('right', rightImg)
    cv2.waitKey(0)
    img1_rect,img2_rect = rectify_images(leftImg,rightImg)
    cv2.imwrite(f'/Users/michaelargi/projects/panneaux/stereo_cam/undistorted_CUBE/{i}_rectified_left.jpg', img1_rect)
    cv2.imwrite(f'/Users/michaelargi/projects/panneaux/stereo_cam/undistorted_CUBE/{i}_rectified_right.jpg', img2_rect)