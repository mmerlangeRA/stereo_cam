import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.calibrate_Cube import read_calibration, undistort_and_crop
from panneaux.stereo_cam.src.auto_calibration_utils import compute_auto_calibration_for_images
from src.utils import load_and_preprocess_cube_front_images


# Load images
calibration_path="/Users/michaelargi/projects/panneaux/stereo_cam/calibration_matrix.yaml"

#Get images for calibration or undistortion

images = glob.glob('/Users/michaelargi/projects/panneaux/stereo_cam/Calibrate_CUBE/*.png')


mtx, dist,rmse = read_calibration(calibration_path)

images = load_and_preprocess_cube_front_images()

images = [undistort_and_crop(img, mtx, dist) for img in images]

focal_lengths = []

costs=[]
for i in range(0,len(images),2):
    refined_K,cost,refined_rvec,refined_tvec =compute_auto_calibration_for_images(images[i:i+2])
    costs.append(cost)

""" focal_lengths.append(focal_length)

max_cost = max(costs)
focal_length = 0
total_weight = 0
for i in range(len(focal_lengths)):
    if costs[i]>10:
        continue
    weight = (max_cost-costs[i])/max_cost
    total_weight += weight
    focal_length += focal_lengths[i]* weight

focal_length /= total_weight """
