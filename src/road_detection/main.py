import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from src.pidnet.main import segment_image
from src.utils.geo import create_Q_matrix
from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.depth_estimation.selective_igev import Selective_igev
from src.utils.path_utils import get_static_folder_path
from typing import Tuple, List, Optional
import numpy.typing as npt

from src.utils.curve_fitting import find_best_2_polynomial_curves
from src.utils.disparity import compute_3d_position_from_disparity
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian
from src.road_detection.segment import segment_road_image
from src.road_detection.common import AttentionWindow


def compute_road_width_from_stereo_cubes(imgL: npt.NDArray[np.uint8], imgR: npt.NDArray[np.uint8], calibration:Calibration,degree=1,debug= False) :
    """
    Estimates road edges from stereo images.

    Parameters:
    - imgL: Left stereo image.
    - imgR: Right stereo image.

    Important : input images must be rectified before and their size must be power of 8.
    """
    test_igev = Selective_igev(None, None)
    input_pair = InputPair(left_image=imgL, right_image=imgR, status="started", calibration=calibration)
    stereo_output = test_igev.compute_disparity(input_pair)
    disparity_map = stereo_output.disparity_pixels

    focal_length = calibration.fx
    baseline = calibration.baseline_meters
    c_x = calibration.cx0
    c_y = calibration.cy

    #depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
    # Assuming you have a function to perform semantic segmentation
    thresh = segment_road_image(imgL, debug=debug)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour[:, 0, :]
    contour_x = contour_points[:, 0]
    contour_y = contour_points[:, 1]

    first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_best_2_polynomial_curves(contour,degree=degree)
    min_y_inliers_first = np.min(y_inliers_first)
    min_y_inliers_second = np.min(y_inliers_second)
    max_y_inliers_first = np.max(y_inliers_first)
    max_y_inliers_second = np.max(y_inliers_second)

    minY = max(min_y_inliers_first, min_y_inliers_second)
    maxY = min(max_y_inliers_first, max_y_inliers_second)

    if maxY < minY:
        print("no road")

    distances = []
    points = []
    for y in range(minY +0, maxY - 0):
        x_first_poly = first_poly_model.predict([[y]])[0]
        x_second_poly = second_poly_model.predict([[y]])[0]
        p1, d1 = compute_3d_position_from_disparity(x_first_poly, y, disparity_map, focal_length, c_x, c_y, baseline)
        p2, d2 = compute_3d_position_from_disparity(x_second_poly, y, disparity_map, focal_length, c_x, c_y, baseline)
        if np.abs(d2 - d1) > 10:
            print(d2, d1)
        points.append([p1, p2])
        distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    if debug:
        print(f'found {len(contours)}')
        
        contour_image = imgL.copy()
        # Draw contours with random colors
        for contour in contours:
            # Generate a random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(contour_image, [contour], -1, color, 3)
        print(np.mean(distances))
        print(points)
        print(distances)
        
        cv2.imshow('contours', contour_image)
        cv2.imshow('disparity_map', disparity_map)
        cv2.imwrite(get_static_folder_path("disparity.png"), disparity_map)
        cv2.imwrite(get_static_folder_path("contours.png"), contour_image)

    return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y


def compute_road_width_from_eac(img: npt.NDArray[np.uint8], window:AttentionWindow, camHeight=2.,degree=1,kernel_width=20,debug=False) :
    """
    Estimates road edges from EAC image.
    We assume road is a plane and that the camera view direction is // to it

    Parameters:
    - img: EAC image.
    """
    #windowed = img[window.top:window.bottom, window.left:window.right]
    windowed = window.crop_image(img)
    if debug:
        cv2.imwrite(get_static_folder_path("windowed.png"), windowed)
    # Assuming you have a function to perform semantic segmentation
    thresh_windowed = segment_road_image(windowed,kernel_width,debug=debug)
    thresh = np.zeros(img.shape[:2], dtype=np.uint8)
    thresh[window.top:window.bottom, window.left:window.right] = thresh_windowed

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour[:, 0, :]
    contour_x = contour_points[:, 0]
    contour_y = contour_points[:, 1]

    first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_best_2_polynomial_curves(contour,degree=degree)
    min_y_inliers_first = np.min(y_inliers_first)
    min_y_inliers_second = np.min(y_inliers_second)
    max_y_inliers_first = np.max(y_inliers_first)
    max_y_inliers_second = np.max(y_inliers_second)

    minY = max(min_y_inliers_first, min_y_inliers_second)
    maxY = min(max_y_inliers_first, max_y_inliers_second)

    if maxY < minY:
        print("no road")

    distances = []
    points = []
    imgHeight,imgWidth = img.shape[:2]
    plane_dy = -camHeight
    for y in range(minY +0, maxY - 0):
        x_first_poly = first_poly_model.predict([[y]])[0]
        x_second_poly = second_poly_model.predict([[y]])[0]
        
        theta1, phi1 = pixel_to_spherical (imgWidth, imgHeight,x_first_poly, y)
        ray1 = spherical_to_cartesian(theta1, phi1)

        theta2, phi2 = pixel_to_spherical (imgWidth, imgHeight, x_second_poly, y)
        ray2 = spherical_to_cartesian(theta2, phi2)

        lambda1 = plane_dy/ray1[1]
        lambda2 = plane_dy/ray2[1]

        p1 = lambda1*ray1
        p2 = lambda2*ray2

        points.append([p1, p2])
        distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    if debug:
        print(f'found {len(contours)}')
        cv2.imshow('thresh', thresh)
        contour_image = img.copy()
        # Draw contours with random colors
        for contour in contours:
            # Generate a random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(contour_image, [contour], -1, color, 3)
        print(np.mean(distances))
        print(points)
        print(distances)
        
        cv2.imshow('contours', contour_image)
        cv2.imwrite(get_static_folder_path("contours.png"), contour_image)
    return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y