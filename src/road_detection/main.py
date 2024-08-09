import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from src.pidnet.main import segment_image
from src.utils.geo import create_Q_matrix
from src.depth_estimation.depth_estimator import InputPair
from src.depth_estimation.selective_igev import Selective_igev
from src.utils.path_utils import get_static_folder_path
from typing import Tuple, List, Optional
import numpy.typing as npt

from src.utils.curve_fitting import find_best_2_polynomial_curves
from src.utils.disparity import compute_3d_position_from_disparity


def segment_road_image(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Computes binary image corresponding to road.

    Parameters:
    - img: rgb image
    returns
    - binary image

    """
    segmented_image, pred = segment_image(img)
    # Create a mask for the road class
    road_mask = (pred == 0).astype(np.uint8)
    # Check if the mask has the same dimensions as the segmented image
    assert road_mask.shape == segmented_image.shape[:2], "Mask size does not match the image size."

    masked_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

    gray = cv2.cvtColor(masked_segmented, cv2.COLOR_BGR2GRAY)
    # we clean the image by removing small blobs

    kernel_size = (10,10) #this size could be computed dynamically from image size

    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Apply the dilation operation to the edged image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

def get_road_edges(imgL: npt.NDArray[np.uint8], imgR: npt.NDArray[np.uint8], debug= True) -> None:
    """
    Estimates road edges from stereo images.

    Parameters:
    - imgL: Left stereo image.
    - imgR: Right stereo image.
    """
    test_igev = Selective_igev(None, None)
    input_pair = InputPair(left_image=imgL, right_image=imgR, status="started", calibration=None)
    stereo_output = test_igev.compute_disparity(input_pair)
    disparity_map = stereo_output.disparity_pixels
    cv2.imwrite(get_static_folder_path("disparity.png"), disparity_map)

    focal_length = 700.0  # in pixels
    baseline = 1.12  # in meters
    c_x = 331.99987265  # principal point x-coordinate
    c_y = 387.5000997  # principal point y-coordinate

    depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
    # Assuming you have a function to perform semantic segmentation
    thresh = segment_road_image(imgL)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    contour = max(contours, key=cv2.contourArea)
    first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_best_2_polynomial_curves(contour, imgL)
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
    for y in range(minY + 60, maxY - 60):
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
        cv2.imwrite(get_static_folder_path("contours.png"), contour_image)
