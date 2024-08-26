from abc import abstractmethod
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.depth_estimation.selective_igev import Selective_igev
from src.utils.path_utils import get_static_folder_path
from typing import Tuple, List, Optional
import numpy.typing as npt

from src.utils.curve_fitting import find_best_2_polynomial_curves
from src.utils.disparity import compute_3d_position_from_disparity
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian
from src.road_detection.RoadSegmentator import RoadSegmentator
from src.road_detection.common import AttentionWindow
from src.calibration.StereoCalibrator import StereoFullCalibration


class RoadDetector:
    roadSegmentator : RoadSegmentator
    window:AttentionWindow
    degree:int
    thresh_windowed: npt.NDArray[np.uint8]
    def __init__(self,roadSegmentator : RoadSegmentator,window:AttentionWindow, degree=1,debug=False):
        self.roadSegmentator = roadSegmentator
        self.window = window
        self.degree = degree
        self.debug = debug
        
    @abstractmethod
    def compute_road_width(self,img: npt.NDArray[np.uint8]):
        pass

class EACRoadDetector(RoadDetector):
    """
    Estimates road width from EAC image.
    We assume road is a plane and that the camera view direction is // to it
    - window : attention window. Road segmentation and contours will be searched only within this window
    - camHeight : estimated camera's height to the road plane
    - degree : degree of the polynom for curve fitting
    """
    img: npt.NDArray[np.uint8]
    camHeight=2.


    def __init__(self, roadSegmentator: RoadSegmentator, window:AttentionWindow, camHeight=2.,degree=1,debug=False):
        super().__init__(roadSegmentator, window=window,degree=degree,debug=debug)
        self.camHeight = camHeight
        

    def compute_road_width(self,img: npt.NDArray[np.uint8]) :
        windowed = self.window.crop_image(img)
        if self.debug:
            cv2.imwrite(get_static_folder_path("windowed.png"), windowed)
        # Assuming you have a function to perform semantic segmentation
        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        thresh = np.zeros(img.shape[:2], dtype=np.uint8)
        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(contours) == 0:
            print("no contour found on road")
        elif self.debug:
            print(f'found {len(contours)} contours')
        contour = max(contours, key=cv2.contourArea)
        contour_points = contour[:, 0, :]
        contour_x = contour_points[:, 0]
        contour_y = contour_points[:, 1]

        first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_best_2_polynomial_curves(contour,degree=self.degree)
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
        plane_dy = -self.camHeight
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
        
        if self.debug:
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

class StereoRoadDetector(RoadDetector):
    calibration:StereoFullCalibration
    def __init__(self, roadSegmentator: RoadSegmentator, window:AttentionWindow,calibration:StereoFullCalibration,degree=1,debug=False):
        super().__init__(roadSegmentator,window=window, degree=degree,debug=debug)
        self.calibration = calibration


    def compute_road_width(self,img_left_right: npt.NDArray[np.uint8]) :
        """
        img is split into 2 left and right images
        """
        cv2.imshow("img_left_right",img_left_right)
        height, width = img_left_right.shape[:2]
        # Ensure the width is even so that it can be evenly split into two halves
        assert width % 2 == 0, "Image width is not even. Cannot split into two equal halves."

        # Calculate the middle index for the split
        middle = width // 2

        # Split the image into left and right halves
        imgL = img_left_right[:, :middle]
        cv2.imshow("imgL",imgL)
        cv2.waitKey(0)
        imgR = img_left_right[:, middle:]

        test_igev = Selective_igev(None, None)
        input_pair = InputPair(left_image=imgL, right_image=imgR, status="started", calibration=self.calibration)
        stereo_output = test_igev.compute_disparity(input_pair)
        disparity_map = stereo_output.disparity_pixels

        K = self.calibration.stereo_rectified_K

        fx = K[0][0]
        fy = K[1][1]
        baseline = -self.calibration.stereo_rectified_tvec[0][0]
        print("baseline",baseline)
        c_x = K[0][2]
        c_y = K[1][2]
        z0= self.calibration.stereo_rectified_Z0
        print("z0", z0) 

        windowed = self.window.crop_image(imgL)

        if self.debug:
            cv2.imwrite(get_static_folder_path("windowed.png"), windowed)
        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        thresh = np.zeros(imgL.shape[:2], dtype=np.uint8)

        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("no road")

        contour = max(contours, key=cv2.contourArea)
        contour_points = contour[:, 0, :]
        contour_x = contour_points[:, 0]
        contour_y = contour_points[:, 1]

        first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_best_2_polynomial_curves(contour,degree=self.degree)
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

        for y in range(minY, maxY):
            x_first_poly = first_poly_model.predict([[y]])[0]
            x_second_poly = second_poly_model.predict([[y]])[0]
            p1, d1 = compute_3d_position_from_disparity(x_first_poly, y, disparity_map, fx, fy,c_x, c_y, baseline,z0)
            p2, d2 = compute_3d_position_from_disparity(x_second_poly, y, disparity_map, fx, fy,c_x, c_y, baseline,z0)
            # if np.abs(d2 - d1) > 10:
            #     print(d2, d1)

            distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
            p1=np.round(p1,2)
            p2=np.round(p2, 2)
            points.append([p1, p2])
        
        if self.debug:
            print(f'found {len(contours)}')
            
            contour_image = imgL.copy()
            # Draw contours with random colors
            for contour in contours:
                # Generate a random color
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.drawContours(contour_image, [contour], -1, color, 3)
            print(np.mean(distances))
            print(points)
            distances = [np.round(d,1) for d in distances]
            print(distances)
            
            cv2.imshow('contours', contour_image)
            cv2.imshow('disparity_map', disparity_map)
            cv2.imwrite(get_static_folder_path("disparity.png"), disparity_map)
            cv2.imwrite(get_static_folder_path("contours.png"), contour_image)

        return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y

