from abc import abstractmethod
import math
import random
import time
import cv2
import numpy as np
from typing import Sequence, Tuple, List, Optional
import numpy.typing as npt

from src.depth_estimation.selective_igev import Selective_igev
from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.utils.path_utils import get_output_path
from src.utils.curve_fitting import find_best_2_polynomial_curves
from src.utils.disparity import compute_3d_position_from_disparity_map
from src.road_detection.RoadSegmentator import RoadSegmentator
from src.road_detection.common import AttentionWindow
from src.calibration.cube.StereoCalibrator import StereoFullCalibration
from src.utils.image_processing import colorize_disparity_map
from src.utils.typing import cvImage

class RoadEquationInPlane:
    #equation left : x= slope_x_from_z*z + intercept_z
    #equation right : x= slope_x_from_z*z + intercept_z + delta_x
    
    slope_x_from_z: float
    delta_x: float
    intercept_z:float

    def __init__(self, slope_x_from_z: float, delta_x: float, intercept_z:float) -> None:
        self.slope_x_from_z = slope_x_from_z
        self.delta_x = delta_x
        self.intercept_z = intercept_z
    
    def get_road_width(self) -> float:
        return math.cos(self.slope_x_from_z) * self.delta_x
    


class RoadDetector:
    roadSegmentator : RoadSegmentator = None
    window:AttentionWindow
    degree:int
    thresh_windowed: cvImage
    frame_id: int = 0

    def __init__(self,roadSegmentator : RoadSegmentator,window:AttentionWindow, degree=1,debug=False):
        self.roadSegmentator = roadSegmentator
        self.window = window
        self.degree = degree
        self.debug = debug
        
    def set_frame_id(self, frame_id: int) -> None:
        self.frame_id = frame_id
        if self.roadSegmentator is not None:
            self.roadSegmentator.set_frame_id(frame_id)

    @abstractmethod
    def compute_road_width(self,img: cvImage)->float:
        pass

    @abstractmethod
    def compute_road_equation(self,img: cvImage):
        pass
       
class StereoRoadDetector(RoadDetector):
    calibration:StereoFullCalibration
    def __init__(self, roadSegmentator: RoadSegmentator, window:AttentionWindow,calibration:StereoFullCalibration,degree=1,debug=False):
        super().__init__(roadSegmentator,window=window, degree=degree,debug=debug)
        self.calibration = calibration


    def compute_road_width(self,img_left_right: cvImage) :
        """
        img is split into 2 left and right images
        """
        if self.debug:
            cv2.imshow("img_left_right",img_left_right)
        height, width = img_left_right.shape[:2]
        # Ensure the width is even so that it can be evenly split into two halves
        assert width % 2 == 0, "Image width is not even. Cannot split into two equal halves."

        # Calculate the middle index for the split
        middle = width // 2

        # Split the image into left and right halves
        imgL = img_left_right[:, :middle]
        imgR = img_left_right[:, middle:]
        print("imgL", imgL.shape)
        print("imgR", imgR.shape)
        test_igev = Selective_igev(None, None)
        input_pair = InputPair(left_image=imgL, right_image=imgR, status="started", calibration=self.calibration)
        stereo_output = test_igev.compute_disparity(input_pair)
        disparity_map = stereo_output.disparity_pixels
        
        K = self.calibration.stereo_rectified_K
        if K is None or len(K) == 0:
            raise ValueError("no calibration data")

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
            cv2.imwrite(get_output_path(f"{self.frame_id}_windowed.png"), windowed)
        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        thresh = np.zeros(imgL.shape[:2], dtype=np.uint8)

        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            raise ValueError("no road")

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
            p1, d1 = compute_3d_position_from_disparity_map(x_first_poly, y, disparity_map, fx, fy,c_x, c_y, baseline,z0)
            p2, d2 = compute_3d_position_from_disparity_map(x_second_poly, y, disparity_map, fx, fy,c_x, c_y, baseline,z0)
            # if np.abs(d2 - d1) > 10:
            #     print(d2, d1)

            distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
            p1=np.round(p1,2)
            p2=np.round(p2, 2)
            points.append([p1, p2])
        
        if self.debug:
            print(f'found {len(contours)} contours')
            
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

            cv2.imwrite(get_output_path("contours.png"), contour_image)
            
            colorized_disparity_map = colorize_disparity_map(disparity_map)
            # Display the colorized disparity map
            cv2.imshow('Colorized Disparity Map', colorized_disparity_map)

            cv2.imwrite(get_output_path("colorized_disparity_map.png"), colorized_disparity_map)
            

        return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y

