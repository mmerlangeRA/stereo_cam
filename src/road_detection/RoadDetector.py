from abc import abstractmethod
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, List, Optional
import numpy.typing as npt
from scipy.optimize import minimize,least_squares

from src.depth_estimation.selective_igev import Selective_igev
from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.utils.path_utils import get_static_folder_path, get_ouput_path
from src.utils.curve_fitting import Road_line_params, compute_residuals, find_best_2_best_contours, find_best_2_polynomial_curves, fit_polynomial_ransac
from src.utils.disparity import compute_3d_position_from_disparity_map
from src.utils.coordinate_transforms import cartesian_to_equirectangular, equirect_to_road_plane_points2D, get_transformation_matrix, pixel_to_spherical, spherical_to_cartesian
from src.road_detection.RoadSegmentator import RoadSegmentator
from src.road_detection.common import AttentionWindow
from src.calibration.cube.StereoCalibrator import StereoFullCalibration
from src.utils.image_processing import colorize_disparity_map
from src.utils.TransformClass import Transform



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

    @abstractmethod
    def compute_road_equation(self,img: npt.NDArray[np.uint8]):
        pass

def compute_eac_road_residual(params, imgWidth, imgHeight, contour_x, contour_y):
    rvec = params[:3]
    camHeight = params[3]
    roadWidth = params[4]
    a= params[5]
    b=params[6]
    c=params[7]
    road_points,width, height = equirect_to_road_plane_points2D(imgWidth, imgHeight, rvec, camHeight, contour_x, contour_y)
    x_points = road_points[:,0]#road width
    average_x = np.mean(x_points)
    z_points = road_points[:,1]
    mask = x_points < average_x

    # Extract x and z points where x < average_x
    filtered_x_points = x_points[mask]
    filtered_z_points = z_points[mask]
    #first_poly_model, inliers_first = fit_polynomial_ransac(filtered_z_points,filtered_x_points,degree=2,residual_threshold=1.)

    global_diff=0.0
    diffs=[]
    #estimated_xs=first_poly_model.predict(z_points)
    for i in range(len(x_points)):
        #estimated_x=first_poly_model.predict([[z_points[i]]])[0]
        z=z_points[i]
        estimated_x = a*z*z + b*z + c
        diff = min(abs(estimated_x-x_points[i]), abs(estimated_x+roadWidth-x_points[i]))
        
        diffs.append(diff)
        global_diff+=diff
    print(global_diff)
    return global_diff
       
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
        
    def compute_line_width(self,imgWidth:int, imgHeight:int, x1:int, x2:int, y1:int, y2:int):
        plane_dy = self.camHeight
        theta1, phi1 = pixel_to_spherical (imgWidth, imgHeight,x1, y1)
        ray1 = spherical_to_cartesian(theta1, phi1)

        theta2, phi2 = pixel_to_spherical (imgWidth, imgHeight, x2, y2)
        ray2 = spherical_to_cartesian(theta2, phi2)

        lambda1 = plane_dy/ray1[1]
        lambda2 = plane_dy/ray2[1]

        p1 = lambda1*ray1
        p2 = lambda2*ray2

        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        return distance, p1, p2
   
    def optimizeRoadDesign(self,imgWidth:int, imgHeight:int, initial_rvec:np.array,initial_camHeight:float,initial_road_width:float,contour_x,contour_y):

        initial_a=0.1
        initial_b=0.1
        initial_c=0.1
        initial_params = np.concatenate([initial_rvec.flatten(), [initial_camHeight, initial_road_width,initial_a,initial_b,initial_c]])

        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (1.5, 2.5), (2.0, 20.0),(-100.0,100.),(-100.0,100.),(-100.0,100.)]

        result = minimize(
            compute_eac_road_residual,
            initial_params,
            args=(imgWidth, imgHeight, contour_x, contour_y),
            bounds=bounds,
            method='L-BFGS-B'  # L-BFGS-B is a good choice when you have bounds
        )
        print(result)
        # Optimized parameters
        optimized_rvec = result.x[:3]
        optimized_camHeight = result.x[3]
        optimized_road_width = result.x[4]
        residual_distance_normalized = result.fun / len(contour_x)
        return optimized_rvec,optimized_camHeight,optimized_road_width,residual_distance_normalized

    def get_road_contours(self,img: npt.NDArray[np.uint8]) :
        '''
        Computes road contours
        Parameters:
        - img: image to manage.

        Returns:
        - contours points
        '''
        windowed = self.window.crop_image(img)
        if self.debug:
            cv2.imwrite(get_static_folder_path("windowed.png"), windowed)
        # Assuming you have a function to perform semantic segmentation
        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        cv2.imwrite(get_ouput_path("thresh_windowed.png"), self.thresh_windowed)
        thresh = np.zeros(img.shape[:2], dtype=np.uint8)
        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_left_right_contours(self,img: npt.NDArray[np.uint8]) :
        '''
        Computes left and right contours of an image
        Parameters:
        - img: image to manage.

        Returns:
        - contours_left: points of left contour
        - contours_right: points of right contour
        '''
        contours = self.get_road_contours(img=img)
    
        if len(contours) == 0:
            print("no contour found on road")
        elif self.debug:
            print(f'found {len(contours)} contours')
        contour = max(contours, key=cv2.contourArea)
        contour_points = contour[:, 0, :]

        left_poly_model, right_poly_model, inliers_left_mask, inliers_right_mask = find_best_2_best_contours(contour,degree=self.degree)
        contour_left = contour_points[inliers_left_mask]
        contour_right= contour_points[inliers_right_mask]
        if self.debug:
            print(f'found {len(contours)}')
            contour_image = img.copy()
            # Draw contours with random colors
            cv2.drawContours(contour_image, [contour_left], -1, (255,0,0), 3)
            cv2.drawContours(contour_image, [contour_right], -1, (0,255,0), 3)            
            cv2.imwrite(get_ouput_path("contours.png"), contour_image)

        return contour_left,contour_right

    def fit_road_curve_to_left_right_contours(self, contour_left,contour_right,initial_guess:Road_line_params,camRight:Transform,image_width:int, image_height:int)->Road_line_params :
        '''
        x= az+b
        y=h
        '''

        bonds=[[-0.3,0.3],[-3.,0],[-2.2,-1.3]]
        lower_bounds = np.array([-0.3, -3, -2.2])
        upper_bounds = np.array([0.3, 0, 2.3])
        result = least_squares(
            compute_residuals,
            initial_guess.as_array(),
            args=(contour_left, contour_right, camRight,image_width,image_height),
            method='trf',  # 'trf' supports bounds
            bounds=(lower_bounds, upper_bounds)
        )
        # Extract optimized parameters
        a_opt, b_opt, h_opt = result.x


        print(f"Optimized line parameters: a = {a_opt}, b = {b_opt}, h = {h_opt}")

        return Road_line_params(a_opt,b_opt, h_opt)

    
    def compute_road_width(self,img: npt.NDArray[np.uint8]) :
        contours = self.get_road_contours(img=img)
    
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
        
        use_optimal_distance = False
        for y1 in range(minY, maxY):
            x1 = first_poly_model.predict([[y1]])[0]
            if use_optimal_distance:
                min_distance = float('inf')
                x2=y2=0
                for test_y2 in range(minY, maxY):
                    test_x2 = second_poly_model.predict([[test_y2]])[0]
                    
                    d = (x1-test_x2)*(x1-test_x2)+(y1-test_y2)*(y1-test_y2)
                    print(test_x2, test_y2,d)
                    if d < min_distance:
                        min_distance = d
                        x2=test_x2
                        y2=test_y2

            else:
                x2= second_poly_model.predict([[y1]])[0]
                y2 = y1
            distance, p1, p2 = self.compute_line_width(imgWidth, imgHeight, x1,x2, y1,y2)

            points.append([p1, p2])
            distances.append(distance)
        
        if self.debug:
            print(f'found {len(contours)}')
            #cv2.imshow('thresh', thresh)
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
            cv2.imwrite(get_static_folder_path("windowed.png"), windowed)
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

            cv2.imwrite(get_static_folder_path("contours.png"), contour_image)
            
            colorized_disparity_map = colorize_disparity_map(disparity_map)
            # Display the colorized disparity map
            cv2.imshow('Colorized Disparity Map', colorized_disparity_map)

            cv2.imwrite(get_static_folder_path("colorized_disparity_map.png"), colorized_disparity_map)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y

