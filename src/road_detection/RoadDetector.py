from abc import abstractmethod
import math
import random
import cv2
import numpy as np
from typing import Tuple, List, Optional
import numpy.typing as npt
from scipy.optimize import minimize,least_squares

from src.depth_estimation.selective_igev import Selective_igev
from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.utils.path_utils import get_ouput_path
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
       
class EquirectRoadDetector(RoadDetector):
    """
    Estimates road width from equirectangular image.
    We assume road is a plane and that the camera view direction is // to it
    - window : attention window. Road segmentation and contours will be searched only within this window
    - camHeight : estimated camera's height to the road plane
    - degree : degree of the polynom for curve fitting
    """
    img: npt.NDArray[np.uint8]
    camHeight=1.65
    road_window_top= 0.53

    def __init__(self, roadSegmentator: RoadSegmentator, window:AttentionWindow, camHeight=2.,degree=1,road_window_top=0.53,debug=False):
        super().__init__(roadSegmentator, window=window,degree=degree,debug=debug)
        self.camHeight = camHeight
        self.road_window_top=road_window_top
        
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
            cv2.imwrite(get_ouput_path("windowed.png"), windowed)
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
    
    def display_road_image(self,left_img_contour_left,left_img_contour_right,camHeight:float,optimized_rotation_vector,img_width:int,img_height:int):
        road_points_left, road_points_right = self.compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,optimized_rotation_vector,img_width,img_height)
        road_points_2D = np.concatenate((road_points_left, road_points_right), axis=0)

        road_points_x = road_points_2D[:,0]
        road_points_z = road_points_2D[:,1]

        minX= np.min(road_points_x)
        maxX= np.max(road_points_x)
        minZ= np.min(road_points_z)
        maxZ= np.max(road_points_z)

        width = math.ceil(maxX-minX)
        height = math.ceil(maxZ-minZ)

        road_points_2D[:,0] = road_points_x - minX
        road_points_2D[:,1] = road_points_z - minZ

        display_coeff=10
        road_image = np.zeros((height*display_coeff, width*display_coeff, 3), dtype=np.uint8)

        for p in road_points_2D:
            x = int(p[0]*display_coeff)
            y = int(p[1]*display_coeff)
            road_image[y, x] = [255, 0, 0]
        cv2.imwrite(get_ouput_path('road_image.png'), road_image)

    def compute_left_right_road_points2D(self,img_contour_left,img_contour_right,camHeight,road_rvec,img_width,img_height):
        left_contour_x = img_contour_left[:, 0]
        left_contour_y = img_contour_left[:, 1]
        road_points_left = equirect_to_road_plane_points2D(imgWidth=img_width, imgHeight=img_height, 
                                            road_rvec=road_rvec,camHeight=camHeight,
                                            contour_x=left_contour_x, contour_y=left_contour_y)
        

        right_contour_x = img_contour_right[:, 0]
        right_contour_y = img_contour_right[:, 1]
        road_points_right = equirect_to_road_plane_points2D(imgWidth=img_width, imgHeight=img_height,
                                            road_rvec=road_rvec,camHeight=camHeight,
                                            contour_x=right_contour_x, contour_y=right_contour_y)
        
        road_points_left= road_points_left[road_points_left[:, 1].argsort()]
        road_points_right= road_points_right[road_points_right[:, 1].argsort()]

        return road_points_left, road_points_right

    def compute_slopes_difference(self,cam_rotation_vector, left_img_contour_left, left_img_contour_right, camHeight, img_width, img_height):
        road_points_left, road_points_right = self.compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,cam_rotation_vector,img_width, img_height)
        nb_left = road_points_left.shape[0]
        nb_right = road_points_right.shape[0]

        xLeftpoints = road_points_left[:, 0]
        yLeftpoints = road_points_left[:, 1]
        xRightpoints = road_points_right[:, 0]
        yRightpoints = road_points_right[:, 1]

        xleft_1 = xLeftpoints[0]
        yleft_1 = yLeftpoints[0]
        xleft_2 = xLeftpoints[nb_left-1]
        yleft_2 = yLeftpoints[nb_left-1]

        xright_1 = xRightpoints[0]
        yright_1 = yRightpoints[0]
        xright_2 = xRightpoints[nb_right-1]
        yright_2 = yRightpoints[nb_right-1]

        slopes_left = (xleft_2 - xleft_1)/(yleft_2 - yleft_1) 
        slopes_right = (xright_2 - xright_1)/(yright_2 - yright_1) 
        
        slopes_diff = np.abs(slopes_left - slopes_right)
        #print(slopes_left, slopes_right,slopes_diff)

        return slopes_diff

    # Function to optimize cam_rotation_vector
    def optimize_cam_rotation(self,left_img_contour_left, left_img_contour_right, camHeight:float, initial_cam_rotation_vector,bounds,img_width:int,img_height:int):
        # Optimize the cam_rotation_vector using least_squares
        result = least_squares(
            fun=lambda cam_rotation_vector: self.compute_slopes_difference(cam_rotation_vector, left_img_contour_left, left_img_contour_right, camHeight,img_width,img_height),
            x0=initial_cam_rotation_vector,
            bounds=bounds,  # Add bounds here
            method='trf'  # 'trf' is the method that supports bounds
        )
        # Return the optimized cam_rotation_vector
        return result.x


        img_height, img_width = img.shape[:2]

    def compute_road_width(self,img: npt.NDArray[np.uint8]) :
        left_img_contour_left,left_img_contour_right = self.get_left_right_contours(img)
        img_height, img_width = img.shape[:2]

        #filter to select only closest points to the car
        road_top = int(self.road_window_top * img_height)
        left_img_contour_left = left_img_contour_left[left_img_contour_left[:, 1] > road_top]
        left_img_contour_right = left_img_contour_right[left_img_contour_right[:, 1] > road_top]

        left_img_contour_left= left_img_contour_left[left_img_contour_left[:, 1].argsort()]
        left_img_contour_right= left_img_contour_right[left_img_contour_right[:, 1].argsort()]

        # project on road plane and optimize cam rotation
        cam0Transform = Transform(0.,0,0,0.,0.,0.)

        initial_cam_rotation_vector= np.array(cam0Transform.rotationVector)
        lower_bounds = np.array([-np.pi/4,-np.pi/10000, -np.pi/10000])  # Lower bounds for the rotation vector
        upper_bounds = np.array([np.pi/4,np.pi/10000,  np.pi/10000])  # Upper bounds for the rotation vector
        bounds = (lower_bounds, upper_bounds)

        # Optimize the camera rotation vector
        optimized_rotation_vector = self.optimize_cam_rotation(left_img_contour_left, left_img_contour_right, self.camHeight, initial_cam_rotation_vector, bounds, img_width, img_height)

        print("optimized_rotation_vector",optimized_rotation_vector)
        road_rvec=optimized_rotation_vector

        if self.debug:
            #debug and display
            self.display_road_image(left_img_contour_left,left_img_contour_right,self.camHeight,optimized_rotation_vector,img_width,img_height)

        #now let's estimate distance between left and right road. They should be more or less parallel now
        road_points_left, road_points_right=self.compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,self.camHeight,road_rvec, img_width,img_height)

        x_left = np.mean(road_points_left[:,0])
        slope_left,intercept_left  = np.polyfit(road_points_left[:,0], road_points_left[:,1], 1)
        slope_right,intercept_right = np.polyfit(road_points_right[:,0], road_points_right[:,1], 1)

        if (slope_left==0):#impossible... slope should be >>1
            slope_left=0.0001
        else:
            y_left = slope_left*x_left+intercept_left
            x_right = (intercept_right-y_left+x_left/slope_left)/(1/slope_left-slope_right)
            y_right = slope_right*x_right+intercept_right

        road_width = np.sqrt((x_right-x_left)*(x_right-x_left) + (y_left-y_right)*(y_left-y_right)) 
        return road_width

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
            cv2.imwrite(get_ouput_path("windowed.png"), windowed)
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

            cv2.imwrite(get_ouput_path("contours.png"), contour_image)
            
            colorized_disparity_map = colorize_disparity_map(disparity_map)
            # Display the colorized disparity map
            cv2.imshow('Colorized Disparity Map', colorized_disparity_map)

            cv2.imwrite(get_ouput_path("colorized_disparity_map.png"), colorized_disparity_map)
            

        return np.mean(distances),first_poly_model, second_poly_model,contour_x,contour_y

