import math
import random
import time
import cv2
import numpy as np
from typing import Sequence, Tuple, List, Optional
import numpy.typing as npt
from scipy.optimize import minimize,least_squares

from src.logger import get_logger
from src.utils.path_utils import get_output_path
from src.utils.curve_fitting import Road_line_params, compute_residuals, find_best_2_best_polynomial_contours, find_best_2_polynomial_curves, fit_polynomial_ransac
from src.utils.disparity import compute_3d_position_from_disparity_map
from src.utils.coordinate_transforms import cartesian_to_equirectangular, equirect_to_road_plane_points2D, get_transformation_matrix, pixel_to_spherical, spherical_to_cartesian
from src.road_detection.RoadSegmentator import RoadSegmentator
from src.road_detection.common import AttentionWindow
from src.utils.TransformClass import Transform
from src.utils.typing import cvImage
from src.road_detection.RoadDetector import RoadDetector

logger = get_logger(__name__)

class EquirectMonoRoadDetector(RoadDetector):
    """
    Estimates road width from an equirectangular image.
    
    Assumptions:
    - The road is assumed to be a plane.
    - The camera's view direction is assumed to be parallel to the road.
    
    Attributes:
    - road_contour_top (float): Attention window. Road segmentation and contours 
      will be searched only within this portion of the image (0-1 range).
    """

    min_z_for_road:float
    max_z_for_road:float

    estimated_road_vector : np.array =np.array([6.,0.,1.65,0.,0.,0.])
    lower_bounds : np.array = np.array([2, -5., 1.65, -20*np.pi/180, -45*np.pi/180,-20*np.pi/180])
    upper_bounds : np.array = np.array([ 12,  0., 2.5, 20*np.pi/180, 45*np.pi/180, 20*np.pi/180])

    image_height:int
    image_width:int

    left_poly_model:any
    right_poly_model:any
    left_img_contour_left:Sequence[np.ndarray]
    left_img_contour_right:Sequence[np.ndarray]

    lower_bounds = np.array([-np.pi/4, -np.pi/10000, -np.pi/10000])
    upper_bounds = np.array([np.pi/4, np.pi/10000, np.pi/10000])

    def __init__(self, 
                 roadSegmentator: RoadSegmentator, 
                 window: AttentionWindow, 
                 degree: int = 1, 
                 min_z_for_road=5., 
                 max_z_for_road=15.,
                 image_width=5376,
                 image_height=2688,
                 debug: bool = False) -> None:
        """
        Initializes the EquirectRoadDetector.

        Parameters:
        - roadSegmentator: A road segmentation model instance.
        - window: Attention window for focusing the detection.
        - camHeight: The estimated height of the camera from the road plane (default=1.65).
        - degree: Degree for polynomial fitting when processing contours (default=1).
        - road_contour_top: The top boundary for filtering road contours. 
        We need larger window for good segmentation, then we consider only contours below road_contour_top.
        - debug: If True, saves debug images and prints debug information (default=False).
        """
        super().__init__(roadSegmentator, window=window, degree=degree, debug=debug)
        self.min_z_for_road = min_z_for_road
        self.max_z_for_road = max_z_for_road
        self.image_height = image_height
        self.image_width = image_width

    def _road_vector_to_road_width_and_transform(self,road_vector:np.array) -> Tuple[float,Transform]:
        road_width, xc,yc,yaw,pitch, roll = road_vector
        road_transform = Transform(xc=xc,yc=yc, pitch=pitch, yaw = yaw, roll=roll)
        return road_width,road_transform
    
    def _road_vector_from_road_width_and_transform(self,road_width:float,road_transform:Transform) -> np.array:
        xc,yc,yaw,pitch, roll = road_transform.xc,road_transform.yc,road_transform.pitch,road_transform.yaw,road_transform.roll
        road_vector = [road_width, xc,yc,yaw,pitch, roll]
        return road_vector
    
    def set_road_vector_and_bounds(self, road_width:float, road_transform:Transform, minRoadWidth=3., maxRoadWidth=10., maxdX=4.,maxDy=0.001, maxPitch=20*np.pi/180, maxYaw=40*np.pi/180, maxRoll=20*np.pi/180 )->None:
        self.estimated_road_vector = self._road_vector_from_road_width_and_transform(road_width, road_transform)
        self.lower_bounds = np.array([minRoadWidth, road_transform.xc-maxdX, road_transform.yc-maxDy, road_transform.pitch-maxPitch, road_transform.yaw-maxYaw, road_transform.roll-maxRoll])
        self.upper_bounds = np.array([maxRoadWidth, road_transform.xc+maxdX, road_transform.yc+maxDy, road_transform.pitch+maxPitch, road_transform.yaw+maxYaw, road_transform.roll+maxRoll])
    
    
    def _get_road_top_and_bottom_v(self)->Tuple[float, float]:
        road_width,road_transform = self._road_vector_to_road_width_and_transform(self.estimated_road_vector)
        plane_y = road_transform.yc
        theta = math.atan(plane_y/self.max_z_for_road) 
        road_contour_top = int(theta*self.image_height/np.pi +self.image_height/2.)
        theta = math.atan(plane_y/self.min_z_for_road) 
        road_contour_bottom = int(theta*self.image_height/np.pi +self.image_height/2.)
        return road_contour_top, road_contour_bottom
    
    def _get_road_contours_of_one_image(self, img: cvImage,prefix="") -> Sequence[np.ndarray]:
        '''
        Computes road contours from an input image.

        Parameters:
        - img: The input image (equirectangular).

        Returns:
        - Sequence[np.ndarray]: A sequence of contour points.
        '''
        windowed = self.window.crop_image(img)
        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed,prefix=prefix)
        if self.debug:
            cv2.imwrite(get_output_path(f'{self.frame_id}_{prefix}_windowed.png'), windowed)
            cv2.imwrite(get_output_path(f'{self.frame_id}_{prefix}_thresh_windowed.png'), self.thresh_windowed)
            

        thresh = np.zeros(img.shape[:2], dtype=np.uint8)
        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.debug:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(get_output_path(f'{self.frame_id}_{prefix}_thresh.png'), thresh)
            blended = cv2.addWeighted(img, 0.7, thresh, 0.3, 0)
            cv2.drawContours(blended, contours, -1, (255, 0, 0), 3)
            cv2.imwrite(get_output_path(f'{self.frame_id}_{prefix}_contours.png'), blended)
        return contours

    def _get_left_right_contours_of_one_image(self, img: cvImage, prefix="") -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        '''
        Computes the left and right contours from an image.

        Parameters:
        - img: The input image.

        Returns:
        - Tuple: Two sequences of contour points, one for the left and one for the right.
        '''
        
        contours = self._get_road_contours_of_one_image(img,prefix=prefix)

        """         
        img_height = img.shape[0]
        road_top = int(self.road_contour_top * img_height)

        for contour in contours:
            contour[:, 0, :]
            contour = contour[contour[:, 1] > road_top] 
        """

        if len(contours) ==0 :
            logger.error(f'{self.frame_id} No contour found on road for {prefix} image.')
            return None,None
            
        contour = max(contours, key=cv2.contourArea)

        # Filter out points outside the specified range: too far or too close (z)
        road_top, road_bottom = self._get_road_top_and_bottom_v()
        points = contour.reshape(-1, 2)
        mask = (points[:, 1] >= road_top) & (points[:, 1] <= road_bottom)
        contour_points = points[mask]

        left_poly_model, right_poly_model, inliers_left_mask, inliers_right_mask = find_best_2_best_polynomial_contours(contour_points, degree=self.degree)
        if left_poly_model is None:
            return None,None
        
        self.left_poly_model = left_poly_model
        self.right_poly_model = right_poly_model
        
        contour_left = contour_points[inliers_left_mask]
        contour_right = contour_points[inliers_right_mask]

        if self.debug:
            logger.debug(f'{self.frame_id} Found {len(contours)} contours')
            # Draw contours 
            contour_image = img.copy()
            cv2.drawContours(contour_image, [contour_left], -1, (255, 0, 0), 3)
            cv2.drawContours(contour_image, [contour_right], -1, (0, 255, 0), 3)
            cv2.imwrite(get_output_path(f'{self.frame_id}_{prefix}_contours_left_right.png'), contour_image)

        return contour_left, contour_right

    
    def _debug_display_road_points2D(self, 
                                     left_img_contour_left: Sequence[np.ndarray], 
                                     left_img_contour_right: Sequence[np.ndarray], 
                                     optimized_road_vector:np.array, 
                                     img_width: int, 
                                     img_height: int) -> None:
        '''
        Displays road points in a 2D image for debugging.

        Parameters:
        - left_img_contour_left: Left road contour points in the image.
        - left_img_contour_right: Right road contour points in the image.
        - optimized_road_transform: Optimized road_transform.
        - img_width: Image width.
        - img_height: Image height.
        '''
        optimized_road_width, optimized_road_transform = self._road_vector_to_road_width_and_transform(optimized_road_vector)
        half_road_width = optimized_road_width / 2
        
        display_coeff = 10
        minX = -half_road_width*2
        maxX = half_road_width*2

        #show left
        road_points_left, road_points_right = self._compute_left_right_road_points2D(left_img_contour_left, left_img_contour_right, optimized_road_transform, img_width, img_height)
        road_points_2D = np.concatenate((road_points_left, road_points_right), axis=0)

        road_points_x = road_points_2D[:, 0]
        road_points_z = road_points_2D[:, 1]

        minZ = np.min(road_points_z)
        maxZ = np.max(road_points_z)

        width = math.ceil(maxX - minX)
        height = math.ceil(maxZ - minZ)

        road_points_2D[:, 0] = road_points_x - minX
        road_points_2D[:, 1] = road_points_z - minZ
        
        road_image_height =height * display_coeff
        road_image_width =width * display_coeff
        road_image = np.zeros((road_image_height, road_image_width, 3), dtype=np.uint8)

        model_left_x = int((-half_road_width-minX)*display_coeff)
        model_right_x = int((half_road_width-minX)*display_coeff)
        cv2.line(road_image,(model_left_x,0),(model_left_x,road_image_height-1),(0,0,255),1)
        cv2.line(road_image,(model_right_x,0),(model_right_x,road_image_height-1),(0,0,255),1)  
        
        for p in road_points_2D:
            x_left = int(p[0] * display_coeff)
            y = int(p[1] * display_coeff)
            road_image[y, x_left] = [255, 0, 0]

        cv2.imwrite(get_output_path(f'{self.frame_id}_plane_road_image.png'), road_image)

    def _compute_left_right_road_points2D(self, 
                                          img_contour_left: Sequence[np.ndarray], 
                                          img_contour_right: Sequence[np.ndarray], 
                                          road_plane_transform:Transform, 
                                          img_width: int, 
                                          img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects road contours onto a 2D road plane.

        Parameters:
        - img_contour_left: Left road contour points in the image.
        - img_contour_right: Right road contour points in the image.
        - camHeight: Camera height.
        - road_rvec: Road rotation vector.
        - img_width: Image width.
        - img_height: Image height.

        Returns:
        - Tuple: Two 2D arrays of road points (left and right).
        """
        left_contour_x = img_contour_left[:, 0]
        left_contour_y = img_contour_left[:, 1]
        road_points_left = equirect_to_road_plane_points2D(imgWidth=img_width, imgHeight=img_height, 
                                                           plane_transform=road_plane_transform, 
                                                           contour_x=left_contour_x, contour_y=left_contour_y)

        right_contour_x = img_contour_right[:, 0]
        right_contour_y = img_contour_right[:, 1]
        road_points_right = equirect_to_road_plane_points2D(imgWidth=img_width, imgHeight=img_height,
                                                            plane_transform=road_plane_transform, 
                                                            contour_x=right_contour_x, contour_y=right_contour_y)

        road_points_left = road_points_left[road_points_left[:, 1].argsort()]
        road_points_right = road_points_right[road_points_right[:, 1].argsort()]

        return road_points_left, road_points_right
    
    def _compute_residuals(self, 
                                road_params: np.ndarray, 
                                left_img_contour_left: np.ndarray, 
                                left_img_contour_right: np.ndarray, 
                                img_width: int, 
                                img_height: int) -> float:
        '''
        Computes the difference in slopes between the left and right road contours in the image.
        Parameters:
        - road_params (np.ndarray): [road width,tx,ty, pitch, yaw,roll].
        - img_contour_left (np.ndarray): Contour points of the left side of the road in the image.
        - img_contour_right (np.ndarray): Contour points of the right side of the road in the image.
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - float: The absolute difference between the slopes of the left and right contours.
        '''
        road_width, road_plane_transform_world = self._road_vector_to_road_width_and_transform(road_params)

        
        
        left_road_points_left, left_road_points_right = self._compute_left_right_road_points2D(
            left_img_contour_left, left_img_contour_right, road_plane_transform_world, img_width, img_height
        )

        all_points  = np.concatenate((left_road_points_left, left_road_points_right), axis=0)
    
        x_p = all_points[:, 0]
        z = all_points[:, 1]

        half_road_width = road_width / 2
        # Compute x positions of the left and right lines
        x_line_left = -half_road_width
        x_line_right = half_road_width

        # Compute squared differences
        diff_left = (x_line_left - x_p) ** 2
        diff_right = (x_line_right - x_p) ** 2

        # Compute minimum squared difference for each point
        min_diffs = np.minimum(diff_left, diff_right)

        # Sum up the residuals
        residue = np.sum(min_diffs)

        return residue

    def _optimize_road_vector(self, 
                            left_img_contour_left: np.ndarray, 
                            left_img_contour_right: np.ndarray, 
                            initial_road_width:float, 
                            initial_road_transform: Transform,
                            bounds: Tuple[np.ndarray, np.ndarray], 
                            img_width: int, 
                            img_height: int) -> Tuple[np.ndarray,float] :
        '''
        Optimizes the road plane equation (3 params) and plane transform (3 params : ty, pitch, roll).
        By comparing contours points projected to road plane to road equation in plane

        Parameters:
        - left_img_contour_left (np.ndarray): Contour points of the left side of the road in the left image.
        - left_img_contour_right (np.ndarray): Contour points of the right side of the road in the left image.
        - initial_road_width (float): initially estimated road width.
        - initial_road_transform (Transform): Initial guess for the road transform.
        - bounds (Tuple[np.ndarray, np.ndarray]): Bounds for the optimization, in the form (lower_bounds, upper_bounds).
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - road_width: The optimized road width.
        - Transfrom : The optimized road transform
        - cost: the remaining residual. Useful to assess the quality of optimization
        '''
        # Optimize the cam_rotation_vector using least_squares
        initial_params = self._road_vector_from_road_width_and_transform(road_width=initial_road_width,road_transform=initial_road_transform)
        
        result = least_squares(
            fun=lambda road_params: self._compute_residuals(
                road_params,
                left_img_contour_left, left_img_contour_right, 
                img_width, img_height
            ),
            x0=initial_params,
            bounds=bounds,
            method='trf'  # 'trf' method supports bounds
        )
        self.estimated_road_vector = result.x
        # Return the optimized camera rotation vector
        return self.estimated_road_vector, result.cost
    
    def compute_road_width(self, img_left: cvImage) -> Tuple[float,float]:
        '''
        Computes the road width from an image.

        Parameters:
        - img_left_right: The input image (equirectangular) containing horizontally concatenated left and right images.

        Returns:
        - float: Estimated road width in meters.
        - float: cost per point. The lower, the safer the optimization is fine
        '''
        logger.info(f'{self.frame_id} Computing eq road width')
        not_found =(-1.,1.)
        initial_road_width, initial_road_transform = self._road_vector_to_road_width_and_transform(self.estimated_road_vector)
        
        img_height, img_width = img_left.shape[:2]

        start_segementation_time = time.time()
        logger.debug(f'{self.frame_id} testing contours.')    
        self.left_img_contour_left, self.left_img_contour_right = self._get_left_right_contours_of_one_image(img_left,"left")
       
        #check if we found contours for left image
        if self.left_img_contour_left is None or  self.left_img_contour_right is  None:
            logger.error(f'{self.frame_id} Could not left and right contours.')   
            return not_found
        
        end_segmentation_time = time.time()
        logger.debug(f'{self.frame_id} Segmentation time: {round(end_segmentation_time - start_segementation_time,1)} seconds')
        
        self.left_img_contour_left = self.left_img_contour_left[self.left_img_contour_left[:, 1].argsort()]
        self.left_img_contour_right = self.left_img_contour_right[self.left_img_contour_right[:, 1].argsort()]

        # Project on the road plane and optimize road vector
        bounds = (self.lower_bounds, self.upper_bounds)
        
        #we must ensure to have roughly same nb points for left and right camera => better optimization
        min_nb_points = min(20,len(self.left_img_contour_left), len(self.left_img_contour_right))
        
        if min_nb_points<5:
            logger.error(f'{self.frame_id} Not enough points to compute road width, min_nb_points is {min_nb_points}')
            return not_found

        arrays = [self.left_img_contour_left,self.left_img_contour_right]
        returned_arrays = []
        for a in arrays:
            indices = random.sample(range(len(a)), min_nb_points)
            returned_arrays.append(np.array([a[i] for i in indices]))
        left_img_contour_left,left_img_contour_right = returned_arrays
        
        optimized_road_vector, cost = self._optimize_road_vector(
            left_img_contour_left, left_img_contour_right, 
            initial_road_width=initial_road_width,
            initial_road_transform=initial_road_transform,
            bounds=bounds, img_width=img_width, img_height=img_height)
        
        nb_points = len(left_img_contour_left)+len(left_img_contour_right)
        cost = cost/nb_points
        logger.debug(f'{self.frame_id} optimized_road_vector {optimized_road_vector} width cost per point {cost}')
        optimized_road_width, optimized_road_transform = self._road_vector_to_road_width_and_transform(optimized_road_vector)
        
        if self.debug:
            # Debug and display road points
            self._debug_display_road_points2D(left_img_contour_left, left_img_contour_right,
                                              optimized_road_vector, 
                                              img_width, img_height)

        self.optimized_road_vector = optimized_road_vector
        return optimized_road_width, cost