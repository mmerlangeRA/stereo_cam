import math
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
    - road_down_y (float): Estimated camera height from the road plane in meters.
    - road_contour_top (float): Attention window. Road segmentation and contours 
      will be searched only within this portion of the image (0-1 range).
    """
    road_down_y: float = 1.65

    min_z_for_road:float
    max_z_for_road:float
    image_height:int
    image_width:int

    left_poly_model:any
    right_poly_model:any
    contour_left:Sequence[np.ndarray]
    contour_right:Sequence[np.ndarray]

    lower_bounds = np.array([-np.pi/4, -np.pi/10000, -np.pi/10000])
    upper_bounds = np.array([np.pi/4, np.pi/10000, np.pi/10000])

    def __init__(self, 
                 roadSegmentator: RoadSegmentator, 
                 window: AttentionWindow, 
                 road_down_y: float = 1.65, 
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
        self.road_down_y = road_down_y
        self.min_z_for_road = min_z_for_road
        self.max_z_for_road = max_z_for_road
        self.image_height = image_height
        self.image_width = image_width

    
    def _get_road_top_and_bottom_v(self)->Tuple[float, float]:
        plane_y = self.road_down_y
        theta = math.atan(plane_y/self.max_z_for_road) 
        road_contour_top = int(theta*self.image_height/np.pi +self.image_height/2.)
        theta = math.atan(plane_y/self.min_z_for_road) 
        road_contour_bottom = int(theta*self.image_height/np.pi +self.image_height/2.)
        return road_contour_top, road_contour_bottom
    
    def _get_road_contours(self, img: cvImage) -> Sequence[np.ndarray]:
        """
        Computes road contours from an input image.

        Parameters:
        - img: The input image (equirectangular).

        Returns:
        - Sequence[np.ndarray]: A sequence of contour points.
        """
        windowed = self.window.crop_image(img)
        if self.debug:
            cv2.imwrite(get_output_path("windowed.png"), windowed)

        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        cv2.imwrite(get_output_path("thresh_windowed.png"), self.thresh_windowed)

        thresh = np.zeros(img.shape[:2], dtype=np.uint8)
        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _get_left_right_contours(self, img: cvImage, prefix="") -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """
        Computes the left and right contours from an image.

        Parameters:
        - img: The input image.

        Returns:
        - Tuple: Two sequences of contour points, one for the left and one for the right.
        """
        contours = self._get_road_contours(img)

        if len(contours) == 0:
            print("No contour found on road.")

        contour = max(contours, key=cv2.contourArea)
        road_top, road_bottom = self._get_road_top_and_bottom_v()
        points = contour.reshape(-1, 2)
        mask = (points[:, 1] >= road_top) & (points[:, 1] <= road_bottom)
        contour_points = points[mask]

        left_poly_model, right_poly_model, inliers_left_mask, inliers_right_mask = find_best_2_best_polynomial_contours(contour_points, degree=self.degree)
        if left_poly_model is None:
            return None,None
        
        contour_left = contour_points[inliers_left_mask]
        contour_right = contour_points[inliers_right_mask]

        if self.debug:
            print(f'Found {len(contours)} contours')
            contour_image = img.copy()
            # Draw contours with random colors
            cv2.drawContours(contour_image, [contour_left], -1, (255, 0, 0), 3)
            cv2.drawContours(contour_image, [contour_right], -1, (0, 255, 0), 3)
            cv2.imwrite(get_output_path(f"{prefix}_contours.png"), contour_image)

        return contour_left, contour_right

    def _debug_display_road_points2D(self, 
                                     left_img_contour_left: Sequence[np.ndarray], 
                                     left_img_contour_right: Sequence[np.ndarray], 
                                     optimized_transform:Transform, 
                                     img_width: int, 
                                     img_height: int) -> None:
        """
        Displays road points in a 2D image for debugging.

        Parameters:
        - left_img_contour_left: Left road contour points in the image.
        - left_img_contour_right: Right road contour points in the image.
        - optimized_transformr: Optimized plane transform.
        - img_width: Image width.
        - img_height: Image height.
        """
        road_points_left, road_points_right = self._compute_left_right_road_points2D(left_img_contour_left, left_img_contour_right, optimized_transform, img_width, img_height)
        road_points_2D = np.concatenate((road_points_left, road_points_right), axis=0)

        road_points_x = road_points_2D[:, 0]
        road_points_z = road_points_2D[:, 1]

        minX = np.min(road_points_x)
        maxX = np.max(road_points_x)
        minZ = np.min(road_points_z)
        maxZ = np.max(road_points_z)

        width = math.ceil(maxX - minX)
        height = math.ceil(maxZ - minZ)

        road_points_2D[:, 0] = road_points_x - minX
        road_points_2D[:, 1] = road_points_z - minZ

        display_coeff = 10
        road_image = np.zeros((height * display_coeff, width * display_coeff, 3), dtype=np.uint8)

        for p in road_points_2D:
            x = int(p[0] * display_coeff)
            y = int(p[1] * display_coeff)
            road_image[y, x] = [255, 0, 0]

        cv2.imwrite(get_output_path('road_image.png'), road_image)

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
    
    def _compute_slopes_difference(self, 
                                road_rotation_vector: np.ndarray, 
                                left_img_contour_left: np.ndarray, 
                                left_img_contour_right: np.ndarray, 
                                road_down_y: float, 
                                img_width: int, 
                                img_height: int) -> float:
        """
        Computes the difference in slopes between the left and right road contours in the image.

        Parameters:
        - road_rotation_vector (np.ndarray): Camera rotation vector.
        - left_img_contour_left (np.ndarray): Contour points of the left side of the road in the image.
        - left_img_contour_right (np.ndarray): Contour points of the right side of the road in the image.
        - road_down_y (float): Height of the camera above the road plane.
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - float: The absolute difference between the slopes of the left and right contours.
        """
        road_plane_transform = Transform(0.,road_down_y,0.,road_rotation_vector[0],road_rotation_vector[1],road_rotation_vector[2])
        road_points_left, road_points_right = self._compute_left_right_road_points2D(
            left_img_contour_left, left_img_contour_right, road_plane_transform, img_width, img_height
        )
        
        nb_left = road_points_left.shape[0]
        nb_right = road_points_right.shape[0]

        xLeftpoints = road_points_left[:, 0]
        yLeftpoints = road_points_left[:, 1]
        xRightpoints = road_points_right[:, 0]
        yRightpoints = road_points_right[:, 1]

        # First and last points for the left contour
        xleft_1, yleft_1 = xLeftpoints[0], yLeftpoints[0]
        xleft_2, yleft_2 = xLeftpoints[nb_left - 1], yLeftpoints[nb_left - 1]

        # First and last points for the right contour
        xright_1, yright_1 = xRightpoints[0], yRightpoints[0]
        xright_2, yright_2 = xRightpoints[nb_right - 1], yRightpoints[nb_right - 1]

        # Calculate slopes for left and right contours
        slopes_left = (xleft_2 - xleft_1) / (yleft_2 - yleft_1)
        slopes_right = (xright_2 - xright_1) / (yright_2 - yright_1)
        
        # Absolute difference between slopes
        slopes_diff = np.abs(slopes_left - slopes_right)

        return slopes_diff

    def _optimize_road_rotation(self, 
                            left_img_contour_left: np.ndarray, 
                            left_img_contour_right: np.ndarray, 
                            road_down_y: float, 
                            initial_road_rotation_vector: np.ndarray, 
                            bounds: Tuple[np.ndarray, np.ndarray], 
                            img_width: int, 
                            img_height: int) -> np.ndarray:
        """
        Optimizes the camera rotation vector by minimizing the difference in slopes between the left and right contours.

        Parameters:
        - left_img_contour_left (np.ndarray): Contour points of the left side of the road in the image.
        - left_img_contour_right (np.ndarray): Contour points of the right side of the road in the image.
        - road_down_y (float): Height of the camera above the road plane.
        - initial_road_rotation_vector (np.ndarray): Initial guess for the road rotation vector.
        - bounds (Tuple[np.ndarray, np.ndarray]): Bounds for the optimization, in the form (lower_bounds, upper_bounds).
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - np.ndarray: The optimized camera rotation vector.
        """
        # Optimize the road_rotation_vector using least_squares
        result = least_squares(
            fun=lambda road_rotation_vector: self._compute_slopes_difference(
                road_rotation_vector, left_img_contour_left, left_img_contour_right, road_down_y, img_width, img_height
            ),
            x0=initial_road_rotation_vector,
            bounds=bounds,
            method='trf'  # 'trf' method supports bounds
        )
        
        # Return the optimized camera rotation vector
        return result.x
    
    def compute_road_width(self, img: cvImage) -> float:
        """
        Computes the road width from an image.

        Parameters:
        - img: The input image (equirectangular).

        Returns:
        - float: Estimated road width in meters.
        """
        not_found= tuple([-1., 1.])
        if self.debug:
            start_segementation_time = time.time()

        self.left_img_contour_left, self.left_img_contour_right = self._get_left_right_contours(img)
        if self.left_img_contour_left is None or self.left_img_contour_right is None:
            logger.error(f'{self.frame_id} No contours found on road.')
            return not_found
        
        img_height, img_width = img.shape[:2]

        if self.debug:
            end_segementation_time = time.time()
            print(f'Segmentation time: {end_segementation_time - start_segementation_time} seconds')

        # Filter to select only the closest points to the car
        self.left_img_contour_left = self.left_img_contour_left[self.left_img_contour_left[:, 1].argsort()]
        self.left_img_contour_right = self.left_img_contour_right[self.left_img_contour_right[:, 1].argsort()]

        # Project on the road plane and optimize camera rotation
        cam0Transform = Transform(0., 0, 0, 0., 0., 0.)
        initial_cam_rotation_vector = np.array(cam0Transform.rotationVector)

        bounds = (self.lower_bounds, self.upper_bounds)

        optimized_rotation_vector = self._optimize_road_rotation(self.left_img_contour_left, self.left_img_contour_right, 
                                                                self.road_down_y, initial_cam_rotation_vector, 
                                                                bounds, img_width, img_height)

        road_rvec = optimized_rotation_vector

        optimized_transform = Transform(0., self.road_down_y, 0., road_rvec[0], road_rvec[1], road_rvec[2])

        if self.debug:
            
            # Debug and display road points
            self._debug_display_road_points2D(left_img_contour_left, left_img_contour_right, 
                                              optimized_transform, 
                                              img_width, img_height)

        # Now estimate the distance between left and right road points
        road_points_left, road_points_right = self._compute_left_right_road_points2D(self.left_img_contour_left, 
                                                                                     left_img_contour_right, 
                                                                                     optimized_transform, 
                                                                                     img_width, img_height)

        x_left = np.mean(road_points_left[:, 0])
        slope_left, intercept_left = np.polyfit(road_points_left[:, 0], road_points_left[:, 1], 1)
        slope_right, intercept_right = np.polyfit(road_points_right[:, 0], road_points_right[:, 1], 1)

        if slope_left == 0:
            slope_left = 0.0001  # Prevent division by zero, set a small value
        else:
            y_left = slope_left * x_left + intercept_left
            x_right = (intercept_right - y_left + x_left / slope_left) / (1 / slope_left - slope_right)
            y_right = slope_right * x_right + intercept_right

        road_width = np.sqrt((x_right - x_left) ** 2 + (y_left - y_right) ** 2)
        return road_width,0
