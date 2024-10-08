import math
import random
import time
import cv2
import numpy as np
from typing import Sequence, Tuple, List, Optional
import numpy.typing as npt
from scipy.optimize import minimize,least_squares

from src.utils.path_utils import get_ouput_path
from src.utils.coordinate_transforms import cartesian_to_equirectangular, equirect_to_road_plane_points2D, get_3D_points_to_cam_transform_referential, get_transformation_matrix, pixel_to_spherical, spherical_to_cartesian
from src.road_detection.RoadSegmentator import RoadSegmentator
from src.road_detection.common import AttentionWindow
from src.utils.TransformClass import Transform
from src.utils.typing import cvImage
from src.utils.intersection_utils import get_plane_P0_and_N_from_transform
from src.road_detection.RoadDetector import RoadDetector,RoadEquationInPlane
from src.utils.curve_fitting import find_best_2_best_contours

class EquirectStereoRoadDetector(RoadDetector):
    """
    Estimates road width from 2 equirectangular images.
    
    Assumptions:
    - The road is assumed to be a plane.
    - The camera's view direction is assumed to be parallel to the road.
    
    Attributes:
    - road_down_y (float): Estimated camera height from the road plane in meters.
    - road_contour_top (float): Attention window. Road segmentation and contours 
      will be searched only within this portion of the image (0-1 range).
    """
    road_down_y: float = 1.65
    road_contour_top: float = 0.53
    estimated_cam2_transform:Transform
    road_vector : np.array
    left_img_contour_left: Sequence[np.ndarray]=[]
    left_img_contour_right: Sequence[np.ndarray] =[]
    right_img_contour_left: Sequence[np.ndarray] =[]
    right_img_contour_right: Sequence[np.ndarray] =[]

    def __init__(self, 
                 roadSegmentator: 'RoadSegmentator', 
                 window: 'AttentionWindow', 
                 road_down_y: float = 1.65, 
                 degree: int = 1, 
                 road_contour_top: float = 0.53, 
                 estimated_cam2_transform=Transform(1.12,0.,0.,0.,0.,0.),
                 debug: bool = False) -> None:
        """
        Initializes the EquirectRoadDetector.

        Parameters:
        - roadSegmentator: A road segmentation model instance.
        - window: Attention window for focusing the detection.
        - road_down_y: The estimated height of the camera from the road plane (default=1.65).
        - degree: Degree for polynomial fitting when processing contours (default=1).
        - road_contour_top: The top boundary for filtering road contours. 
        - estimated_cam2_transform : estimated transform of cam right
        We need larger window for good segmentation, then we consider only contours below road_contour_top.
        - debug: If True, saves debug images and prints debug information (default=False).
        """
        super().__init__(roadSegmentator, window=window, degree=degree, debug=debug)
        self.road_down_y = road_down_y
        self.road_contour_top = road_contour_top
        self.estimated_cam2_transform = estimated_cam2_transform

    def _road_vector_to_road_equation_and_transform(self,road_vector:np.array) -> Tuple[RoadEquationInPlane,Transform]:
        slope_x_from_z,intercept_z, road_width,yc, pitch, roll=road_vector
        road_equation = RoadEquationInPlane(slope_x_from_z=slope_x_from_z,width=road_width,intercept_z=intercept_z)
        road_transform = Transform(yc=yc, pitch=pitch, roll=roll)
        return road_equation,road_transform
    
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
            cv2.imwrite(get_ouput_path("windowed.png"), windowed)

        self.thresh_windowed = self.roadSegmentator.segment_road_image(windowed)
        cv2.imwrite(get_ouput_path("thresh_windowed.png"), self.thresh_windowed)

        thresh = np.zeros(img.shape[:2], dtype=np.uint8)
        thresh[self.window.top:self.window.bottom, self.window.left:self.window.right] = self.thresh_windowed

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _get_left_right_contours(self, img: cvImage) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """
        Computes the left and right contours from an image.

        Parameters:
        - img: The input image.

        Returns:
        - Tuple: Two sequences of contour points, one for the left and one for the right.
        """
        contours = self._get_road_contours(img)

        if len(contours) <2:
            print("No contour found on road.")
            return None,None
            

        contour = max(contours, key=cv2.contourArea)
        contour_points = contour[:, 0, :]

        left_poly_model, right_poly_model, inliers_left_mask, inliers_right_mask = find_best_2_best_contours(contour, degree=self.degree)
        contour_left = contour_points[inliers_left_mask]
        contour_right = contour_points[inliers_right_mask]

        if self.debug:
            print(f'Found {len(contours)} contours')
            contour_image = img.copy()
            # Draw contours with random colors
            cv2.drawContours(contour_image, [contour_left], -1, (255, 0, 0), 3)
            cv2.drawContours(contour_image, [contour_right], -1, (0, 255, 0), 3)
            cv2.imwrite(get_ouput_path("contours.png"), contour_image)

        return contour_left, contour_right

    def _debug_display_road_points2D(self, 
                                     left_img_contour_left: Sequence[np.ndarray], 
                                     left_img_contour_right: Sequence[np.ndarray], 
                                     right_img_contour_left: Sequence[np.ndarray], 
                                     right_img_contour_right: Sequence[np.ndarray], 
                                     optimized_road_vector:np.array, 
                                     img_width: int, 
                                     img_height: int) -> None:
        """
        Displays road points in a 2D image for debugging.

        Parameters:
        - left_img_contour_left: Left road contour points in the image.
        - left_img_contour_right: Right road contour points in the image.
        - optimized_road_transform: Optimized road_transform.
        - img_width: Image width.
        - img_height: Image height.
        """
        slope_x_from_z,intercept_z, road_width,yc, pitch, roll=optimized_road_vector
        optimized_road_transform= Transform(0.,yc,0.,pitch,0.,roll)
        #show left
        road_points_left, road_points_right = self._compute_left_right_road_points2D(left_img_contour_left, left_img_contour_right, optimized_road_transform, img_width, img_height)
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
        road_image = np.zeros((height * display_coeff, (width+10) * display_coeff, 3), dtype=np.uint8)

        for p in road_points_2D:
            x_left = int(p[0] * display_coeff)
            y = int(p[1] * display_coeff)
            road_image[y, x_left] = [255, 0, 0]

        #show right
        P0_camLeft, N_camLeft =get_plane_P0_and_N_from_transform(optimized_road_transform)
        camRightRotationInversed =self.estimated_cam2_transform.inverseRotationMatrix

        P0_camRight = camRightRotationInversed @ (P0_camLeft - self.estimated_cam2_transform.translationVector)
        #N_camRight = camRightRotationInversed @ N_camLeft
        rvec_right, _ = cv2.Rodrigues(camRightRotationInversed)
        rvec_right = rvec_right.flatten()
        road_plane_transform_camRight = Transform(xc=P0_camRight[0], yc=P0_camRight[1], zc=P0_camRight[2], pitch=rvec_right[0], yaw=rvec_right[1], roll=rvec_right[2])
        
        road_points_left, road_points_right = self._compute_left_right_road_points2D(right_img_contour_left, right_img_contour_right, road_plane_transform_camRight, img_width, img_height)
        road_points_2D = np.concatenate((road_points_left, road_points_right), axis=0)

        road_points_x = road_points_2D[:, 0]
        road_points_z = road_points_2D[:, 1]

        road_points_2D[:, 0] = road_points_x - minX
        road_points_2D[:, 1] = road_points_z - minZ

        for p in road_points_2D:
            x_left = int(p[0] * display_coeff)
            y = int(p[1] * display_coeff)
            if x_left>0 and x_left<(width* display_coeff) and y>0 and y<(height * display_coeff):
                road_image[y, x_left] = [0, 0, 255]

        nb_points=20
        for index in range(nb_points):
            z= (maxZ-minZ)*index/nb_points+minZ
            y = int((z - minZ) * display_coeff)
            x_left= slope_x_from_z*z+intercept_z
            x_left-=minX
            x_left*=display_coeff
            x_right = x_left+road_width*display_coeff
            x_left= int(x_left)
            
            if x_left>0 and x_left<(width* display_coeff) and y>0 and y<(height * display_coeff):
                road_image[y, x_left] = [0,255, 0]
            
            x_right = int(x_right)
            if x_right>0 and x_right<(width* display_coeff) and y>0 and y<(height * display_coeff):
                road_image[y, x_right] = [0,255, 0]

        cv2.imwrite(get_ouput_path('road_image.png'), road_image)

    def _debug_display_projected_road_on_image(self, 
                                    img:cvImage,
                                     img_contour_left: Sequence[np.ndarray], 
                                     img_contour_right: Sequence[np.ndarray], 
                                     cam_transform : Transform,
                                     optimized_road_vector:np.array) -> cvImage:
        """
        Displays contours and projected road equation on an image

        Parameters:
        - img : background image
        - img_contour_left: Left road contour points in the image.
        - img_contour_right: Right road contour points in the image.
        - cam_transform: transform of the camera
        - optimized_road_transform: Optimized road_transform.
        """
        image_height, image_width = img.shape[:2]

        debug_image = img.copy()

        road_equation, road_transform = self._road_vector_to_road_equation_and_transform(optimized_road_vector)
        contours = np.concatenate((img_contour_left, img_contour_right), axis=0)
        
        for p in contours:
            x = int(p[0])
            y = int(p[1])
            cv2.circle(debug_image, (x, y), 3, (255, 0, 0), -1)

        nb_projected_road = 100
        road_points_3D_origin_plane = []
        for index in range(nb_projected_road):
            z= 50*index/nb_projected_road
            x_left= road_equation.slope_x_from_z*z+road_equation.intercept_z
            x_right = x_left+road_equation.width
            road_points_3D_origin_plane.append([x_left,0,z])
            road_points_3D_origin_plane.append([x_right,0,z])

        t = road_transform.translationVector
        t = t[:, np.newaxis] 
        rotationMatrix = road_transform.rotationMatrix
        road_points_3D_origin_plane = np.array(road_points_3D_origin_plane)
        road_points_world_space = np.dot(rotationMatrix, (road_points_3D_origin_plane.T + t)).T

        cam_points = get_3D_points_to_cam_transform_referential(road_points_world_space,cam_transform)

        u,v = cartesian_to_equirectangular(cam_points[:,0],cam_points[:,1],cam_points[:,2],
                                                     image_width=image_width, image_height=image_height,to_int=True)
        for i in range(len(u)):
            cv2.circle(debug_image, (u[i], v[i]), 3, (0, 255, 0), -1)

        road_contour_top = int(self.road_contour_top*image_height)

        cv2.line(debug_image,(0,road_contour_top),(image_width-1,road_contour_top),(0, 0, 255),3)
        return debug_image
           
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
        - road_plane_transform (Transform): Road plane transform.
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
                                camRightTransform:Transform,
                                left_img_contour_left: np.ndarray, 
                                left_img_contour_right: np.ndarray, 
                                right_img_contour_left: np.ndarray, 
                                right_img_contour_right: np.ndarray, 
                                img_width: int, 
                                img_height: int) -> float:
        """
        Computes the difference in slopes between the left and right road contours in the image.
        Parameters:
        - road_params (np.ndarray): [slope_x_from_z,intercept_z, width,
                          .ty, .pitch, .roll].
        - img_contour_left (np.ndarray): Contour points of the left side of the road in the image.
        - img_contour_right (np.ndarray): Contour points of the right side of the road in the image.
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - float: The absolute difference between the slopes of the left and right contours.
        """
        slope_x_from_z,intercept_z, road_width,ty, pitch, roll = road_params
        road_plane_transform = Transform(0.,ty,0., pitch, 0.,roll)

        P0_camLeft, N_camLeft =get_plane_P0_and_N_from_transform(road_plane_transform)
        camRightRotationInversed =camRightTransform.inverseRotationMatrix

        P0_camRight = camRightRotationInversed @ (P0_camLeft - camRightTransform.translationVector)
        #N_camRight = camRightRotationInversed @ N_camLeft
        rvec_right, _ = cv2.Rodrigues(camRightRotationInversed)
        rvec_right = rvec_right.flatten()
        road_plane_transform_camRight = Transform(xc=P0_camRight[0], yc=P0_camRight[1], zc=P0_camRight[2], pitch=rvec_right[0], yaw=rvec_right[1], roll=rvec_right[2])
        
        left_road_points_left, left_road_points_right = self._compute_left_right_road_points2D(
            left_img_contour_left, left_img_contour_right, road_plane_transform, img_width, img_height
        )

        right_road_points_left, right_road_points_right = self._compute_left_right_road_points2D(
            right_img_contour_left, right_img_contour_right, road_plane_transform_camRight, img_width, img_height
        )
        all_points  = np.concatenate((left_road_points_left, left_road_points_right, right_road_points_left, right_road_points_right), axis=0)
    
        residue = 0.
        for p in all_points:
            z=p[1]
            x_line_left= slope_x_from_z*z+intercept_z
            x_line_right= slope_x_from_z*z+intercept_z+road_width
            #residue += min(abs(x_line_left-p[0]),abs(x_line_right-p[0]))
            residue += min((x_line_left-p[0])*(x_line_left-p[0]),(x_line_right-p[0])*(x_line_right-p[0]))

        return residue

    def _optimize_road_equation_and_rotation(self, 
                            left_img_contour_left: np.ndarray, 
                            left_img_contour_right: np.ndarray, 
                            right_img_contour_left: np.ndarray, 
                            right_img_contour_right: np.ndarray, 
                            initial_road_equation:RoadEquationInPlane, 
                            initial_road_transform: Transform,
                            bounds: Tuple[np.ndarray, np.ndarray], 
                            img_width: int, 
                            img_height: int) -> np.ndarray:
        """
        Optimizes the road plane equation (3 params) and plane transform (3 params : ty, pitch, roll).
        By comparing contours points projected to road plane to road equation in plane

        Parameters:
        - left_img_contour_left (np.ndarray): Contour points of the left side of the road in the left image.
        - left_img_contour_right (np.ndarray): Contour points of the right side of the road in the left image.
        - right_img_contour_left (np.ndarray): Contour points of the left side of the road in the right image.
        - right_img_contour_right (np.ndarray): Contour points of the right side of the road in the right image.
        - initial_road_equation (RoadEquationInPlane): initially estimated RoadEquationInPlane.
        - initial_road_transform (Transform): Initial guess for the road transform.
        - bounds (Tuple[np.ndarray, np.ndarray]): Bounds for the optimization, in the form (lower_bounds, upper_bounds).
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.

        Returns:
        - RoadEquationInPlane: The optimized road equation.
        - Transfrom : The optimized road transform
        """
        # Optimize the cam_rotation_vector using least_squares
        initial_params = [
            initial_road_equation.slope_x_from_z,
            initial_road_equation.intercept_z, 
            initial_road_equation.width,
            initial_road_transform.yc, 
            initial_road_transform.pitch, 
            initial_road_transform.roll]
        
        
        result = least_squares(
            fun=lambda road_params: self._compute_residuals(
                road_params,
                self.estimated_cam2_transform,
                left_img_contour_left, left_img_contour_right, 
                right_img_contour_left, right_img_contour_right,
                img_width, img_height
            ),
            x0=initial_params,
            bounds=bounds,
            method='trf'  # 'trf' method supports bounds
        )
        self.road_vector   = result.x
        # Return the optimized camera rotation vector
        return self.road_vector
    
    def compute_road_width(self, img_left_right: cvImage) -> float:
        """
        Computes the road width from an image.

        Parameters:
        - img: The input image (equirectangular) containing horizontally concatenated left and right images.

        Returns:
        - float: Estimated road width in meters.
        """
        height, width = img_left_right.shape[:2]
        # Ensure the width is even so that it can be evenly split into two halves
        assert width % 2 == 0, "Image width is not even. Cannot split into two equal halves."

        # Calculate the middle index for the split
        middle = width // 2

        # Split the image into left and right halves
        img_left = img_left_right[:, :middle]
        img_right = img_left_right[:, middle:]
        img_height, img_width = img_left.shape[:2]

        if self.debug:
            start_segementation_time = time.time()
            
        self.left_img_contour_left, self.left_img_contour_right = self._get_left_right_contours(img_left)
        self.right_img_contour_left, self.right_img_contour_right = self._get_left_right_contours(img_right)

        if self.left_img_contour_left is None or self.right_img_contour_left is None:
            return -1.

        if self.debug:
            end_segementation_time = time.time()
            print(f'Segmentation time: {end_segementation_time - start_segementation_time} seconds')

        # Filter to select only the closest points to the car
        road_top = int(self.road_contour_top * img_height)
        self.left_img_contour_left = self.left_img_contour_left[self.left_img_contour_left[:, 1] > road_top]
        self.left_img_contour_right = self.left_img_contour_right[self.left_img_contour_right[:, 1] > road_top]
        self.right_img_contour_left = self.right_img_contour_left[self.right_img_contour_left[:, 1] > road_top]
        self.right_img_contour_right = self.right_img_contour_right[self.right_img_contour_right[:, 1] > road_top]

        self.left_img_contour_left = self.left_img_contour_left[self.left_img_contour_left[:, 1].argsort()]
        self.left_img_contour_right = self.left_img_contour_right[self.left_img_contour_right[:, 1].argsort()]
        self.right_img_contour_left = self.right_img_contour_left[self.right_img_contour_left[:, 1].argsort()]
        self.right_img_contour_right = self.right_img_contour_right[self.right_img_contour_right[:, 1].argsort()]

        # Project on the road plane and optimize camera rotation
        lower_bounds = np.array([-0.5, -10., 2., 1.65, -20*np.pi/180, -10*np.pi/180])
        upper_bounds = np.array([ 0.5,  0.,  10., 2.5, 20*np.pi/180, 10*np.pi/180])
        bounds = (lower_bounds, upper_bounds)
        initial_road_equation = RoadEquationInPlane(0., 6., -3.)
        initial_road_transform = Transform(0., self.road_down_y, 0, 0., 0., 0.)

        #we must ensure to have roughly same nb points for left and right camera
        min_nb_points = min(20,len(self.left_img_contour_left), len(self.left_img_contour_right), len(self.right_img_contour_left), len(self.right_img_contour_right))
        arrays = [self.left_img_contour_left,self.left_img_contour_right,self.right_img_contour_left,self.right_img_contour_right]
        returned_arrays = []
        for a in arrays:
            indices = random.sample(range(len(a)), min_nb_points)
            returned_arrays.append(np.array([a[i] for i in indices]))
        left_img_contour_left,left_img_contour_right,right_img_contour_left,right_img_contour_right = returned_arrays
        
        optimized_road_vector = self._optimize_road_equation_and_rotation(
            left_img_contour_left, left_img_contour_right, 
            right_img_contour_left, right_img_contour_right, 
            initial_road_equation=initial_road_equation,
            initial_road_transform=initial_road_transform,
            bounds=bounds, img_width=img_width, img_height=img_height)

        print("optimized_road_vector",optimized_road_vector)
        slope_x_from_z,intercept_z, road_width,yc, pitch, roll=optimized_road_vector

        if self.debug:
            # Debug and display road points
            print(optimized_road_vector)
            self._debug_display_road_points2D(left_img_contour_left, left_img_contour_right, 
                                              right_img_contour_left, right_img_contour_right,
                                              optimized_road_vector, 
                                              img_width, img_height)

        
        return road_width

