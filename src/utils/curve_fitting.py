import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple, List, Optional
import numpy.typing as npt

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple
import numpy.typing as npt
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from src.utils.TransformClass import Transform
from src.utils.coordinate_transforms import cartesian_to_equirectangular
from src.utils.path_utils import get_ouput_path


class Road_line_params:
    '''
    Assume equation is 
    y=height (fixed)
    x= az+b
    '''
    height: float
    a: float
    b: float

    def __init__(self, a: float, b: float,height: float):
        self.height = height
        self.a = a
        self.b = b

    def __str__(self):
        return f"a={self.a}, b={self.b}, height={self.height}"
    
    def as_array(self):
        return np.array([self.a, self.b,self.height])
    
def vizualize_road_equirectangular(road_line_params: Road_line_params, camRight:Transform,img, debug_name:str="debug_vizualize"):
    a=road_line_params.a
    b=road_line_params.b
    h=road_line_params.height

    R2 = camRight.rotationMatrix
    t2 = camRight.translationVector

    image_height, image_width= img.shape[:2]
    debug_img = img.copy()

    # Generate 3D points along the line
    z_min, z_max = 0, 50  # Adjust range as needed
    num_points = 500
    z_values = np.linspace(z_min, z_max, num_points)
    x_values = a * z_values + b
    y_values = np.full_like(x_values, h)
    points_3d = np.stack((x_values, y_values, z_values), axis=-1)

    # Project into image 2
    points_cam2 = (R2 @ points_3d.T).T + t2  # Apply rotation and translation
    u2, v2 = cartesian_to_equirectangular(points_cam2[:,0],points_cam2[:,1],points_cam2[:,2],image_width, image_height, to_int=True)
    for u, v in zip(u2, v2):
        cv2.circle(debug_img, (u, v), 2, (0, 255, 0), -1)

    cv2.imwrite(get_ouput_path(f'{debug_name}.png'), debug_img)

def compute_residuals(params, data_cam1, data_cam2, camRight:Transform,image_width,image_height)->np.array:
    """
    Computes the residuals (distances) between the detected points and the projected line.

    Parameters:
    - params: Array containing the line parameters [a, b, h]
    - data_cam1: Detected points in image 1 (Nx2 array)
    - data_cam2: Detected points in image 2 (Mx2 array)
    - camera_params: Dictionary containing image dimensions and camera extrinsics

    Returns:
    - residuals: Concatenated array of residuals for both images
    """
    a, b, h = params
    R2 = camRight.rotationMatrix
    t2 = camRight.translationVector

    # Generate 3D points along the line
    z_min, z_max = 0, 50  # Adjust range as needed
    num_points = len(data_cam1)
    num_points= 500
    z_values = np.linspace(z_min, z_max, num_points)
    x_values = a * z_values + b
    y_values = np.full_like(x_values, h)
    points_3d = np.stack((x_values, y_values, z_values), axis=-1)

    # Project into image 1
    u1, v1 = cartesian_to_equirectangular(x_values,y_values,z_values,image_width, image_height, to_int=False)
    
    # Project into image 2
    points_cam2 = (R2 @ points_3d.T).T + t2  # Apply rotation and translation
    u2, v2 = cartesian_to_equirectangular(points_cam2[:,0],points_cam2[:,1],points_cam2[:,2],image_width, image_height, to_int=False)

    # For each image
    residuals_cam1 = compute_point_line_distances(data_cam1, u1, v1)
    residuals_cam2 = compute_point_line_distances(data_cam2, u2, v2)
    nb_points = len(residuals_cam1) + len(residuals_cam2)
    residuals = np.concatenate((residuals_cam1, residuals_cam2))
    print(np.sum(residuals)/(len(data_cam1)+len(data_cam2)))
    return residuals

def compute_point_line_distances(points, u_line, v_line):
    """
    Computes the minimal distance from each point to the projected line.

    Parameters:
    - points: Detected points (Kx2 array)
    - u_line, v_line: Projected line points

    Returns:
    - distances: Array of minimal distances
    """
    # Combine line points
    line_points = np.stack((u_line, v_line), axis=-1)

    # Compute distances
    distances = []
    for point in points:
        # Compute distance to all line points
        dists = cdist([point], line_points)
        # Find minimal distance
        min_dist = np.min(dists)
        distances.append(min_dist)
    return np.array(distances)


def fit_polynomial_ransac(
    x: npt.NDArray[np.float32], 
    y: npt.NDArray[np.float32], 
    degree: int = 2, 
    residual_threshold: float = 10
) -> Tuple[make_pipeline, npt.NDArray[np.bool_]]:
    """
    Fits a polynomial curve to the given x, y data using RANSAC.

    Parameters:
    - x: Array of x values.
    - y: Array of y values.
    - degree: Degree of the polynomial.
    - residual_threshold: RANSAC residual threshold.

    Returns:
    - poly_model: Fitted polynomial model.
    - inlier_mask: Mask of inliers detected by RANSAC.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must have the same non-zero length")
    poly_model = make_pipeline(
        PolynomialFeatures(degree), 
        RANSACRegressor(residual_threshold=residual_threshold)
    )
    poly_model.fit(x[:, np.newaxis], y)
    inlier_mask = poly_model.named_steps['ransacregressor'].inlier_mask_
    return poly_model, inlier_mask

def find_best_2_best_contours(
    contour: npt.NDArray[np.int32], 
    degree: int = 2, 
    max_pixel_distance: float = 10
) -> Tuple[make_pipeline, make_pipeline, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Finds two polynomial curves fit to the given contour points.

    Parameters:
    - contour: Array of contour points.
    - degree: Degree of the polynomial.
    - max_pixel_distance: RANSAC residual threshold.

    Returns:
    - first_poly_model: Fitted model for the left polynomial curve.
    - second_poly_model: Fitted model for the right polynomial curve.
    - inliers_first: Boolean mask of inliers for the left polynomial fit.
    - inliers_second_full: Boolean mask of inliers for the right polynomial fit over the initial data.
    """
    contour_points = contour[:, 0, :]
    x = contour_points[:, 0]
    y = contour_points[:, 1]

    # Fit the first polynomial curve using RANSAC
    first_poly_model, inliers_first = fit_polynomial_ransac(y, x, degree, residual_threshold=max_pixel_distance)

    # Identify outliers from the first fit
    outlier_indices = np.where(~inliers_first)[0]
    x_outliers = x[outlier_indices]
    y_outliers = y[outlier_indices]

    # Fit the second polynomial curve using RANSAC on the outliers
    second_poly_model, inliers_second = fit_polynomial_ransac(
        y_outliers, x_outliers, degree, residual_threshold=max_pixel_distance
    )

    # Map inliers from the second fit back to the original data indices
    inliers_second_full = np.zeros_like(inliers_first, dtype=bool)
    inliers_second_full[outlier_indices] = inliers_second

    mean_x_1= np.mean(x[inliers_first])
    mean_x_2= np.mean(x[inliers_second_full])

    swicth_order = mean_x_1 > mean_x_2
    if swicth_order:
        return second_poly_model, first_poly_model, inliers_second_full, inliers_first

    return first_poly_model, second_poly_model, inliers_first, inliers_second_full

def find_best_2_polynomial_curves(contour: npt.NDArray[np.int32],degree=2,max_pixel_distance=10) -> Tuple[make_pipeline, make_pipeline, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Finds and plots two polynomial curves fit to the given contour points.

    Parameters:
    - contour: Array of contour points.
    - img: Image to plot the curves on. Optional.

    Returns:
    - first_poly_model: Fitted model for the left polynomial curve.
    - second_poly_model: Fitted model for the right polynomial curve.
    - y_inliers_first: Inlier y-values for the left polynomial.
    - y_inliers_second: Inlier y-values for the right polynomial.
    """
    contour_points = contour[:, 0, :]
    y = contour_points[:, 1]
   

    first_poly_model,second_poly_model, inliers_first, inliers_second =find_best_2_best_contours(contour=contour,degree=degree,max_pixel_distance=max_pixel_distance)
    y_inliers_first = y[inliers_first]
    y_inliers_second = y[inliers_second]
    return first_poly_model, second_poly_model, y_inliers_first, y_inliers_second

