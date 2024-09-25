from typing import Tuple
import cv2
import numpy as np
from scipy.optimize import minimize
from src.road_detection.common import AttentionWindow
from scipy.spatial.transform import Rotation as R
from src.matching.match_simple_pytorch import FeatureExtractor
from src.utils.equirectangular.equirectangular_mapper import EquirectangularMapper
from src.utils.equirectangular.SignTools import SignTransform

def compute_error(
    equirect_image: np.ndarray,
    projected_image: np.ndarray,
    bounding_box: AttentionWindow,
    weight_pixel: float = 0.,
    weight_feature: float = 1.0
) -> float:
    """
    Computes the error between the projected image and the observed image within the bounding box,
    including a penalty for low overlap.
    
    Parameters:
    - equirect_image: The original equirectangular image (H, W, C).
    - projected_image: The projected roadsign image onto the equirectangular image (H, W, C).
    - bounding_box: AttentionWindow defining the bounding box.
    - overlap_weight: Weight for the overlap penalty term.
    
    Returns:
    - error: The computed error.
    """
    error_ssd=0.
    x_min=bounding_box.left
    y_min=bounding_box.top
    x_max=bounding_box.right
    y_max=bounding_box.bottom

   # Extract the regions within the bounding box
    observed_region = equirect_image[y_min:y_max, x_min:x_max]
    projected_region = projected_image[y_min:y_max, x_min:x_max]
    
    # Compute pixel-wise SSD error
    if weight_pixel>0:
        error_ssd = np.sum((observed_region.astype(float) - projected_region.astype(float)) ** 2)
    
    featureExtractor= FeatureExtractor()
    # Extract features from the observed and projected regions
    features_observed = featureExtractor.extract_features(observed_region)
    features_projected = featureExtractor.extract_features(projected_region)
    
    # Compute feature-based distance
    feature_distance = featureExtractor.compute_feature_distance(features_observed, features_projected)
    
    # Combine errors with weights
    total_error = weight_pixel * error_ssd + weight_feature * feature_distance
    
    return total_error

def optimize_roadsign_position_and_orientation(
    initial_Transform: SignTransform,
    equirect_image: np.ndarray,
    roadsign_image: np.ndarray,
    bounding_box: AttentionWindow,
    plane_width: float,
    plane_height: float,
    equirect_width: int,
    equirect_height: int
) -> SignTransform:
    """
    Optimizes the position and orientation of the roadsign to minimize the error between the
    projected image and the observed image within the bounding box.
    
    Parameters:
    - initial_params: Initial guess for the roadsign parameters (x, y, z, yaw, pitch, roll).
    - equirect_image: The original equirectangular image (H, W, C).
    - roadsign_image: The 2D image of the roadsign (h, w, C).
    - bounding_box: Tuple (x_min, y_min, x_max, y_max) defining the bounding box.
    - plane_width: The width of the roadsign in world units.
    - plane_height: The height of the roadsign in world units.
    - equirect_width, equirect_height: Dimensions of the equirectangular image.
    
    Returns:
    - optimized_params: The optimized parameters (x, y, z, yaw, pitch, roll).
    """

    equirectangularMapper=EquirectangularMapper(equirect_width,equirect_height)

    # Objective function to minimize
    def objective_function(params):
        # Unpack parameters
        x, y, z, yaw, pitch, roll = params
        
        # Project the roadsign image onto the equirectangular image
        projected_image = equirectangularMapper.map_image_to_equirectangular(
            roadsign_image,
            plane_center=np.array([x, y, z], dtype=np.float64),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            plane_width=plane_width,
            plane_height=plane_height
        )
        
        # Compute the error within the bounding box
        error = compute_error(equirect_image, projected_image, bounding_box)
        
        return error
    
    # Initial parameter guess
    initial_guess = initial_Transform.as_array()
    
    # Run the optimization
    result = minimize(
        objective_function,
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 200, 'disp': True}
    )
    
    optimized_params = result.x
    xc,yc,zc,yaw,pitch, roll = optimized_params
    return SignTransform(xc,yc,zc,yaw,pitch, roll)
