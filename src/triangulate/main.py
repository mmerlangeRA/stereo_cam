import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Union
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian

def rotation_matrix_from_params(params: List[float]) -> np.ndarray:
    """Construct a rotation matrix from parameters."""
    return R.from_euler('xyz', params, degrees=False).as_matrix()
    return R.from_euler('yxz',params, degrees=False).as_matrix()

def triangulate_point_old(ray1: np.ndarray, ray2: np.ndarray, t: np.ndarray, R_matrix: np.ndarray, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Triangulate a 3D point from two rays and the relative camera transformation."""
    # Create the matrix A for the linear system
    A = np.zeros((3, 3))
    A[:, 0] = ray1
    A[:, 1] = -R_matrix @ ray2
    A[:, 2] = np.cross(ray1, R_matrix @ ray2)
    b = t

    # Solve the system using least squares
    lambdas, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    lambda1, lambda2, _ = lambdas

    # Calculate the 3D point using lambda1
    point_3d_1 = lambda1 * ray1
    point_3d_2 = R_matrix @ (lambda2 * ray2) + t
    #point_3d_2 = lambda2 * ray2

    residual_distance_in_m = np.linalg.norm(point_3d_1 - point_3d_2)
    #max_acceptable_distance=20.

    #min_d = min(np.linalg.norm(point_3d_1), np.linalg.norm(point_3d_2))
    #residual_distance_normalized /= min_d
    '''
    if min_d>max_acceptable_distance:
        r= max_acceptable_distance/min_d
        ratio = r*min_d+(1.-r)*max_acceptable_distance
        residual_distance_normalized/=ratio
    else:
        residual_distance_normalized/=min_d
    '''
    #residual_distance_normalized/=min_d

    #if min_d>max_acceptable_distance:
    #    residual_distance_normalized*=10.*min_d/max_acceptable_distance

    if verbose:
        print("ray1", ray1)
        print("ray2", ray2)
        print(A)
        print(f"Lambda1: {lambda1}, Lambda2: {lambda2}")
        print("b", b)
        predicted_b = A @ lambdas
        print("predicted_b", predicted_b)

    return point_3d_1, lambda2*ray2, residual_distance_in_m


def triangulate_point_from_rays(
    ray1: Union[np.ndarray, np.array], 
    ray2: Union[np.ndarray, np.array], 
    t: np.ndarray, 
    R_matrix: np.ndarray, 
    verbose: bool=False
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulate a 3D point (or points) from two rays and the relative camera transformation.
    
    This function handles both single rays and arrays of rays.

    Parameters:
    - ray1: A 3D vector or an array of 3D vectors representing the first ray(s) from camera 1.
    - ray2: A 3D vector or an array of 3D vectors representing the second ray(s) from camera 2.
    - t: A translation vector (between cameras).
    - R_matrix: Rotation matrix (relative orientation between cameras).
    - verbose: If True, prints debugging information.

    Returns:
    - point_3d_1: The triangulated 3D point(s) from camera 1's perspective.
    - point_3d_2: The triangulated 3D point(s) from camera 2's perspective.
    - residual_distance_in_m: The distance between triangulated points from both cameras (if single point).
    """
    def solve_system(ray1, ray2, t, R_matrix):
        """Solve for a single point."""
        # Create the matrix A for the linear system
        A = np.zeros((3, 3))
        A[:, 0] = ray1
        A[:, 1] = -R_matrix @ ray2
        A[:, 2] = np.cross(ray1, R_matrix @ ray2)
        b = t

        # Solve the system using least squares
        lambdas, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lambda1, lambda2, _ = lambdas

        # Calculate the 3D point using lambda1 and lambda2
        point_3d_1 = lambda1 * ray1
        point_3d_2 = R_matrix @ (lambda2 * ray2) + t
        residual_distance_in_m = np.linalg.norm(point_3d_1 - point_3d_2)

        return point_3d_1, point_3d_2, residual_distance_in_m

    # Handle scalar inputs
    if ray1.ndim == 1 and ray2.ndim == 1:
        return solve_system(ray1, ray2, t, R_matrix)

    # Handle array inputs
    elif ray1.ndim == 2 and ray2.ndim == 2:
        points_3d_1 = []
        points_3d_2 = []
        residuals = []
        
        for r1, r2 in zip(ray1, ray2):
            p1, p2, residual = solve_system(r1, r2, t, R_matrix)
            points_3d_1.append(p1)
            points_3d_2.append(p2)
            residuals.append(residual)

        return np.array(points_3d_1), np.array(points_3d_2), np.array(residuals)

    else:
        raise ValueError("ray1 and ray2 must either both be 1D arrays (for a single point) or both be 2D arrays (for multiple points).")

def get_3d_point_cam1_2_from_coordinates_old(keypoints_cam1: Tuple[float, float], keypoints_cam2: Tuple[float, float], image_width: int, image_height: int, R: np.ndarray, t: np.ndarray, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get 3D points from camera coordinates."""
    point_image1 = np.array(keypoints_cam1)
    point_image2 = np.array(keypoints_cam2)

    theta1, phi1 = pixel_to_spherical(image_width, image_height, point_image1[0], point_image1[1])
    theta2, phi2 = pixel_to_spherical(image_width, image_height, point_image2[0], point_image2[1])

    ray1 = spherical_to_cartesian(theta1, phi1)
    ray2 = spherical_to_cartesian(theta2, phi2)
    point_3d_cam1, point_3d_cam2, residual_distance_in_m = triangulate_point_from_rays(ray1, ray2, t, R, verbose=verbose)
    return point_3d_cam1, point_3d_cam2, residual_distance_in_m

import numpy as np
from typing import Union, Tuple

def get_3d_point_cam1_2_from_coordinates(
    keypoints_cam1: Union[Tuple[float, float], np.ndarray],
    keypoints_cam2: Union[Tuple[float, float], np.ndarray],
    image_width: int,
    image_height: int,
    R: np.ndarray,
    t: np.ndarray,
    verbose: bool=False
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get 3D points from camera coordinates. Handles both single keypoints and arrays of keypoints.

    Parameters:
    - keypoints_cam1: Tuple of (x, y) coordinates or array of keypoints from camera 1.
    - keypoints_cam2: Tuple of (x, y) coordinates or array of keypoints from camera 2.
    - image_width: Width of the equirectangular image.
    - image_height: Height of the equirectangular image.
    - R: Rotation matrix between the two cameras.
    - t: Translation vector between the two cameras.
    - verbose: If True, prints debugging information.

    Returns:
    - point_3d_cam1: 3D points from the perspective of camera 1.
    - point_3d_cam2: 3D points from the perspective of camera 2.
    - residual_distance_in_m: Residual distance between the triangulated points.
    """
    
    def process_single_pair(kp1, kp2):
        """Helper function to process a single keypoint pair."""
        theta1, phi1 = pixel_to_spherical(image_width, image_height, kp1[0], kp1[1])
        theta2, phi2 = pixel_to_spherical(image_width, image_height, kp2[0], kp2[1])

        ray1 = spherical_to_cartesian(theta1, phi1)
        ray2 = spherical_to_cartesian(theta2, phi2)

        point_3d_cam1, point_3d_cam2, residual_distance_in_m = triangulate_point_from_rays(ray1, ray2, t, R, verbose=verbose)
        return point_3d_cam1, point_3d_cam2, residual_distance_in_m

    # Handle scalar keypoints (single point)
    if isinstance(keypoints_cam1, tuple) and isinstance(keypoints_cam2, tuple):
        return process_single_pair(keypoints_cam1, keypoints_cam2)

    # Handle array keypoints (multiple points)
    elif isinstance(keypoints_cam1, np.ndarray) and isinstance(keypoints_cam2, np.ndarray):
        theta1, phi1 = pixel_to_spherical(image_width, image_height, keypoints_cam1[:,0], keypoints_cam1[:,1])
        theta2, phi2 = pixel_to_spherical(image_width, image_height, keypoints_cam2[:,0], keypoints_cam2[:,1])

        ray1 = spherical_to_cartesian(theta1, phi1)
        ray2 = spherical_to_cartesian(theta2, phi2)

        point_3d_cam1, point_3d_cam2, residual_distance_in_m = triangulate_point_from_rays(ray1, ray2, t, R, verbose=verbose)
        return point_3d_cam1, point_3d_cam2, residual_distance_in_m

    else:
        raise ValueError("keypoints_cam1 and keypoints_cam2 must either both be tuples (for a single point) or both be arrays (for multiple points).")

