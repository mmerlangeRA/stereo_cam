import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from typing import List, Tuple
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian

def rotation_matrix_from_params(params: List[float]) -> np.ndarray:
    """Construct a rotation matrix from parameters."""
    return SciPyRotation.from_euler('xyz', params, degrees=False).as_matrix()

def triangulate_point(ray1: np.ndarray, ray2: np.ndarray, t: np.ndarray, R_matrix: np.ndarray, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
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

    residual_distance_normalized = np.linalg.norm(point_3d_1 - point_3d_2)
    residual_distance_normalized /= np.linalg.norm(point_3d_1)

    if verbose:
        print("ray1", ray1)
        print("ray2", ray2)
        print(A)
        print(f"Lambda1: {lambda1}, Lambda2: {lambda2}")
        print("b", b)
        predicted_b = A @ lambdas
        print("predicted_b", predicted_b)

    return point_3d_1, lambda2 * ray2, residual_distance_normalized

def get_3d_point_cam1_2_from_coordinates(keypoints_cam1: Tuple[float, float], keypoints_cam2: Tuple[float, float], image_width: int, image_height: int, R: np.ndarray, t: np.ndarray, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get 3D points from camera coordinates."""
    point_image1 = np.array(keypoints_cam1)
    point_image2 = np.array(keypoints_cam2)

    theta1, phi1 = pixel_to_spherical(image_width, image_height, point_image1[0], point_image1[1])
    theta2, phi2 = pixel_to_spherical(image_width, image_height, point_image2[0], point_image2[1])

    ray1 = spherical_to_cartesian(theta1, phi1)
    ray2 = spherical_to_cartesian(theta2, phi2)
    point_3d_cam1, point_3d_cam2, residual_distance_normalized = triangulate_point(ray1, ray2, t, R, verbose)
    return point_3d_cam1, point_3d_cam2, residual_distance_normalized
