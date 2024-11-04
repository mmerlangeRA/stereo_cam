import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Union
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian


def compute_single_closest_point(v1:np.array, v2:np.array, t:np.array)-> Tuple[np.array, np.array, float]:
    """
    Computes the closest points between two lines defined by single vectors.

    Parameters:
    - v1: Direction vector of Line 1 (shape: (3,)).
    - v2: Direction vector of Line 2 (shape: (3,)).
    - t: Point through which Line 2 passes (shape: (3,)).

    Returns:
    - P1: Point on Line 1 closest to Line 2 (shape: (3,)).
    - P2: Point on Line 2 closest to Line 1 (shape: (3,)).
    - distance: Minimal distance between the lines (scalar).
    """
    # Compute dot products
    A = np.dot(v1, v1)
    B = np.dot(v1, v2)
    D = np.dot(v2, v2)
    C = np.dot(v1, t)
    E = np.dot(v2, t)

    # Set up the system of equations
    coeff_matrix = np.array([[A, -B],
                             [B, -D]])
    rhs = np.array([C, E])

    # Solve for lambda1 and lambda2
    try:
        lambdas = np.linalg.solve(coeff_matrix, rhs)
        lambda1, lambda2 = lambdas
    except np.linalg.LinAlgError as e:
        raise ValueError("Lines are parallel or coincident, no unique minimal distance.") from e

    # Compute the closest points
    P1 = lambda1 * v1
    P2 = t + lambda2 * v2

    # Compute the minimal distance
    distance = np.linalg.norm(P1 - P2)

    return P1, P2, distance

def compute_closest_points_array(v1_array:np.ndarray, v2_array:np.ndarray, t:np.array)-> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Computes the closest points between pairs of lines defined by arrays of vectors.

    Parameters:
    - v1_array: Array of direction vectors for Line 1 (shape: (N, 3)).
    - v2_array: Array of direction vectors for Line 2 (shape: (N, 3)).
    - t_array: Array of points through which Line 2 passes (shape: (N, 3)).

    Returns:
    - P1_array: Points on Line 1 closest to Line 2 (shape: (N, 3)).
    - P2_array: Points on Line 2 closest to Line 1 (shape: (N, 3)).
    - distances: Minimal distances between the lines (shape: (N,)).
    """
    N = v1_array.shape[0]

    # Compute dot products
    A = np.einsum('ij,ij->i', v1_array, v1_array)  # (N,)
    B = np.einsum('ij,ij->i', v1_array, v2_array)  # (N,)
    D = np.einsum('ij,ij->i', v2_array, v2_array)  # (N,)
    C = np.einsum('ij,ij->i', v1_array, t)   # (N,)
    E = np.einsum('ij,ij->i', v2_array, t)   # (N,)

    # Set up the system of equations for each pair
    coeff_matrices = np.stack([
        np.array([[A[i], -B[i]],
                  [B[i], -D[i]]]) for i in range(N)
    ])  # Shape: (N, 2, 2)

    rhs = np.stack([C, E], axis=-1)  # Shape: (N, 2)

    # Solve for lambda1 and lambda2 for each pair
    try:
        lambdas = np.linalg.solve(coeff_matrices, rhs)  # Shape: (N, 2)
        lambda1 = lambdas[:, 0]
        lambda2 = lambdas[:, 1]
    except np.linalg.LinAlgError as e:
        raise ValueError("One or more pairs of lines are parallel or coincident, no unique minimal distance.") from e

    # Compute the closest points
    P1_array = v1_array * lambda1[:, np.newaxis]  # (N, 3)
    P2_array = t + v2_array * lambda2[:, np.newaxis]  # (N, 3)

    # Compute the minimal distances
    distances = np.linalg.norm(P1_array - P2_array, axis=1)  # (N,)

    return P1_array, P2_array, distances

def triangulate_point_from_rays(v1:Union[np.ndarray, np.array], v2:Union[np.ndarray, np.array], t:np.array, R_matrix:np.ndarray, verbose= False)-> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.array]]:
    """
    Computes the closest points between two lines in 3D space.

    Line 1: Passes through the origin and has direction vector v1.
    Line 2: Passes through point t and has direction vector v2 (after rotation by R_matrix).

    Parameters:
    - v1: Direction vector(s) of Line 1 (shape: (3,) or (N, 3)).
    - v2: Direction vector(s) of Line 2 before rotation (shape: (3,) or (N, 3)).
    - t: Point(s) through which Line 2 passes (shape: (3,) or (N, 3)).
    - R_matrix: Rotation matrix to be applied to v2 (shape: (3, 3)).

    Returns:
    - P1: Point(s) on Line 1 closest to Line 2 (shape: (3,) or (N, 3)).
    - P2: Point(s) on Line 2 closest to Line 1 (shape: (3,) or (N, 3)).
    - distances: Minimal distance(s) between the lines (scalar or array of shape (N,)).
    """

    # Ensure inputs are numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    t = np.asarray(t)
    R_matrix = np.asarray(R_matrix)

    # Rotate v2 by R_matrix
    if v2.ndim == 1:
        # Single vector
        v2_rotated = R_matrix @ v2
    elif v2.ndim == 2:
        # Array of vectors
        v2_rotated = (R_matrix @ v2.T).T
    else:
        raise ValueError("v2 must be a vector or an array of vectors")

    # Handle different input dimensions
    if v1.ndim == 1:
        # v1 is a single vector
        if v2_rotated.ndim == 1:
            return compute_single_closest_point(v1, v2_rotated, t)
        else:
            raise ValueError("v2_rotated must be same size as v1")
    elif v1.ndim == 2:
        # v1 is an array of vectors
        if v2_rotated.ndim == 2:
            t_array = np.tile(t, (v1.shape[0], 1))
            return compute_closest_points_array(v1, v2_rotated, t_array)
        else:
            raise ValueError("t must be a single point or an array of points matching v1 and v2")
    else:
        raise ValueError("v1 must be a vector or an array of vectors")


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

