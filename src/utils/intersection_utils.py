from typing import Tuple
import numpy as np

from src.utils.TransformClass import Transform



def compute_intersection_rays_plane(rays: np.ndarray, P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Computes the intersection point(s) of ray(s) with a plane(s).
    
    Parameters:
    - D (np.ndarray): Direction vector(s) of the ray(s), shape (..., 3).
    - P0 (np.ndarray): Point(s) on the plane(s), shape (..., 3).
    - N (np.ndarray): Normal vector(s) of the plane(s), shape (..., 3).
    
    Returns:
    - np.ndarray: Intersection point(s), shape (..., 3).
    
    Raises:
    - ValueError: If the ray is parallel to the plane or the intersection is behind the ray origin.
    """
    # Ensure inputs are NumPy arrays
    rays = np.asarray(rays)
    P0 = np.asarray(P0)
    N = np.asarray(N)
    
    # Normalize the direction vectors D
    D_norm = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    
    # Compute the dot products
    numerator = np.einsum('...i,...i', N, P0)
    denominator = np.einsum('...i,...i', N, D_norm)
    
    # Check for rays parallel to the plane
    parallel = np.isclose(denominator, 0)
    if np.any(parallel):
        raise ValueError("The ray is parallel to the plane.")
    
    # Compute t for each ray-plane intersection
    t = numerator / denominator
    
    # Check for intersections behind the ray origin
    behind = t < 0
    if np.any(behind):
        raise ValueError("The intersection point is behind the ray origin.")
    
    # Compute the intersection points
    intersection_points = D_norm * t[..., np.newaxis]
    return intersection_points

