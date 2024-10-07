from typing import Tuple
import numpy as np

from src.utils.TransformClass import Transform

def get_plane_P0_and_N_from_transform(plane_transform:Transform)->Tuple[np.array,np.array]:
    P0 = np.array(plane_transform.translationVector)
    # Compute the plane's normal vector N
    R = plane_transform.rotationMatrix
    N0 = np.array([0, 1, 0])  # Initial normal vector
    N = R @ N0
    return P0, N

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

def compute_plane_coordinates(P: np.ndarray, plane_transform:Transform) -> np.ndarray:
    """
    Computes the coordinates of points P in the plane's local coordinate system.
    
    Parameters:
    - P (np.ndarray): Points in 3D space, shape (..., 3).
    - plane_transform: transform of the plane.
    
    Returns:
    - np.ndarray: Coordinates of points in the plane's referential, shape (..., 2).
    """
    # Ensure inputs are NumPy arrays
    P = np.asarray(P)
    P0,N = get_plane_P0_and_N_from_transform(plane_transform)
    
    # Compute vector V from P0 to P
    V = P - P0
    
    # Compute the plane's basis vectors u and v
    # Find a vector not parallel to N for cross product
    # We can use the global X-axis unless N is parallel to it
    reference_vector = np.array([1, 0, 0])
    N_norm = N / np.linalg.norm(N, axis=-1, keepdims=True)
    
    # If N is parallel to reference_vector, use Y-axis instead
    parallel = np.isclose(np.abs(np.einsum('...i,...i', N_norm, reference_vector)), 1.0)
    if np.any(parallel):
        reference_vector = np.array([0, 1, 0])
    
    # Compute u: orthogonal to N and in the plane
    u = np.cross(N_norm, reference_vector)
    u_norm = u / np.linalg.norm(u, axis=-1, keepdims=True)
    
    # Compute v: orthogonal to both N and u (lies in the plane)
    v = np.cross(N_norm, u_norm)
    v_norm = v / np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Project V onto u and v to get coordinates in the plane
    u_coord = np.einsum('...i,...i', V, u_norm)
    v_coord = np.einsum('...i,...i', V, v_norm)
    
    # Stack coordinates to form (u, v) pairs
    plane_coords = np.stack((u_coord, v_coord), axis=-1)
    return plane_coords