from typing import Tuple
import numpy as np


def pixel_to_spherical(image_width: int, image_height: int, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi - np.pi # longitude
    phi = (pixel_y / image_height) * np.pi - np.pi / 2 # latitude
    return theta, phi

def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to 3D cartesian coordinates."""
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return np.array([x, y, z])