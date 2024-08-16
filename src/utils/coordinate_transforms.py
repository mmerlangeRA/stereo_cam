from typing import Tuple
import numpy as np
import cv2


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



def get_transformation_matrix(rvec:np.array, tvec:np.array)->np.array:
    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()
    return transformation_matrix


def get_rvec_tvec_from_transformation_matrix(transformation_matrix:np.array)->Tuple[np.array,np.array]:
    R = transformation_matrix[:3, :3]
    tvec = transformation_matrix[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec

def invert_rvec_tvec(rvec:np.array, tvec:np.array)->Tuple[np.array,np.array]:

    transformation_matrix = get_transformation_matrix(rvec, tvec)
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)
    rvec_inv,tvec_inv = get_rvec_tvec_from_transformation_matrix(inv_transformation_matrix)
    return rvec_inv, tvec_inv

def get_identity_extrinsic_matrix()->np.array:
    return np.eye(3, 4)

def get_extrinsic_matrix_from_rvec_tvec(rvec:np.array, tvec:np.array)->np.array:
    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack((R, tvec))
    return RT


if __name__ == "__main__":
    # Example usage
    rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    tvec = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    rvec_inv, tvec_inv = invert_rvec_tvec(rvec, tvec)

    print("Inverted rvec:\n", rvec_inv)
    print("Inverted tvec:\n", tvec_inv)

    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()

    R_inv, _ = cv2.Rodrigues(rvec_inv)
    transformation_matrix_inverse =np.eye(4)
    transformation_matrix_inverse[:3, :3] = R_inv
    transformation_matrix_inverse[:3, 3] = tvec_inv.flatten()

    m = transformation_matrix.dot(transformation_matrix_inverse)
    print(m)

    transformation_matrix = get_transformation_matrix(rvec, tvec)
    transformation_matrix_inverse = get_transformation_matrix(rvec_inv, tvec_inv)
    m = transformation_matrix.dot(transformation_matrix_inverse)
    print(m)

