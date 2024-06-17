import os
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as SciPyRotation

def rotation_matrix_from_params(params):
    """Construct a rotation matrix from parameters."""
    return SciPyRotation.from_euler('xyz', params,degrees=False).as_matrix()

def pixel_to_spherical(image_width, image_height, pixel_x, pixel_y):
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi
    phi = (pixel_y / image_height) * np.pi
    return theta, phi

def pixel_to_spherical_revised(image_width, image_height, pixel_x, pixel_y):
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi - np.pi #longitude
    phi = (pixel_y / image_height) * np.pi - np.pi / 2 #latitude
    return theta, phi

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to 3D cartesian coordinates."""
    x = np.cos(phi) * np.sin(theta)
    y = -np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return np.array([x, y, z])

def triangulate_point(ray1, ray2, t, R_matrix,verbose=False):
    """Triangulate a 3D point from two rays and the relative camera transformation."""
    # Formulate the system of linear equations
    A = np.vstack((ray1, -R_matrix @ ray2)).T
    b = np.array(t).T
    if verbose:
        print("R_matrix",R_matrix)
        print("triangulate_point")
        print("t",t)
        print("ray1",ray1)
        print("ray2",ray2)
        print("A",A)
        print("b",b)
    
    # Solve for lambda1 and lambda2
    lambdas, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    lambda1, lambda2 = lambdas
    
    # Calculate the 3D point using lambda1
    point_3d_1 = lambda1 * ray1
    point_3d_2 = lambda2 * ray2 + b
    if verbose:
        print("residuals", residuals)
        print("triangulate_point debug")
        print(lambda1,lambda2)
        print(b)
        print("test",A @ lambdas)
        print("point_3d_1", point_3d_1)
        print("point_3d_2", point_3d_2)
    return (point_3d_1+point_3d_2)/2.,residuals

def get_3d_point_cam1_2_from_coordinates(keypoints_cam1, keypoints_cam2,image_width, image_height,R, t):
    point_image1 = np.array(keypoints_cam1) 
    point_image2 = np.array(keypoints_cam2) 
    theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
    theta2, phi2 = pixel_to_spherical_revised(image_width, image_height, point_image2[0], point_image2[1])
    ray1 = spherical_to_cartesian(theta1, phi1)
    ray2 = spherical_to_cartesian(theta2, phi2)
    point_3d_cam1,residuals = triangulate_point(ray1, ray2, t, R)
    point_3d_cam2 = R @ (point_3d_cam1 - t) 
    return point_3d_cam1,point_3d_cam2,residuals

