import os
import numpy as np
import cv2 as cv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SciPyRotation

def rotation_matrix_from_params(params):
    """Construct a rotation matrix from parameters."""
    return SciPyRotation.from_euler('xyz', params).as_matrix()

def objective_function(params, ray1, ray2, t, known_distance):
    """Objective function for optimization."""
    R_matrix = rotation_matrix_from_params(params)
    point_3d = triangulate_point(ray1, ray2, t, R_matrix)
    estimated_distance = np.linalg.norm(point_3d)
    return (estimated_distance - known_distance) ** 2

def pixel_to_spherical(image_width, image_height, pixel_x, pixel_y):
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi
    phi = (pixel_y / image_height) * np.pi
    return theta, phi

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to 3D cartesian coordinates."""
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])

def triangulate_point(ray1, ray2, t, R):
    """Triangulate a 3D point from two rays and the relative camera transformation."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    A[:3, :3] = np.column_stack((ray1, -R @ ray2, np.cross(ray1, R @ ray2)))
    b[:3] = t
    
    # Solve the system using SVD
    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return X

# Main execution
img_folder = os.path.join(os.getcwd(), 'Photos', 'P1')
left_image_path = 'D_P1_CAM_G_2_EAC.png'
right_image_path = 'D_P1_CAM_D_2_EAC.png'

left_image = cv.imread(os.path.join(img_folder, left_image_path))
right_image = cv.imread(os.path.join(img_folder, right_image_path))

# Image dimensions
image_height,image_width = left_image.shape[:2]

print(image_width, image_height)

#center letter O
point_image1 = np.array([3217, 1331])  # Keypoints in image from Camera 1 (left)
point_image2 = np.array([3055, 1340])  # Keypoints in image from Camera 2 (right)

#top right stop (red)
point_image1 = np.array([3223, 1300])  # Keypoints in image from Camera 1 (left)
point_image2 = np.array([3062, 1308])  # Keypoints in image from Camera 2 (right)

# Convert pixel coordinates to spherical coordinates
theta1, phi1 = pixel_to_spherical(image_width, image_height, point_image1[0], point_image1[1])
theta2, phi2 = pixel_to_spherical(image_width, image_height, point_image2[0], point_image2[1])

# Convert spherical coordinates to 3D cartesian coordinates
ray1 = spherical_to_cartesian(theta1, phi1)
ray2 = spherical_to_cartesian(theta2, phi2)

# Relative position and rotation (example values)
t = np.array([1.12, 0, 0])  # Example translation: 1.12 meters along the x-axis
R = np.eye(3)  # Assuming no rotation for simplicity

# Triangulate the 3D point
point_3d = triangulate_point(ray1, ray2, t, R)

# Output the 3D coordinates
print("3D coordinates of the point:", point_3d)
print(np.linalg.norm(point_3d))

known_distance = 9.05

# Initial guess for rotation parameters (no rotation)
initial_params = np.zeros(3)

# Optimize for the rotation parameters
result = minimize(objective_function, initial_params, args=(ray1, ray2, t, known_distance), method='BFGS')

# Extract the optimized rotation matrix
optimized_R = rotation_matrix_from_params(result.x)

# Triangulate the 3D point using the optimized rotation matrix
point_3d = triangulate_point(ray1, ray2, t, optimized_R)

print("Optimized rotation matrix R:\n", optimized_R)
print("3D coordinates of the point:", point_3d)
print(np.linalg.norm(point_3d))

# Calculate the position of the 3D point relative to the second camera
point_3d_cam2 = optimized_R @ (point_3d - t)  # Rotate and translate the point to the second camera's coordinate system
distance_cam2_to_point = np.linalg.norm(point_3d_cam2)

# Output the distance
print("Distance from the second camera to the 3D point:", distance_cam2_to_point)

# Extract Euler angles (in degrees) from the optimized rotation matrix
euler_angles_rad = SciPyRotation.from_matrix(optimized_R).as_euler('xyz')
euler_angles_deg = np.degrees(euler_angles_rad)

# Output the Euler angles in degrees
print("Rotation angles (degrees):\n", euler_angles_deg)

