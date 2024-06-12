import os
import numpy as np
import cv2 as cv

# Compute intrinsic parameters for the 360-degree camera
def compute_360_intrinsics(image_width, image_height):
    cx = image_width / 2
    cy = image_height / 2
    fx = image_width / (2 * np.pi)
    fy = image_height / np.pi
    f=211.
    fx = f
    fy = f

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    return K

# Convert equirectangular coordinates to spherical coordinates
def equirectangular_to_spherical(x, y, width, height):
    theta = (x / width) * 2.0 * np.pi - np.pi  # Longitude (-π to π)
    phi = (y / height) * np.pi - (np.pi / 2)  # Latitude (-π/2 to π/2)
    return theta, phi

# Convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(theta, phi):
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.array([x, y, z])

# Scale Cartesian direction vectors to intersect the focal plane
def cartesian_to_focal_plane_coordinates(direction, focal_length,width,height):
    return direction * focal_length / direction[2]

def cartesian_to_focal_plane_coordinates_2(direction, focal_length,width,height):
    plane_position= direction * focal_length / direction[2]
    plane_position[0]+=width/2
    plane_position[1]+=height/2
    return plane_position

# Transform 3D points from Camera 1's reference frame to Camera 2's reference frame
def transform_to_cam2(points_3d_cam1, R, T):
    points_3d_cam2 = (R @ points_3d_cam1.T + T.reshape(-1, 1)).T
    return points_3d_cam2

# Triangulate 3D points from keypoints in 360-degree images
def triangulate_360(keypoints_cam1, keypoints_cam2, K1, K2, R, T, width, height):
    directions_cam1 = []
    directions_cam2 = []

    for kp1, kp2 in zip(keypoints_cam1, keypoints_cam2):
        theta1, phi1 = equirectangular_to_spherical(kp1[0], kp1[1], width, height)
        theta2, phi2 = equirectangular_to_spherical(kp2[0], kp2[1], width, height)

        dir1 = spherical_to_cartesian(theta1, phi1)
        dir2 = spherical_to_cartesian(theta2, phi2)

        focal_length = K1[0, 0]
        dir1_scaled = cartesian_to_focal_plane_coordinates_2(dir1, focal_length,width,height)
        dir2_scaled = cartesian_to_focal_plane_coordinates_2(dir2, focal_length,width,height)

        directions_cam1.append(dir1_scaled)
        directions_cam2.append(dir2_scaled)

    directions_cam1 = np.array(directions_cam1).T
    directions_cam2 = np.array(directions_cam2).T

    # Convert to homogeneous coordinates
    directions_cam1_hom = np.vstack((directions_cam1[:2], np.ones((1, directions_cam1.shape[1]))))
    directions_cam2_hom = np.vstack((directions_cam2[:2], np.ones((1, directions_cam2.shape[1]))))
    print("vectors")
    print(directions_cam1_hom[:2])
    print(directions_cam2_hom[:2])
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(-1, 1)))

    points_4d_hom = cv.triangulatePoints(P1, P2, directions_cam1_hom[:2], directions_cam2_hom[:2])
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    return points_3d.T

# Main execution
img_folder = os.path.join(os.getcwd(), 'Photos', 'P1')
left_image_path = 'D_P1_CAM_G_2_EAC.png'
right_image_path = 'D_P1_CAM_D_2_EAC.png'

left_image = cv.imread(os.path.join(img_folder, left_image_path))
right_image = cv.imread(os.path.join(img_folder, right_image_path))

height, width = left_image.shape[:2]
K1 = compute_360_intrinsics(width, height)
K2 = compute_360_intrinsics(width, height)


# Example extrinsic parameters (assume no rotation and translation along the x-axis)
R = np.identity(3)
T = np.array([1.12, 0, 0])  # Translation vector in meters

# Example keypoints in the equirectangular images
keypoints_cam1 = np.array([[3217, 1334]])  # Keypoints in image from Camera 1
keypoints_cam2 = np.array([[3056, 1341]])  # Keypoints in image from Camera 2

# Triangulate 3D points in Camera 1's reference frame
points_3d_cam1 = triangulate_360(keypoints_cam1, keypoints_cam2, K1, K2, R, T, width, height)

# Transform 3D points to Camera 2's reference frame
points_3d_cam2 = transform_to_cam2(points_3d_cam1, R, T)

# Print distances to verify
for p in points_3d_cam1:
    distance = np.linalg.norm(p)
    print(f'Distance in Cam 1: {distance}')

for p in points_3d_cam2:
    distance = np.linalg.norm(p)
    print(f'Distance in Cam 2: {distance}')

print("3D positions of the keypoints in Camera 1's reference frame (in meters):", points_3d_cam1)
print("3D positions of the keypoints in Camera 2's reference frame (in meters):", points_3d_cam2)
