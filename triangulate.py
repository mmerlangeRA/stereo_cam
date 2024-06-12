import os
import numpy as np
import cv2 as cv

from src.calibrate import get_cube_subs, read_calibration, undistort

img_folder = os.path.join(os.getcwd(), 'Photos')
img_folder= os.path.join(img_folder, 'P1')
left_image_path = 'D_P1_CAM_G_2_EAC.png'
right_image_path = 'D_P1_CAM_D_2_EAC.png'

is_Cube= False
config_index= 2
left_image = cv.imread(os.path.join(img_folder,left_image_path))
right_image = cv.imread(os.path.join(img_folder,right_image_path))
if is_Cube:
    left_image = get_cube_subs(left_image)[config_index]
    right_image = get_cube_subs(right_image)[config_index]
else:
    sub_left = left_image
    sub_right = right_image

left_pos = [1208, 622]
right_pos = [1192, 606]

cv.imwrite("left.png",sub_left)
cv.imwrite("right.png",sub_right)

calibration_path="calibration_matrix_eac.yaml"

mtx, dist,rmse = read_calibration(calibration_path)
undistorted_left = undistort(sub_left, mtx, dist)
undistorted_right = undistort(sub_right, mtx, dist)
cv.imwrite("undistorted_left.png",undistorted_left)
cv.imwrite("undistorted_right.png",undistorted_right)

# Intrinsic parameters of Camera 1
K1 = np.asarray(mtx)

# Intrinsic parameters of Camera 2
K2 = np.asarray(mtx)

# Rotation and translation from Camera 1 to Camera 2
R = np.identity(3)
T = np.array([0,0,1.12])  # 3x1 translation vector

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 projection matrix [K1 | 0]
P2 = K2 @ np.hstack((R, T.reshape(-1, 1)))          # Camera 2 projection matrix [K2 | RT]

# Keypoints in the images
p1 = np.array(left_pos)  # Keypoint in image from Camera 1
p2 = np.array(right_pos)  # Keypoint in image from Camera 2

# Convert keypoints to homogeneous coordinates
p1_h = np.append(p1, 1)
p2_h = np.append(p2, 1)

# Triangulate the 3D point
points_4d_hom = cv.triangulatePoints(P1, P2, p1_h[:2].reshape(2, 1), p2_h[:2].reshape(2, 1))

# Convert from homogeneous coordinates to 3D
points_3d = points_4d_hom[:3] / points_4d_hom[3]

distance = np.linalg.norm(points_3d)


print("3D position of the keypoint:", points_3d.flatten())

print(f'distance is {distance}')
