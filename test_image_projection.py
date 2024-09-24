import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from python_server.utils.path_helper import get_static_path
from src.road_detection.common import AttentionWindow
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian
from src.matching.match_simple_pytorch import VGGMatcher
from src.utils.equirectangular.equirectangular_mapper import EquirectangularMapper
from src.utils.equirectangular.minimize_projection import  optimize_roadsign_position_and_orientation


equirectangular_path = r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_G_2_EAC.png"
equirect_image =cv2.imread(equirectangular_path)
equirect_height,equirect_width= equirect_image.shape[:2]
equirectangularMapper=EquirectangularMapper(equirect_width,equirect_height)

signWindow= AttentionWindow(3171,3251,1295,1369,False)

#find the sign in the image
matcher = VGGMatcher(reference_folder_path=r'C:\Users\mmerl\projects\stereo_cam\panneaux')
#matcher.crop_all_images()

query_image = signWindow.crop_image(equirect_image)
id=matcher.find_matching(query_image)
img_path = matcher.get_image_path(id)
sign_img = cv2.imread(img_path)  
print(img_path)

#estimate the plane parameters from the sign window
sign_center =signWindow.center 
sign_width = signWindow.width
sign_top_left = signWindow.top_left
sign_bottom_left = signWindow.bottom_left

real_sign_width=0.7

theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_top_left[0],sign_top_left[1])
x_tl,y_tl,z_tl = spherical_to_cartesian(theta,phi)

theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_bottom_left[0],sign_bottom_left[1])
x_bl,y_bl,z_bl = spherical_to_cartesian(theta,phi)

theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_center[0],sign_center[1])
x_c,y_c,z_c = spherical_to_cartesian(theta,phi)

#assume same z and size is real_sign_width
x_tl*=np.sign(z_tl)
y_tl*=np.sign(z_tl)
z_tl =abs(z_tl)

x_bl*=np.sign(z_bl)
y_bl*=np.sign(z_bl)
z_bl =abs(z_bl)

x_c*=np.sign(z_c)
y_c*=np.sign(z_c)
z_c =abs(z_c)

x_bl *= z_tl/z_bl
y_bl *= z_tl/z_bl
z_bl = z_tl

x_c *= z_tl/z_c
y_c *= z_tl/z_c
z_c = z_tl

ratio =real_sign_width /(y_bl-y_tl)

x_tl*=ratio
y_tl*=ratio
z_tl*=ratio

x_bl*=ratio
y_bl*=ratio
z_bl*= ratio

x_c*=ratio
y_c*=ratio
z_c*= ratio

h,w= sign_img.shape[:2]
y_c=y_bl

# Define plane parameters
plane_center = np.array([x_c, y_c, z_c])  # Position the plane 1 unit in front of the camera
plane_normal = np.array([0, 0, -1])   # Facing the camera
plane_up_vector = np.array([0, -1, 0])  # Up direction of the plane
plane_width = real_sign_width  # Adjust as needed
plane_height = plane_width*h/w


# Map the 2D image onto the equirectangular image
equirect_sign_image = equirectangularMapper.map_image_to_equirectangular(
    sign_img, plane_center, plane_normal, plane_up_vector, plane_width, plane_height)

blended = cv2.addWeighted(equirect_image, 0.5, equirect_sign_image, 0.5, 0)
# Save or display the result

cv2.imwrite(get_static_path('equirect_image.png'), blended)
#cv2.imshow('Equirectangular Image', blended)

# Initial estimate of the position
x_estimate, y_estimate, z_estimate = x_c, y_c, z_c  # Example values

plane_normal = plane_normal / np.linalg.norm(plane_normal)
plane_up_vector = plane_up_vector / np.linalg.norm(plane_up_vector)
plane_right_vector = np.cross(plane_up_vector, plane_normal)
plane_right_vector /= np.linalg.norm(plane_right_vector)
plane_up_vector = np.cross(plane_normal, plane_right_vector)
plane_up_vector /= np.linalg.norm(plane_up_vector)
rotation_matrix = np.column_stack((plane_right_vector, plane_up_vector, plane_normal))
rotation = R.from_matrix(rotation_matrix)
yaw_estimate, pitch_estimate, roll_estimate = rotation.as_euler('ZYX', degrees=False)

#yaw_estimate, pitch_estimate, roll_estimate = 0.0, 0.0, 0.0  # In radians
initial_params = np.array([
    x_estimate,
    y_estimate,
    z_estimate,
    yaw_estimate,
    pitch_estimate,
    roll_estimate
])

print("optimizing from",initial_params)
# Optimize position and orientation
optimized_params = optimize_roadsign_position_and_orientation(
    initial_params,
    equirect_image,
    sign_img,
    signWindow,
    plane_width,
    plane_height,
    equirect_width,
    equirect_height
)
#127994870.0
# Extract optimized position and orientation
optimized_x, optimized_y, optimized_z, optimized_yaw, optimized_pitch, optimized_roll = optimized_params

print(f"Optimized Position: x={optimized_x}, y={optimized_y}, z={optimized_z}")
print(f"Optimized Orientation (radians): yaw={optimized_yaw}, pitch={optimized_pitch}, roll={optimized_roll}")

# Project the roadsign image with optimized parameters
optimized_plane_center = np.array([optimized_x, optimized_y, optimized_z], dtype=np.float64)

# Recompute rotation matrix and orientation vectors
rotation = R.from_euler('zyx', [optimized_roll, optimized_pitch, optimized_yaw], degrees=False)
rotation_matrix = rotation.as_matrix()
optimized_plane_normal = rotation_matrix @ np.array([0, 0, -1], dtype=np.float64)
optimized_plane_up_vector = rotation_matrix @ np.array([0, 1, 0], dtype=np.float64)

# Map the image
projected_image = equirectangularMapper.map_image_to_equirectangular(
    sign_img,
    plane_center=optimized_plane_center,
    plane_normal=optimized_plane_normal,
    plane_up_vector=optimized_plane_up_vector,
    plane_width=plane_width,
    plane_height=plane_height
)


# Overlay the projected image onto the equirectangular image
blended_image = cv2.addWeighted(equirect_image, 0.7, projected_image, 0.3, 0)


# Define the top and bottom points of the sign in the plane's coordinate system
plane_up_vector_rotated = rotation_matrix @ optimized_plane_up_vector

top_sign = optimized_plane_center - plane_height * 0.5 * plane_up_vector_rotated
bottom_sign = optimized_plane_center + plane_height * 0.5 * plane_up_vector_rotated

# Convert points_3D to a NumPy array
points_3D = np.array([top_sign, bottom_sign])  # Shape: (2, 3)

us,vs = equirectangularMapper.map_3d_points_to_equirectangular(points_3D)

for u,v in zip(us,vs):
    cv2.circle(blended_image, (int(u), int(v)), 5, (0, 0, 255), -1)
    
cv2.imwrite(get_static_path('optimized.png'), blended_image)


