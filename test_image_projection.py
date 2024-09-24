# Load your 2D image (e.g., using OpenCV)
import cv2
import numpy as np

from src.utils.image_processing import map_image_to_equirectangular
from python_server.utils.path_helper import get_static_path
from src.road_detection.common import AttentionWindow
from src.utils.coordinate_transforms import pixel_to_spherical, spherical_to_cartesian

equirectangular_path = r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_G_2_EAC.png"
sign_img = cv2.imread(r"C:\Users\mmerl\projects\yolo_test\src\matching\panneaux\France_road_sign_AB4.svg_200.png")  

equirect_image =cv2.imread(equirectangular_path)
# Define equirectangular image size
equirect_height,equirect_width= equirect_image.shape[:2]

signWindow= AttentionWindow(3171,3251,1295,1369,False)

#estimate the plane parameters from the sign window
plane_center =signWindow.center 
plane_width = signWindow.width
plane_top_left = signWindow.top_left
plane_bottom_left = signWindow.bottom_left

real_sign_width=0.7

theta,phi = pixel_to_spherical(equirect_width, equirect_height,plane_top_left[0],plane_top_left[1])
x_tl,y_tl,z_tl = spherical_to_cartesian(theta,phi)

theta,phi = pixel_to_spherical(equirect_width, equirect_height,plane_bottom_left[0],plane_bottom_left[1])
x_bl,y_bl,z_bl = spherical_to_cartesian(theta,phi)

theta,phi = pixel_to_spherical(equirect_width, equirect_height,plane_center[0],plane_center[1])
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

cropped_sign = signWindow.crop_image(equirect_image)
cv2.imshow('cropped', cropped_sign)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Define plane parameters
plane_center = np.array([x_c, y_c, z_c])  # Position the plane 1 unit in front of the camera
#plane_center = np.array([0, 0, 1])  # Position the plane 1 unit in front of the camera
plane_normal = np.array([0, 0, -1])   # Facing the camera
plane_up_vector = np.array([0, -1, 0])  # Up direction of the plane
plane_width = real_sign_width  # Adjust as needed
plane_height = plane_width*h/w


# Map the 2D image onto the equirectangular image
equirect_sign_image = map_image_to_equirectangular(
    sign_img, plane_center, plane_normal, plane_up_vector, plane_width, plane_height,
    equirect_width, equirect_height)

blended = cv2.addWeighted(equirect_image, 0.5, equirect_sign_image, 0.5, 0)
# Save or display the result

cv2.imwrite(get_static_path('equirect_image.png'), blended)
#cv2.imshow('Equirectangular Image', blended)

