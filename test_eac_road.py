# from bootstrap import set_paths
# set_paths()
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EACRoadDetector
from src.road_detection.common import AttentionWindow
from src.utils.curve_fitting import find_best_2_polynomial_curves
from src.utils.coordinate_transforms import equirect_to_road_plane_points2D
from src.utils.path_utils import get_ouput_path
from src.utils.TransformClass import Transform
from scipy.optimize import least_squares


def display_road_image(left_img_contour_left,left_img_contour_right,camHeight,optimized_rotation_vector):
    road_points_left, road_points_right = compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,optimized_rotation_vector)
    road_points_2D = np.concatenate((road_points_left, road_points_right), axis=0)

    road_points_x = road_points_2D[:,0]
    road_points_z = road_points_2D[:,1]

    minX= np.min(road_points_x)
    maxX= np.max(road_points_x)
    minZ= np.min(road_points_z)
    maxZ= np.max(road_points_z)

    width = math.ceil(maxX-minX)
    height = math.ceil(maxZ-minZ)

    road_points_2D[:,0] = road_points_x - minX
    road_points_2D[:,1] = road_points_z - minZ

    display_coeff=10
    road_image = np.zeros((height*display_coeff, width*display_coeff, 3), dtype=np.uint8)

    for p in road_points_2D:
        x = int(p[0]*display_coeff)
        y = int(p[1]*display_coeff)
        road_image[y, x] = [255, 0, 0]
    cv2.imwrite(get_ouput_path('road_image.png'), road_image)


def compute_left_right_road_points2D(img_contour_left,img_contour_right,camHeight,road_rvec):
    left_contour_x = img_contour_left[:, 0]
    left_contour_y = img_contour_left[:, 1]
    road_points_left = equirect_to_road_plane_points2D(imgWidth=width, imgHeight=height, 
                                         road_rvec=road_rvec,camHeight=camHeight,
                                         contour_x=left_contour_x, contour_y=left_contour_y)
    

    right_contour_x = img_contour_right[:, 0]
    right_contour_y = img_contour_right[:, 1]
    road_points_right = equirect_to_road_plane_points2D(imgWidth=width, imgHeight=height,
                                          road_rvec=road_rvec,camHeight=camHeight,
                                          contour_x=right_contour_x, contour_y=right_contour_y)
    
    road_points_left= road_points_left[road_points_left[:, 1].argsort()]
    road_points_right= road_points_right[road_points_right[:, 1].argsort()]

    return road_points_left, road_points_right

def compute_slopes_difference(cam_rotation_vector, road_points_left, road_points_right, camHeight):
    road_points_left, road_points_right = compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,cam_rotation_vector)
    nb_left = road_points_left.shape[0]
    nb_right = road_points_right.shape[0]

    xLeftpoints = road_points_left[:, 0]
    yLeftpoints = road_points_left[:, 1]
    xRightpoints = road_points_right[:, 0]
    yRightpoints = road_points_right[:, 1]

    xleft_1 = xLeftpoints[0]
    yleft_1 = yLeftpoints[0]
    xleft_2 = xLeftpoints[nb_left-1]
    yleft_2 = yLeftpoints[nb_left-1]

    xright_1 = xRightpoints[0]
    yright_1 = yRightpoints[0]
    xright_2 = xRightpoints[nb_right-1]
    yright_2 = yRightpoints[nb_right-1]

    slopes_left = (xleft_2 - xleft_1)/(yleft_2 - yleft_1) 
    slopes_right = (xright_2 - xright_1)/(yright_2 - yright_1) 
    
    slopes_diff = np.abs(slopes_left - slopes_right)
    #print(slopes_left, slopes_right,slopes_diff)

    return slopes_diff

# Function to optimize cam_rotation_vector
def optimize_cam_rotation(left_img_contour_left, left_img_contour_right, camHeight, initial_cam_rotation_vector,bounds):
    # Optimize the cam_rotation_vector using least_squares
    result = least_squares(
        fun=lambda cam_rotation_vector: compute_slopes_difference(cam_rotation_vector, left_img_contour_left, left_img_contour_right, camHeight),
        x0=initial_cam_rotation_vector,
        bounds=bounds,  # Add bounds here
        method='trf'  # 'trf' is the method that supports bounds
    )
    # Return the optimized cam_rotation_vector
    return result.x


    height, width = img.shape[:2]

img_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_0_EAC.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

height, width = img.shape[:2]

window_left=0.4
window_right = 0.6
window_top = 0.53
window_bottom = 0.6
debug = True
camHeight = 1.65
degree= 1

# Attention window for segementation and road detection
limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

cv2.imwrite(get_ouput_path('road_window.png'), window.crop_image(img))
#processing

roadSegmentator = SegFormerRoadSegmentator(kernel_width=10, use_1024=False, debug=debug)
roadDetector = EACRoadDetector(roadSegmentator=roadSegmentator,window=window,camHeight=camHeight, degree=degree, debug=debug)
#average_width, first_poly_model, second_poly_model, contour_x, contour_y = roadDetector.compute_road_width(img)

#get contours
left_img_contour_left,left_img_contour_right = roadDetector.get_left_right_contours(img)

# project on road plane and optimize cam rotation
cam0Transform = Transform(0.,0,0,0.,0.,0.)

initial_cam_rotation_vector= np.array(cam0Transform.rotationVector)
lower_bounds = np.array([-np.pi/4,-np.pi/10000, -np.pi/10000])  # Lower bounds for the rotation vector
upper_bounds = np.array([np.pi/4,np.pi/10000,  np.pi/10000])  # Upper bounds for the rotation vector
bounds = (lower_bounds, upper_bounds)

# Optimize the camera rotation vector
optimized_rotation_vector = optimize_cam_rotation(left_img_contour_left, left_img_contour_right, camHeight, initial_cam_rotation_vector, bounds)

print("optimized_rotation_vector",optimized_rotation_vector)
road_rvec=optimized_rotation_vector

#debug and display
display_road_image(left_img_contour_left,left_img_contour_right,camHeight,optimized_rotation_vector)


road_points_left, road_points_right=compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,road_rvec)
x_left_mean = np.mean(road_points_left[:,0])
x_right_mean = np.mean(road_points_right[:, 0])
print("x_left_mean", x_left_mean)
print("x_right_mean", x_right_mean)
width = x_right_mean - x_left_mean
print("width", width)

road_points_left, road_points_right=compute_left_right_road_points2D(left_img_contour_left,left_img_contour_right,camHeight,road_rvec)
x_left_mean = np.mean(road_points_left[:,0])
x_right_mean = np.mean(road_points_right[:, 0])
print("x_left_mean", x_left_mean)
print("x_right_mean", x_right_mean)
width = x_right_mean - x_left_mean
print("width", width)
