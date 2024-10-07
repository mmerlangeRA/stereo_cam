# from bootstrap import set_paths
# set_paths()
import cv2
import numpy as np

from src.road_detection.RoadSegmentator import SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EquirectMonoRoadDetector
from src.road_detection.common import AttentionWindow

from src.utils.curve_fitting import Road_line_params, find_best_2_polynomial_curves, vizualize_road_equirectangular
from src.utils.TransformClass import Transform
from src.utils.path_utils import get_ouput_path



if __name__ == '__main__':
    img_left_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_EAC.png'
    img_left = cv2.imread(img_left_path, cv2.IMREAD_COLOR)

    img_right_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_D_2_EAC.png'
    img_right = cv2.imread(img_right_path, cv2.IMREAD_COLOR)

    if img_left is None:
        print(f"Error: Unable to load image at {img_left_path}")
        exit(1)

    height, width = img_left.shape[:2]

    max_width = width
    max_height = int(height/width*max_width)

    img_left = cv2.resize(img_left, (max_width, max_height))
    height, width = img_left.shape[:2]

    window_left=0.2
    window_right = 0.8
    window_top = 0.3
    window_bottom = 0.57
    debug = True
    camHeight = 1.65
    degree = 1

    # Attention window for segementation and road detection
    limit_left = int(window_left * width)
    limit_right = int(window_right * width)
    limit_top = int(window_top * height)
    limit_bottom = int(window_bottom * height)
    window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

    cam_left_transform = Transform(0., 0.0, 0.0, 0.0, 0.0, 0.0)
    cam_right_transform = Transform(1.12, 0.0, 0.0, 0.0, 0.0, 0.0)

    initial_guess = Road_line_params(0.1,-2.5,camHeight)

    
    #processing

    roadSegmentator = SegFormerRoadSegmentator(kernel_width=10, use_1024=True, debug=True)
    roadDetector = EquirectMonoRoadDetector(roadSegmentator=roadSegmentator,window=window,road_down_y=camHeight, degree=degree, debug=False)
    
    cv2.imwrite(get_ouput_path( "window_left.png"),window.crop_image(img_left))
    cv2.imwrite(get_ouput_path( "window_right.png"),window.crop_image(img_right))
    
    left_countours=roadDetector._get_road_contours(img_left)
    right_countours=roadDetector._get_road_contours(img_right)
    left_contour = max(left_countours, key=cv2.contourArea)
    right_contour = max(right_countours, key=cv2.contourArea)
    
    left_contour_points = left_contour[:, 0, :]
    left_contour_x = left_contour_points[:, 0]
    left_contour_y = left_contour_points[:, 1]
    left_poly_model_left, left_poly_model_right, left_y_inliers_left, left_y_inliers_right = find_best_2_polynomial_curves(left_contour,degree=degree)
    
    right_contour_points = right_contour[:, 0, :]
    right_contour_x = right_contour_points[:, 0]
    right_contour_y = right_contour_points[:, 1]
    right_poly_model_left, right_poly_model_right, right_y_inliers_left, right_y_inliers_right = find_best_2_polynomial_curves(right_contour,degree=degree)
    
    min_y_inliers_first = np.min(left_y_inliers_left)
    min_y_inliers_second = np.min(left_y_inliers_right)
    max_y_inliers_first = np.max(left_y_inliers_left)
    max_y_inliers_second = np.max(left_y_inliers_right)

    minY = max(min_y_inliers_first, min_y_inliers_second)
    maxY = min(max_y_inliers_first, max_y_inliers_second)
    #maxY = 1534
    minY = 1480
    maxY = 1580

    left_contour_left=[]
    left_contour_right=[]
    right_contour_left=[]
    right_contour_right=[]

    for y1 in range(minY, maxY,2):
        left_x_left = int(left_poly_model_left.predict([[y1]])[0])
        left_x_right = int(left_poly_model_right.predict([[y1]])[0])
        left_contour_left.append((left_x_left, int(y1)))
        left_contour_right.append((left_x_right, int(y1)))

        right_x_left = int(right_poly_model_left.predict([[y1]])[0])
        right_x_right = int(right_poly_model_right.predict([[y1]])[0])
        right_contour_left.append((right_x_left, int(y1)))
        right_contour_right.append((right_x_right, int(y1)))
        

    #left_contour_left, left_contour_right = roadDetector.get_left_right_contours(img_left)
    #right_contour_left, right_contour_right = roadDetector.get_left_right_contours(img_right)

    for p in left_contour_left:
        cv2.circle(img_left, p, 2, (255, 0, 0), -1)
    for p in right_contour_left:
        cv2.circle(img_right, p, 2, (255, 0, 0), -1)
    
    cv2.imwrite(get_ouput_path("superleft.png"),img_left)
    vizualize_road_equirectangular(initial_guess, cam_left_transform, img_left,'initleft')
    vizualize_road_equirectangular(initial_guess, cam_right_transform, img_right, "initright")
    
    fitted_params = roadDetector.fit_road_curve_to_left_right_contours(left_contour_left,right_contour_left,initial_guess=initial_guess, camRight=cam_right_transform,image_width=width, image_height=height)
    print(fitted_params)
    vizualize_road_equirectangular(fitted_params, cam_left_transform, img_left,"left")
    vizualize_road_equirectangular(fitted_params, cam_right_transform, img_right, "right")


