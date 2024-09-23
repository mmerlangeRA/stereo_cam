# from bootstrap import set_paths
# set_paths()
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.road_detection.common import AttentionWindow
from src.calibration.StereoCalibrator import StereoFullCalibration
from src.utils.image_processing import colorize_disparity_map, detect_sign, reshapeToWindow
from src.depth_estimation.depth_estimator import InputPair
from src.depth_estimation.selective_igev import Selective_igev
from src.utils.disparity import compute_3d_position_from_disparity
from src.utils.path_utils import get_static_folder_path


imgL_path=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_left.png'
imgR_path=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_right.png'

imgL_path=r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_rectified_left.png"
imgR_path=r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_rectified_right.png"

calibration_path = r'C:\Users\mmerl\projects\stereo_cam\calibration\calibrator_matrix.json'
imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)

window_left=0.1
window_right=0.9
window_top=0.35
window_bottom=0.6

debug=True

height,width = imgL.shape[:2]

limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

imgL = reshapeToWindow(imgL,window=window,max_width=width)
imgR = reshapeToWindow(imgR, window=window,max_width=width)

cv2.imwrite(get_static_folder_path("imgL.png"), imgL)


if imgL is None:
    print(f"Error: Unable to load image at {imgL_path}")
    exit(1)
if imgR is None:
    print(f"Error: Unable to load image at {imgR}")
    exit(1)

calibration = StereoFullCalibration.from_json (open(calibration_path, 'r').read())


print("debug",debug) 
def compute_sign_size(imgL,imgR,calibration:StereoFullCalibration,sign_window:AttentionWindow,nb_sides:int,index:int,debug=True):
    test_igev = Selective_igev(None, None)
    input_pair = InputPair(left_image=imgL, right_image=imgR, status="started", calibration=calibration)
    stereo_output = test_igev.compute_disparity(input_pair)
    disparity_map = stereo_output.disparity_pixels

    K = calibration.stereo_rectified_K
    if K is None or len(K) == 0:
        raise ValueError("no calibration data")

    fx = K[0][0]
    fy = K[1][1]
    baseline = -calibration.stereo_rectified_tvec[0][0]
    print("baseline",baseline)
    c_x = K[0][2]
    c_y = K[1][2]
    z0= calibration.stereo_rectified_Z0

    if debug:
        cv2.imwrite(get_static_folder_path(f"disparity_map{index}.png"), disparity_map)
        colorized_disparity_map = colorize_disparity_map(disparity_map)
        cv2.imwrite(get_static_folder_path(f"colorized_disparity_map{index}.png"), colorized_disparity_map)

    detected_sign,shape_points, disparities = detect_sign(disparity_map, sign_window,nb_sides=nb_sides)

    if debug:
        #print(shape_points)
        for point in shape_points:
            cv2.circle(imgL, tuple(point), 5, (0, 255, 0), -1)  # Draw green circles on the refined corners
        cv2.imwrite(get_static_folder_path(f"imgL{index}.png"), imgL)
        cv2.imwrite(get_static_folder_path(f"detected_sign{index}.png"), detected_sign)

    #get top left and bottom right points of the quadrilateral
    top_left = shape_points[0] 
    bottom_left = shape_points[3]

    p1 = compute_3d_position_from_disparity(top_left[0], top_left[1], disparities[0], fx, fy,c_x, c_y, baseline,z0)
    p2 = compute_3d_position_from_disparity(bottom_left[0], bottom_left[1], disparities[3], fx, fy,c_x, c_y, baseline,z0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    print("disparities",disparities[0],disparities[3])
    size = np.linalg.norm(p1 - p2)

    return p1, p2, size



sign_array=[
    [0,760,771,706,717,4],
    [-1,954,1038,133,218,4],
    [2,1240,1326,624,714,6],
    [3,836,866,730,757,4],
    [4,860,900,744,778,4],
    [13,954,1038,133,218,4],
    [9,838,871,733,766,0],
]


results=[]
for sign_info in sign_array:
    index,left,right,top,bottom,nb_sides = sign_info
    if index == -1:
        continue
    sign_window = AttentionWindow(left, right, top, bottom)
    imgL_path=fr"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\{index}_rectified_left.png"
    imgR_path=fr"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\{index}_rectified_right.png"
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    p1, p2, size = compute_sign_size(imgL, imgR, calibration,sign_window=sign_window, nb_sides=nb_sides, index=index, debug=debug)
    print(index,nb_sides, size)
    print("p1", p1)
    print("p2", p2)
    results.append((index, nb_sides, size))

df = pd.DataFrame(results, columns=['index', 'nb_sides', 'size'])
df.to_csv(get_static_folder_path("sign_sizes.csv"), index=False)

