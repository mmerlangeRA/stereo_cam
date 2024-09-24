# from bootstrap import set_paths
# set_paths()
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.road_detection.common import AttentionWindow
from src.calibration.StereoCalibrator import StereoFullCalibration
from src.utils.path_utils import get_static_folder_path
from src.utils.cube_image import get_cube_front_image
from test_stereo_sign import compute_sign_size


pathL=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_left.png'
pathR=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_right.png'

calibration_path = r'C:\Users\mmerl\projects\stereo_cam\calibration\calibrator_matrix_003.json'
imgL = cv2.imread(pathL)
imgR = cv2.imread(pathR)


calibration = StereoFullCalibration.from_json (open(calibration_path, 'r').read())


window_left=0.0
window_right=1.0
window_top=0.
window_bottom=1.0

debug=True

height,width = imgL.shape[:2]

limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
attentionWindow = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom,False)

sign_array=[
    [100,147,170,186,210,4],
]

index,left, right, top, bottom,nb_sides=sign_array[0]
sign_window = AttentionWindow(left, right, top, bottom,False)


p1, p2, size = compute_sign_size(imgL, imgR, calibration,attentionWindow=attentionWindow,sign_window=sign_window, nb_sides=nb_sides, index=index, debug=debug)
