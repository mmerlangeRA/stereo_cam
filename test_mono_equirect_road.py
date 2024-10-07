# from bootstrap import set_paths
# set_paths()
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EquirectRoadDetector
from src.road_detection.common import AttentionWindow
from src.utils.curve_fitting import find_best_2_polynomial_curves, fit_polynomial_ransac
from src.utils.coordinate_transforms import equirect_to_road_plane_points2D
from src.utils.path_utils import get_ouput_path
from src.utils.TransformClass import Transform
from scipy.optimize import least_squares

img_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_1_EAC.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

height, width = img.shape[:2]

window_left=0.4
window_right = 0.6
road_window_top = 0.53
window_top = 0.4
window_bottom = 0.6
debug = True
camHeight = 1.65
degree= 2

# Attention window for segementation and road detection
limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

cv2.imwrite(get_ouput_path('road_window.png'), window.crop_image(img))

roadSegmentator = SegFormerRoadSegmentator(kernel_width=20, use_1024=True, debug=debug)
roadDetector = EquirectRoadDetector(roadSegmentator=roadSegmentator,window=window,camHeight=camHeight, degree=degree, road_window_top=road_window_top,debug=debug)

road_width = roadDetector.compute_road_width(img)

print("road_width", road_width)

# 668-444 =>5.5
# (738 - 402)/(668-444)*5.5=>8.25