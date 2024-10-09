from bootstrap import set_paths
set_paths()
import cv2
import numpy as np
import time

from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EquirectMonoRoadDetector, EquirectStereoRoadDetector
from src.road_detection.common import AttentionWindow
from src.utils.path_utils import get_output_path

from src.utils.TransformClass import Transform

img_left_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_1_EAC.png'
img_right_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_D_1_EAC.png'
img_left = cv2.imread(img_left_path, cv2.IMREAD_COLOR)
img_right = cv2.imread(img_right_path, cv2.IMREAD_COLOR)

height, width = img_left.shape[:2]

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

cv2.imwrite(get_output_path('road_window_left.png'), window.crop_image(img_left))
cv2.imwrite(get_output_path('road_window_right.png'), window.crop_image(img_right))
initialization_time = time.time()
roadSegmentator = SegFormerRoadSegmentator(kernel_width=20, use_1024=False, debug=debug)
end_time = time.time()
print("Time taken for segmentation initialization: ", end_time - initialization_time, "seconds")

start_time = time.time()
estimated_cam_transform = Transform(1.12,0.,0., 0.,0.,0)
roadDetector = EquirectStereoRoadDetector(roadSegmentator=roadSegmentator,
                                          window=window,road_down_y=camHeight, 
                                          degree=degree, road_contour_top=road_window_top,
                                          estimated_cam2_transform=estimated_cam_transform,
                                        debug=debug)
concatenated_horizontal_img = np.concatenate((img_left, img_right), axis=1)
road_width = roadDetector.compute_road_width(concatenated_horizontal_img)
end_time = time.time()
print("Time taken for road detection: ", end_time - start_time, "seconds")

print("road_width", road_width)

# NB on D_P5_CAM_G_1_EAC, 
# 668-444 =>5.5
# (738 - 395)/(668-444)*5.5=>8.25