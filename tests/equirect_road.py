import torch
from bootstrap import set_paths
set_paths()
import cv2

from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.equirect_mono_road_detector import EquirectMonoRoadDetector, Transform
from src.road_detection.common import AttentionWindow
from src.utils.path_utils import get_output_path
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("device", device)

img_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_EAC.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

height, width = img.shape[:2]

window_left=0.4
window_right = 0.6
window_top = 0.4
window_bottom = 0.6
debug = True
camHeight = 1.85
polynomial_degree= 1

# Attention window for segementation and road detection
limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

cv2.imwrite(get_output_path('road_window.png'), window.crop_image(img))

initialization_time = time.time()
roadSegmentator = SegFormerRoadSegmentator(kernel_width=20, use_1024=False, debug=debug)
end_time = time.time()
print("Time taken for segmentation initialization: ", end_time - initialization_time, "seconds")

start_time = time.time()
roadDetector = EquirectMonoRoadDetector(roadSegmentator=roadSegmentator,window=window, degree=polynomial_degree,debug=debug)
roadDetector.set_road_vector_and_bounds(road_width=6.,road_transform=Transform(0.,camHeight,0.,0.,0.,0.))
road_width,optimized_transform = roadDetector.compute_road_width(img)
print("optimized_transform",optimized_transform)    
end_time = time.time()
print("Time taken for road detection: ", end_time - start_time, "seconds")

print("road_width", road_width)

# NB on D_P5_CAM_G_1_EAC, 
# 668-444 =>5.5
# (738 - 395)/(668-444)*5.5=>8.25