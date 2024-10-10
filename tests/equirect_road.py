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
roadDetector.set_road_vector_and_bounds(road_width=6.3,road_transform=Transform(0.,camHeight,0.,0.,0.,0.))
print(roadDetector.lower_bounds,roadDetector.upper_bounds)
road_width,cost = roadDetector.compute_road_width(img)
print("cost",cost)    
end_time = time.time()
print("Time taken for road detection: ", end_time - start_time, "seconds")

print("road_width", road_width)

# NB on D_P5_CAM_G_1_EAC, 
# 668-444 =>5.5
# (738 - 395)/(668-444)*5.5=>8.25
'''
6.
[ 6.02785584e+00  3.77175554e-03  1.84996809e+00 -5.19450809e-03
  1.39877531e-02  2.78473262e-03] 
cost 0.0001886373501555088
Time taken for road detection:  33.07236886024475 seconds
road_width 6.027855837782305

3.

[ 3.         -4.          1.849      -0.34906585 -0.6981317  -0.34906585] 
[10.          4.          1.851       0.34906585  0.6981317   0.34906585]
[ 3.67786884  0.22480434  1.84940468 -0.04085067  0.13056794 -0.01028536] 
width cost per point 0.01675100131850772

6.
[ 6.03692622e+00 -7.61098495e-04  1.84995332e+00  2.26052163e-03
  1.59974894e-02  1.35459889e-02] width cost per point 0.0011826671917053628

[ 6.50334814e+00 -2.26646954e-02  1.84998832e+00  6.45369679e-03
  2.89218126e-03  1.71825735e-02] width cost per point 0.00048516911327349066

 [ 6.31359983e+00 -1.65056198e-02  1.84997757e+00  4.24037166e-03
  7.64424067e-03  1.48226552e-02] width cost per point 0.000576670602025873

'''