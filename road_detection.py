import os
import cv2
from src.road_detection.main import get_road_edges


if __name__ == '__main__':
    folder=r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE'
    imgL_path = os.path.join(folder,'13_rectified_left.jpg')
    imgR_path = os.path.join(folder,'13_rectified_right.jpg')
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    get_road_edges(imgL, imgR)