import os
import cv2
from src.road_detection.main import get_road_edges_from_stereo_cubes
from src.calibration.stereo_standard_refinement import compute_auto_calibration_for_2_stereo_standard_images
from src.depth_estimation.depth_estimator import Calibration


if __name__ == '__main__':
    folder=r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE'
    imgL_path = os.path.join(folder,'13_rectified_left.jpg')
    imgR_path = os.path.join(folder,'13_rectified_right.jpg')
    imgL = cv2.imread(imgL_path,cv2.IMREAD_COLOR)
    imgR = cv2.imread(imgR_path,cv2.IMREAD_COLOR)
    height,width = imgL.shape[:2]
    new_width = 640
    new_height = int((height / width) * new_width)
    imgL = cv2.resize(imgL, (new_width, new_height))
    imgR = cv2.resize(imgR, (new_width, new_height))
    imgL = imgL[:480, :]
    imgR = imgR[:480, :]
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    print(imgL.shape)
    cv2.waitKey(0)
    calibration_path = r'C:\Users\mmerl\projects\stereo_cam\calibration\stereodemo_calibration.json'
    if os.path.exists(calibration_path) and False:
        calibration = Calibration.from_json (open(calibration_path, 'r').read())
    else:
        height, width = imgL.shape[:2]
        K,_,refined_rvec,refined_tvec =compute_auto_calibration_for_2_stereo_standard_images(imgL,imgR)
        fx = K[0, 0]
        fy = K[1, 1]
        cx0 = K[0, 2]
        cx1 = cx0  # Assume both cameras share the same cx if not specified
        cy = K[1, 2]

        calibration = Calibration(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx0=cx0,
            cx1=cx1,
            cy=cy,
            baseline_meters=1.12
        )
    get_road_edges_from_stereo_cubes(imgL, imgR,calibration)