import argparse
from bootstrap import set_paths
set_paths()
import os
import cv2
from src.road_detection.main import get_road_edges_from_stereo_cubes
from src.calibration.stereo_standard_refinement import compute_auto_calibration_for_2_stereo_standard_images
from src.depth_estimation.depth_estimator import Calibration

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Road Detection Script with Polynomial Curve Fitting')

    parser.add_argument('--img_left_path', type=str, required=True, help='Path to the left input image.')
    parser.add_argument('--img_right_path', type=str, required=True, help='Path to the right input image.')
    parser.add_argument('--calibration_path', type=str, required=False, help='Path to the calibration file.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    imgL_path = args.img_left_path
    imgR_path = args.img_right_path
    imgL = cv2.imread(imgL_path,cv2.IMREAD_COLOR)
    imgR = cv2.imread(imgR_path,cv2.IMREAD_COLOR)
    if imgL is None:
        print(f"Error: Unable to load image at {args.img_left_path}")
        exit(1)
    if imgR is None:
        print(f"Error: Unable to load image at {args.img_right_path}")
        exit(1)
    # we resize for fast processing and ensure it's multiple of 8
    height,width = imgL.shape[:2]
    new_width = 640
    new_height = int((height / width) * new_width)
    imgL = cv2.resize(imgL, (new_width, new_height))
    imgR = cv2.resize(imgR, (new_width, new_height))
    imgL = imgL[:480, :]
    imgR = imgR[:480, :]
    #cv2.imshow('imgL', imgL)
    #cv2.imshow('imgR', imgR)

    calibration_path = args.calibration_path
    if args.calibration_path and os.path.exists(calibration_path) :
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()