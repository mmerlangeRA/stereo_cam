import argparse

from matplotlib import pyplot as plt
import numpy as np
from bootstrap import set_paths
set_paths()
import cv2
from src.road_detection.common import AttentionWindow
from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.RoadDetector import StereoRoadDetector
from src.calibration.StereoCalibrator import StereoFullCalibration

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Road Detection Script with Polynomial Curve Fitting')

    parser.add_argument('--img_left_path', type=str, required=True, help='Path to the left input image.')
    parser.add_argument('--img_right_path', type=str, required=True, help='Path to the right input image.')
    parser.add_argument('--window_left', type=float, default=0.1, help='Attention window left.')
    parser.add_argument('--window_right', type=float, default=0.9, help='Attention window top.')
    parser.add_argument('--window_top', type=float, default=0.5, help='Attention window top.')
    parser.add_argument('--window_bottom', type=float, default=0.75, help='Attention window bottom.')
    parser.add_argument('--degree', type=int, default=1, help='Degree of the polynomial for curve fitting.')
    parser.add_argument('--calibration_path', type=str, required=False, help='Path to the calibration file.')
    parser.add_argument('--debug', type=bool, default=True, help='Show debug info.')
    parser.add_argument('--segformer', type=bool, default=True, help='Use Segfomer.')
    parser.add_argument('--segformer_1024', type=bool, default=False, help='Use Segfomer with 1024 model.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    imgL_path = args.img_left_path
    imgR_path = args.img_right_path
    imgL = cv2.imread(imgL_path,cv2.IMREAD_COLOR)
    imgR = cv2.imread(imgR_path,cv2.IMREAD_COLOR)
    height,width = imgL.shape[:2]


    if imgL is None:
        print(f"Error: Unable to load image at {args.img_left_path}")
        exit(1)
    if imgR is None:
        print(f"Error: Unable to load image at {args.img_right_path}")
        exit(1)

    calibration_path = args.calibration_path
    calibration = StereoFullCalibration.from_json (open(calibration_path, 'r').read())
   

    #cv2.imshow('imgL', imgL)
    #cv2.imshow('imgR', imgR)
    img = np.hstack((imgL, imgR))
    
    #processing
    if args.segformer:
        roadSegmentator = SegFormerRoadSegmentator(kernel_width=10, use_1024=args.segformer_1024, debug=args.debug)
    else:
        roadSegmentator = PIDNetRoadSegmentator(kernel_width=10,debug=args.debug)

    limit_left = int(args.window_left * width)
    limit_right = int(args.window_right * width)
    limit_top = int(args.window_top * height)
    limit_bottom = int(args.window_bottom * height)
    window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)
    print(window)

    print("debug",args.debug)

    roadDetector = StereoRoadDetector(roadSegmentator=roadSegmentator,window=window,calibration=calibration, debug=args.debug)
    average_width, first_poly_model, second_poly_model, x, y = roadDetector.compute_road_width(img)

    if args.debug:
        # Debug infos
        # Generate y values for plotting the polynomial curves
        y_range = np.linspace(np.min(y), np.max(y), 500)

        # Predict x values using the polynomial models
        x_first_poly = first_poly_model.predict(y_range[:, np.newaxis])
        x_second_poly = second_poly_model.predict(y_range[:, np.newaxis])

        # Plot the polynomial curves on the image
        plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
        plt.plot(x_first_poly, y_range, color='red', linewidth=2, label='First Polynomial')
        plt.plot(x_second_poly, y_range, color='blue', linewidth=2, label='Second Polynomial')
        plt.scatter(x, y, color='yellow', s=5, label='Contour Points')
        plt.legend()
        plt.title('Polynomial Curves Fit to Contour Points')
        plt.savefig(r'C:\Users\mmerl\projects\stereo_cam\output\polynomial_fit.png')
        plt.show()
    
    
    print("average width",average_width)
