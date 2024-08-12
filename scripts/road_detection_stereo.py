import argparse

from matplotlib import pyplot as plt
import numpy as np
from bootstrap import set_paths
set_paths()
import os
import cv2
from src.utils.path_utils import get_calibration_folder_path
from src.road_detection.main import compute_road_width_from_stereo_cubes
from src.calibration.stereo_standard_refinement import compute_auto_calibration_for_2_stereo_standard_images
from src.depth_estimation.depth_estimator import Calibration

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Road Detection Script with Polynomial Curve Fitting')

    parser.add_argument('--img_left_path', type=str, required=True, help='Path to the left input image.')
    parser.add_argument('--img_right_path', type=str, required=True, help='Path to the right input image.')
    parser.add_argument('--degree', type=int, default=1, help='Degree of the polynomial for curve fitting.')
    parser.add_argument('--calibration_path', type=str, required=False, help='Path to the calibration file.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    imgL_path = args.img_left_path
    imgR_path = args.img_right_path
    imgL = cv2.imread(imgL_path,cv2.IMREAD_COLOR)
    imgR = cv2.imread(imgR_path,cv2.IMREAD_COLOR)
    height,width = imgL.shape[:2]
    new_width = 640
    ratio = new_width/width
    trunc_height = 480

    if imgL is None:
        print(f"Error: Unable to load image at {args.img_left_path}")
        exit(1)
    if imgR is None:
        print(f"Error: Unable to load image at {args.img_right_path}")
        exit(1)
    calibration_path = args.calibration_path
    if args.calibration_path and os.path.exists(calibration_path) :
        calibration = Calibration.from_json (open(calibration_path, 'r').read())
    else:
        height, width = imgL.shape[:2]
        K,cost,refined_rvec,refined_tvec =compute_auto_calibration_for_2_stereo_standard_images(imgL,imgR)
        print("cost",cost)
        fx = K[0, 0]
        fy = K[1, 1]
        cx0 = K[0, 2]
        cx1 = cx0  # Assume both cameras share the same cx if not specified
        cy = K[1, 2]
        # we resize for fast processing and ensure it's multiple of 8
        
        fx = fx * ratio
        fy = fy * ratio
        cx0 = cx0 * ratio
        cx1 = cx1 * ratio
        cy = cy * ratio
        cy = trunc_height/2.
        calibration = Calibration(
            width=new_width,
            height=trunc_height,
            fx=fx,
            fy=fy,
            cx0=cx0,
            cx1=cx1,
            cy=cy,
            baseline_meters=1.12
        )
        calibration_path = args.calibration_path if args.calibration_path else get_calibration_folder_path('calibration.json')
        open(calibration_path, 'w').write(calibration.to_json())
        print("calibration", calibration)
    new_height = int((height / width) * new_width)
    imgL = cv2.resize(imgL, (new_width, new_height))
    imgR = cv2.resize(imgR, (new_width, new_height))
    imgL = imgL[:trunc_height, :]
    imgR = imgR[:trunc_height, :]
    
    #cv2.imshow('imgL', imgL)
    #cv2.imshow('imgR', imgR)

    
    #processing
    average_width,first_poly_model, second_poly_model,x,y = compute_road_width_from_stereo_cubes(imgL,imgR, calibration=calibration,degree=args.degree, debug=True)

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
