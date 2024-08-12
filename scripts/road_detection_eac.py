from bootstrap import set_paths
set_paths()
import cv2
from matplotlib import pyplot as plt
import numpy as np
from src.road_detection.seg_former import seg_segment_image
from src.road_detection.main import compute_road_width_from_eac
from src.road_detection.common import AttentionWindow
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Road Detection Script with Polynomial Curve Fitting')

    parser.add_argument('--img_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--camHeight', type=float, default=1.75, help='Camera height in meters.')
    parser.add_argument('--degree', type=int, default=1, help='Degree of the polynomial for curve fitting.')
    parser.add_argument('--max_width', type=int, default=2048, help='Max image width for resizing.')
    parser.add_argument('--window_left', type=float, default=0.4, help='Attention window left.')
    parser.add_argument('--window_right', type=float, default=0.6, help='Attention window top.')
    parser.add_argument('--window_top', type=float, default=0.3, help='Attention window top.')
    parser.add_argument('--window_bottom', type=float, default=0.6, help='Attention window bottom.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: Unable to load image at {args.img_path}")
        exit(1)

    height, width = img.shape[:2]

    max_width = args.max_width
    max_height = int(height/width*max_width)

    img = cv2.resize(img, (max_width, max_height))
    height, width = img.shape[:2]

    # Attention window for segementation and road detection
    limit_left = int(args.window_left * width)
    limit_right = int(args.window_right * width)
    limit_top = int(args.window_top * height)
    limit_bottom = int(args.window_bottom * height)
    window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)
    print(window)
    # Other params
    degree = args.degree
    camHeight = args.camHeight

    #processing
    average_width,first_poly_model, second_poly_model,x,y = compute_road_width_from_eac(img,window,camHeight=1.65,degree=1,debug=True)

    # Debug infos
    # Generate y values for plotting the polynomial curves
    y_range = np.linspace(np.min(y), np.max(y), 500)

    # Predict x values using the polynomial models
    x_first_poly = first_poly_model.predict(y_range[:, np.newaxis])
    x_second_poly = second_poly_model.predict(y_range[:, np.newaxis])

    # Plot the polynomial curves on the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x_first_poly, y_range, color='red', linewidth=2, label='First Polynomial')
    plt.plot(x_second_poly, y_range, color='blue', linewidth=2, label='Second Polynomial')
    plt.scatter(x, y, color='yellow', s=5, label='Contour Points')
    plt.legend()
    plt.title('Polynomial Curves Fit to Contour Points')
    plt.savefig(r'C:\Users\mmerl\projects\stereo_cam\output\polynomial_fit.png')
    plt.show()
    
    
    print("average width",average_width)

