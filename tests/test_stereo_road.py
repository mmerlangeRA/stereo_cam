from bootstrap import set_paths
set_paths()
import cv2
from matplotlib import pyplot as plt
import numpy as np
from src.road_detection.RoadSegmentator import PIDNetRoadSegmentator, SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EquirectMonoRoadDetector, StereoRoadDetector
from src.road_detection.common import AttentionWindow
from src.calibration.cube.StereoCalibrator import StereoFullCalibration
from src.utils.image_processing import reshapeToWindow

imgL_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_rectified_left.png'
imgR_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_rectified_right.png'

imgL_path=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_left.png'
imgR_path=r'C:\Users\mmerl\projects\stereo_cam\Photos\kitti\kitti_000046_right.png'

imgL_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_left.png'
imgR_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_right.png'

imgL_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_left.png'
imgR_path = r'C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\13_right.png'

imgL_path=r'C:\Users\mmerl\projects\stereo_cam\static\photos\13_rectified_left.jpg'
imgR_path=r'C:\Users\mmerl\projects\stereo_cam\static\photos\13_rectified_right.jpg'

calibration_path = r'C:\Users\mmerl\projects\stereo_cam\calibration\calibrator_matrix.json'
imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)

window_left=0.1
window_right=0.9
window_top=0.5
window_bottom=0.75

height,width = imgL.shape[:2]

limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

imgL = reshapeToWindow(imgL,window=window,max_width=1280)
imgR = reshapeToWindow(imgR, window=window,max_width=1280)

height,width = imgL.shape[:2]
segformer= True
segformer_1024=True

window_left=0.1
window_right=0.9
window_top=0.5
window_bottom=0.75

window_left=0.
window_right=1.
window_top=0.
window_bottom=1.
debug = True

if imgL is None:
    print(f"Error: Unable to load image at {imgL_path}")
    exit(1)
if imgR is None:
    print(f"Error: Unable to load image at {imgR}")
    exit(1)

calibration = StereoFullCalibration.from_json (open(calibration_path, 'r').read())


#cv2.imshow('imgL', imgL)
#cv2.imshow('imgR', imgR)
img = np.hstack((imgL, imgR))

#processing
if segformer:
    roadSegmentator = SegFormerRoadSegmentator(kernel_width=10, use_1024=segformer_1024, debug=debug)
else:
    roadSegmentator = PIDNetRoadSegmentator(kernel_width=10,debug=debug)

limit_left = int(window_left * width)
limit_right = int(window_right * width)
limit_top = int(window_top * height)
limit_bottom = int(window_bottom * height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)
print(window)

print("debug",debug)

roadDetector = StereoRoadDetector(roadSegmentator=roadSegmentator,window=window,calibration=calibration, debug=debug)
average_width, first_poly_model, second_poly_model, x, y = roadDetector.compute_road_width(img)

if debug:
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
